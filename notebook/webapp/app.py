"""VoxBind sampling-results browser.

Run with:
    streamlit run notebook/webapp/app.py
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data import (  # noqa: E402
    EXPS_ROOT,
    count_raw_samples,
    find_gt_sdf,
    find_pocket_pdb,
    list_experiments,
    list_targets,
    load_samples,
    mol_to_sdf_text,
    ngl_html,
    parse_pocket_pdb,
)
from metrics import (  # noqa: E402
    compute_target_metrics,
    load_metrics,
    metrics_are_fresh,
    metrics_path,
)


# ─── Streamlit config ─────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="VoxBind Results",
    initial_sidebar_state="expanded",
)


# ─── Cached helpers ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _file_text(path_str: str, _mtime: float) -> str:
    return Path(path_str).read_text()


@st.cache_data(show_spinner=False)
def _pocket_arr(path_str: str, _mtime: float):
    coords, elems = parse_pocket_pdb(Path(path_str))
    return coords, elems


@st.cache_data(show_spinner=False)
def _load_samples_cached(target_str: str, _mtime: float):
    mols = load_samples(Path(target_str))
    smis = [Chem.MolToSmiles(m) for m in mols]
    return mols, smis


def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _mol_to_pil(mol: Chem.Mol, size: tuple[int, int] = (380, 300)) -> Image.Image:
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


def _run_target_eval(target: Path):
    pdb = find_pocket_pdb(target)
    if pdb is None:
        st.error(f"No pocket PDB in {target.name}; cannot evaluate.")
        return
    coords, elems = parse_pocket_pdb(pdb)
    with st.spinner(f"Computing TargetDiff metrics for {target.name}…"):
        compute_target_metrics(target, coords, elems)
    st.cache_data.clear()
    st.rerun()


# ─── Sidebar: experiment → target → sample ────────────────────────────────────
st.sidebar.title("Browse")
st.sidebar.caption(f"`{EXPS_ROOT}`")

exps = list_experiments()
if not exps:
    st.sidebar.error("No experiments under voxbind/exps/.")
    st.stop()

exp_names = [e.name for e in exps]
default_exp = "260514_voxbind_100ep340"
default_idx = exp_names.index(default_exp) if default_exp in exp_names else 0
exp_name = st.sidebar.selectbox("Experiment", exp_names, index=default_idx)
exp = exps[exp_names.index(exp_name)]

targets = list_targets(exp)
if not targets:
    st.sidebar.warning("This experiment has no `target_*` folders yet.")
    st.stop()


def _target_label(i: int) -> str:
    t = targets[i]
    n_raw = count_raw_samples(t)
    cached = metrics_path(t).exists()
    marker = " ✓" if cached else ""
    return f"{t.name} ({n_raw}){marker}"


target_idx = st.sidebar.radio(
    "Target",
    range(len(targets)),
    format_func=_target_label,
)
target = targets[target_idx]
sdf_path = target / "samples.sdf"
sdf_mtime = _safe_mtime(sdf_path)

samples, sample_smis = _load_samples_cached(str(target), sdf_mtime)
if not samples:
    st.sidebar.info("No valid samples for this target.")
    sample_idx: int | None = None
else:
    sample_idx = st.sidebar.radio(
        "Sample",
        range(len(samples)),
        format_func=lambda i: (
            f"{i}: {sample_smis[i][:32]}{'…' if len(sample_smis[i]) > 32 else ''}"
        ),
    )

st.sidebar.divider()
st.sidebar.caption("Cached metrics: `target_*/metrics.json`")
if st.sidebar.button("Compute metrics for all targets"):
    pbar = st.sidebar.progress(0.0, text="Starting…")
    for i, t in enumerate(targets):
        pdb = find_pocket_pdb(t)
        if (t / "samples.sdf").exists() and pdb is not None:
            coords, elems = parse_pocket_pdb(pdb)
            try:
                compute_target_metrics(t, coords, elems)
            except Exception as e:  # noqa: BLE001
                st.sidebar.warning(f"{t.name}: {e}")
        pbar.progress((i + 1) / len(targets), text=f"{i+1}/{len(targets)}")
    st.sidebar.success("Done.")
    st.cache_data.clear()
    st.rerun()


# ─── Main area ────────────────────────────────────────────────────────────────
pocket_pdb = find_pocket_pdb(target)
gt_sdf = find_gt_sdf(target)

st.title(f"{exp.name} · {target.name}")
if pocket_pdb is None:
    st.error("No `*_pocket10.pdb` in this target directory.")
    st.stop()

pdb_mtime = _safe_mtime(pocket_pdb)
gt_mtime = _safe_mtime(gt_sdf) if gt_sdf else 0.0
pdb_text = _file_text(str(pocket_pdb), pdb_mtime)
gt_text = _file_text(str(gt_sdf), gt_mtime) if gt_sdf else ""
pocket_coords, pocket_elems = _pocket_arr(str(pocket_pdb), pdb_mtime)

header_bits = [
    f"Pocket `{pocket_pdb.name}` · {len(pocket_coords)} heavy atoms",
    f"GT `{gt_sdf.name}`" if gt_sdf else "no GT ligand",
    f"{len(samples)} valid / {count_raw_samples(target)} raw samples",
]
st.caption(" · ".join(header_bits))

metrics = load_metrics(target)
fresh = metrics_are_fresh(target)


# 3D viewer + per-sample info (side-by-side)
VIEWER_PX = 450  # square viewer edge length

if sample_idx is None:
    st.info("This target has no valid samples to visualise.")
else:
    sample_mol = samples[sample_idx]
    sample_text = mol_to_sdf_text(sample_mol)

    # Resolve cached per-sample metrics by index (fall back to SMILES match).
    sample_match = None
    if metrics is not None:
        if sample_idx < len(metrics.get("samples", [])):
            cand = metrics["samples"][sample_idx]
            if cand["smiles"] == sample_smis[sample_idx]:
                sample_match = cand
        if sample_match is None:
            sample_match = next(
                (s for s in metrics["samples"]
                 if s["smiles"] == sample_smis[sample_idx]),
                None,
            )

    viewer_col, info_col = st.columns([1, 1], gap="large")

    with viewer_col:
        has_gt = bool(gt_text)
        show_gt = st.toggle(
            "Show reference ligand",
            value=True,
            key="show_gt",
            disabled=not has_gt,
            help=(
                "Toggle the magenta GT ligand (camera framing always uses pocket + sample)"
                if has_gt else "No GT ligand for this target"
            ),
        )
        st.iframe(
            ngl_html(
                pdb_text, gt_text, sample_text,
                height=VIEWER_PX,
                show_gt=show_gt and has_gt,
            ),
            width=VIEWER_PX + 10,
            height=VIEWER_PX + 10,
        )
        st.caption(f"Sample {sample_idx} of {len(samples) - 1}")

    with info_col:
        # 2D depiction + SMILES, compact side-by-side
        d2_col, smi_col = st.columns([1, 1], gap="small")
        with d2_col:
            st.image(_mol_to_pil(sample_mol, size=(230, 190)))
        with smi_col:
            st.caption("SMILES")
            st.code(sample_smis[sample_idx], language="text")
            st.caption(f"{sample_mol.GetNumHeavyAtoms()} heavy atoms")

        st.markdown("##### TargetDiff metrics")
        if metrics is None:
            st.warning("Not yet evaluated.")
            if st.button("Evaluate this pocket", key="ev-info-1"):
                _run_target_eval(target)
        elif sample_match is None:
            st.info("Sample missing from cache.")
            if st.button("Recompute", key="ev-info-2"):
                _run_target_eval(target)
        else:
            if not fresh:
                st.warning("Cache may be stale.")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("QED", f"{sample_match['qed']:.3f}")
            m2.metric("SA", f"{sample_match['sa']:.2f}")
            m3.metric("LogP", f"{sample_match['logp']:.2f}")
            m4.metric("Lipinski", f"{sample_match['lipinski']}/5")
            m5.metric("HAtoms", sample_match["n_atoms"])

            st.markdown("##### Pocket-ligand contacts")
            ix = sample_match["interactions"]
            md = ix.get("min_dist")
            md_str = (
                f"{md:.2f}"
                if isinstance(md, (int, float)) and np.isfinite(md)
                else "—"
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("min Å", md_str)
            c2.metric("contacts <4Å", ix["n_contacts"])
            c3.metric(
                "clashes <2Å",
                ix["n_clashes"],
                delta=("⚠" if ix["n_clashes"] > 0 else None),
                delta_color="inverse",
            )
            if ix["closest"]:
                with st.expander(f"Top {len(ix['closest'])} closest atom pairs"):
                    st.dataframe(
                        pd.DataFrame(ix["closest"]),
                        hide_index=True,
                        height=min(280, 38 * (len(ix["closest"]) + 1)),
                    )


# Aggregate metrics
st.subheader("Per-target aggregates")
if metrics is None:
    st.warning("Not yet evaluated.")
    if st.button("Evaluate this pocket", key="ev-agg"):
        _run_target_eval(target)
else:
    agg = metrics["aggregates"]
    if agg.get("n_valid", 0) == 0:
        st.info(f"{agg.get('n_total', 0)} raw, 0 valid — no aggregates.")
    else:
        def _fmt(v, prec=3):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return "—"
            return f"{v:.{prec}f}"
        cols = st.columns(6)
        cols[0].metric("Validity", _fmt(agg.get("validity")),
                       help=f"{agg['n_valid']} valid / {agg['n_total']} raw")
        cols[1].metric("Uniqueness", _fmt(agg.get("uniqueness")))
        cols[2].metric("Diversity", _fmt(agg.get("diversity")))
        cols[3].metric("QED (mean)", _fmt(agg.get("qed_mean")))
        cols[4].metric("SA (mean)", _fmt(agg.get("sa_mean")))
        cols[5].metric("LogP (mean)", _fmt(agg.get("logp_mean"), 2))
    bar1, bar2 = st.columns([3, 1])
    bar1.caption(f"Computed: {metrics['computed_at']}"
                 + ("" if fresh else "  ·  ⚠ stale"))
    if bar2.button("Recompute", key="ev-recompute-agg"):
        _run_target_eval(target)
