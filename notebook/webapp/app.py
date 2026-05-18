"""VoxBind sampling-results browser.

Run with:
    streamlit run notebook/webapp/app.py
"""

from __future__ import annotations

import sys
from datetime import datetime
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
    list_sample_runs,
    list_targets,
    load_gt_mols,
    load_samples,
    mol_to_sdf_text,
    ngl_html,
    parse_pocket_pdb,
)
from metrics import (  # noqa: E402
    DOCKING_MODES,
    compute_reference,
    compute_target_metrics,
    docking_available,
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

# ─── Visual polish (CSS only — layout structure untouched) ────────────────────
st.markdown(
    """
    <style>
      /* tighter page margins */
      .block-container { padding-top: 2rem; padding-bottom: 2.5rem; }
      /* calmer, tighter headings */
      .block-container h1 { font-size: 1.55rem; font-weight: 600; margin-bottom: .1rem; }
      .block-container h2 { font-size: 1.15rem; font-weight: 600; margin: .5rem 0 .3rem; }
      .block-container h5 { font-size: .82rem; font-weight: 600; color: #555;
                            text-transform: uppercase; letter-spacing: .04em;
                            margin: .8rem 0 .15rem; }
      /* compact metrics */
      [data-testid="stMetricValue"] { font-size: 1.15rem; }
      [data-testid="stMetricLabel"] p { font-size: .72rem; }
      /* subtle captions */
      [data-testid="stCaptionContainer"] { color: #8a8a8a; font-size: .8rem; }
      /* tighter sidebar */
      section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
      section[data-testid="stSidebar"] h1 { font-size: 1.15rem; }
      /* trim vertical gaps between stacked elements */
      .block-container [data-testid="stVerticalBlock"] { gap: .55rem; }
    </style>
    """,
    unsafe_allow_html=True,
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


def _fmt(v, prec: int = 3) -> str:
    """Render a metric value, showing None/NaN/inf as an em dash."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{prec}f}"


def _delta_str(sample_val, ref_val, prec: int = 2) -> str | None:
    """A signed 'Δ vs ref' string for st.metric, or None if a value is missing.

    Pair with delta_color: "normal" for higher-is-better metrics (QED, SA,
    Lipinski), "inverse" for lower-is-better (Vina affinity), "off" otherwise.
    """
    if not isinstance(sample_val, (int, float)) or not isinstance(ref_val, (int, float)):
        return None
    return f"{sample_val - ref_val:+.{prec}f} vs ref"


# Per-metric "good direction" for the sample-vs-reference comparison table:
# +1 higher-is-better, -1 lower-is-better, 0 no preferred direction (uncoloured).
_METRIC_DIR = {
    "QED": 1, "SA": 1, "LogP": 0, "Lipinski": 1, "HAtoms": 0,
    "Vina Score": -1, "Vina Min": -1, "Vina Dock": -1,
}
_TABLE_FMT = {
    "QED": "{:.3f}", "SA": "{:.2f}", "LogP": "{:.2f}",
    "Lipinski": "{:.1f}", "HAtoms": "{:.0f}",
    "Vina Score": "{:.2f}", "Vina Min": "{:.2f}", "Vina Dock": "{:.2f}",
}


def _comparison_df(metrics: dict) -> pd.DataFrame:
    """Two-row comparison table: the reference ligand and the sample mean.

    One column per metric; the Vina Score / Vina Min / Vina Dock columns
    appear only when docking was run. The Mean row averages each column over
    the samples that have a value.
    """
    chem = (("QED", "qed"), ("SA", "sa"), ("LogP", "logp"),
            ("Lipinski", "lipinski"), ("HAtoms", "n_atoms"))
    vina = (("score_only", "Vina Score"), ("minimize", "Vina Min"),
            ("dock", "Vina Dock"))

    def _row(label: str, rec: dict) -> dict:
        row: dict = {"Ligand": label}
        for col, key in chem:
            row[col] = rec.get(key)
        v = rec.get("vina")
        if isinstance(v, dict) and "error" not in v:
            for key, col in vina:
                if isinstance(v.get(key), (int, float)):
                    row[col] = v[key]
        return row

    sample_rows = [_row(f"Sample {i}", s)
                   for i, s in enumerate(metrics.get("samples", []))]

    rows = []
    ref = metrics.get("reference")
    if isinstance(ref, dict) and "error" not in ref:
        rows.append(_row("Reference", ref))
    if sample_rows:
        metric_cols: list[str] = []
        for r in sample_rows:
            metric_cols += [c for c in r
                            if c != "Ligand" and c not in metric_cols]
        mean_row: dict = {"Ligand": "Mean"}
        for c in metric_cols:
            vals = [r[c] for r in sample_rows
                    if isinstance(r.get(c), (int, float))]
            if vals:
                mean_row[c] = sum(vals) / len(vals)
        rows.append(mean_row)
    return pd.DataFrame(rows)


def _style_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Colour the Mean row green/red vs the Reference row.

    The Reference row is shaded; the Mean row is bold.
    """
    css = pd.DataFrame("", index=df.index, columns=df.columns)
    if "Ligand" not in df.columns:
        return css
    mean_idx = set(df.index[df["Ligand"] == "Mean"])
    for mi in mean_idx:
        css.loc[mi, :] = "font-weight: 600"
    ref_hits = df.index[df["Ligand"] == "Reference"]
    if len(ref_hits) == 0:
        return css
    ri = ref_hits[0]
    css.loc[ri, :] = "background-color: #eef1f7; font-weight: 600"
    numeric = (int, float, np.number)
    for col in df.columns:
        direction = _METRIC_DIR.get(col, 0)
        if direction == 0:
            continue
        rv = df.at[ri, col]
        if not isinstance(rv, numeric) or pd.isna(rv):
            continue
        for idx in df.index:
            if idx == ri:
                continue
            v = df.at[idx, col]
            if not isinstance(v, numeric) or pd.isna(v):
                continue
            diff = (v - rv) * direction
            if diff > 0:
                color = "#d6efd9"
            elif diff < 0:
                color = "#f6d9da"
            else:
                continue
            weight = "; font-weight: 600" if idx in mean_idx else ""
            css.at[idx, col] = f"background-color: {color}{weight}"
    return css


def _mol_to_pil(mol: Chem.Mol, size: tuple[int, int] = (380, 300)) -> Image.Image:
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


def _run_target_eval(target: Path, docking: str = "none", exhaustiveness: int = 8):
    pdb = find_pocket_pdb(target)
    if pdb is None:
        st.error(f"No pocket PDB in {target.name}; cannot evaluate.")
        return
    coords, elems = parse_pocket_pdb(pdb)
    label = "metrics" if docking == "none" else f"metrics + {docking}"
    with st.spinner(f"Computing {label} for {target.name}…"):
        compute_target_metrics(target, coords, elems,
                               docking=docking, receptor_pdb=pdb,
                               exhaustiveness=exhaustiveness)
    st.cache_data.clear()
    st.rerun()


def _run_reference_eval(target: Path, docking: str = "none",
                        exhaustiveness: int = 8) -> None:
    """Recompute only the reference-ligand row of an existing metrics.json."""
    pdb = find_pocket_pdb(target)
    label = "reference ligand" if docking == "none" else f"reference + {docking}"
    try:
        with st.spinner(f"Computing {label} for {target.name}…"):
            compute_reference(target, docking=docking, receptor_pdb=pdb,
                              exhaustiveness=exhaustiveness)
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Reference compute failed — {e}")
        return
    st.cache_data.clear()
    st.rerun()


def _compute_all_targets(targets: list[Path], docking: str,
                         exhaustiveness: int) -> None:
    """Compute metrics.json for every target, with an inline progress bar."""
    pbar = st.progress(0.0, text="Starting…")
    for i, t in enumerate(targets):
        pdb = find_pocket_pdb(t)
        if (t / "samples.sdf").exists() and pdb is not None:
            coords, elems = parse_pocket_pdb(pdb)
            try:
                compute_target_metrics(t, coords, elems, docking=docking,
                                       receptor_pdb=pdb,
                                       exhaustiveness=exhaustiveness)
            except Exception as e:  # noqa: BLE001
                st.warning(f"{t.name}: {e}")
        pbar.progress((i + 1) / len(targets), text=f"{i + 1}/{len(targets)}")
    st.cache_data.clear()
    st.rerun()


def _compute_all_references(targets: list[Path], docking: str,
                            exhaustiveness: int) -> None:
    """Recompute the reference-ligand row for every target with a metrics.json.

    Targets without a metrics.json are skipped — compute_reference needs an
    existing one to merge into (run a target compute first).
    """
    eligible = [t for t in targets if metrics_path(t).exists()]
    if not eligible:
        st.warning("No targets have a metrics.json yet — run a target "
                   "compute first.")
        return
    pbar = st.progress(0.0, text="Starting…")
    for i, t in enumerate(eligible):
        pdb = find_pocket_pdb(t)
        try:
            compute_reference(t, docking=docking, receptor_pdb=pdb,
                              exhaustiveness=exhaustiveness)
        except Exception as e:  # noqa: BLE001
            st.warning(f"{t.name}: {e}")
        pbar.progress((i + 1) / len(eligible), text=f"{i + 1}/{len(eligible)}")
    st.cache_data.clear()
    st.rerun()


@st.dialog("Recompute this target?")
def _confirm_target_eval(target: Path, docking: str,
                         exhaustiveness: int) -> None:
    """Confirm before overwriting a target's existing metrics.json.

    Confirming stashes the request and reruns, so the compute runs in the main
    page flow — its spinner shows out in the page, not inside this modal.
    """
    st.warning(f"`{target.name}` already has cached metrics — recomputing "
               "overwrites its `metrics.json`.")
    if docking != "none":
        st.caption(f"Vina mode `{docking}` is selected; docking can be slow.")
    go, cancel = st.columns(2)
    if go.button("Recompute", type="primary", width="stretch"):
        st.session_state["_pending_compute"] = (
            "target", (target, docking, exhaustiveness))
        st.rerun()
    if cancel.button("Cancel", width="stretch"):
        st.rerun()


@st.dialog("Compute all targets?")
def _confirm_all_eval(targets: list[Path], docking: str,
                      exhaustiveness: int) -> None:
    """Confirm before a batch compute that would overwrite cached metrics.

    Confirming stashes the request and reruns, so the progress bar shows out in
    the main page rather than inside this modal.
    """
    n_cached = sum(1 for t in targets if metrics_path(t).exists())
    st.warning(f"This computes metrics for all {len(targets)} targets — "
               f"{n_cached} already cached will be overwritten.")
    if docking != "none":
        st.caption(f"Vina mode `{docking}` across all targets can be slow.")
    go, cancel = st.columns(2)
    if go.button("Compute all", type="primary", width="stretch"):
        st.session_state["_pending_compute"] = (
            "all", (targets, docking, exhaustiveness))
        st.rerun()
    if cancel.button("Cancel", width="stretch"):
        st.rerun()


# ─── Sidebar: experiment → target → sample ────────────────────────────────────
st.sidebar.title("Browse")
st.sidebar.caption(f"`{EXPS_ROOT}`")

# Force a re-scan without touching a selector. Directory listings refresh on
# every rerun and the file caches are mtime-keyed, but Streamlit only reruns on
# a widget event — this button is that event. Clearing the data cache also
# re-reads sample SDFs whose mtime didn't move (in-place writes, coarse NFS).
if st.sidebar.button(
    "🔄 Refresh results",
    width="stretch",
    help="Re-scan the experiments directory and reload samples from disk.",
):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"Scanned at {datetime.now():%H:%M:%S}")

exps = list_experiments()
if not exps:
    st.sidebar.error("No experiments under voxbind/exps/.")
    st.stop()

exp_names = [e.name for e in exps]
default_exp = "260514_voxbind_100ep340"
default_idx = exp_names.index(default_exp) if default_exp in exp_names else 0
exp_name = st.sidebar.selectbox("Experiment", exp_names, index=default_idx)
exp = exps[exp_names.index(exp_name)]

# sampling-run selector — an experiment can hold several runs under samples/
sample_runs = list_sample_runs(exp)
if not sample_runs:
    st.sidebar.warning("This experiment has no sampling runs yet.")
    st.stop()
# Default to the most recently created/modified run directory.
_samples_root = exp / "samples"


def _run_mtime(r: str) -> float:
    try:
        return (_samples_root / r).stat().st_mtime
    except OSError:
        return 0.0


default_run = max(sample_runs, key=_run_mtime)
run = st.sidebar.selectbox(
    "Sample run", sample_runs, index=sample_runs.index(default_run)
)

targets = list_targets(exp, run)
if not targets:
    st.sidebar.warning(f"Run `{run}` has no `target_*` folders yet.")
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
        format_func=lambda i: f"Sample {i}",
    )

st.sidebar.divider()
st.sidebar.caption("Cached metrics: `target_*/metrics.json`")

# Optional Vina docking — attaches vina_* fields to metrics.json on (re)compute.
# AutoDock Vina is CPU-only; vina_dock with high exhaustiveness can be slow.
docking_mode = st.sidebar.selectbox(
    "Vina docking", DOCKING_MODES, index=0,
    help="none = chemical metrics only · vina_score = + score_only · "
         "vina_min = + minimize · vina_dock = + full re-docking (slow). "
         "vina_dock yields all three paper scores.",
)
docking_exh = 8
if docking_mode != "none":
    docking_exh = st.sidebar.slider(
        "Exhaustiveness", 1, 32, 8,
        help="Vina search effort for vina_dock — higher is slower, more thorough.",
    )
    if not docking_available():
        st.sidebar.warning(
            "Vina toolchain not installed — docking records an error per "
            "sample. Install instructions are in the `metrics.py` docstring."
        )

if st.sidebar.button("Compute metrics for all targets", width="stretch"):
    if any(metrics_path(t).exists() for t in targets):
        _confirm_all_eval(targets, docking_mode, docking_exh)
    else:
        _compute_all_targets(targets, docking_mode, docking_exh)
if st.sidebar.button(f"Compute metrics for {target.name}", width="stretch"):
    if metrics_path(target).exists():
        _confirm_target_eval(target, docking_mode, docking_exh)
    else:
        _run_target_eval(target, docking_mode, docking_exh)
if st.sidebar.button("Compute reference ligand", width="stretch"):
    if metrics_path(target).exists():
        _run_reference_eval(target, docking_mode, docking_exh)
    else:
        st.sidebar.warning("Run a target compute first — the reference "
                           "is stored in its metrics.json.")
if st.sidebar.button("Compute reference ligand for all targets",
                     width="stretch"):
    if any(metrics_path(t).exists() for t in targets):
        st.session_state["_pending_compute"] = (
            "all_ref", (targets, docking_mode, docking_exh))
        st.rerun()
    else:
        st.sidebar.warning("Run a target compute first — the reference "
                           "is stored in each target's metrics.json.")


# ─── Main area ────────────────────────────────────────────────────────────────
pocket_pdb = find_pocket_pdb(target)
gt_sdf = find_gt_sdf(target)

st.title(f"{exp.name} · {target.name}")

# A (re)compute confirmed in a dialog runs here, in the main page flow, so its
# spinner / progress bar shows out in the page instead of inside the modal.
_pending = st.session_state.pop("_pending_compute", None)
if _pending is not None:
    _kind, _args = _pending
    if _kind == "target":
        _run_target_eval(*_args)
    elif _kind == "all":
        _compute_all_targets(*_args)
    elif _kind == "all_ref":
        _compute_all_references(*_args)

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
# The viewer stretches to its column width (responsive); only the height is
# fixed, since an iframe needs an explicit pixel height.
VIEWER_H = 720   # NGL viewer height

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

    viewer_col, info_col = st.columns([2, 1], gap="large")

    with viewer_col:
        st.iframe(
            ngl_html(pdb_text, gt_text, sample_text),
            width="stretch",
            height=VIEWER_H,
        )
        st.caption(
            f"Sample {sample_idx} of {len(samples) - 1}"
            "  ·  representation toggles in the bar above the viewer"
        )

    with info_col:
        # Generated vs reference — a 2-column grid: image row, then SMILES row.
        gt_mols = load_gt_mols(target)
        gen_col, ref_col = st.columns(2, gap="small")
        with gen_col:
            st.caption("Generated")
            st.image(_mol_to_pil(sample_mol, size=(220, 180)),
                     width="stretch")
            # wrap_lines=True → long SMILES wraps; st.code keeps a copy button.
            st.code(sample_smis[sample_idx], language="text", wrap_lines=True)
        with ref_col:
            if gt_mols:
                lbl = "Reference"
                if len(gt_mols) > 1:
                    lbl += f" (1 of {len(gt_mols)})"
                st.caption(lbl)
                st.image(_mol_to_pil(gt_mols[0], size=(220, 180)),
                         width="stretch")
                st.code(Chem.MolToSmiles(gt_mols[0]), language="text",
                        wrap_lines=True)
            else:
                st.caption("Reference")
                st.info("No reference ligand for this target.")

        st.subheader("Per-sample metrics")
        if metrics is None:
            st.warning("Not yet evaluated — use **Compute metrics** "
                       "in the sidebar.")
        elif sample_match is None:
            st.info("Sample missing from cache — recompute via "
                    "**Compute metrics** in the sidebar.")
        else:
            if not fresh:
                st.warning("Cache may be stale.")
            # Every tile carries a Δ-vs-reference badge (green = better).
            ref = metrics.get("reference")
            ref = ref if isinstance(ref, dict) and "error" not in ref else {}

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("QED", f"{sample_match['qed']:.3f}",
                      delta=_delta_str(sample_match["qed"], ref.get("qed"), 3),
                      delta_color="normal")
            m2.metric("SA", f"{sample_match['sa']:.2f}",
                      delta=_delta_str(sample_match["sa"], ref.get("sa"), 2),
                      delta_color="normal")
            m3.metric("LogP", f"{sample_match['logp']:.2f}",
                      delta=_delta_str(sample_match["logp"], ref.get("logp"), 2),
                      delta_color="off")
            m4.metric("Lipinski", f"{sample_match['lipinski']}/5",
                      delta=_delta_str(sample_match["lipinski"],
                                       ref.get("lipinski"), 0),
                      delta_color="normal")
            m5.metric("HAtoms", sample_match["n_atoms"],
                      delta=_delta_str(sample_match["n_atoms"],
                                       ref.get("n_atoms"), 0),
                      delta_color="off")

            vina = sample_match.get("vina")
            if vina is not None:
                st.markdown("##### Vina docking")
                if "error" in vina:
                    st.caption(f"⚠ docking failed — {vina['error']}")
                else:
                    # Lower (more negative) affinity is better → inverse colour.
                    ref_vina = ref.get("vina")
                    ref_vina = ref_vina if isinstance(ref_vina, dict) else {}
                    vc = st.columns(3)
                    for col, key in zip(vc, ("score_only", "minimize", "dock")):
                        col.metric(
                            key,
                            f"{vina[key]:.2f}" if key in vina else "—",
                            delta=_delta_str(vina.get(key), ref_vina.get(key), 2),
                            delta_color="inverse",
                        )

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


# Per-pocket metrics — set-level stats + the reference-vs-mean comparison
st.subheader("Per-pocket metrics")
if metrics is None:
    st.warning("Not yet evaluated — use **Compute metrics** in the sidebar.")
else:
    agg = metrics["aggregates"]
    if agg.get("n_valid", 0) == 0:
        st.info(f"{agg.get('n_total', 0)} raw, 0 valid — nothing to compare.")
    else:
        # Set-level metrics — no per-sample form, so they stay as a strip.
        scols = st.columns(6)
        scols[0].metric("Validity", _fmt(agg.get("validity")),
                        help=f"{agg['n_valid']} valid / {agg['n_total']} raw")
        scols[1].metric("Uniqueness", _fmt(agg.get("uniqueness")))
        scols[2].metric("Diversity", _fmt(agg.get("diversity")))

        # The reference ligand vs the sample mean. A Mean cell is green when
        # the sample mean beats the reference on that metric, red when worse;
        # the shaded row is the reference.
        st.markdown("##### Reference vs sample mean")
        cmp_df = _comparison_df(metrics)
        cmp_fmt = {k: v for k, v in _TABLE_FMT.items() if k in cmp_df.columns}
        st.dataframe(
            cmp_df.style.apply(_style_comparison, axis=None)
                  .format(cmp_fmt, na_rep="—"),
            width="stretch",
            hide_index=True,
            height=36 * (len(cmp_df) + 1) + 3,
        )
        st.caption("green = sample mean beats the reference · red = worse · "
                   "shaded row = reference ligand")
        ref = metrics.get("reference")
        if isinstance(ref, dict) and "error" in ref:
            st.caption(f"⚠ reference ligand not scored — {ref['error']}")
        elif ref is None and "reference" not in metrics:
            st.caption("⚠ no reference row — recompute this target to add it.")
        elif ref is None:
            st.caption("⚠ this target has no reference ligand.")

    st.caption(f"Computed: {metrics['computed_at']}"
               + ("" if fresh else "  ·  ⚠ stale"))
