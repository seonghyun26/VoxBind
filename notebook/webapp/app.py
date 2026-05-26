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
    cache_satisfies,
    compute_reference,
    compute_target_metrics,
    docking_available,
    load_metrics,
    metrics_are_fresh,
    metrics_path,
    ref_satisfies,
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
      /* Sidebar split into two side-by-side columns:
           - LEFT  (`sidebar-ctrl-col`): browsing context + compute section.
           - RIGHT (`sidebar-tgt-col`):  target + sample radios.
         Sidebar is widened to 520 px so the two columns have breathing
         room (default 336 px would crush selectbox labels). A 1 px hairline
         between the columns marks the boundary. The width override is
         scoped to `[aria-expanded="true"]` so Streamlit's collapse
         (translateX by its *stored* width — default 300 px — would otherwise
         leave 220 px of our 520 px sidebar visible). When collapsed we
         shrink to 0 width so the collapse is clean. Backgrounds inherit
         from the sidebar (Streamlit 1.57 dropped the bg CSS var, so we
         normalize to its actual default gray with a dark-mode fallback).
         Outer sidebar scrolling is disabled and replaced by per-column
         scrolling so users only ever see one scrollbar at a time. */
      section[data-testid="stSidebar"] {
        background-color: var(--secondary-background-color, #F0F2F6);
        transition: width 300ms, min-width 300ms !important;
      }
      section[data-testid="stSidebar"][aria-expanded="true"] {
        width: 520px !important;
        min-width: 520px !important;
      }
      section[data-testid="stSidebar"][aria-expanded="false"] {
        width: 0 !important;
        min-width: 0 !important;
        overflow: hidden;
      }
      section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        height: 100vh;
        overflow: hidden;
      }
      section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
      section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
      section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div,
      section[data-testid="stSidebar"] .st-key-sidebar-ctrl-col,
      section[data-testid="stSidebar"] .st-key-sidebar-tgt-col {
        background-color: inherit;
      }
      section[data-testid="stSidebar"] .st-key-sidebar-ctrl-col {
        padding-right: .5rem;
        border-right: 1px solid rgba(127,127,127,0.25);
        max-height: calc(100vh - 100px);
        overflow-y: auto;
      }
      section[data-testid="stSidebar"] .st-key-sidebar-tgt-col {
        padding-left: .5rem;
        max-height: calc(100vh - 100px);
        overflow-y: auto;
      }
      @media (prefers-color-scheme: dark) {
        section[data-testid="stSidebar"] {
          background-color: var(--secondary-background-color, #262730);
        }
      }
      /* Selectbox dropdown: let the popover grow to fit the longest option
         (one line per option, no ellipsis) so full experiment / sample-run
         names are readable. Width is driven by content, not the trigger. */
      [data-baseweb="popover"] ul[role="listbox"] { min-width: max-content; }
      [data-baseweb="popover"] li[role="option"] { white-space: nowrap; }
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


# Display formats for the comparison table's metric columns.
_TABLE_FMT = {
    "QED": "{:.3f}", "SA": "{:.2f}", "LogP": "{:.2f}",
    "Lipinski": "{:.1f}", "HAtoms": "{:.0f}", "Sim→Ref": "{:.3f}",
    "Vina Score": "{:.2f}", "Vina Min": "{:.2f}", "Vina Dock": "{:.2f}",
}

# Comparison-table cell backgrounds, per Streamlit theme. `summary` shades the
# Reference and Mean rows in the same dark gray so both summary rows stand
# apart from the per-sample rows below them. `selected` highlights the row
# matching the currently-picked sample in the sidebar.
_COMPARE_PALETTE = {
    "light": {"summary": "#c8c8c8", "selected": "#fce97a"},
    "dark":  {"summary": "#3a3f4b", "selected": "#6b5a1f"},
}


def _comparison_df(metrics: dict) -> pd.DataFrame:
    """Comparison table: the reference ligand, the sample mean, then every sample.

    One column per metric; the Vina Score / Vina Min / Vina Dock columns
    appear only when docking was run. The Mean row averages each column over
    the samples that have a value.
    """
    chem = (("QED", "qed"), ("SA", "sa"), ("LogP", "logp"),
            ("Lipinski", "lipinski"), ("HAtoms", "n_atoms"),
            ("Sim→Ref", "sim_to_ref"))
    vina = (("score_only", "Vina Score"), ("minimize", "Vina Min"),
            ("dock", "Vina Dock"))

    def _row(label: str, rec: dict) -> dict:
        row: dict = {"Ligand": label}
        for col, key in chem:
            row[col] = rec.get(key)
        if label == "Reference":
            row["Sim→Ref"] = 1.0  # the reference is identical to itself
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
    rows.extend(sample_rows)
    return pd.DataFrame(rows)


def _theme_is_dark() -> bool:
    """True when Streamlit's active theme is dark (defaults to light)."""
    try:
        return st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001
        return False


def _style_comparison(df: pd.DataFrame, palette: dict,
                      selected_idx: int | None = None) -> pd.DataFrame:
    """Shade Reference & Mean rows in the same gray; highlight the selected sample.

    `selected_idx` is the index of the sample currently chosen in the sidebar;
    the row labelled ``Sample {selected_idx}`` gets the highlight colour so the
    user can quickly find the row matching the 3-D view above.
    """
    css = pd.DataFrame("", index=df.index, columns=df.columns)
    if "Ligand" not in df.columns:
        return css
    summary_mask = df["Ligand"].isin(("Reference", "Mean"))
    css.loc[summary_mask, :] = (
        f"background-color: {palette['summary']}; font-weight: 600"
    )
    if selected_idx is not None:
        sel_mask = df["Ligand"] == f"Sample {selected_idx}"
        css.loc[sel_mask, :] = (
            f"background-color: {palette['selected']}; font-weight: 600"
        )
    return css


def _mol_to_pil(mol: Chem.Mol, size: tuple[int, int] = (380, 300)) -> Image.Image:
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


@st.dialog("Computing metrics…")
def _progress_dialog(kind: str, args: tuple) -> None:
    """Run a (re)compute inside a modal so the page dims behind a live bar.

    Streamlit dims the whole page whenever a dialog is open. The compute runs
    here synchronously; its progress callback streams per-molecule updates into
    the bar, and the dialog closes via st.rerun() once the compute finishes.

    `kind` selects the variant and dictates `args`:
      - "target":  (target, docking, exh, force)
      - "ref":     (target, docking, exh)
      - "all":     (targets, docking, exh, force, skip_existing)
      - "all_ref": (targets, docking, exh, skip_existing)
    """
    st.caption("Working — this window closes when the compute finishes.")

    if kind == "target":
        target, docking, exh, force = args
        st.markdown(f"##### {target.name}")
        bar = st.progress(0.0, text="Starting…")

        def _cb(done: int, total: int, label: str) -> None:
            frac = done / total if total else 1.0
            bar.progress(min(frac, 1.0), text=f"{label} — {done}/{total}")

        pdb = find_pocket_pdb(target)
        try:
            if pdb is None:
                st.error(f"No pocket PDB in {target.name}; cannot evaluate.")
                return
            coords, elems = parse_pocket_pdb(pdb)
            compute_target_metrics(target, coords, elems, docking=docking,
                                   receptor_pdb=pdb, exhaustiveness=exh,
                                   progress_cb=_cb, force=force)
        except Exception as e:  # noqa: BLE001
            st.error(f"Compute failed — {e}")
            return

    elif kind == "ref":
        target, docking, exh = args
        st.markdown(f"##### {target.name}")
        bar = st.progress(0.0, text="Starting…")

        def _cb(done: int, total: int, label: str) -> None:
            frac = done / total if total else 1.0
            bar.progress(min(frac, 1.0), text=f"{label} — {done}/{total}")

        pdb = find_pocket_pdb(target)
        try:
            compute_reference(target, docking=docking, receptor_pdb=pdb,
                              exhaustiveness=exh, progress_cb=_cb)
        except Exception as e:  # noqa: BLE001
            st.error(f"Compute failed — {e}")
            return

    else:  # "all" | "all_ref" — batch over every target
        if kind == "all":
            targets, docking, exh, force, skip_existing = args
            eligible = [t for t in targets
                        if (t / "samples.sdf").exists()
                        and find_pocket_pdb(t) is not None]
            empty_msg = "No targets have both samples.sdf and a pocket PDB."
            # Force overrides skip-existing (the whole point of Force is to
            # redo everything); otherwise filter out already-satisfied targets.
            if skip_existing and not force:
                eligible = [t for t in eligible
                            if not cache_satisfies(t, docking)]
                if not eligible:
                    empty_msg = ("All targets already satisfy the selected "
                                 "Vina mode — nothing to do (turn off 'Skip "
                                 "already-done' to recompute).")
        else:  # "all_ref" — compute_reference merges into existing metrics.json
            targets, docking, exh, skip_existing = args
            eligible = [t for t in targets if metrics_path(t).exists()]
            empty_msg = ("No targets have a metrics.json yet — run a target "
                         "compute first.")
            if skip_existing:
                eligible = [t for t in eligible
                            if not ref_satisfies(t, docking)]
                if not eligible:
                    empty_msg = ("Every cached reference already has the "
                                 "selected Vina mode — nothing to do (turn "
                                 "off 'Skip already-done' to recompute).")
            force = False  # compute_reference doesn't accept force

        if not eligible:
            st.warning(empty_msg)
            return

        outer = st.progress(0.0, text=f"0/{len(eligible)} targets")
        inner = st.progress(0.0, text="Starting…")
        for i, t in enumerate(eligible):
            def _cb(done: int, total: int, label: str, _t: Path = t) -> None:
                frac = done / total if total else 1.0
                inner.progress(min(frac, 1.0),
                               text=f"{_t.name}: {label} — {done}/{total}")
            pdb = find_pocket_pdb(t)
            try:
                if kind == "all":
                    coords, elems = parse_pocket_pdb(pdb)
                    compute_target_metrics(t, coords, elems, docking=docking,
                                           receptor_pdb=pdb, exhaustiveness=exh,
                                           progress_cb=_cb, force=force)
                else:  # "all_ref"
                    compute_reference(t, docking=docking, receptor_pdb=pdb,
                                      exhaustiveness=exh, progress_cb=_cb)
            except Exception as e:  # noqa: BLE001
                st.warning(f"{t.name}: {e}")
            outer.progress((i + 1) / len(eligible),
                           text=f"{i + 1}/{len(eligible)} targets")

    st.cache_data.clear()
    st.rerun()


@st.dialog("Force-recompute this target?")
def _confirm_target_eval(target: Path, docking: str,
                         exhaustiveness: int, force: bool) -> None:
    """Only shown when **Force recompute** is on for a target that already has
    a cache. Smart compute is non-destructive (preserves cached chem / vina
    that already satisfies the request), so it never triggers this dialog.
    """
    st.warning(f"`{target.name}` already has cached metrics — **Force "
               "recompute** is on, so its `metrics.json` will be rebuilt from "
               "scratch and any cached vina results re-run.")
    if docking != "none":
        st.caption(f"Vina mode `{docking}` is selected; docking can be slow.")
    go, cancel = st.columns(2)
    if go.button("Force recompute", type="primary", width="stretch"):
        st.session_state["_pending_compute"] = (
            "target", (target, docking, exhaustiveness, force))
        st.rerun()
    if cancel.button("Cancel", width="stretch"):
        st.rerun()


@st.dialog("Force-recompute all targets?")
def _confirm_all_eval(targets: list[Path], docking: str,
                      exhaustiveness: int, force: bool,
                      skip_existing: bool) -> None:
    """Only shown when **Force recompute** is on for a batch that has cached
    targets. Smart-compute batches (force off) skip this and just run."""
    n_cached = sum(1 for t in targets if metrics_path(t).exists())
    st.warning(f"**Force recompute** will rebuild metrics for all "
               f"{len(targets)} targets — {n_cached} cached `metrics.json` "
               "files will be overwritten and any cached vina re-run.")
    if docking != "none":
        st.caption(f"Vina mode `{docking}` across all targets can be slow.")
    if skip_existing:
        st.caption("Note: 'Skip already-done' is ignored when Force is on.")
    go, cancel = st.columns(2)
    if go.button("Force recompute all", type="primary", width="stretch"):
        st.session_state["_pending_compute"] = (
            "all", (targets, docking, exhaustiveness, force, skip_existing))
        st.rerun()
    if cancel.button("Cancel", width="stretch"):
        st.rerun()


# ─── Sidebar: experiment → target → sample ────────────────────────────────────
# Two side-by-side columns inside a widened sidebar:
#   - LEFT  (`sidebar-ctrl-col`): browsing context + compute-metrics section.
#   - RIGHT (`sidebar-tgt-col`):  target + sample radios, scrolling inside
#                                 their own max-height box.
# Streamlit containers accept multiple `with` blocks; we set up both column
# containers up front, fill the left with browse context, switch to the right
# for the radios (which defines `target`), then re-enter the left to append
# the compute section that needs to reference `target`.

with st.sidebar:
    _ctrl_col, _tgt_col = st.columns([1, 1], gap="small")
    with _ctrl_col:
        ctrl = st.container(key="sidebar-ctrl-col")
    with _tgt_col:
        tgt = st.container(key="sidebar-tgt-col")

with ctrl:
    st.title("Browse")
    st.caption(f"`{EXPS_ROOT}`")

    # Force a re-scan without touching a selector. Directory listings refresh
    # on every rerun and the file caches are mtime-keyed, but Streamlit only
    # reruns on a widget event — this button is that event. Clearing the data
    # cache also re-reads sample SDFs whose mtime didn't move (in-place
    # writes, coarse NFS).
    if st.button(
        "🔄 Refresh results",
        width="stretch",
        help="Re-scan the experiments directory and reload samples from disk.",
    ):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Scanned at {datetime.now():%H:%M:%S}")

    exps = list_experiments()
    if not exps:
        st.error("No experiments under voxbind/exps/.")
        st.stop()

    exp_names = [e.name for e in exps]
    default_exp = "260514_voxbind_100ep340"
    default_idx = exp_names.index(default_exp) if default_exp in exp_names else 0
    exp_name = st.selectbox("Experiment", exp_names, index=default_idx)
    exp = exps[exp_names.index(exp_name)]

    # sampling-run selector — an experiment can hold several runs under samples/
    sample_runs = list_sample_runs(exp)
    if not sample_runs:
        st.warning("This experiment has no sampling runs yet.")
        st.stop()
    # Default to the most recently created/modified run directory.
    _samples_root = exp / "samples"

    def _run_mtime(r: str) -> float:
        try:
            return (_samples_root / r).stat().st_mtime
        except OSError:
            return 0.0

    default_run = max(sample_runs, key=_run_mtime)
    run = st.selectbox(
        "Sample run", sample_runs, index=sample_runs.index(default_run)
    )

    targets = list_targets(exp, run)
    if not targets:
        st.warning(f"Run `{run}` has no `target_*` folders yet.")
        st.stop()


with tgt:
    def _target_label(i: int) -> str:
        t = targets[i]
        n_raw = count_raw_samples(t)
        cached = metrics_path(t).exists()
        marker = " ✓" if cached else ""
        return f"{t.name} ({n_raw}){marker}"

    target_idx = st.radio(
        "Target",
        range(len(targets)),
        format_func=_target_label,
    )
    target = targets[target_idx]
    sdf_path = target / "samples.sdf"
    sdf_mtime = _safe_mtime(sdf_path)

    samples, sample_smis = _load_samples_cached(str(target), sdf_mtime)
    if not samples:
        st.info("No valid samples for this target.")
        sample_idx: int | None = None
    else:
        sample_idx = st.radio(
            "Sample",
            range(len(samples)),
            format_func=lambda i: f"Sample {i}",
        )


# Re-enter the left (control) column so the compute section sits beside the
# target/sample column — but added *after* the radios run, so the button
# labels and actions can reference the freshly selected `target`.
with ctrl:
    st.divider()
    st.title("Evaluate")

    # Optional Vina docking — attaches vina_* fields to metrics.json on
    # (re)compute. AutoDock Vina is CPU-only; vina_dock with high
    # exhaustiveness can be slow.
    docking_mode = st.selectbox(
        "Vina docking", DOCKING_MODES, index=0,
        help="none = chemical metrics only · vina_score = + score_only · "
             "vina_min = + minimize · vina_dock = + full re-docking (slow). "
             "vina_dock yields all three paper scores.",
    )
    docking_exh = 8
    if docking_mode != "none":
        docking_exh = st.slider(
            "Exhaustiveness", 1, 32, 8,
            help="Vina search effort for vina_dock — higher is slower, more thorough.",
        )
        if not docking_available():
            st.warning(
                "Vina toolchain not installed — docking records an error per "
                "sample. Install instructions are in the `metrics.py` docstring."
            )

    force = st.checkbox(
        "Force recompute", value=False,
        help="Off (default) = *smart compute*: per-sample chem and per-sample "
             "vina from a fresh cache are reused; only what's missing for the "
             "selected mode is (re)computed. On = rebuild every sample from "
             "scratch even when the cache already has it.",
    )
    skip_existing = st.checkbox(
        "Skip already-done (batch)", value=True,
        help="When computing for *all* targets, skip those whose metrics.json "
             "already satisfies the selected Vina mode. Lets an interrupted "
             "'Compute all' resume without redoing finished pockets. Ignored "
             "when 'Force recompute' is on.",
    )

    if st.button("Compute metrics for all targets", width="stretch"):
        if force and any(metrics_path(t).exists() for t in targets):
            _confirm_all_eval(targets, docking_mode, docking_exh,
                              force, skip_existing)
        else:
            st.session_state["_pending_compute"] = (
                "all", (targets, docking_mode, docking_exh,
                        force, skip_existing))
            st.rerun()
    if st.button(f"Compute metrics for {target.name}", width="stretch"):
        if force and metrics_path(target).exists():
            _confirm_target_eval(target, docking_mode, docking_exh, force)
        else:
            st.session_state["_pending_compute"] = (
                "target", (target, docking_mode, docking_exh, force))
            st.rerun()
    if st.button("Compute reference ligand", width="stretch"):
        if metrics_path(target).exists():
            st.session_state["_pending_compute"] = (
                "ref", (target, docking_mode, docking_exh))
            st.rerun()
        else:
            st.warning("Run a target compute first — the reference "
                       "is stored in its metrics.json.")
    if st.button("Compute reference ligand for all targets",
                 width="stretch"):
        if any(metrics_path(t).exists() for t in targets):
            st.session_state["_pending_compute"] = (
                "all_ref", (targets, docking_mode, docking_exh, skip_existing))
            st.rerun()
        else:
            st.warning("Run a target compute first — the reference "
                       "is stored in each target's metrics.json.")


# ─── Main area ────────────────────────────────────────────────────────────────
pocket_pdb = find_pocket_pdb(target)
gt_sdf = find_gt_sdf(target)

st.title(f"{exp.name} · {target.name}")

# A compute request — from a sidebar button or a confirm dialog — is stashed
# in session_state and runs here on the next rerun, inside `_progress_dialog`:
# a modal that dims the page while a live bar tracks scoring / docking.
_pending = st.session_state.pop("_pending_compute", None)
if _pending is not None:
    _progress_dialog(*_pending)

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

            m1, m2, m3, m4, m5, m6 = st.columns(6)
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
            m6.metric("Sim→Ref", _fmt(sample_match.get("sim_to_ref"), 3),
                      help="RDKit-fingerprint Tanimoto similarity to the "
                           "reference ligand (1.0 = identical fingerprint)")

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
        ha = agg.get("high_affinity")
        if ha is not None:
            scols[3].metric(
                "High Affinity", f"{ha * 100:.1f}%",
                help=f"{agg.get('high_affinity_n', 0)}/"
                     f"{agg.get('high_affinity_total', 0)} samples beat the "
                     "reference Vina Dock score · needs vina_dock",
            )

        # Reference and Mean rows share a dark gray background so both summary
        # rows stand apart; the row matching the sidebar-selected sample is
        # highlighted so users can spot it without scanning.
        st.markdown("##### Reference, mean & per-sample metrics")
        cmp_df = _comparison_df(metrics)
        cmp_fmt = {k: v for k, v in _TABLE_FMT.items() if k in cmp_df.columns}
        pal = _COMPARE_PALETTE["dark" if _theme_is_dark() else "light"]
        st.dataframe(
            cmp_df.style.apply(_style_comparison, palette=pal,
                               selected_idx=sample_idx, axis=None)
                  .format(cmp_fmt, na_rep="—"),
            width="stretch",
            hide_index=True,
            height=min(36 * (len(cmp_df) + 1) + 3, 560),
        )
        st.caption("grey rows = reference ligand & sample mean · "
                   "highlighted row = currently selected sample")
        ref = metrics.get("reference")
        if isinstance(ref, dict) and "error" in ref:
            st.caption(f"⚠ reference ligand not scored — {ref['error']}")
        elif ref is None and "reference" not in metrics:
            st.caption("⚠ no reference row — recompute this target to add it.")
        elif ref is None:
            st.caption("⚠ this target has no reference ligand.")

        # Sample-level vina failures: surface them explicitly so a target that
        # was "evaluated but every dock crashed" is clearly distinguishable
        # from "docking wasn't requested" (the comparison table renders both
        # cases as a blank Vina cell). Each row shows sample index + error;
        # the same shape appears under reference if its dock errored too.
        _sample_errs = [(i, (s["vina"] or {}).get("error", ""))
                        for i, s in enumerate(metrics.get("samples", []))
                        if isinstance(s.get("vina"), dict)
                        and "error" in s["vina"]]
        _n_samples = len(metrics.get("samples", []))
        if _sample_errs:
            with st.expander(
                f"⚠ Vina failed for {len(_sample_errs)} of {_n_samples} samples "
                f"— click for per-sample errors",
                expanded=False,
            ):
                err_rows = [{"sample": f"sample[{i}]", "error": err}
                            for i, err in _sample_errs]
                # show distinct error strings up top so the failure mode is
                # obvious at a glance (e.g. 10 identical PrepProt AttributeErrors)
                distinct = sorted({e for _, e in _sample_errs})
                if len(distinct) == 1:
                    st.caption(f"all {len(_sample_errs)} share the same error: {distinct[0]}")
                else:
                    st.caption(f"{len(distinct)} distinct error message(s) across "
                               f"{len(_sample_errs)} failed samples")
                st.dataframe(pd.DataFrame(err_rows), hide_index=True,
                             width="stretch")

    st.caption(f"Computed: {metrics['computed_at']}"
               + ("" if fresh else "  ·  ⚠ stale"))
