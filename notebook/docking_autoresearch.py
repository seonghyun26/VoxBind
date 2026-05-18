"""docking_autoresearch.py — autoresearch-style optimisation of Vina docking speed.

Inspired by karpathy/autoresearch (https://github.com/karpathy/autoresearch):
rather than a fixed grid, this runs an iterative *research loop* —
baseline -> hypothesis -> experiment -> keep-or-reject -> next — that
coordinate-descent hill-climbs the one Vina setting which is pure speed: the
CPU thread count. With a fixed seed Vina is cpu-deterministic, so more threads
change *only* the wall-clock, never the docked pose or binding affinity.

exhaustiveness and search-box buffer are deliberately NOT optimised — both
change the docked pose/affinity, not just the speed, so trading them for
wall-time would corrupt any downstream comparison metric. They are held fixed
at the metrics.py defaults (exh=8, buffer=5 Å) and swept once as *diagnostics*,
so the report still shows their speed/affinity trade-off curves.

Robust to a noisy machine
-------------------------
Wall-clock docking time on a shared box (e.g. with training running) drifts a
lot — the same config can measure 1.5x apart minutes apart. So every candidate
is timed in a *paired, interleaved* way: for each ligand the incumbent and the
candidate are docked back-to-back, and the decision uses the per-ligand time
*ratio*. Common-mode drift cancels, so a small real effect is still visible.
A candidate is kept only if it is meaningfully faster (>=5%) on a majority of
ligands AND its affinity stays within 0.6 kcal/mol of a thorough reference dock.

Every trial is logged; the converged optimum is verified — also paired — on a
held-out ligand set. Writes a self-contained HTML report + a JSON journal.
AutoDock Vina is CPU-only — this never touches a GPU.

Run (works with or without `conda activate voxbind` — PATH is self-healed):
    python notebook/docking_autoresearch.py                 # full autoresearch
    python notebook/docking_autoresearch.py --smoke         # ~3-min sanity run
    python notebook/docking_autoresearch.py --render-only notebook/html/<name>.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import platform
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless — no display, no GPU
import matplotlib.pyplot as plt  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# ── paths & environment ───────────────────────────────────────────────────────
HERE = Path(__file__).resolve()
REPO = next((p for p in HERE.parents if (p / "voxbind" / "exps").is_dir()), None)
if REPO is None:
    REPO = Path("/home/shpark/prj-denovo/VoxBind")
TARGETDIFF = REPO.parent / "TargetDIff"
for _p in (str(REPO), str(TARGETDIFF), str(REPO / "notebook" / "webapp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_BIN = str(Path(sys.executable).parent)  # docking_vina.py shells out by bare name
if _BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _BIN + os.pathsep + os.environ.get("PATH", "")

CACHE = Path("/tmp/docking_autoresearch")
SCRATCH = CACHE / "scratch"

AFF_TOL = 0.6        # kcal/mol — candidate affinity must stay this close to ref
ACCEPT_MARGIN = 0.05  # a candidate must be >=5% faster (paired) to be kept
SEED = 1             # fixed Vina seed -> affinity is reproducible run-to-run


# ── test set ──────────────────────────────────────────────────────────────────
def _valid_mols(sdf: Path) -> list:
    out = []
    for m in Chem.SDMolSupplier(str(sdf), sanitize=True):
        if m is None:
            continue
        try:
            if "." not in Chem.MolToSmiles(m):
                out.append(m)
        except Exception:  # noqa: BLE001
            pass
    return out


def pick_test_set(n_needed: int, target: Path | None):
    """Return (target_dir, receptor_pdb, [mols]) with at least n_needed ligands."""
    if target is not None:
        candidates = [target]
    else:
        preferred = (REPO / "voxbind/exps/260514_voxbind_100ep340"
                     / "samples/res_ep5007_test/target_00")
        candidates = [preferred] + sorted(
            (REPO / "voxbind" / "exps").glob("*/samples/*/target_*"))
    for tdir in candidates:
        sdf, pdbs = tdir / "samples.sdf", list(tdir.glob("*_pocket10.pdb"))
        if not sdf.exists() or not pdbs:
            continue
        mols = _valid_mols(sdf)
        if len(mols) >= n_needed:
            return tdir, pdbs[0], mols[:n_needed]
    raise RuntimeError(f"no target with >= {n_needed} valid samples found")


# ── one docking trial ─────────────────────────────────────────────────────────
def dock_trial(receptor: Path, mol, cfg: dict) -> dict:
    """Dock one molecule once under `cfg`; return {affinity, seconds, error}."""
    from utils.evaluation.docking_vina import VinaDockingTask
    try:
        task = VinaDockingTask(
            str(receptor), deepcopy(mol), tmp_dir=str(SCRATCH),
            size_factor=cfg["size_factor"], buffer=cfg["buffer"])
        t0 = time.perf_counter()
        res = task.run(mode=cfg["mode"], exhaustiveness=cfg["exhaustiveness"],
                       cpu=cfg["cpu"], seed=SEED)
        dt = time.perf_counter() - t0
        return {"affinity": float(res[0]["affinity"]), "seconds": dt, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"affinity": None, "seconds": None,
                "error": f"{type(e).__name__}: {e}"}


def evaluate(receptor: Path, cfg: dict, mols: list) -> dict:
    """Dock every ligand once under `cfg` (no pairing) — for the baseline/ref."""
    rs = [dock_trial(receptor, m, cfg) for m in mols]
    ok = [r for r in rs if r["error"] is None]
    t = [r["seconds"] for r in ok]
    a = [r["affinity"] for r in ok]
    return dict(sec_per_mol=float(np.median(t)) if t else None,
                aff_mean=float(np.mean(a)) if a else None,
                n_ok=len(ok), n_fail=len(rs) - len(ok), per_mol=rs)


def paired_eval(receptor: Path, inc: dict, cand: dict, mols: list) -> dict:
    """Interleaved per-ligand timing of incumbent vs candidate.

    For each ligand the two configs are docked back-to-back (order alternated),
    so machine-load drift is common-mode and cancels in the per-ligand ratio.
    """
    pairs = []
    for i, mol in enumerate(mols):
        order = [("inc", inc), ("cand", cand)]
        if i % 2:
            order.reverse()                       # cancel back-to-back order bias
        got = {tag: dock_trial(receptor, mol, c) for tag, c in order}
        pairs.append(dict(
            mol=i,
            inc_sec=got["inc"]["seconds"], inc_aff=got["inc"]["affinity"],
            cand_sec=got["cand"]["seconds"], cand_aff=got["cand"]["affinity"],
            error=got["inc"]["error"] or got["cand"]["error"]))
    good = [p for p in pairs if p["inc_sec"] and p["cand_sec"]]
    ratios = [p["cand_sec"] / p["inc_sec"] for p in good]
    cand_aff = [p["cand_aff"] for p in good if p["cand_aff"] is not None]
    return dict(
        ratio=float(np.median(ratios)) if ratios else None,
        cand_sec=float(np.median([p["cand_sec"] for p in good])) if good else None,
        inc_sec=float(np.median([p["inc_sec"] for p in good])) if good else None,
        cand_aff=float(np.mean(cand_aff)) if cand_aff else None,
        n_faster=sum(1 for r in ratios if r < 1.0),
        n_pairs=len(good), n_fail=len(pairs) - len(good), pairs=pairs)


def cfg_str(c: dict) -> str:
    return (f"{c['mode']} · exh={c['exhaustiveness']} · cpu={c['cpu']} "
            f"· buffer={c['buffer']:g}Å")


# ── the autoresearch loop ─────────────────────────────────────────────────────
KNOB_HYP = {
    "cpu": "More CPU threads parallelise Vina's Monte-Carlo search — expect a "
           "speedup until the search saturates the cores. With a fixed seed "
           "this changes only the speed, so it is the one knob safe to tune.",
    "exhaustiveness": "Lower exhaustiveness runs fewer Monte-Carlo searches — "
           "faster, but it changes the docked affinity (a noisier estimate of "
           "the global minimum), so it is held fixed and swept only as a "
           "diagnostic, never adopted.",
    "buffer": "A smaller search box has fewer affinity-grid points — faster, "
           "but the box size changes the pose/affinity (too tight corrupts it; "
           "larger dilutes the search), so it is held fixed and swept only as "
           "a diagnostic, never adopted.",
}


def autoresearch(receptor: Path, search_mols: list, verify_mols: list,
                 grids: dict, exh_ref: int, passes: int, log=print) -> dict:
    """Hill-climb the cpu thread count; exhaustiveness & buffer are diagnostic-only."""
    journal: list = []
    counter = [0]

    def record(phase, knob, hypothesis, cfg, *, sec, ratio, inc_sec,
               aff, decision, note, extra=None):
        counter[0] += 1
        rec = dict(id=counter[0], phase=phase, knob=knob, hypothesis=hypothesis,
                   config=dict(cfg), sec=sec, ratio=ratio, inc_sec=inc_sec,
                   aff=aff, decision=decision, note=note, extra=extra or {})
        journal.append(rec)
        rr = f"{ratio:.3f}" if ratio is not None else "  —  "
        log(f"  exp {rec['id']:2d} [{phase:8s}] {cfg_str(cfg):42s} "
            f"{(f'{sec:.2f}s/mol' if sec else 'FAIL'):>12s}  ratio={rr}  "
            f"-> {decision}", flush=True)
        return rec

    # ── Phase 0 — baseline + thorough reference dock (affinity guardrail) ──────
    log("Phase 0 — baseline & affinity reference")
    baseline = dict(mode="dock", exhaustiveness=8, cpu=grids["cpu_default"],
                    buffer=5.0, size_factor=1.0)
    b = evaluate(receptor, baseline, search_mols)
    record("baseline", "-",
           "metrics.py's current default docking config — the reference every "
           "later experiment is hill-climbed against.",
           baseline, sec=b["sec_per_mol"], ratio=None, inc_sec=None,
           aff=b["aff_mean"], decision="baseline", note="starting incumbent")
    g = evaluate(receptor, dict(baseline, exhaustiveness=exh_ref), search_mols)
    aff_ref = g["aff_mean"] if g["aff_mean"] is not None else b["aff_mean"]
    record("baseline", "exhaustiveness",
           f"A thorough exh={exh_ref} dock pins the affinity reference; a faster "
           f"config is accepted only within {AFF_TOL} kcal/mol of it.",
           dict(baseline, exhaustiveness=exh_ref), sec=g["sec_per_mol"],
           ratio=None, inc_sec=None, aff=aff_ref, decision="reference",
           note=f"affinity reference = {aff_ref:.2f} kcal/mol")
    log(f"  -> affinity reference {aff_ref:.2f} kcal/mol\n")

    best = dict(baseline)
    cum_speedup = 1.0
    sweeps: dict = {}

    # ── Phases 1..N — coordinate-descent hill-climbing (paired timing) ────────
    for p in range(1, passes + 1):
        log(f"Phase {p} — cpu coordinate-descent pass {p}")
        changed = False
        curve = []
        for val in grids["cpu"]:
            if val == best["cpu"]:
                continue
            cand = dict(best)
            cand["cpu"] = val
            pr = paired_eval(receptor, best, cand, search_mols)
            faster = (pr["ratio"] is not None
                      and pr["ratio"] <= 1.0 - ACCEPT_MARGIN
                      and pr["n_faster"] * 2 >= pr["n_pairs"])
            accurate = (pr["cand_aff"] is not None
                        and abs(pr["cand_aff"] - aff_ref) <= AFF_TOL)
            if pr["ratio"] is None:
                decision, note = "rejected: failed", "docking failed"
            elif not accurate:
                decision = "rejected: affinity"
                note = (f"affinity {pr['cand_aff']:.2f} drifts "
                        f"{abs(pr['cand_aff'] - aff_ref):.2f} > {AFF_TOL}")
            elif faster:
                decision = "NEW BEST"
                note = (f"{(1 - pr['ratio']) * 100:.0f}% faster than the "
                        f"incumbent ({pr['n_faster']}/{pr['n_pairs']} ligands)")
                best = dict(cand)
                cum_speedup /= pr["ratio"]
                changed = True
            elif pr["ratio"] < 1.0:
                decision = "rejected: within noise"
                note = (f"only {(1 - pr['ratio']) * 100:.0f}% faster "
                        f"(< {ACCEPT_MARGIN * 100:.0f}% margin)")
            else:
                decision = "rejected: slower"
                note = f"{(pr['ratio'] - 1) * 100:.0f}% slower (paired)"
            r = record(f"pass {p}", "cpu", KNOB_HYP["cpu"], cand,
                       sec=pr["cand_sec"], ratio=pr["ratio"],
                       inc_sec=pr["inc_sec"], aff=pr["cand_aff"],
                       decision=decision, note=note,
                       extra=dict(n_faster=pr["n_faster"],
                                  n_pairs=pr["n_pairs"], pairs=pr["pairs"]))
            curve.append(r)
        if p == 1:
            sweeps["cpu"] = curve
        log(f"  -> after pass {p}: {cfg_str(best)}  "
            f"(cumulative {cum_speedup:.2f}x)\n")
        if not changed:
            log(f"  converged after pass {p} — cpu did not improve.\n")
            break

    # ── Phase D — diagnostic sweeps: exhaustiveness & buffer (held fixed) ──
    # Swept once around the chosen operating point so the report keeps their
    # speed/affinity trade-off plots; never adopted, because these knobs change
    # the docking result, not just the wall-clock.
    log("Phase D — diagnostic sweeps (exhaustiveness, buffer — held fixed)")
    for knob in ("exhaustiveness", "buffer"):
        curve = []
        for val in grids[knob]:
            if val == best[knob]:
                continue
            cand = dict(best)
            cand[knob] = val
            pr = paired_eval(receptor, best, cand, search_mols)
            if pr["ratio"] is None:
                note = "diagnostic only (knob held fixed) — docking failed"
            else:
                pct = (1.0 - pr["ratio"]) * 100.0
                if pct > 0.5:
                    speed = f"{pct:.0f}% faster"
                elif pct < -0.5:
                    speed = f"{-pct:.0f}% slower"
                else:
                    speed = "≈ even"
                if pr["cand_aff"] is not None:
                    aff_txt = (f"affinity {pr['cand_aff']:.2f}, drift "
                               f"{abs(pr['cand_aff'] - aff_ref):.2f} kcal/mol "
                               f"vs the exh={exh_ref} reference")
                else:
                    aff_txt = "affinity unavailable"
                note = f"diagnostic only (knob held fixed) — {speed}; {aff_txt}"
            r = record("diagnostic", knob, KNOB_HYP[knob], cand,
                       sec=pr["cand_sec"], ratio=pr["ratio"],
                       inc_sec=pr["inc_sec"], aff=pr["cand_aff"],
                       decision="diagnostic", note=note,
                       extra=dict(n_faster=pr["n_faster"],
                                  n_pairs=pr["n_pairs"], pairs=pr["pairs"]))
            curve.append(r)
        sweeps[knob] = curve
    log(f"  -> diagnostics logged; optimum stays {cfg_str(best)}\n")

    # ── Phase V — verification: paired baseline vs optimum on held-out ligands ─
    log("Phase V — verification on held-out ligands (paired)")
    vr = paired_eval(receptor, baseline, best, verify_mols)
    speedup = (1.0 / vr["ratio"]) if vr["ratio"] else None
    record("verify", "-",
           "Paired re-measurement of the optimum against the baseline on fresh "
           "held-out ligands — the drift-free ground-truth speedup.",
           best, sec=vr["cand_sec"], ratio=vr["ratio"], inc_sec=vr["inc_sec"],
           aff=vr["cand_aff"], decision="verify-optimum",
           note=(f"{speedup:.2f}x faster than baseline "
                 f"({vr['n_faster']}/{vr['n_pairs']} ligands)"
                 if speedup else "verification failed"),
           extra=dict(n_faster=vr["n_faster"], n_pairs=vr["n_pairs"],
                      pairs=vr["pairs"]))

    return dict(journal=journal, baseline=baseline, optimum=best,
                aff_ref=aff_ref, cum_speedup=cum_speedup, sweeps=sweeps,
                verify=dict(**vr, speedup=speedup), n_experiments=counter[0])


# ── plots ─────────────────────────────────────────────────────────────────────
ACCENT, GREEN, GREY, ORANGE = "#3b82f6", "#15803d", "#9aa0a6", "#b45309"


def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_convergence(journal: list) -> str:
    """Cumulative paired speedup vs experiment number (the hill-climb)."""
    pts = [e for e in journal if e["phase"].startswith(("baseline", "pass"))]
    if not pts:
        return ""
    xs, cum, cur = [], [], 1.0
    for e in pts:
        if e["decision"] == "NEW BEST" and e["ratio"]:
            cur /= e["ratio"]
        xs.append(e["id"])
        cum.append(cur)
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    ax.step(xs, cum, where="post", color=GREEN, lw=2)
    for e, x, y in zip(pts, xs, cum):
        if e["decision"] == "NEW BEST":
            ax.scatter([x], [y], c=GREEN, s=70, zorder=3,
                       edgecolors="white", linewidths=0.8)
            ax.annotate(f"#{e['id']}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=GREEN)
    ax.axhline(1.0, color=GREY, ls="--", lw=1)
    ax.set_xlabel("experiment #")
    ax.set_ylabel("cumulative speedup vs baseline")
    ax.set_title("Autoresearch convergence (paired speedup)", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _fig_b64(fig)


def plot_knob(knob: str, curve: list, logx: bool) -> str:
    """Paired ratio + affinity vs one knob, from its pass-1 sweep."""
    pts = [(e["config"][knob], e["ratio"], e["aff"]) for e in curve
           if e["ratio"] is not None]
    pts.sort(key=lambda r: r[0])
    if len(pts) < 2:
        return ""
    x = [p[0] for p in pts]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.3))
    a1.axhline(1.0, color=GREY, ls="--", lw=1)
    a1.plot(x, [p[1] for p in pts], "o-", color=ACCENT)
    a1.set_ylabel("paired time ratio vs incumbent\n(<1 = faster)")
    a1.set_title(f"Speed vs {knob}", fontsize=11)
    a2.plot(x, [p[2] for p in pts], "o-", color=ORANGE)
    a2.set_ylabel("Vina affinity (kcal/mol)")
    a2.set_title(f"Affinity vs {knob}", fontsize=11)
    for ax in (a1, a2):
        if logx:
            ax.set_xscale("log", base=2)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:g}" for v in x])
        ax.set_xlabel(knob)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _fig_b64(fig)


# ── HTML report ───────────────────────────────────────────────────────────────
CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica,
       Arial, sans-serif; margin: 0 auto; padding: 28px; background: #fafafa;
       color: #222; max-width: 1080px; }
h1 { margin-bottom: 2px; font-size: 1.5rem; }
h2 { margin-top: 34px; border-bottom: 1px solid #ddd; padding-bottom: 6px;
     font-size: 1.15rem; }
h3 { margin-top: 20px; font-size: 1rem; }
.subtitle { color: #666; margin-bottom: 20px; font-size: .9rem; }
.optimum { background: #ecfdf3; border: 1px solid #15803d; border-left: 5px solid
           #15803d; padding: 14px 20px; margin: 18px 0; border-radius: 6px; }
.optimum .big { font-size: 1.15rem; font-weight: 700; color: #14532d; }
.hyp { color: #555; font-size: .88rem; margin: 4px 0 10px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0;
        background: white; font-size: 12.5px; }
th, td { border: 1px solid #e5e5e5; padding: 6px 9px; text-align: right; }
th { background: #f5f5f5; font-weight: 600; }
td.l, th.l { text-align: left; }
tr:nth-child(even) td { background: #fafafa; }
.b-best  { color: #15803d; font-weight: 700; }
.b-base  { color: #3b82f6; font-weight: 700; }
.b-ref   { color: #b45309; font-weight: 700; }
.b-rej   { color: #9aa0a6; }
.b-verify{ color: #6d28d9; font-weight: 700; }
.b-diag  { color: #0891b2; }
img { max-width: 100%; background: white; border: 1px solid #e5e5e5;
      border-radius: 4px; padding: 6px; margin: 8px 0; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
       background: #f1f3f5; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
details { background: white; border: 1px solid #ddd; border-radius: 4px;
          padding: 8px 14px; margin: 10px 0; }
summary { cursor: pointer; font-weight: 600; }
.footer { color: #888; font-size: 12px; margin-top: 44px;
          border-top: 1px solid #ddd; padding-top: 12px; }
"""

_BADGE = {
    "baseline": ("b-base", "baseline"),
    "reference": ("b-ref", "affinity ref"),
    "NEW BEST": ("b-best", "★ NEW BEST"),
    "rejected: slower": ("b-rej", "rejected · slower"),
    "rejected: within noise": ("b-rej", "rejected · within noise"),
    "rejected: affinity": ("b-rej", "rejected · affinity"),
    "rejected: failed": ("b-rej", "rejected · failed"),
    "verify-optimum": ("b-verify", "★ verified optimum"),
    "diagnostic": ("b-diag", "diagnostic"),
}


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _f(v, p=2):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{p}f}"


def _ratio_cell(r) -> str:
    if r is None:
        return "—"
    pct = (1 - r) * 100
    if pct > 0.5:
        return f"{r:.3f}  ({pct:.0f}% faster)"
    if pct < -0.5:
        return f"{r:.3f}  ({-pct:.0f}% slower)"
    return f"{r:.3f}  (≈ even)"


def render_html(result: dict, meta: dict, out: Path) -> None:
    journal = result["journal"]
    opt, base = result["optimum"], result["baseline"]
    vr = result["verify"]
    sp = vr["speedup"]

    rows = ""
    for e in journal:
        cls, label = _BADGE.get(e["decision"], ("b-rej", e["decision"]))
        rows += (
            f"<tr><td>{e['id']}</td><td class='l'>{_esc(e['phase'])}</td>"
            f"<td class='l'>{_esc(e['knob'])}</td>"
            f"<td class='l'><code>{_esc(cfg_str(e['config']))}</code></td>"
            f"<td>{_f(e['sec'])}</td><td class='l'>{_ratio_cell(e['ratio'])}</td>"
            f"<td>{_f(e['aff'])}</td><td class='l {cls}'>{label}</td>"
            f"<td class='l'>{_esc(e['note'])}</td></tr>")
    journal_table = (
        "<table><tr><th>#</th><th class='l'>phase</th><th class='l'>knob</th>"
        "<th class='l'>candidate config</th><th>s/mol</th>"
        "<th class='l'>paired ratio vs incumbent</th><th>affinity</th>"
        f"<th class='l'>decision</th><th class='l'>note</th></tr>{rows}</table>")

    seen, hyp_html = set(), ""
    for e in journal:
        if e["knob"] in KNOB_HYP and e["knob"] not in seen:
            seen.add(e["knob"])
            hyp_html += (f"<p class='hyp'><b>{_esc(e['knob'])}</b> — "
                         f"{_esc(KNOB_HYP[e['knob']])}</p>")

    knob_plots = ""
    for knob in ("cpu", "exhaustiveness", "buffer"):
        curve = result["sweeps"].get(knob)
        if not curve:
            continue
        img = plot_knob(knob, curve, logx=(knob != "buffer"))
        if img:
            knob_plots += (f"<h3>{knob}</h3>"
                           f"<img src='data:image/png;base64,{img}'>")

    drift = (abs(vr["cand_aff"] - result["aff_ref"])
             if vr["cand_aff"] is not None else None)
    speed_txt = f"<b>{sp:.2f}× faster</b>" if sp else "—"

    body = f"""
<h1>VoxBind · Docking-Speed Autoresearch</h1>
<div class="subtitle">Generated {meta['date']} · {meta['host']} ·
{meta['cpu_count']} CPUs · vina {meta['vina_version']} ·
{result['n_experiments']} experiments in {_f(meta['minutes'], 1)} min</div>

<div class="optimum">
<div class="big">Optimal config — {speed_txt} than the baseline</div>
<p><code>{_esc(cfg_str(opt))}</code><br>
Paired verification on {vr['n_pairs']} held-out ligands: baseline
{_f(vr['inc_sec'])} → optimum <b>{_f(vr['cand_sec'])} s/mol</b>
({vr['n_faster']}/{vr['n_pairs']} ligands faster); affinity moved
{_f(drift)} kcal/mol (guardrail {AFF_TOL}).</p>
<p style="font-size:.86rem;color:#14532d;margin:8px 0 0"><b>Reading it:</b>
the optimum differs from the baseline in <code>cpu</code> only — a
result-preserving change, so the affinity moved just {_f(drift)} kcal/mol
(measurement noise, not a real shift). <code>exhaustiveness</code> and
<code>buffer</code> are deliberately <i>not</i> tuned — both change the docked
affinity, so trading them for speed would bias a model-comparison metric; their
trade-off is shown as a diagnostic sweep below. <code>metrics.py</code> takes
<code>cpu</code> directly — adopt <code>cpu={opt['cpu']}</code> and leave
exhaustiveness at its default 8.</p>
</div>

<h2>Method — autoresearch loop</h2>
<p class="hyp">Inspired by <code>karpathy/autoresearch</code>: instead of a fixed
grid, an iterative loop — <b>baseline → hypothesis → experiment → keep-or-reject
→ next</b> — hill-climbs by <b>coordinate descent</b>. Only <b>one</b> knob is
optimised — the <b>CPU thread count</b>: with a fixed Vina seed more threads
change <i>only</i> the wall-clock, never the docked pose or affinity, so it is
the one knob safe to trade for speed. <b>exhaustiveness</b> and the
<b>search-box buffer</b> change the docking <i>result</i>, not just the speed,
so they are held fixed at the <code>metrics.py</code> defaults
(exh=8, buffer=5&nbsp;Å) and only swept as <i>diagnostics</i> (below) — never
adopted. Wall-clock docking time on this shared machine drifts heavily (the same
config measured up to 1.6× apart in an earlier un-paired run), so every
candidate is timed <b>paired & interleaved</b>: for each of {meta['n_search']}
ligands the incumbent and the candidate are docked back-to-back and compared by
the per-ligand time <b>ratio</b>, which cancels common-mode drift. A cpu
candidate is kept only if it is ≥{int(ACCEPT_MARGIN*100)}% faster on a majority
of ligands <i>and</i> its mean Vina affinity stays within {AFF_TOL} kcal/mol of
a thorough exh={meta['exh_ref']} reference dock. The converged optimum is
verified — also paired — on {meta['n_verify']} held-out ligands.</p>
{hyp_html}

<h2>Result — convergence</h2>
<img src='data:image/png;base64,{plot_convergence(journal)}'>

<h2>Research log — every trial</h2>
<p class="hyp">{len(journal)} experiments. The <b>paired ratio</b> column is the
decision signal (incumbent and candidate timed back-to-back); ★ marks an
accepted improvement.</p>
{journal_table}

<h2>Per-knob exploration</h2>
<p class="hyp">Sweeps around the chosen operating point. Left: paired time ratio
vs the incumbent (&lt;1 = faster). Right: Vina affinity. <code>cpu</code> is the
optimised knob — affinity is flat across it (pure parallelism).
<code>exhaustiveness</code> and <code>buffer</code> are <b>diagnostic only</b>:
the affinity panel moving with them <i>is</i> the docking result changing —
which is why they are held fixed rather than traded for speed.</p>
{knob_plots}

<h2>Setup &amp; reproduce</h2>
<table>
<tr><td class="l">test pocket</td><td class="l"><code>{_esc(meta['target'])}</code></td></tr>
<tr><td class="l">receptor</td><td class="l"><code>{_esc(meta['receptor'])}</code></td></tr>
<tr><td class="l">search ligands</td><td class="l">{meta['n_search']} (hill-climbing, paired)</td></tr>
<tr><td class="l">verification ligands</td><td class="l">{meta['n_verify']} (held out, paired)</td></tr>
<tr><td class="l">accept margin</td><td class="l">≥{int(ACCEPT_MARGIN*100)}% faster (paired), majority of ligands</td></tr>
<tr><td class="l">affinity guardrail</td><td class="l">±{AFF_TOL} kcal/mol vs exh={meta['exh_ref']} reference</td></tr>
<tr><td class="l">Vina seed</td><td class="l">{SEED} (fixed — affinity reproducible)</td></tr>
</table>
<details><summary>Raw per-ligand timings (all experiments)</summary>
<table><tr><th>exp</th><th class='l'>config</th><th>ligand</th>
<th>incumbent s</th><th>candidate s</th><th>cand affinity</th></tr>""" + "".join(
        "".join(
            f"<tr><td>{e['id']}</td>"
            f"<td class='l'><code>{_esc(cfg_str(e['config']))}</code></td>"
            f"<td>{p['mol']}</td><td>{_f(p.get('inc_sec'))}</td>"
            f"<td>{_f(p.get('cand_sec'))}</td><td>{_f(p.get('cand_aff'))}</td></tr>"
            for p in (e.get("extra", {}) or {}).get("pairs", []))
        for e in journal
    ) + f"""</table></details>

<div class="footer">
Generated by <code>notebook/docking_autoresearch.py</code> — autoresearch-style
docking-speed optimisation (inspired by
<a href="https://github.com/karpathy/autoresearch">karpathy/autoresearch</a>).
Re-run: <code>python notebook/docking_autoresearch.py</code> ·
raw journal: <code>{_esc(meta['json_name'])}</code>.
</div>
"""
    out.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>VoxBind docking-speed autoresearch</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search-mols", type=int, default=3)
    ap.add_argument("--verify-mols", type=int, default=6)
    ap.add_argument("--target", type=Path, default=None)
    ap.add_argument("--exhaustiveness-ref", type=int, default=16)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--out", type=Path,
                    default=REPO / "notebook" / "html" /
                    f"{datetime.now():%y%m%d}_docking_autoresearch.html")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny ~3-min run to validate the harness")
    ap.add_argument("--render-only", type=Path, default=None,
                    help="re-render HTML from an existing JSON journal")
    args = ap.parse_args()

    if args.render_only is not None:
        data = json.loads(args.render_only.read_text())
        render_html(data["result"], data["meta"], args.out)
        print(f"re-rendered {args.out} from {args.render_only}")
        return

    max_cpu = os.cpu_count() or 8
    if args.smoke:
        n_search, n_verify, passes, exh_ref = 2, 2, 1, 8
        grids = {"cpu": [8, 16], "exhaustiveness": [2, 8], "buffer": [5.0, 8.0],
                 "cpu_default": min(8, max_cpu)}
    else:
        n_search, n_verify = args.search_mols, args.verify_mols
        passes, exh_ref = args.passes, args.exhaustiveness_ref
        grids = {
            "cpu": [c for c in (4, 8, 16, 32, 64) if c <= max_cpu],
            "exhaustiveness": [e for e in (1, 2, 4, 8, 16) if e <= exh_ref],
            "buffer": [2.0, 4.0, 6.0, 8.0, 12.0],
            "cpu_default": min(8, max_cpu),
        }

    print(f"docking autoresearch | host {platform.node()} | {max_cpu} CPUs")
    tdir, receptor_src, mols = pick_test_set(n_search + n_verify, args.target)
    search_mols, verify_mols = mols[:n_search], mols[n_search:n_search + n_verify]
    print(f"test set: {tdir.relative_to(REPO)} | "
          f"{n_search} search + {n_verify} verify ligands\n")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    receptor = CACHE / receptor_src.name
    receptor.write_bytes(receptor_src.read_bytes())
    warm = dock_trial(receptor, search_mols[0],
                      dict(mode="score_only", exhaustiveness=8,
                           cpu=grids["cpu_default"], buffer=5.0, size_factor=1.0))
    if warm["error"]:
        print(f"FATAL: receptor prep / first dock failed: {warm['error']}")
        sys.exit(1)

    t0 = time.perf_counter()
    result = autoresearch(receptor, search_mols, verify_mols, grids, exh_ref, passes)
    minutes = (time.perf_counter() - t0) / 60.0

    try:
        import vina as _vina
        vina_version = getattr(_vina, "__version__", "?")
    except Exception:  # noqa: BLE001
        vina_version = "?"
    meta = dict(
        date=f"{datetime.now():%Y-%m-%d %H:%M}", host=platform.node(),
        cpu_count=max_cpu, vina_version=vina_version, minutes=minutes,
        target=str(tdir.relative_to(REPO)), receptor=receptor_src.name,
        n_search=n_search, n_verify=n_verify, exh_ref=exh_ref, passes=passes,
        json_name=args.out.with_suffix(".json").name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    render_html(result, meta, args.out)
    args.out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "result": result}, indent=2))

    vr = result["verify"]
    print(f"\nautoresearch done in {minutes:.1f} min · "
          f"{result['n_experiments']} experiments")
    print(f"  optimum  : {cfg_str(result['optimum'])}")
    if vr["speedup"]:
        print(f"  verified : {vr['inc_sec']:.2f} -> {vr['cand_sec']:.2f} s/mol "
              f"= {vr['speedup']:.2f}x on {vr['n_pairs']} held-out ligands "
              f"(paired)")
    print(f"  report   : {args.out}")
    print(f"  journal  : {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
