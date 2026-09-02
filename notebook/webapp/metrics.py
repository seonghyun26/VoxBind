"""On-demand TargetDiff metric computation with on-disk caching.

Cache file: `target_*/metrics.json`. Considered stale if older than samples.sdf.

Optional AutoDock Vina docking can be attached per sample. It is OFF by default
because it is slow and needs an extra toolchain (vina, meeko, pdb2pqr,
AutoDockTools). When the toolchain is missing each sample simply records a
`vina.error` string and every other metric still works.

CLI (offline batch docking):
    python metrics.py <target_dir | run_dir> [--docking vina_score|vina_min|vina_dock]
                       [--exhaustiveness N] [--cpu N] [--workers N] [--skip-existing]

Docking parallelism: a target's samples are docked across a process pool
(`workers` procs x `cpu` Vina threads each). Every dock is the identical
computation — only the scheduling changes — so metrics.json is unaffected.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger

# data.py also injects VoxBind + TargetDiff onto sys.path on import.
from data import find_pocket_pdb, load_gt_mols, parse_pocket_pdb
from utils.evaluation.scoring_func import get_chem  # from TargetDiff

RDLogger.DisableLog("rdApp.*")

# `vina` fields, the `reference` baseline, and per-sample `sim_to_ref` are
# additive & optional, so the cache format is unchanged: a metrics.json with
# or without them is version 2.
METRICS_VERSION = 2

# none       -> chemical metrics only
# vina_score -> + Vina score_only        (the generated pose, as-is)
# vina_min   -> + score_only & minimize  (local optimisation; no pose search)
# vina_dock  -> + score_only, minimize & full re-docking — all three paper scores
DOCKING_MODES = ("none", "vina_score", "vina_min", "vina_dock")

# Pose-quality eval (PoseCheck + PoseBusters), computed in the dedicated
# `moleval` conda env via the pose_eval.py subprocess worker. Additive &
# optional, exactly like vina above — a metrics.json with or without the
# per-sample `posecheck`/`posebusters` blocks is still version 2, so requesting
# pose never invalidates a cached (and possibly expensively docked) target.
# none        -> no pose eval
# posecheck   -> + PoseCheck (clashes, strain energy, interaction profile)
# posebusters -> + PoseBusters (dock-mode validity checks)
# all         -> both
POSE_MODES = ("none", "posecheck", "posebusters", "all")

# The moleval interpreter (override with $MOLEVAL_PY). Its env `bin/` is
# prepended to PATH for the worker so posecheck can shell out to `hydride` /
# `reduce` (protein protonation).
_MOLEVAL_PY = os.environ.get("MOLEVAL_PY", "/opt/conda/envs/moleval/bin/python")
_POSE_WORKER = str(Path(__file__).with_name("pose_eval.py"))
_POSE_IMPORT_ERROR: str | None = None
# Backstop for each pose-eval chunk (protein protonation / PoseBusters can also
# hang, not just strain). Keeping the timeout per chunk prevents one slow group
# from discarding every result in a roughly 100-ligand target.
_POSE_TIMEOUT_S = float(os.environ.get("POSE_TIMEOUT_S", "600"))
# Number of ligands sent to one moleval subprocess. Twenty avoids the old
# all-target timeout without paying protein-protonation overhead per ligand.
_POSE_CHUNK_SIZE = max(1, int(os.environ.get("POSE_CHUNK_SIZE", "20")))

# An optional progress callback: progress_cb(done, total, stage_label). Called
# during the chemical-scoring and docking phases so a UI can show a live bar.
ProgressCb = Callable[[int, int, str], None]


def metrics_path(target_dir: Path) -> Path:
    return target_dir / "metrics.json"


def metrics_are_fresh(target_dir: Path) -> bool:
    mp = metrics_path(target_dir)
    sdf = target_dir / "samples.sdf"
    if not mp.exists() or not sdf.exists():
        return False
    try:
        data = json.loads(mp.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if data.get("version") != METRICS_VERSION:
        return False
    return mp.stat().st_mtime >= sdf.stat().st_mtime


def load_metrics(target_dir: Path) -> dict | None:
    mp = metrics_path(target_dir)
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# ── Smart-compute helpers ─────────────────────────────────────────────────────
# A cached row is reusable only when it carries every field the requested mode
# needs. These predicates drive both the per-sample reuse inside
# `compute_target_metrics` and the batch "skip-existing" toggle in the webapp.

_CHEM_KEYS = ("smiles", "qed", "sa", "logp", "lipinski", "n_atoms", "interactions")


def _has_chem(row: dict) -> bool:
    """A cached sample row carries every chemical-metric field."""
    return all(k in row for k in _CHEM_KEYS)


def _vina_mode_keys(mode: str) -> tuple[str, ...]:
    """Vina sub-fields required to satisfy `mode` (empty for 'none')."""
    if mode == "vina_score": return ("score_only",)
    if mode == "vina_min":   return ("score_only", "minimize")
    if mode == "vina_dock":  return ("score_only", "minimize", "dock")
    return ()


def _has_vina_for_mode(vina, mode: str) -> bool:
    """Cached `vina` dict satisfies `mode` (no error, every required field)."""
    if mode == "none":
        return True
    if not isinstance(vina, dict) or "error" in vina:
        return False
    return all(isinstance(vina.get(k), (int, float))
               for k in _vina_mode_keys(mode))


def _pose_blocks(mode: str) -> tuple[str, ...]:
    """Per-sample pose blocks required to satisfy `mode` (empty for 'none')."""
    if mode == "posecheck":   return ("posecheck",)
    if mode == "posebusters": return ("posebusters",)
    if mode == "all":         return ("posecheck", "posebusters")
    return ()


# Root of the untouched CrossDocked receptors. The pocket10 crops each target dir
# carries are a literal subset of these (matched-atom RMSD 0.0000 Å).
FULL_RECEPTOR_ROOT = os.environ.get(
    "FULL_RECEPTOR_ROOT", "/home1/irteam/VoxBind/targetdiff/data/test_set/test_set")


def find_full_receptor(target_dir) -> Path | None:
    """Whole-receptor PDB for a target dir, or None if it cannot be resolved.

    PoseCheck's clash count is a ligand-atom x protein-atom distance test, so it can
    only see the protein it is handed. The 10 Å crop keeps nearly every atom inside the
    ligand box but drops 35-64% of atoms within van der Waals reach of the pose, which
    makes crop-scored clash counts a structural UNDER-count -- and more so for larger
    ligands, since they reach further into the deleted shell. Strain is unaffected
    (it never sees the protein).

    Two crop-naming layouts exist and both must resolve:
      <subdir>__<ligbase>_pocket10.pdb   our sample dirs      -> split on the first __
      <ligbase>_pocket10.pdb             base_drug/eval/...   -> subdir dropped, glob it
    Returns None rather than falling back to the crop: a run that silently mixed the two
    receptors would be uninterpretable, so the caller must decide what to do.
    """
    crop = find_pocket_pdb(target_dir)
    if crop is None:
        return None
    base = Path(crop).name[:-len("_pocket10.pdb")]
    ligbase = base.split("__", 1)[1] if "__" in base else base
    stem = ligbase[:10] + ".pdb"                    # PDBId_Chain_rec.pdb
    root = Path(FULL_RECEPTOR_ROOT)
    if "__" in base:
        cand = root / base.split("__", 1)[0] / stem
        if cand.exists():
            return cand
    hits = sorted(root.glob(f"*/{stem}"))
    return hits[0] if len(hits) == 1 else None

def _has_pose_for_mode(row: dict, mode: str) -> bool:
    """Cached row already carries every pose block `mode` needs (error-free).

    A block whose top-level value is an error dict (protein-load / worker
    failure) counts as unsatisfied so it retries; deterministic per-field issues
    live under the block's own `errors` key and do not force a recompute.
    """
    if mode == "none":
        return True
    for blk in _pose_blocks(mode):
        b = row.get(blk)
        if not isinstance(b, dict) or "error" in b:
            return False
    return True


def cache_satisfies(target_dir: Path, docking: str, pose: str = "none") -> bool:
    """True when target_dir/metrics.json is fresh AND has chem for every sample
    AND — if `docking`/`pose` are requested — the required vina fields / pose
    blocks for every sample and (when present) the reference ligand.

    Drives the batch "skip-existing" toggle: a target whose cache already
    satisfies the requested modes is skipped without recomputing anything.
    """
    if not metrics_are_fresh(target_dir):
        return False
    data = load_metrics(target_dir)
    if data is None:
        return False
    samples = data.get("samples", [])
    if not samples or not all(_has_chem(s) for s in samples):
        return False
    if docking != "none":
        if not all(_has_vina_for_mode(s.get("vina"), docking) for s in samples):
            return False
        ref = data.get("reference")
        if isinstance(ref, dict) and "error" not in ref:
            if not _has_vina_for_mode(ref.get("vina"), docking):
                return False
    if pose != "none":
        if not all(_has_pose_for_mode(s, pose) for s in samples):
            return False
        ref = data.get("reference")
        if isinstance(ref, dict) and "error" not in ref:
            if not _has_pose_for_mode(ref, pose):
                return False
    return True


def ref_satisfies(target_dir: Path, docking: str) -> bool:
    """True when target_dir's cached reference already has the requested vina
    mode. Used by the batch 'skip-existing' toggle for ref-only recomputes."""
    if not metrics_are_fresh(target_dir):
        return False
    data = load_metrics(target_dir)
    if data is None:
        return False
    ref = data.get("reference")
    if not isinstance(ref, dict) or "error" in ref:
        return False
    return _has_vina_for_mode(ref.get("vina"), docking)


def _tanimoto_diversity(mols: list[Chem.Mol]) -> float:
    if len(mols) < 2:
        return float("nan")
    fps = [Chem.RDKFingerprint(m) for m in mols]
    sims: list[float] = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return 1.0 - float(np.mean(sims))


def _similarity_to_reference(
    mols: list[Chem.Mol], ref_mol: Chem.Mol | None
) -> list[float | None]:
    """Tanimoto similarity (RDKit fingerprint) of each mol to the reference ligand.

    Uses the same `Chem.RDKFingerprint` as `_tanimoto_diversity`, so the two
    similarity numbers are directly comparable. Returns one value per input mol
    — None for the whole list when there is no reference ligand, or for any
    single mol whose fingerprint cannot be built.
    """
    if ref_mol is None:
        return [None] * len(mols)
    try:
        ref_fp = Chem.RDKFingerprint(ref_mol)
    except Exception:  # noqa: BLE001
        return [None] * len(mols)
    out: list[float | None] = []
    for m in mols:
        try:
            out.append(float(
                DataStructs.TanimotoSimilarity(Chem.RDKFingerprint(m), ref_fp)
            ))
        except Exception:  # noqa: BLE001
            out.append(None)
    return out


def _mol_xyz(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])


def _pocket_ligand_contacts(
    pocket_coords: np.ndarray,
    pocket_elems: list[str],
    mol: Chem.Mol,
    contact_cutoff: float = 4.0,
    clash_cutoff: float = 2.0,
) -> dict:
    if len(pocket_coords) == 0:
        return {"n_contacts": 0, "n_clashes": 0, "min_dist": float("nan"), "closest": []}
    lig_xyz = _mol_xyz(mol)
    lig_elems = [a.GetSymbol() for a in mol.GetAtoms()]
    diffs = lig_xyz[:, None, :] - pocket_coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    min_per_atom = dists.min(axis=1)
    closest_pi = dists.argmin(axis=1)
    closest = []
    for i, (md, pi) in enumerate(zip(min_per_atom, closest_pi)):
        if md < contact_cutoff:
            closest.append({
                "lig_atom": f"{lig_elems[i]}{i}",
                "pocket_atom": f"{pocket_elems[int(pi)]}{int(pi)}",
                "dist": round(float(md), 3),
            })
    closest.sort(key=lambda r: r["dist"])
    return {
        "n_contacts": int((dists < contact_cutoff).any(axis=1).sum()),
        "n_clashes": int((dists < clash_cutoff).any(axis=1).sum()),
        "min_dist": float(min_per_atom.min()),
        "closest": closest[:10],
    }


# ── AutoDock Vina docking (optional) ──────────────────────────────────────────
_VINA_IMPORT_ERROR: str | None = None


def _heal_path() -> None:
    """Ensure conda-env binaries (python3, pdb2pqr30, vina) shadow any system
    alternatives on PATH.

    docking_vina.py shells out to them via bare name. If a system Python (e.g.
    /compuworks/anaconda3/bin/python3) appears on PATH *before* the conda env,
    prepare_receptor4.py's subprocess lands on the wrong interpreter — which
    lacks MolKit — and fails silently with ModuleNotFoundError (Popen discards
    stderr and doesn't check rc), producing no .pdbqt. The next docking step
    then crashes with `'PrepProt' object has no attribute 'prot_pqr'` because
    VinaDockingTask.run() skips addH when .pqr already exists and never sets
    that attribute. So always *prepend* the conda env bin, even if it's
    already somewhere later on PATH — being present isn't enough; it has to
    win bare-name resolution.
    """
    _bin = str(Path(sys.executable).parent)
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p != _bin]
    os.environ["PATH"] = os.pathsep.join([_bin, *parts])


def docking_available() -> bool:
    """True if TargetDiff's Vina docking stack imports cleanly.

    Records the failure reason in `_VINA_IMPORT_ERROR` when it does not.
    """
    global _VINA_IMPORT_ERROR
    try:
        from utils.evaluation.docking_vina import VinaDockingTask  # noqa: F401
    except Exception as e:  # noqa: BLE001 — ImportError or native-lib load error
        _VINA_IMPORT_ERROR = f"{type(e).__name__}: {e}"
        return False
    _VINA_IMPORT_ERROR = None
    return True


def run_vina_docking(
    mol: Chem.Mol,
    receptor_pdb: Path | str,
    mode: str = "vina_score",
    exhaustiveness: int = 8,
    tmp_dir: Path | str = "./tmp",
    cpu: int = 8,
) -> dict:
    """Dock one molecule against `receptor_pdb` with AutoDock Vina.

    Returns the affinities computed for `mode` — {"score_only": ..[,
    "minimize": ..][, "dock": ..]} — or {"error": ".."} on any failure
    (missing toolchain, bad pose, ...). Never raises, so callers attach the
    result verbatim to a sample record.

    `mode` selects how far to go: "vina_score" (score_only — the pose as-is),
    "vina_min" (+ local minimize), or "vina_dock" (+ full re-docking).
    `cpu` caps Vina's threads — 8 is a good speed/courtesy point (a docking-speed
    sweep showed returns flatten past ~16; Vina's all-cores default is no faster).
    """
    _heal_path()
    try:
        from utils.evaluation.docking_vina import VinaDockingTask
    except Exception as e:  # noqa: BLE001
        return {"error": f"toolchain unavailable: {type(e).__name__}: {e}"}
    try:
        task = VinaDockingTask(str(receptor_pdb), deepcopy(mol), tmp_dir=str(tmp_dir))
        result = {
            "score_only": float(
                task.run(mode="score_only", exhaustiveness=exhaustiveness,
                         cpu=cpu)[0]["affinity"]
            ),
        }
        if mode in ("vina_min", "vina_dock"):
            result["minimize"] = float(
                task.run(mode="minimize", exhaustiveness=exhaustiveness,
                         cpu=cpu)[0]["affinity"]
            )
        if mode == "vina_dock":
            result["dock"] = float(
                task.run(mode="dock", exhaustiveness=exhaustiveness,
                         cpu=cpu)[0]["affinity"]
            )
        return result
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _prepare_receptor(receptor_cached: Path) -> None:
    """Build the receptor .pqr/.pdbqt once, before the parallel docking pool.

    VinaDockingTask.run() prepares these lazily and guards them with
    os.path.exists — fine serially, but a race when many worker processes hit
    a cold cache at once. Doing it up front makes every worker a pure reader.

    We invoke the prep tools directly (rather than via TargetDiff's PrepProt
    which silences stderr and doesn't check rc) so the actual upstream cause
    of a failure — pdb2pqr30 "gap in biomolecule structure", missing MolKit
    in prepare_receptor4.py, etc. — propagates to the RuntimeError message
    (and into the per-sample `vina.error` + the new top-level `receptor_prep`
    field in metrics.json). Each subprocess's stdout/stderr is also persisted
    to `<receptor>.prep.log` next to the cache for offline inspection.

    Multi-chain fallback: when pdb2pqr30 fails with "gap in biomolecule
    structure" (typical for protein-protein-interface pockets where the carve
    spans non-contiguous chains — see target_71 / TIAM1 4gvd), bypass pdb2pqr30
    and feed the .pdb directly to prepare_receptor4.py with -A hydrogens. That
    uses Gasteiger charges (AutoDock default) instead of AMBER charges, and
    preserves every atom of every chain — no biological bias from chain
    removal. The Vina dock numbers from a fallback target are charge-model-
    distinct from the rest of the run; flagged in receptor_prep.note so a
    reader knows.
    """
    import subprocess as _sp
    import AutoDockTools as _adt
    _heal_path()

    base = str(receptor_cached)[:-4]
    pqr, pdbqt = base + ".pqr", base + ".pdbqt"
    prep_log = Path(base + ".prep.log")
    prepare_receptor4 = os.path.join(
        _adt.__path__[0], "Utilities24/prepare_receptor4.py")
    # Note: the multi-chain fallback path produces .pdbqt without writing .pqr,
    # so only require .pdbqt for the early-return check.
    if os.path.exists(pdbqt):
        return

    log_buf: list[str] = []

    def _capture(cmd: list[str], stage: str) -> _sp.CompletedProcess:
        r = _sp.run(cmd, capture_output=True, text=True)
        log_buf.append(
            f"=== {stage} (rc={r.returncode}) ===\n"
            f"$ {' '.join(cmd)}\n"
            f"--- stdout ---\n{r.stdout}\n"
            f"--- stderr ---\n{r.stderr}\n"
        )
        return r

    def _raise(stage: str, r: _sp.CompletedProcess) -> None:
        prep_log.write_text("\n".join(log_buf))
        err_lines = r.stderr.strip().splitlines()
        # Prefer lines that look like exception messages (no leading whitespace,
        # not 'File "..."' frames) — they carry the actual cause (e.g.
        # "ValueError: Biomolecular structure is incomplete").
        informative = [
            ln for ln in err_lines
            if ln and not ln.startswith((" ", "\t", "File ", "Traceback"))
        ]
        summary = "; ".join(informative[-4:]) if informative \
            else "\n  ".join(err_lines[-15:]) or f"(empty — see {prep_log})"
        raise RuntimeError(
            f"{stage} failed for {receptor_cached.name} (rc={r.returncode}). "
            f"{summary}"
        )

    # Step 1: produce .pqr via pdb2pqr30 (AMBER). On multi-chain-gap, skip
    # this step and remember to go direct in step 2.
    multichain_fallback = False
    if not os.path.exists(pqr):
        r = _capture(
            ["pdb2pqr30", "--ff=AMBER", str(receptor_cached), pqr],
            stage="pdb2pqr30",
        )
        if r.returncode != 0 or not os.path.exists(pqr):
            if "gap in biomolecule" in r.stderr.lower():
                multichain_fallback = True
                log_buf.append(
                    "=== multi-chain fallback triggered: pdb2pqr30 hit a "
                    "structural gap; prepare_receptor4.py will read the .pdb "
                    "directly (Gasteiger charges, all atoms preserved) ===\n"
                )
            else:
                _raise("pdb2pqr30", r)

    # Step 2: produce .pdbqt. Normal path takes the .pqr; the multi-chain
    # fallback path reads the original .pdb with -A hydrogens.
    if not os.path.exists(pdbqt):
        if multichain_fallback:
            cmd = [sys.executable, prepare_receptor4,
                   "-r", str(receptor_cached), "-o", pdbqt,
                   "-A", "hydrogens", "-U", "nphs_lps_waters"]
            stage = "prepare_receptor4.py [direct, multi-chain fallback]"
        else:
            cmd = [sys.executable, prepare_receptor4, "-r", pqr, "-o", pdbqt]
            stage = "prepare_receptor4.py"
        r = _capture(cmd, stage=stage)
        if r.returncode != 0 or not os.path.exists(pdbqt):
            _raise(stage, r)

    # Persist diagnostics for offline inspection (successful or not).
    if log_buf:
        prep_log.write_text("\n".join(log_buf))

    # Stash a side-marker so compute_target_metrics can mention the fallback
    # in metrics.json (different charge model -> reader should know).
    if multichain_fallback:
        Path(base + ".multichain_fallback").write_text(
            "prepare_receptor4.py was invoked directly on the multi-chain .pdb "
            "(Gasteiger charges) because pdb2pqr30 failed with a "
            "'gap in biomolecule structure' error.\n"
        )


def _high_affinity(sample_rows: list[dict], reference: dict | None) -> dict:
    """High Affinity — share of samples whose Vina Dock score beats the reference.

    A sample is "high affinity" when its Vina Dock affinity is lower (more
    negative) than the crystal reference ligand's. Returns {} unless both the
    reference and at least one sample carry a finite `dock` score — i.e. it is
    only defined when docking ran in `vina_dock` mode.
    """
    ref_vina = reference.get("vina") if isinstance(reference, dict) else None
    ref_dock = ref_vina.get("dock") if isinstance(ref_vina, dict) else None
    if not isinstance(ref_dock, (int, float)):
        return {}
    dock_vals = [
        s["vina"]["dock"] for s in sample_rows
        if isinstance(s.get("vina"), dict) and "error" not in s["vina"]
        and isinstance(s["vina"].get("dock"), (int, float))
    ]
    if not dock_vals:
        return {}
    n_better = sum(1 for v in dock_vals if v < ref_dock)
    return {
        "high_affinity": n_better / len(dock_vals),
        "high_affinity_n": n_better,
        "high_affinity_total": len(dock_vals),
    }


def _aggregate_vina(sample_rows: list[dict], reference: dict | None = None) -> dict:
    """Mean Vina affinities over samples that docked successfully.

    When the reference ligand also has a Vina Dock score, the High Affinity
    fraction (samples beating the reference) is added too.
    """
    out: dict = {}
    for src, dst in (("score_only", "vina_score_mean"),
                     ("minimize", "vina_min_mean"),
                     ("dock", "vina_dock_mean")):
        vals = [
            s["vina"][src] for s in sample_rows
            if isinstance(s.get("vina"), dict) and "error" not in s["vina"]
            and isinstance(s["vina"].get(src), (int, float))
        ]
        if vals:
            out[dst] = float(np.mean(vals))
    n_ok = sum(
        1 for s in sample_rows
        if isinstance(s.get("vina"), dict) and "error" not in s["vina"]
    )
    out["vina_n_docked"] = n_ok
    out["vina_n_failed"] = len(sample_rows) - n_ok
    out.update(_high_affinity(sample_rows, reference))
    return out


# ── PoseCheck + PoseBusters bridge (runs in the `moleval` env) ─────────────────
def pose_eval_available() -> bool:
    """True if the moleval interpreter and pose_eval.py worker are both present.

    Records why in `_POSE_IMPORT_ERROR` when not — mirrors `docking_available`.
    """
    global _POSE_IMPORT_ERROR
    if not os.path.exists(_MOLEVAL_PY):
        _POSE_IMPORT_ERROR = f"moleval python not found: {_MOLEVAL_PY} (set $MOLEVAL_PY)"
        return False
    if not os.path.exists(_POSE_WORKER):
        _POSE_IMPORT_ERROR = f"pose_eval.py worker not found: {_POSE_WORKER}"
        return False
    _POSE_IMPORT_ERROR = None
    return True


def run_pose_eval(
    mols: list[Chem.Mol],
    receptor_pdb: Path | str,
    mode: str = "all",
    tmp_dir: Path | str | None = None,
    chunk_size: int | None = None,
) -> list[dict]:
    """Score mols in bounded moleval subprocess chunks.

    Results are merged in input order. A worker failure is local to its chunk,
    so completed chunks are retained and only failed rows retry on the next
    smart-compute pass.
    """
    if not mols:
        return []
    blocks = _pose_blocks(mode)
    if not pose_eval_available():
        reason = f"pose toolchain unavailable: {_POSE_IMPORT_ERROR}"
        return [{b: {"error": reason} for b in blocks} for _ in mols]

    size = _POSE_CHUNK_SIZE if chunk_size is None else int(chunk_size)
    if size < 1:
        raise ValueError(f"pose chunk_size must be >= 1, got {size}")
    n_chunks = (len(mols) + size - 1) // size
    results: list[dict] = []
    for chunk_index, start in enumerate(range(0, len(mols), size), 1):
        stop = min(start + size, len(mols))
        label = f"chunk {chunk_index}/{n_chunks} (mols {start + 1}-{stop})"
        results.extend(_run_pose_eval_chunk(
            mols[start:stop],
            receptor_pdb,
            mode=mode,
            blocks=blocks,
            tmp_dir=tmp_dir,
            label=label,
            chunk_index=chunk_index,
        ))
    return results


def _run_pose_eval_chunk(
    mols: list[Chem.Mol],
    receptor_pdb: Path | str,
    *,
    mode: str,
    blocks: tuple[str, ...],
    tmp_dir: Path | str | None,
    label: str,
    chunk_index: int,
) -> list[dict]:
    """Run one bounded moleval worker and return input-aligned records."""

    def _err(reason: str) -> list[dict]:
        tagged = f"{label}: {reason}"
        return [{b: {"error": tagged} for b in blocks} for _ in mols]

    tmp = Path(tempfile.mkdtemp(
        prefix=f"poseeval_{chunk_index:03d}_",
        dir=str(tmp_dir) if tmp_dir else None,
    ))
    sdf_path, out_path = tmp / "ligands.sdf", tmp / "pose.json"
    try:
        with Chem.SDWriter(str(sdf_path)) as writer:
            for mol in mols:
                writer.write(mol)
        # Prepend the moleval bin so posecheck hydride/reduce shell-outs resolve.
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(
            [str(Path(_MOLEVAL_PY).parent), env.get("PATH", "")])
        proc = subprocess.run(
            [_MOLEVAL_PY, _POSE_WORKER, "--protein", str(receptor_pdb),
             "--ligands", str(sdf_path), "--out", str(out_path), "--pose", mode],
            env=env, capture_output=True, text=True, timeout=_POSE_TIMEOUT_S,
        )
        if proc.returncode != 0 or not out_path.exists():
            tail = " | ".join((proc.stderr or "").strip().splitlines()[-3:])
            return _err(f"worker rc={proc.returncode}: {tail}")
        chunk_results = json.loads(out_path.read_text())
        if len(chunk_results) != len(mols):
            return _err(
                f"worker returned {len(chunk_results)} records for {len(mols)} mols"
            )
        return [{b: row[b] for b in blocks if b in row}
                for row in chunk_results]
    except subprocess.TimeoutExpired:
        return _err(f"worker timed out after {_POSE_TIMEOUT_S:g}s")
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _aggregate_pose(sample_rows: list[dict]) -> dict:
    """Aggregate PoseCheck + PoseBusters over samples that were scored.

    Strain energy is summarised by median (its distribution is heavy-tailed, so
    the median is the robust central value the SBDD literature reports); clashes
    by both mean and median. PoseBusters contributes `pb_valid_rate` — the share
    of samples that pass *every* validity check.
    """
    out: dict = {}

    def _vals(blk: str, key: str) -> list[float]:
        v = []
        for s in sample_rows:
            b = s.get(blk)
            if isinstance(b, dict) and isinstance(b.get(key), (int, float)):
                v.append(b[key])
        return v

    clashes = _vals("posecheck", "clashes")
    if clashes:
        out["clashes_mean"] = float(np.mean(clashes))
        out["clashes_median"] = float(np.median(clashes))
    strain = _vals("posecheck", "strain")
    if strain:
        out["strain_mean"] = float(np.mean(strain))
        out["strain_median"] = float(np.median(strain))
    ninter = _vals("posecheck", "n_interactions")
    if ninter:
        out["n_interactions_mean"] = float(np.mean(ninter))
    pb_valid = [bool(s["posebusters"]["valid"]) for s in sample_rows
                if isinstance(s.get("posebusters"), dict)
                and isinstance(s["posebusters"].get("valid"), bool)]
    if pb_valid:
        out["pb_valid_rate"] = float(np.mean(pb_valid))
        out["pb_valid_n"] = int(sum(pb_valid))
        out["pb_valid_total"] = len(pb_valid)
    return out


def _reference_row(target_dir: Path):
    """Chemical metrics for the crystal reference ligand — the per-target baseline.

    Returns (row, mol): `row` is shaped like a sample row minus `vina` (docking
    is attached later, alongside the samples) and `mol` is kept so the caller
    can dock it in the same pool. Returns (None, None) when the target has no
    reference-ligand SDF.
    """
    gt_mols = load_gt_mols(target_dir)
    if not gt_mols:
        return None, None
    mol = gt_mols[0]  # TargetDiff convention: one reference ligand per pocket
    try:
        smi = Chem.MolToSmiles(mol)
    except Exception:  # noqa: BLE001
        smi = ""
    c = get_chem(mol)
    row = {
        "smiles": smi,
        "qed": float(c["qed"]),
        "sa": float(c["sa"]),
        "logp": float(c["logp"]),
        "lipinski": int(c["lipinski"]),
        "n_atoms": int(mol.GetNumHeavyAtoms()),
    }
    return row, mol


def compute_target_metrics(
    target_dir: Path,
    pocket_coords: np.ndarray,
    pocket_elems: list[str],
    docking: str = "none",
    receptor_pdb: Path | None = None,
    exhaustiveness: int = 8,
    cpu: int = 8,
    workers: int | None = None,
    progress_cb: ProgressCb | None = None,
    force: bool = False,
    pose: str = "none",
    pose_scope: str = "crop",
    dock_scope: str = "crop",
) -> dict:
    """Score every valid sample under `target_dir` and write metrics.json.

    docking: "none" | "vina_score" (score_only + minimize) | "vina_dock"
             (+ full re-docking). When enabled, every sample gains a `vina`
             dict and the aggregates gain vina_*_mean keys. The receptor is
             `receptor_pdb` (defaults to the target's *_pocket10.pdb); its
             prepared .pqr/.pdbqt are cached under `target_dir/.vina_cache/`.
    workers: parallel docking processes, each using `cpu` Vina threads. None →
             cpu_count // cpu. Pure scheduling — every dock is the identical
             computation, so metrics.json matches a serial run.
    force:   when True, ignore any cached metrics.json and recompute every
             sample's chem and vina from scratch. Default False enables
             *smart compute*: when the cache is fresh, per-sample chem rows
             (and per-sample vina results that already satisfy the requested
             mode) are reused — only what's missing is (re)computed.

    The crystal reference ligand is scored the same way under the top-level
    `reference` key — a baseline to read the generated samples against. Each
    sample also carries `sim_to_ref` — its RDKit-fingerprint Tanimoto
    similarity to that reference ligand.

    progress_cb: optional progress_cb(done, total, stage_label) — called once
                 per molecule for the chemical-scoring and docking phases.
    """
    sdf = target_dir / "samples.sdf"
    if not sdf.exists():
        raise FileNotFoundError(f"No samples.sdf in {target_dir}")
    if docking not in DOCKING_MODES:
        raise ValueError(f"docking must be one of {DOCKING_MODES}, got {docking!r}")

    dock_on = docking != "none"
    cache_dir: Path | None = None
    receptor_cached: Path | None = None
    if dock_on:
        # --dock-scope full mirrors --pose-scope full for the affinity side.
        # TargetDiff's VinaDockingTask resolves the WHOLE receptor
        # (<protein_root>/<subdir>/<ligand[:10]>.pdb), not the 10 Å crop, so a
        # crop-scored run is not comparable to published Vina numbers. It wins
        # over an explicitly passed receptor_pdb, which is the caller's default
        # crop lookup rather than a deliberate choice.
        if dock_scope == "full":
            receptor_pdb = find_full_receptor(target_dir)
            if receptor_pdb is None:
                raise FileNotFoundError(
                    f"dock_scope='full' could not resolve a whole receptor for "
                    f"{target_dir} under {FULL_RECEPTOR_ROOT}"
                )
        if receptor_pdb is None:
            receptor_pdb = find_pocket_pdb(target_dir)
        if receptor_pdb is None:
            raise FileNotFoundError(
                f"docking={docking!r} needs a *_pocket10.pdb in {target_dir}"
            )
        # Dock against a copy inside .vina_cache/ so the prepared receptor
        # files (.pqr/.pdbqt) and ligand scratch never pollute the target dir.
        cache_dir = target_dir / ".vina_cache"
        cache_dir.mkdir(exist_ok=True)
        receptor_cached = cache_dir / Path(receptor_pdb).name
        if not receptor_cached.exists():
            shutil.copy(receptor_pdb, receptor_cached)

    n_raw = sum(1 for _ in Chem.SDMolSupplier(str(sdf), sanitize=False))

    # Smart compute: when an existing metrics.json is still fresh (samples.sdf
    # unchanged since the last write), per-sample chemistry and per-sample
    # docking results are reused verbatim from the cache, keyed by canonical
    # SMILES. Only what's missing for the requested `docking` mode gets
    # (re)computed. `force=True` skips this and rebuilds from scratch.
    prev: dict | None = None
    cached_by_smi: dict[str, dict] = {}
    if not force and metrics_are_fresh(target_dir):
        prev = load_metrics(target_dir)
        # A cached pose block measured against a different receptor is not reusable:
        # samples.sdf is unchanged, so nothing else would catch the mismatch.
        if (prev is not None and pose != "none"
                and prev.get("pose_receptor_scope", "crop") != pose_scope):
            prev = None
        # Same for docking: a cached affinity measured against the crop is not a
        # full-receptor affinity, and samples.sdf being unchanged hides that.
        if (prev is not None and dock_on
                and prev.get("dock_receptor_scope", "crop") != dock_scope):
            prev = None
        if prev is not None:
            for s in prev.get("samples", []):
                if isinstance(s, dict) and isinstance(s.get("smiles"), str):
                    cached_by_smi[s["smiles"]] = s

    # ── chemical metrics — cheap, computed serially; reuse cache when possible
    valid_mols: list[Chem.Mol] = []
    sample_rows: list[dict] = []
    for i, m in enumerate(Chem.SDMolSupplier(str(sdf), sanitize=True)):
        if progress_cb is not None:
            progress_cb(i + 1, n_raw, "Scoring molecules")
        if m is None:
            continue
        try:
            smi = Chem.MolToSmiles(m)
        except Exception:
            continue
        if "." in smi:
            continue
        valid_mols.append(m)
        cached = cached_by_smi.get(smi)
        if cached is not None and _has_chem(cached):
            # Cached row carries chem (+ any prior vina / sim_to_ref); reuse it
            # so chem-only recomputes never lose cached docking results.
            sample_rows.append(deepcopy(cached))
        else:
            c = get_chem(m)
            ix = _pocket_ligand_contacts(pocket_coords, pocket_elems, m)
            sample_rows.append({
                "smiles": smi,
                "qed": float(c["qed"]),
                "sa": float(c["sa"]),
                "logp": float(c["logp"]),
                "lipinski": int(c["lipinski"]),
                "n_atoms": int(m.GetNumHeavyAtoms()),
                "interactions": ix,
            })

    # Reference-ligand baseline (chem metrics here; docked with the samples
    # below). It must never break the core metrics, so an unexpected failure is
    # recorded in-band rather than raised.
    try:
        reference, ref_mol = _reference_row(target_dir)
    except Exception as e:  # noqa: BLE001
        reference = {"error": f"reference scoring failed: {type(e).__name__}: {e}"}
        ref_mol = None

    # Reference docking is fixed per target (the crystal ligand never changes),
    # so reuse a cached `reference.vina` whenever available. The docking
    # section below will only re-dock the reference if the requested mode adds
    # fields the cache lacks.
    if isinstance(reference, dict) and isinstance(prev, dict):
        prev_ref = prev.get("reference")
        if isinstance(prev_ref, dict) and isinstance(prev_ref.get("vina"), dict):
            reference["vina"] = prev_ref["vina"]

    # Per-sample Tanimoto similarity (RDKit fingerprint) to the reference
    # ligand. Reused samples already carry it (the reference is fixed per
    # target), so we only compute it for rows where it's missing.
    missing_sim = [i for i, row in enumerate(sample_rows)
                   if "sim_to_ref" not in row]
    if missing_sim:
        sims = _similarity_to_reference(
            [valid_mols[i] for i in missing_sim], ref_mol)
        for idx, sim in zip(missing_sim, sims):
            if sim is not None:
                sample_rows[idx]["sim_to_ref"] = sim

    # ── Vina docking — parallel across molecules ─────────────────────────────
    # Each molecule docks independently, so a process pool gives a large
    # speedup with per-dock-identical results. The receptor .pqr/.pdbqt are
    # prepared once up front so workers never race on that cache; if prep fails
    # (e.g. toolchain missing) we fall back to serial and each dock reports its
    # own error.
    if dock_on:
        # Smart-dock: only dock molecules whose cached `vina` doesn't already
        # satisfy the requested mode. `slot == -1` marks the reference ligand;
        # any other slot is the index into `sample_rows`.
        needed: list[tuple[int, Chem.Mol]] = [
            (i, valid_mols[i]) for i, row in enumerate(sample_rows)
            if not _has_vina_for_mode(row.get("vina"), docking)
        ]
        if ref_mol is not None and not _has_vina_for_mode(
                reference.get("vina"), docking):
            needed.append((-1, ref_mol))

        receptor_prep_error: str | None = None
        if needed:
            try:
                _prepare_receptor(receptor_cached)
                n_workers = workers or max(1, (os.cpu_count() or 8) // max(1, cpu))
            except Exception as e:  # noqa: BLE001 — workers surface the error per-dock
                # Preserve the upstream cause (pdb2pqr30 stderr, missing dep, ...)
                # so the metrics.json reader can see WHY every sample's vina.error
                # is the same downstream "pocket10.pdbqt does not exist" — without
                # this, the actual root cause is only in the .prep.log file.
                receptor_prep_error = f"{type(e).__name__}: {e}"
                n_workers = 1
            n_workers = min(n_workers, len(needed))
            dock_one = partial(
                run_vina_docking, receptor_pdb=receptor_cached, mode=docking,
                exhaustiveness=exhaustiveness, tmp_dir=cache_dir, cpu=cpu,
            )
            n_dock = len(needed)
            if progress_cb is not None:
                progress_cb(0, n_dock, "Docking molecules")

            def _place(slot: int, result: dict) -> None:
                if slot == -1:
                    reference["vina"] = result
                else:
                    sample_rows[slot]["vina"] = result

            # run_vina_docking already catches its own exceptions and returns
            # {"error": ...}, but a worker process that dies hard (segfault,
            # OOM-kill, BrokenProcessPool, signal) makes fut.result() re-raise
            # — that would abort the whole target without writing metrics.json.
            # Catch here so one bad sample never sinks the rest of the target;
            # the crashed sample is recorded as a vina.error and the smart-dock
            # logic (`_has_vina_for_mode`) will retry it on the next run.
            if n_workers <= 1:
                for i, (slot, m) in enumerate(needed):
                    try:
                        result = dock_one(m)
                    except Exception as e:  # noqa: BLE001 — defensive; run_vina_docking should already catch
                        result = {"error": f"dock raised: {type(e).__name__}: {e}"}
                    _place(slot, result)
                    if progress_cb is not None:
                        progress_cb(i + 1, n_dock, "Docking molecules")
            else:
                # submit + as_completed (not pool.map) so the progress bar
                # advances as each dock finishes; futs maps each future back
                # to its slot so results land in the right row.
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futs = {pool.submit(dock_one, m): slot
                            for slot, m in needed}
                    for done, fut in enumerate(as_completed(futs), 1):
                        slot = futs[fut]
                        try:
                            result = fut.result()
                        except Exception as e:  # noqa: BLE001 — worker died (segfault, OOM-kill, BrokenProcessPool, ...)
                            result = {"error": f"worker crashed: {type(e).__name__}: {e}"}
                        _place(slot, result)
                        if progress_cb is not None:
                            progress_cb(done, n_dock, "Docking molecules")

    # ── PoseCheck + PoseBusters — one moleval subprocess over the needed mols ─
    # Runs after chem/vina so reused (SMILES-keyed) rows already carry any prior
    # pose blocks; only rows missing them for the requested `pose` are scored.
    pose_error: str | None = None
    if pose != "none":
        # Clashes are measured against whatever protein is handed in; --pose-scope full
        # swaps the 10 Å crop for the whole receptor so the count is not an under-count.
        if pose_scope == "full":
            pose_receptor = find_full_receptor(target_dir)
            if pose_receptor is None:
                pose_error = (f"pose_scope='full' could not resolve a whole receptor for "
                              f"{target_dir} under {FULL_RECEPTOR_ROOT}")
        else:
            pose_receptor = receptor_pdb or find_pocket_pdb(target_dir)
            if pose_receptor is None:
                pose_error = f"pose={pose!r} needs a *_pocket10.pdb in {target_dir}"
        if pose_receptor is None:
            pass
        else:
            needed_pose: list[tuple[int, Chem.Mol]] = [
                (i, valid_mols[i]) for i, row in enumerate(sample_rows)
                if not _has_pose_for_mode(row, pose)
            ]
            if ref_mol is not None and not _has_pose_for_mode(reference, pose):
                needed_pose.append((-1, ref_mol))
            if needed_pose:
                if progress_cb is not None:
                    progress_cb(0, len(needed_pose), "Pose-checking molecules")
                pose_results = run_pose_eval(
                    [m for _, m in needed_pose], pose_receptor, mode=pose,
                    tmp_dir=cache_dir,
                )
                blocks = _pose_blocks(pose)
                for (slot, _), res in zip(needed_pose, pose_results):
                    dst = reference if slot == -1 else sample_rows[slot]
                    for blk in blocks:
                        if blk in res:
                            dst[blk] = res[blk]
                if progress_cb is not None:
                    progress_cb(len(needed_pose), len(needed_pose),
                                "Pose-checking molecules")

    # ── aggregate metrics (vina_* means see every docked sample) ─────────────
    if sample_rows:
        smiles_list = [s["smiles"] for s in sample_rows]
        aggregates = {
            "n_total": n_raw,
            "n_valid": len(sample_rows),
            "validity": len(sample_rows) / n_raw if n_raw else float("nan"),
            "uniqueness": len(set(smiles_list)) / len(sample_rows),
            "diversity": _tanimoto_diversity(valid_mols),
            "qed_mean": float(np.mean([s["qed"] for s in sample_rows])),
            "sa_mean": float(np.mean([s["sa"] for s in sample_rows])),
            "logp_mean": float(np.mean([s["logp"] for s in sample_rows])),
            "lipinski_mean": float(np.mean([s["lipinski"] for s in sample_rows])),
        }
        sim_vals = [s["sim_to_ref"] for s in sample_rows if "sim_to_ref" in s]
        if sim_vals:
            aggregates["sim_to_ref_mean"] = float(np.mean(sim_vals))
        if dock_on:
            aggregates.update(_aggregate_vina(sample_rows, reference))
        if pose != "none":
            aggregates.update(_aggregate_pose(sample_rows))
    else:
        aggregates = {
            "n_total": n_raw,
            "n_valid": 0,
            "validity": 0.0 if n_raw else float("nan"),
        }

    data = {
        "version": METRICS_VERSION,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "docking": docking,
        "pose": pose,
        "pose_receptor_scope": pose_scope if pose != "none" else None,
        "dock_receptor_scope": dock_scope if dock_on else None,
        "reference": reference,
        "samples": sample_rows,
        "aggregates": aggregates,
    }
    if pose_error:
        data["pose_error"] = pose_error
    # When docking was requested but the receptor-prep step failed, every
    # sample ended up with a downstream "pocket10.pdbqt does not exist" error
    # — the root cause (pdb2pqr30 gap, missing dep, ...) only lived inside the
    # _prepare_receptor exception. Persist it here so a metrics.json reader
    # can see WHY all dockings failed in one glance, without digging into the
    # .prep.log file or rerunning. The key is only present when prep had issues.
    _prep_err = locals().get("receptor_prep_error")
    # Detect the multi-chain-fallback marker dropped by _prepare_receptor.
    _mc_marker = None
    if dock_on and receptor_cached is not None:
        cand = Path(str(receptor_cached)[:-4] + ".multichain_fallback")
        if cand.exists():
            _mc_marker = cand
    if dock_on and (_prep_err or _mc_marker):
        rp: dict = {}
        if _prep_err:
            rp["ok"] = False
            rp["error"] = _prep_err
        elif _mc_marker:
            # Prep succeeded but via the multi-chain fallback (Gasteiger
            # charges, not AMBER). The Vina dock numbers for this target are
            # not strictly charge-model-comparable to other targets in the run.
            rp["ok"] = True
            rp["fallback"] = "multi_chain_direct_pdbqt"
            rp["note"] = ("pdb2pqr30 failed with 'gap in biomolecule "
                          "structure'; prepare_receptor4.py was run directly "
                          "on the multi-chain .pdb (Gasteiger charges). All "
                          "atoms preserved; charge model differs from the "
                          "rest of the run (AMBER).")
        rp["log_hint"] = f".vina_cache/{Path(receptor_cached).stem}.prep.log"
        data["receptor_prep"] = rp
    metrics_path(target_dir).write_text(
        json.dumps(_sanitize_for_json(data), indent=2, allow_nan=False)
    )
    return data


def compute_reference(
    target_dir: Path,
    docking: str = "none",
    receptor_pdb: Path | None = None,
    exhaustiveness: int = 8,
    cpu: int = 8,
    progress_cb: ProgressCb | None = None,
) -> dict:
    """(Re)compute only the reference-ligand row and merge it into metrics.json.

    A fast, samples-untouched update — get the reference baseline (or its Vina
    docking) without re-scoring every generated sample. Requires an existing
    metrics.json, i.e. compute_target_metrics must have been run first.

    progress_cb: optional progress_cb(done, total, stage_label), called around
                 the single reference dock so a UI can show a live bar.
    """
    mp = metrics_path(target_dir)
    if not mp.exists():
        raise FileNotFoundError(
            f"no metrics.json in {target_dir} — compute the target first"
        )
    if docking not in DOCKING_MODES:
        raise ValueError(f"docking must be one of {DOCKING_MODES}, got {docking!r}")
    data = json.loads(mp.read_text())

    reference, ref_mol = _reference_row(target_dir)
    if reference is not None and docking != "none":
        if receptor_pdb is None:
            receptor_pdb = find_pocket_pdb(target_dir)
        if receptor_pdb is None:
            raise FileNotFoundError(
                f"docking={docking!r} needs a *_pocket10.pdb in {target_dir}"
            )
        cache_dir = target_dir / ".vina_cache"
        cache_dir.mkdir(exist_ok=True)
        receptor_cached = cache_dir / Path(receptor_pdb).name
        if not receptor_cached.exists():
            shutil.copy(receptor_pdb, receptor_cached)
        if progress_cb is not None:
            progress_cb(0, 1, "Docking reference")
        reference["vina"] = run_vina_docking(
            ref_mol, receptor_cached, mode=docking,
            exhaustiveness=exhaustiveness, tmp_dir=cache_dir, cpu=cpu,
        )
        if progress_cb is not None:
            progress_cb(1, 1, "Docking reference")
    elif reference is not None:
        # Chem-only recompute (docking="none"): preserve any docking the
        # previous reference had, so High Affinity — measured against the
        # reference's Vina Dock score — is not silently dropped.
        prev = data.get("reference")
        if isinstance(prev, dict) and isinstance(prev.get("vina"), dict):
            reference["vina"] = prev["vina"]

    data["reference"] = reference
    # High Affinity is measured against the reference's Vina Dock score, which
    # may have just changed — recompute it so the cached aggregates stay valid.
    agg = data.get("aggregates")
    if isinstance(agg, dict):
        for k in ("high_affinity", "high_affinity_n", "high_affinity_total"):
            agg.pop(k, None)
        agg.update(_high_affinity(data.get("samples", []), reference))
    mp.write_text(
        json.dumps(_sanitize_for_json(data), indent=2, allow_nan=False)
    )
    return data


def _sanitize_for_json(obj):
    """Replace non-finite floats with None so strict JSON parsers accept the file."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# ── CLI: offline batch docking ────────────────────────────────────────────────
def _iter_target_dirs(root: Path):
    """Yield `root` itself if it holds samples.sdf, else its target_* subdirs."""
    if (root / "samples.sdf").exists():
        yield root
        return
    yield from sorted(d for d in root.glob("target_*") if d.is_dir())


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Compute metrics.json (optionally with Vina docking and/or "
                    "PoseCheck+PoseBusters pose eval) for a target_XX/ dir or a "
                    "run dir of target_* folders.",
    )
    ap.add_argument("path", type=Path,
                    help="a target_XX/ dir, or a run dir containing target_* dirs")
    ap.add_argument("--docking", choices=DOCKING_MODES, default="none",
                    help="none | vina_score (score) | vina_min (+minimize) | "
                         "vina_dock (+full dock)")
    ap.add_argument("--pose", choices=POSE_MODES, default="none",
                    help="none | posecheck (clashes/strain/interactions) | "
                         "posebusters (validity) | all — runs in the moleval env")
    ap.add_argument("--exhaustiveness", type=int, default=8,
                    help="Vina exhaustiveness for dock mode (higher = slower)")
    ap.add_argument("--cpu", type=int, default=8,
                    help="Vina CPU threads per docking worker")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel docking workers (default: cpu_count // cpu)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip targets whose metrics.json already satisfies the "
                         "requested docking AND pose modes")
    ap.add_argument("--pose-scope", choices=("crop", "full"), default="crop",
                    help="receptor for the PoseCheck clash/interaction terms. 'crop' is "
                         "the target's *_pocket10.pdb (what every prior run used); 'full' "
                         "is the whole CrossDocked receptor. Strain is unaffected either "
                         "way -- it never sees the protein.")
    ap.add_argument("--dock-scope", choices=("crop", "full"), default="crop",
                    help="receptor for Vina: crop = the target's *_pocket10.pdb; "
                         "full = the whole receptor under $FULL_RECEPTOR_ROOT, which "
                         "is what TargetDiff's VinaDockingTask scores against")
    ap.add_argument("--force", action="store_true",
                    help="ignore the per-sample cache and recompute every block "
                         "from scratch. Needed when a measurement changed for a "
                         "reason the cache cannot detect (e.g. PoseCheck's strain "
                         "seed), since samples.sdf is unchanged and the cache "
                         "would otherwise be reused verbatim.")
    args = ap.parse_args()

    root = args.path.resolve()
    if not root.is_dir():
        ap.error(f"not a directory: {root}")

    if args.docking != "none" and not docking_available():
        print(f"WARNING: Vina toolchain unavailable ({_VINA_IMPORT_ERROR}).")
        print("         install with:")
        print("           pip install vina meeko pdb2pqr")
        print("           pip install git+https://github.com/Valdes-Tresanco-MS/"
              "AutoDockTools_py3.git")
        print("         proceeding anyway — each sample will record a vina.error.\n")

    if args.pose != "none" and not pose_eval_available():
        print(f"WARNING: pose toolchain unavailable ({_POSE_IMPORT_ERROR}).")
        print("         build the env, e.g.:")
        print("           conda create -n moleval python=3.10 -y")
        print("           conda run -n moleval pip install posebusters "
              "'pandas>=2.2.3' prolif datamol hydride biopython rdkit")
        print("           conda run -n moleval pip install --no-deps posecheck")
        print("           conda install -n moleval -c bioconda -c conda-forge "
              "reduce seaborn xorg-libxrender xorg-libxext -y")
        print("         proceeding anyway — each sample will record a pose error.\n")

    targets = list(_iter_target_dirs(root))
    if not targets:
        ap.error(f"no samples.sdf and no target_* dirs under {root}")

    print(f"{len(targets)} target(s)  |  docking={args.docking}  pose={args.pose}  "
          f"exhaustiveness={args.exhaustiveness}  cpu={args.cpu}  "
          f"workers={args.workers if args.workers else 'auto'}  "
          f"pose_chunk={_POSE_CHUNK_SIZE}  pose_timeout={_POSE_TIMEOUT_S:g}s")
    for i, tdir in enumerate(targets, 1):
        tag = f"[{i}/{len(targets)}] {tdir.name}"
        if args.skip_existing and cache_satisfies(
                tdir, args.docking, args.pose):
            print(f"{tag}: already computed — skipped")
            continue
        pdb = find_pocket_pdb(tdir)
        if pdb is None:
            print(f"{tag}: no *_pocket10.pdb — skipped")
            continue
        coords, elems = parse_pocket_pdb(pdb)
        t0 = time.time()
        try:
            data = compute_target_metrics(
                tdir, coords, elems,
                docking=args.docking, receptor_pdb=pdb,
                exhaustiveness=args.exhaustiveness, cpu=args.cpu,
                workers=args.workers, pose=args.pose, force=args.force,
                pose_scope=args.pose_scope, dock_scope=args.dock_scope,
            )
        except Exception as e:  # noqa: BLE001
            print(f"{tag}: FAILED — {type(e).__name__}: {e}")
            continue
        agg = data["aggregates"]
        msg = f"{tag}: {agg.get('n_valid', 0)}/{agg.get('n_total', 0)} valid"
        if args.docking != "none":
            vs = agg.get("vina_score_mean")
            msg += (f"  vina_score_mean={vs:.2f}"
                    if isinstance(vs, (int, float)) else "  vina: n/a")
            if agg.get("vina_n_failed"):
                msg += f" ({agg['vina_n_failed']} failed)"
            hf = agg.get("high_affinity")
            if isinstance(hf, (int, float)):
                msg += f"  HF={hf * 100:.1f}%"
            ref_vina = (data.get("reference") or {}).get("vina")
            if isinstance(ref_vina, dict) and isinstance(
                    ref_vina.get("score_only"), (int, float)):
                msg += f"  ref_score={ref_vina['score_only']:.2f}"
        if args.pose != "none":
            sm = agg.get("strain_median"); cm = agg.get("clashes_median")
            if isinstance(cm, (int, float)):
                msg += f"  clash_med={cm:.1f}"
            if isinstance(sm, (int, float)):
                msg += f"  strain_med={sm:.0f}"
            pv = agg.get("pb_valid_rate")
            if isinstance(pv, (int, float)):
                msg += f"  PB-valid={pv * 100:.0f}%"
            if data.get("pose_error"):
                msg += f"  pose:ERR({data['pose_error']})"
        print(f"{msg}  [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    _cli()
