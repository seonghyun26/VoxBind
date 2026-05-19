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
import sys
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

# `vina` fields and the `reference` baseline are additive & optional, so the
# cache format is unchanged: a metrics.json with or without them is version 2.
METRICS_VERSION = 2

# none       -> chemical metrics only
# vina_score -> + Vina score_only        (the generated pose, as-is)
# vina_min   -> + score_only & minimize  (local optimisation; no pose search)
# vina_dock  -> + score_only, minimize & full re-docking — all three paper scores
DOCKING_MODES = ("none", "vina_score", "vina_min", "vina_dock")

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


def _tanimoto_diversity(mols: list[Chem.Mol]) -> float:
    if len(mols) < 2:
        return float("nan")
    fps = [Chem.RDKFingerprint(m) for m in mols]
    sims: list[float] = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return 1.0 - float(np.mean(sims))


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
    """Make conda-env binaries (pdb2pqr30, python3) resolvable by bare name.

    docking_vina.py shells out to them; this keeps them found even when the
    process was started without `conda activate`.
    """
    _bin = str(Path(sys.executable).parent)
    if _bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")


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
    """
    _heal_path()
    from utils.evaluation.docking_vina import PrepProt

    base = str(receptor_cached)[:-4]
    pqr, pdbqt = base + ".pqr", base + ".pdbqt"
    if os.path.exists(pqr) and os.path.exists(pdbqt):
        return
    prot = PrepProt(str(receptor_cached))
    if os.path.exists(pqr):
        prot.prot_pqr = pqr
    else:
        prot.addH(pqr)
    if not os.path.exists(pdbqt):
        prot.get_pdbqt(pdbqt)


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

    The crystal reference ligand is scored the same way under the top-level
    `reference` key — a baseline to read the generated samples against.

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

    # ── chemical metrics — cheap, computed serially ──────────────────────────
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

    # ── Vina docking — parallel across molecules ─────────────────────────────
    # Each molecule docks independently, so a process pool gives a large
    # speedup with per-dock-identical results. The receptor .pqr/.pdbqt are
    # prepared once up front so workers never race on that cache; if prep fails
    # (e.g. toolchain missing) we fall back to serial and each dock reports its
    # own error.
    if dock_on:
        dock_mols = list(valid_mols)
        if ref_mol is not None:
            dock_mols.append(ref_mol)
        if dock_mols:
            try:
                _prepare_receptor(receptor_cached)
                n_workers = workers or max(1, (os.cpu_count() or 8) // max(1, cpu))
            except Exception:  # noqa: BLE001 — workers surface the error per-dock
                n_workers = 1
            n_workers = min(n_workers, len(dock_mols))
            dock_one = partial(
                run_vina_docking, receptor_pdb=receptor_cached, mode=docking,
                exhaustiveness=exhaustiveness, tmp_dir=cache_dir, cpu=cpu,
            )
            n_dock = len(dock_mols)
            vina_results: list = [None] * n_dock
            if progress_cb is not None:
                progress_cb(0, n_dock, "Docking molecules")
            if n_workers <= 1:
                for i, m in enumerate(dock_mols):
                    vina_results[i] = dock_one(m)
                    if progress_cb is not None:
                        progress_cb(i + 1, n_dock, "Docking molecules")
            else:
                # submit + as_completed (not pool.map) so the progress bar
                # advances as each dock finishes; futs maps each future back
                # to its slot, keeping vina_results in submission order.
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futs = {pool.submit(dock_one, m): i
                            for i, m in enumerate(dock_mols)}
                    for done, fut in enumerate(as_completed(futs), 1):
                        vina_results[futs[fut]] = fut.result()
                        if progress_cb is not None:
                            progress_cb(done, n_dock, "Docking molecules")
            for row, res in zip(sample_rows, vina_results):
                row["vina"] = res
            if ref_mol is not None:
                reference["vina"] = vina_results[-1]

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
        if dock_on:
            aggregates.update(_aggregate_vina(sample_rows, reference))
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
        "reference": reference,
        "samples": sample_rows,
        "aggregates": aggregates,
    }
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
        description="Compute metrics.json (optionally with Vina docking) for a "
                    "target_XX/ dir or a run dir of target_* folders.",
    )
    ap.add_argument("path", type=Path,
                    help="a target_XX/ dir, or a run dir containing target_* dirs")
    ap.add_argument("--docking", choices=DOCKING_MODES, default="none",
                    help="none | vina_score (score) | vina_min (+minimize) | "
                         "vina_dock (+full dock)")
    ap.add_argument("--exhaustiveness", type=int, default=8,
                    help="Vina exhaustiveness for dock mode (higher = slower)")
    ap.add_argument("--cpu", type=int, default=8,
                    help="Vina CPU threads per docking worker")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel docking workers (default: cpu_count // cpu)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip targets whose metrics.json already has docking results")
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

    targets = list(_iter_target_dirs(root))
    if not targets:
        ap.error(f"no samples.sdf and no target_* dirs under {root}")

    print(f"{len(targets)} target(s)  |  docking={args.docking}  "
          f"exhaustiveness={args.exhaustiveness}  cpu={args.cpu}  "
          f"workers={args.workers if args.workers else 'auto'}")
    for i, tdir in enumerate(targets, 1):
        tag = f"[{i}/{len(targets)}] {tdir.name}"
        if args.skip_existing:
            existing = load_metrics(tdir)
            if existing and existing.get("docking", "none") != "none":
                print(f"{tag}: already docked — skipped")
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
                workers=args.workers,
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
        print(f"{msg}  [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    _cli()
