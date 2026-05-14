"""On-demand TargetDiff metric computation with on-disk caching.

Cache file: `target_*/metrics.json`. Considered stale if older than samples.sdf.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger

from data import parse_pocket_pdb  # noqa: F401  (re-exports useful elsewhere)
from utils.evaluation.scoring_func import get_chem  # from TargetDiff (sys.path set in data.py)

RDLogger.DisableLog("rdApp.*")

METRICS_VERSION = 2


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


def compute_target_metrics(
    target_dir: Path,
    pocket_coords: np.ndarray,
    pocket_elems: list[str],
) -> dict:
    """Score every valid sample under `target_dir` and write metrics.json."""
    sdf = target_dir / "samples.sdf"
    if not sdf.exists():
        raise FileNotFoundError(f"No samples.sdf in {target_dir}")

    n_raw = sum(1 for _ in Chem.SDMolSupplier(str(sdf), sanitize=False))

    valid_mols: list[Chem.Mol] = []
    sample_rows: list[dict] = []
    for m in Chem.SDMolSupplier(str(sdf), sanitize=True):
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
    else:
        aggregates = {
            "n_total": n_raw,
            "n_valid": 0,
            "validity": 0.0 if n_raw else float("nan"),
        }

    data = {
        "version": METRICS_VERSION,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "samples": sample_rows,
        "aggregates": aggregates,
    }
    metrics_path(target_dir).write_text(
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
