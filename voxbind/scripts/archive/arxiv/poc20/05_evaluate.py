#!/usr/bin/env python
"""
05_evaluate.py — Evaluate VoxBind-generated molecules.

Each target_XX/ directory is expected to contain:
  samples.sdf       – generated molecules (all samples merged)
  *_pocket10.pdb    – the receptor pocket PDB (used for Vina scoring)
  *_lig_*.sdf       – ground-truth ligand (for reference; not used in metrics)

Metrics computed
----------------
Per-pocket (then averaged across pockets):
  validity        – fraction of mols parseable by RDKit (no fragments)
  uniqueness      – unique SMILES / valid mols per pocket
  diversity       – mean pairwise Tanimoto dissimilarity (RDK fingerprints)
  QED             – quantitative estimate of drug-likeness  (higher = better)
  SA              – synthetic accessibility (1 = easy, 10 = hard)
  LogP            – Crippen logP
  Lipinski        – # of Lipinski Ro5 rules satisfied (0–5)

Optional (--docking_mode vina_score or vina_dock):
  vina_score      – Vina score_only affinity (kcal/mol, no re-docking)
  vina_min        – Vina minimize affinity (local opt of generated pose)
  vina_dock       – Vina full re-docking affinity

Note: Vina uses the *_pocket10.pdb as receptor, which is a 10 Å-cropped
pocket rather than the full protein. This is sufficient for score_only and
minimize modes; for full dock mode results will be approximate.

Usage
-----
cd voxbind   # run from the voxbind/ directory
python scripts/05_evaluate.py exps/exp_sig0.9/samples/res \\
    [--docking_mode {none,vina_score,vina_dock}] \\
    [--exhaustiveness 16] \\
    [--out eval_results.json] \\
    [--verbose]
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from tqdm import tqdm

# ── targetdiff evaluation utilities ────────────────────────────────────────
TARGETDIFF_ROOT = str(Path(__file__).resolve().parents[3] / "targetdiff")
if TARGETDIFF_ROOT not in sys.path:
    sys.path.insert(0, TARGETDIFF_ROOT)

try:
    from utils.evaluation.scoring_func import get_chem
except ImportError as e:
    print(f"ERROR: Could not import targetdiff utilities from {TARGETDIFF_ROOT}")
    print(f"  {e}")
    sys.exit(1)


# ── molecule loading ────────────────────────────────────────────────────────

def load_mols(sdf_path: str) -> list[tuple[Chem.Mol, str]]:
    """Return (mol, smiles) pairs for valid, complete molecules in an SDF."""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    out = []
    for mol in suppl:
        if mol is None:
            continue
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:
            continue
        if "." not in smi:          # discard disconnected fragments
            out.append((mol, smi))
    return out


# ── diversity ───────────────────────────────────────────────────────────────

def tanimoto_diversity(mols: list[Chem.Mol]) -> float:
    """Mean pairwise Tanimoto *dissimilarity* (= 1 – similarity)."""
    if len(mols) < 2:
        return float("nan")
    fps = [Chem.RDKFingerprint(m) for m in mols]
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return 1.0 - float(np.mean(sims))


# ── Vina docking ────────────────────────────────────────────────────────────

def find_pocket_pdb(target_dir: str) -> str | None:
    """Return path to the *_pocket10.pdb inside a target directory, or None."""
    hits = glob(os.path.join(target_dir, "*_pocket10.pdb"))
    return hits[0] if hits else None


def vina_score_mol(
    mol: Chem.Mol,
    pocket_pdb: str,
    mode: str,
    exhaustiveness: int,
    tmp_dir: str,
) -> float | None:
    """Return Vina affinity (kcal/mol) for one mol, or None on failure."""
    try:
        from utils.evaluation.docking_vina import VinaDockingTask
        task = VinaDockingTask(
            protein_path=pocket_pdb,
            ligand_rdmol=mol,
            tmp_dir=tmp_dir,
        )
        results = task.run(mode=mode, exhaustiveness=exhaustiveness)
        return results[0]["affinity"]
    except Exception:
        return None


# ── per-target evaluation ───────────────────────────────────────────────────

def evaluate_target(
    target_dir: str,
    docking_mode: str,
    exhaustiveness: int,
    tmp_dir: str,
    verbose: bool,
) -> dict:
    name = os.path.basename(target_dir)
    sdf_path = os.path.join(target_dir, "samples.sdf")

    if not os.path.exists(sdf_path):
        if verbose:
            print(f"  [{name}] samples.sdf not found, skipping.")
        return {"target": name, "skipped": True}

    # count raw entries (including unparseable)
    raw_suppl = list(Chem.SDMolSupplier(sdf_path, sanitize=False))
    n_total = len(raw_suppl)

    valid_mols = load_mols(sdf_path)
    n_valid = len(valid_mols)

    if n_valid == 0:
        return {
            "target": name,
            "n_total": n_total,
            "n_valid": 0,
            "validity": 0.0,
        }

    mol_list   = [m for m, _ in valid_mols]
    smiles_list = [s for _, s in valid_mols]
    n_unique   = len(set(smiles_list))

    chem_list     = [get_chem(m) for m in mol_list]
    qed_vals      = [c["qed"]      for c in chem_list]
    sa_vals       = [c["sa"]       for c in chem_list]
    logp_vals     = [c["logp"]     for c in chem_list]
    lipinski_vals = [c["lipinski"] for c in chem_list]

    entry = {
        "target":     name,
        "n_total":    n_total,
        "n_valid":    n_valid,
        "n_unique":   n_unique,
        "validity":   n_valid / n_total if n_total else 0.0,
        "uniqueness": n_unique / n_valid,
        "diversity":  tanimoto_diversity(mol_list),
        "qed":        float(np.mean(qed_vals)),
        "sa":         float(np.mean(sa_vals)),
        "logp":       float(np.mean(logp_vals)),
        "lipinski":   float(np.mean(lipinski_vals)),
    }

    if docking_mode != "none":
        pocket_pdb = find_pocket_pdb(target_dir)
        if pocket_pdb is None:
            if verbose:
                print(f"  [{name}] no pocket PDB found, skipping Vina.")
        else:
            vina_scores, vina_min_scores, vina_dock_scores = [], [], []
            for mol in mol_list:
                s = vina_score_mol(mol, pocket_pdb, "score_only", exhaustiveness, tmp_dir)
                if s is not None:
                    vina_scores.append(s)

                if docking_mode == "vina_dock":
                    s2 = vina_score_mol(mol, pocket_pdb, "minimize", exhaustiveness, tmp_dir)
                    if s2 is not None:
                        vina_min_scores.append(s2)
                    s3 = vina_score_mol(mol, pocket_pdb, "dock", exhaustiveness, tmp_dir)
                    if s3 is not None:
                        vina_dock_scores.append(s3)

            entry["vina_score"] = float(np.mean(vina_scores)) if vina_scores else None
            if docking_mode == "vina_dock":
                entry["vina_min"]  = float(np.mean(vina_min_scores))  if vina_min_scores  else None
                entry["vina_dock"] = float(np.mean(vina_dock_scores)) if vina_dock_scores else None

    return entry


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sample_dir", type=str,
                        help="Path to samples directory (contains target_XX/ subdirs)")
    parser.add_argument("--docking_mode", choices=["none", "vina_score", "vina_dock"],
                        default="none",
                        help="Vina docking mode (default: none)")
    parser.add_argument("--exhaustiveness", type=int, default=16,
                        help="Vina exhaustiveness (default: 16)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: <sample_dir>/eval_results.json)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")

    sample_dir = os.path.abspath(args.sample_dir)
    out_path   = args.out or os.path.join(sample_dir, "eval_results.json")
    tmp_dir    = os.path.join(sample_dir, ".vina_tmp")

    target_dirs = sorted(glob(os.path.join(sample_dir, "target_*")))
    if not target_dirs:
        print(f"No target_* directories found in {sample_dir}")
        sys.exit(1)
    print(f"Found {len(target_dirs)} target directories in {sample_dir}")
    print(f"Docking mode : {args.docking_mode}\n")

    per_target = []
    for td in tqdm(target_dirs, desc="Evaluating"):
        entry = evaluate_target(
            td,
            docking_mode=args.docking_mode,
            exhaustiveness=args.exhaustiveness,
            tmp_dir=tmp_dir,
            verbose=args.verbose,
        )
        per_target.append(entry)

    # ── aggregate ────────────────────────────────────────────────────────────
    valid_entries = [e for e in per_target if not e.get("skipped") and e.get("n_valid", 0) > 0]

    def _mean(key: str) -> float | None:
        vals = [
            e[key] for e in valid_entries
            if e.get(key) is not None and not (isinstance(e[key], float) and np.isnan(e[key]))
        ]
        return float(np.mean(vals)) if vals else None

    summary_keys = ["validity", "uniqueness", "diversity", "qed", "sa", "logp", "lipinski"]
    if args.docking_mode != "none":
        summary_keys += ["vina_score"]
    if args.docking_mode == "vina_dock":
        summary_keys += ["vina_min", "vina_dock"]

    summary = {
        "n_targets_total":    len(per_target),
        "n_targets_evaluated": len(valid_entries),
    }
    for k in summary_keys:
        summary[k] = _mean(k)

    results = {"summary": summary, "per_target": per_target}
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── print ─────────────────────────────────────────────────────────────────
    print("\n=== Evaluation Summary ===")
    print(f"  {'targets evaluated':<20}: {summary['n_targets_evaluated']} / {summary['n_targets_total']}")
    for k in summary_keys:
        v = summary[k]
        if v is not None:
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: N/A")
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()
