"""poc_evaluate.py — Evaluate VoxBind PoC samples using TargetDiff metrics.

Computes QED, SA, LogP, Lipinski for baseline and X-ray conditioned samples,
and optionally runs Vina docking (score_only + minimize).

Usage
-----
    cd voxbind
    # chemical metrics only (no docking):
    python scripts/poc_evaluate.py \\
        --compare_dir exps/poc_xray/compare \\
        --targetdiff_root ../targetdiff

    # with Vina docking (needs TargetDiff protein root):
    python scripts/poc_evaluate.py \\
        --compare_dir     exps/poc_xray/compare \\
        --targetdiff_root ../targetdiff \\
        --docking_mode    vina_score \\
        --protein_root    ../targetdiff/data/crossdocked_v1.1_rmsd1.0

Outputs
-------
    exps/poc_xray/compare/eval_results.pt   raw per-molecule results
    exps/poc_xray/compare/eval_summary.txt  printed table
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from copy import deepcopy

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


# ── TargetDiff import ─────────────────────────────────────────────────────────

def setup_targetdiff(targetdiff_root: str):
    path = os.path.abspath(targetdiff_root)
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from utils.evaluation import scoring_func
        return scoring_func
    except ImportError as e:
        raise ImportError(
            f"Cannot import TargetDiff utils from '{path}'. "
            f"Check --targetdiff_root.\n  {e}"
        )


# ── SDF loading ───────────────────────────────────────────────────────────────

def load_sdf(path: str) -> list:
    """Return list of valid RDKit mol objects from an SDF file."""
    if not os.path.exists(path):
        return []
    suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
    mols = [m for m in suppl if m is not None]
    return mols


def find_sdf(compare_dir: str, condition: str) -> str:
    return os.path.join(compare_dir, condition, "samples.sdf")


# ── Find pocket PDB for Vina ──────────────────────────────────────────────────

def find_pocket_pdb(compare_dir: str) -> str | None:
    """Find the pocket PDB saved by poc_compare_samples.py."""
    pdbs = glob(os.path.join(compare_dir, "*pocket10.pdb"))
    return pdbs[0] if pdbs else None


def find_ligand_filename(compare_dir: str) -> str | None:
    """Reconstruct original CrossDocked ligand filename from saved reference."""
    sdfs = [f for f in glob(os.path.join(compare_dir, "*.sdf"))
            if "pocket10" not in f and "samples" not in f]
    if not sdfs:
        return None
    # e.g. GSTP1_HUMAN_2_210_0__14gs_A_rec_20gs_cbd_lig_tt_min_0.sdf
    # → GSTP1_HUMAN_2_210_0/14gs_A_rec_20gs_cbd_lig_tt_min_0.sdf
    basename = os.path.basename(sdfs[0])
    return basename.replace("__", "/", 1)


# ── Vina docking ─────────────────────────────────────────────────────────────

def _resolve_protein_path(ligand_filename: str, protein_root: str) -> str:
    """Return the protein PDB path, preferring the full receptor then pocket10."""
    target_dir = os.path.dirname(ligand_filename)
    stem = os.path.splitext(os.path.basename(ligand_filename))[0]
    # Standard CrossDocked path: PDBid_Chain_rec.pdb (first 10 chars of stem)
    full_rec = os.path.join(protein_root, target_dir, stem[:10] + ".pdb")
    if os.path.exists(full_rec):
        return full_rec
    # Fallback: pocket10 file from VoxBind's crossdocked_pocket10
    pocket10 = os.path.join(protein_root, target_dir, stem + "_pocket10.pdb")
    return pocket10


def run_vina(mol, ligand_filename: str, protein_root: str,
             docking_mode: str, exhaustiveness: int, targetdiff_root: str):
    sys.path.insert(0, os.path.abspath(targetdiff_root))
    from utils.evaluation.docking_vina import VinaDockingTask
    try:
        protein_path = _resolve_protein_path(ligand_filename, protein_root)
        task = VinaDockingTask(protein_path, deepcopy(mol))
        score_only = task.run(mode="score_only", exhaustiveness=exhaustiveness)
        minimize   = task.run(mode="minimize",   exhaustiveness=exhaustiveness)
        result = {
            "score_only": score_only[0]["affinity"],
            "minimize":   minimize[0]["affinity"],
        }
        if docking_mode == "vina_dock":
            dock = task.run(mode="dock", exhaustiveness=exhaustiveness)
            result["dock"] = dock[0]["affinity"]
        return result
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# ── Diversity ─────────────────────────────────────────────────────────────────

def compute_diversity(mols: list, radius: int = 2, n_bits: int = 2048) -> float:
    """Mean pairwise Tanimoto dissimilarity (internal diversity, 0–1)."""
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits) for m in mols]
    n = len(fps)
    if n < 2:
        return 0.0
    sims = []
    for i in range(n):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    return float(1.0 - np.mean(sims))


# ── Per-condition evaluation ──────────────────────────────────────────────────

def evaluate_condition(
    mols: list,
    label: str,
    scoring_func,
    ligand_filename: str | None,
    protein_root: str | None,
    docking_mode: str,
    exhaustiveness: int,
    targetdiff_root: str,
) -> list:
    results = []
    print(f"\n  [{label}]  {len(mols)} molecules")

    # Pre-flight: check protein file exists before running any Vina
    if docking_mode != "none" and ligand_filename and protein_root:
        protein_path = _resolve_protein_path(ligand_filename, protein_root)
        if not os.path.exists(protein_path):
            print(f"  WARNING: Vina skipped — protein not found at:\n"
                  f"    {protein_path}\n"
                  f"  Set PROTEIN_ROOT to crossdocked_pocket10 or crossdocked_v1.1_rmsd1.0.")
            docking_mode = "none"
        else:
            print(f"  Protein : {protein_path}")

    for i, mol in enumerate(mols):
        try:
            chem = scoring_func.get_chem(mol)
        except Exception as e:
            print(f"    mol {i}: chem failed — {e}")
            continue

        entry = {"mol": mol, "smiles": Chem.MolToSmiles(mol), "chem": chem}

        if docking_mode != "none" and ligand_filename and protein_root:
            vr = run_vina(mol, ligand_filename, protein_root,
                          docking_mode, exhaustiveness, targetdiff_root)
            entry["vina"] = vr
            if "error" not in vr:
                vina_str = (f"  score={vr.get('score_only', 'err'):.2f}"
                            f"  min={vr.get('minimize', 'err'):.2f}")
            else:
                vina_str = f"  vina_err={vr['error']}"
                if "traceback" in vr:
                    print(vr["traceback"])
        else:
            vina_str = ""

        print(f"    mol {i:2d}  QED={chem['qed']:.3f}  SA={chem['sa']:.3f}"
              f"  LogP={chem['logp']:.2f}  Lip={chem['lipinski']}/5{vina_str}")
        results.append(entry)

    return results


# ── Summary table ─────────────────────────────────────────────────────────────

def summarise(label: str, results: list, docking_mode: str) -> dict:
    if not results:
        return {}

    qed  = [r["chem"]["qed"]      for r in results]
    sa   = [r["chem"]["sa"]       for r in results]
    logp = [r["chem"]["logp"]     for r in results]
    lip  = [r["chem"]["lipinski"] for r in results]
    mols = [r["mol"]              for r in results]

    row = {
        "n":               len(results),
        "Diversity":       compute_diversity(mols),
        "QED mean":        np.mean(qed),
        "QED median":      np.median(qed),
        "SA mean":         np.mean(sa),
        "SA median":       np.median(sa),
        "LogP mean":       np.mean(logp),
        "LogP median":     np.median(logp),
        "Lipinski mean":   np.mean(lip),
        "Lipinski median": np.median(lip),
    }

    if docking_mode != "none":
        vina_scores = [r["vina"].get("score_only") for r in results
                       if "vina" in r and "error" not in r["vina"]]
        vina_min    = [r["vina"].get("minimize")   for r in results
                       if "vina" in r and "error" not in r["vina"]]
        if vina_scores:
            row["Vina score mean"]   = np.mean(vina_scores)
            row["Vina score median"] = np.median(vina_scores)
        if vina_min:
            row["Vina min mean"]     = np.mean(vina_min)
            row["Vina min median"]   = np.median(vina_min)

    return row


def print_table(summaries: dict, out_path: str):
    conditions = list(summaries.keys())
    metrics    = sorted({k for s in summaries.values() for k in s if k != "n"})

    col_w = max(26, max(len(c) for c in conditions) + 2)
    header = f"{'Metric':<28}" + "".join(f"{c:>{col_w}}" for c in conditions)
    sep    = "-" * len(header)

    lines = [sep, header, sep]
    lines.append(f"{'n (molecules)':<28}" +
                 "".join(f"{summaries[c].get('n', 0):>{col_w}d}" for c in conditions))
    for m in metrics:
        vals = "".join(
            f"{summaries[c].get(m, float('nan')):>{col_w}.3f}"
            for c in conditions
        )
        lines.append(f"{m:<28}{vals}")
    lines.append(sep)

    table = "\n".join(lines)
    print("\n" + table)

    with open(out_path, "w") as f:
        f.write(table + "\n")
    print(f"\nSummary saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate PoC samples with TargetDiff metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--compare_dir",     default="exps/poc_xray/compare",
                   help="Output directory of poc_compare_samples.py")
    p.add_argument("--targetdiff_root", default="../targetdiff",
                   help="Root of the TargetDiff repository")
    p.add_argument("--docking_mode",    default="none",
                   choices=["none", "vina_score", "vina_dock"],
                   help="Vina docking mode (none = skip docking)")
    p.add_argument("--protein_root",    default=None,
                   help="TargetDiff CrossDocked protein root "
                        "(required for docking_mode != none)")
    p.add_argument("--exhaustiveness", type=int, default=16)
    p.add_argument("--conditions",     nargs="+",
                   default=["baseline", "xray_cond"],
                   help="Sub-directory names to compare")
    return p.parse_args()


def main():
    args = parse_args()

    scoring_func = setup_targetdiff(args.targetdiff_root)

    if args.docking_mode != "none" and args.protein_root is None:
        raise ValueError("--protein_root is required when --docking_mode != none")

    # ── Locate reference files for Vina ──────────────────────────────────────
    ligand_filename = find_ligand_filename(args.compare_dir)
    if ligand_filename:
        print(f"Reference ligand filename : {ligand_filename}")
    else:
        print("WARNING: could not find reference ligand SDF in compare_dir.")

    # ── Load and evaluate each condition ─────────────────────────────────────
    all_results = {}
    for cond in args.conditions:
        sdf_path = find_sdf(args.compare_dir, cond)
        mols = load_sdf(sdf_path)
        if not mols:
            print(f"\n  [{cond}]  no molecules found at {sdf_path}")
            all_results[cond] = []
            continue

        all_results[cond] = evaluate_condition(
            mols=mols,
            label=cond,
            scoring_func=scoring_func,
            ligand_filename=ligand_filename,
            protein_root=args.protein_root,
            docking_mode=args.docking_mode,
            exhaustiveness=args.exhaustiveness,
            targetdiff_root=args.targetdiff_root,
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    summaries = {
        cond: summarise(cond, results, args.docking_mode)
        for cond, results in all_results.items()
    }
    table_path = os.path.join(args.compare_dir, "eval_summary.txt")
    print_table(summaries, table_path)

    # ── Save raw results ──────────────────────────────────────────────────────
    import torch
    raw_path = os.path.join(args.compare_dir, "eval_results.pt")
    torch.save({
        "conditions": args.conditions,
        "results":    all_results,
        "summaries":  summaries,
        "args":       vars(args),
    }, raw_path)
    print(f"Raw results  → {raw_path}")


if __name__ == "__main__":
    main()
