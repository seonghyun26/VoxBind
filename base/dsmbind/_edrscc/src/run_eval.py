"""run_eval.py — apply the pretrained DSMBind drug (protein-ligand) energy model to
the lp_edrscc_v2 TEST split, zero-shot, and correlate predicted binding energy with pK.

DSMBind is unsupervised: it scores a complex with a learned binding energy (lower =
stronger). We follow the repo's recipe (DrugAllAtomEnergyModel.virtual_screen):
  parsePDB(pocket) -> get_seq_coords_and_angles -> (target_seq, target_coords[n_res,14,3]);
  ligand from SDF -> binder_mol; DrugDataset.process -> 50-NN pocket patch; ESM-2-3B
  target embedding; model.predict -> energy; score = -energy.

ADAPTATION: we feed the 10 A pocket PDB ({pid}_pocket.pdb), not the full protein, as the
target. The model is pocket-based (patch_size=50), and our proteins reach 3322 residues
which would OOM ESM-2-3B (no truncation in load_esm_embedding). Pocket input keeps the
binding site, bounds the ESM cost, and matches the model's pocket-level design.

Metrics: Pearson / Spearman of (-energy) vs experimental pK. RMSE is not meaningful
(energy is not in pK units), so we report correlation only. Zero-shot, no training.
"""
import os
import sys
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
import scipy.stats
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
DSM_ROOT = os.path.dirname(os.path.dirname(HERE))           # base/dsmbind/
REPO = os.path.dirname(os.path.dirname(DSM_ROOT))           # VoxBind/
sys.path.insert(0, DSM_ROOT)

from rdkit import Chem, RDLogger                            # noqa: E402
RDLogger.DisableLog("rdApp.*")
from prody import parsePDB, confProDy                       # noqa: E402
confProDy(verbosity="none")
from sidechainnet.utils.measure import get_seq_coords_and_angles  # noqa: E402
from bindenergy import DrugAllAtomEnergyModel, DrugDataset, load_esm_embedding  # noqa: E402

SPLIT = os.path.join(REPO, "voxbind", "splits", "lp_edrscc_v2.csv")
LP = os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "raw", "LP_PDBBind.csv")
STRUCT_BASES = [
    os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "structures", "pbpp-2020"),
    os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "structures", "misato_qm_built"),
]
CKPT = os.path.join(DSM_ROOT, "ckpts", "model.drug.allatom")


def resolve(pid):
    for base in STRUCT_BASES:
        d = os.path.join(base, pid)
        pocket = os.path.join(d, f"{pid}_pocket.pdb")
        sdf = os.path.join(d, f"{pid}_ligand.sdf")
        mol2 = os.path.join(d, f"{pid}_ligand.mol2")
        if os.path.exists(pocket) and (os.path.exists(sdf) or os.path.exists(mol2)):
            return pocket, sdf, mol2
    return None, None, None


def load_mol(sdf, mol2):
    """Robust 3D ligand load: SDF strict -> SDF relaxed(+sanitize try) -> mol2."""
    if os.path.exists(sdf):
        m = next(iter(Chem.SDMolSupplier(sdf, sanitize=True, removeHs=True)), None)
        if m is not None and m.GetNumConformers() > 0:
            return m
        m = next(iter(Chem.SDMolSupplier(sdf, sanitize=False, removeHs=False)), None)
        if m is not None and m.GetNumConformers() > 0:
            try:
                Chem.SanitizeMol(m)
            except Exception:
                pass
            return m
    if os.path.exists(mol2):
        m = Chem.MolFromMol2File(mol2, sanitize=True)
        if m is None:
            m = Chem.MolFromMol2File(mol2, sanitize=False)
        if m is not None and m.GetNumConformers() > 0:
            return m
    return None


def build_entry(pid, pK):
    pocket, sdf, mol2 = resolve(pid)
    if pocket is None:
        return None, "no_structure"
    try:
        hchain = parsePDB(pocket, model=1)
        _, hcoords, hseq, _, _ = get_seq_coords_and_angles(hchain)
        hcoords = hcoords.reshape((len(hseq), 14, 3))
    except Exception as e:
        return None, f"protein:{type(e).__name__}"
    try:
        mol = load_mol(sdf, mol2)
        if mol is None:
            return None, "ligand_none"
    except Exception as e:
        return None, f"ligand:{type(e).__name__}"
    return {"pdb": pid, "binder_mol": mol, "target_seq": hseq,
            "target_coords": hcoords, "affinity": float(pK)}, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)

    sp = pd.read_csv(SPLIT); sp["pid"] = sp["pid"].astype(str).str.lower()
    lp = pd.read_csv(LP).rename(columns={"Unnamed: 0": "pid", "value": "pK"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    pk = dict(zip(lp["pid"], lp["pK"]))
    test_pids = sp[sp["split"] == "test"]["pid"].tolist()
    if args.limit:
        test_pids = test_pids[:args.limit]

    print(f"loading DSMBind drug model from {CKPT}")
    model_ckpt, _, model_args = torch.load(CKPT, map_location="cpu")
    model = DrugAllAtomEnergyModel(model_args).cuda()
    model.load_state_dict(model_ckpt)
    model.eval()

    entries, fails = [], {}
    for pid in tqdm(test_pids, desc="build entries"):
        if pid not in pk or not np.isfinite(pk[pid]):
            fails[pid] = "no_label"; continue
        e, status = build_entry(pid, pk[pid])
        if e is None:
            fails[pid] = status
        else:
            entries.append(e)
    print(f"built {len(entries)} entries | failures {len(fails)}: "
          f"{dict(pd.Series(list(fails.values())).value_counts()) if fails else {}}")

    print("computing ESM-2-3B target embeddings (pocket sequences)...")
    embedding = load_esm_embedding(entries, ["target_seq"])
    proc = DrugDataset.process(entries, model_args.patch_size)

    rows = []
    with torch.no_grad():
        for entry in tqdm(proc, desc="predict energy"):
            try:
                binder, target = DrugDataset.make_bind_batch([entry], embedding, model_args)
                energy = model.predict(binder, target).item()
            except Exception as e:
                fails[entry["pdb"]] = f"predict:{type(e).__name__}"; continue
            rows.append((entry["pdb"], entry["affinity"], energy, -energy))

    df = pd.DataFrame(rows, columns=["pid", "pK", "energy", "score"])
    n = len(df)
    pear = float(np.corrcoef(df["pK"], df["score"])[0, 1])
    spear = float(scipy.stats.spearmanr(df["pK"], df["score"])[0])
    summary = {
        "model": "DSMBind (NeurIPS 2023, pretrained drug all-atom, zero-shot)",
        "split": "lp_edrscc_v2", "subset": "test",
        "n_scored": n, "n_failed": len(fails),
        "test_pearson": pear, "test_spearman": spear,
        "note": "score = -binding_energy vs experimental pK; pocket.pdb target; ESM-2-3B; "
                "unsupervised energy -> correlation only (no RMSE; energy not in pK units).",
    }
    os.makedirs(os.path.join(DSM_ROOT, "results"), exist_ok=True)
    with open(os.path.join(DSM_ROOT, "results", "dsmbind_edrscc.json"), "w") as f:
        json.dump(summary, f, indent=2)
    df.to_csv(os.path.join(DSM_ROOT, "results", "dsmbind_edrscc_preds.csv"), index=False)
    with open(os.path.join(DSM_ROOT, "results", "dsmbind_fails.json"), "w") as f:
        json.dump(fails, f, indent=2)

    print(f"\n===== DSMBind zero-shot  lp_edrscc_v2 (test) =====")
    print(f"n scored = {n}  | failed = {len(fails)}")
    print(f"test Pearson  r  {pear:.4f}")
    print(f"test Spearman    {spear:.4f}")
    print("(energy vs pK correlation; unsupervised, zero-shot)")


if __name__ == "__main__":
    main()
