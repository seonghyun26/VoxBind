"""extract_features.py — cache FROZEN DSMBind-encoder features for an MLP probe.

DSMBind is an unsupervised energy model, not a representation model. To put it on the
SAME footing as ProFSA / our voxel encoder (frozen encoder + trained MLP head, 3 seeds),
we extract a fixed-size pooled embedding from the pretrained drug all-atom encoder and
cache it per complex. A downstream head (probe_mlp.py) then regresses pK from these.

Per complex we replicate DrugAllAtomEnergyModel.predict up to the encoder output
  h = encoder((bind_X, bind_S, bind_A), (tgt_X, tgt_S, tgt_A))  -> [1, N+M, 14, H]
and pool (masked mean over valid atoms):
  z_bind  = mean_{binder atoms}  h[:, :N]     [H]
  z_tgt   = mean_{pocket atoms}  h[:, N:]     [H]
  energy  = predict(...)                       [1]  (the zero-shot score, kept as 1 feature)
  feat = [z_bind ; z_tgt ; energy]             [2H+1]
The encoder is frozen; predict() uses the given (true) pose, so features are deterministic.

Same data path as run_eval.py: pocket.pdb target, ESM-2-3B pocket embedding, DrugDataset.
Caches _edrscc/features/<split>.pt = {"feat": {pid: np[2H+1]}, "pK": {pid: float},
"split": {pid: train|val|test}, "hidden_size": H}. Run in the `dsmbind` env.
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
DSM_ROOT = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(os.path.dirname(DSM_ROOT))
sys.path.insert(0, DSM_ROOT)

from rdkit import Chem, RDLogger                            # noqa: E402
RDLogger.DisableLog("rdApp.*")
from prody import parsePDB, confProDy                       # noqa: E402
confProDy(verbosity="none")
from sidechainnet.utils.measure import get_seq_coords_and_angles  # noqa: E402
from bindenergy import DrugAllAtomEnergyModel, DrugDataset, load_esm_embedding  # noqa: E402
from chemprop.features import mol2graph                     # noqa: E402

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
        m = Chem.MolFromMol2File(mol2, sanitize=True) or Chem.MolFromMol2File(mol2, sanitize=False)
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


@torch.no_grad()
def pool_features(model, binder, target):
    """Masked-mean pool of the frozen encoder output + the energy scalar -> [2H+1]."""
    bind_X, mol_batch, bind_A, _ = binder
    tgt_X, tgt_S, tgt_A, _ = target
    bind_S = model.mpn(mol2graph(mol_batch))
    B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
    bind_A2 = bind_A * (bind_X.norm(dim=-1) > 1e-4).long()
    tgt_A2 = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long()
    h = model.encoder((bind_X, bind_S, bind_A2, None),
                      (tgt_X, tgt_S, tgt_A2, None))          # [B, N+M, 14, H]
    bmask = (bind_A2 > 0).float().unsqueeze(-1)               # [B,N,14,1]
    tmask = (tgt_A2 > 0).float().unsqueeze(-1)                # [B,M,14,1]
    hb, ht = h[:, :N], h[:, N:]
    z_bind = (hb * bmask).sum((1, 2)) / bmask.sum((1, 2)).clamp(min=1.0)   # [B,H]
    z_tgt = (ht * tmask).sum((1, 2)) / tmask.sum((1, 2)).clamp(min=1.0)    # [B,H]
    return z_bind, z_tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="lp_edrscc_v2", help="frozen split scheme name")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)

    split_csv = os.path.join(REPO, "voxbind", "splits", f"{args.split}.csv")
    sp = pd.read_csv(split_csv); sp["pid"] = sp["pid"].astype(str).str.lower()
    lp = pd.read_csv(LP).rename(columns={"Unnamed: 0": "pid", "value": "pK"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    pk = dict(zip(lp["pid"], lp["pK"]))
    split_map = dict(zip(sp["pid"], sp["split"]))
    pids = list(sp["pid"])
    if args.limit:
        pids = pids[:args.limit]

    print(f"loading DSMBind drug model from {CKPT}")
    model_ckpt, _, model_args = torch.load(CKPT, map_location="cpu")
    model = DrugAllAtomEnergyModel(model_args).cuda()
    model.load_state_dict(model_ckpt)
    model.eval()
    H = model_args.hidden_size

    entries, fails = [], {}
    for pid in tqdm(pids, desc="build entries"):
        if pid not in pk or not np.isfinite(pk[pid]):
            fails[pid] = "no_label"; continue
        e, status = build_entry(pid, pk[pid])
        if e is None:
            fails[pid] = status
        else:
            entries.append(e)
    print(f"built {len(entries)} | failures {len(fails)}: "
          f"{dict(pd.Series(list(fails.values())).value_counts()) if fails else {}}")

    print("computing ESM-2-3B pocket embeddings...")
    embedding = load_esm_embedding(entries, ["target_seq"])
    proc = DrugDataset.process(entries, model_args.patch_size)

    feat, pKmap, spmap = {}, {}, {}
    for entry in tqdm(proc, desc="extract features"):
        pid = entry["pdb"]
        try:
            binder, target = DrugDataset.make_bind_batch([entry], embedding, model_args)
            z_bind, z_tgt = pool_features(model, binder, target)
            energy = model.predict(binder, target).detach()      # [1]
            vec = torch.cat([z_bind[0], z_tgt[0], energy], dim=-1).float().cpu().numpy()
        except Exception as ex:
            fails[pid] = f"extract:{type(ex).__name__}"; continue
        feat[pid] = vec
        pKmap[pid] = float(entry["affinity"])
        spmap[pid] = split_map[pid]

    out_dir = os.path.join(DSM_ROOT, "_edrscc", "features")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.split}.pt")
    torch.save({"feat": feat, "pK": pKmap, "split": spmap,
                "hidden_size": H, "dim": 2 * H + 1}, out)
    from collections import Counter
    counts = Counter(spmap.values())
    print(f"\nsaved {len(feat)} feats ({dict(counts)}) dim={2*H+1} -> {out}")
    print(f"failures {len(fails)}: {dict(pd.Series(list(fails.values())).value_counts()) if fails else {}}")


if __name__ == "__main__":
    main()
