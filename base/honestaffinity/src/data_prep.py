"""data_prep.py — build per-pid records (protein seq, pocket-residue mask, ligand SMILES,
pK) for a frozen split, and precompute + cache ESM-2-650M per-residue embeddings.

HonestAffinity-Pocket needs, for each pid in a split:
  - protein sequence               (1-letter, ordered)
  - pocket-position marker mask     (binary, len == protein sequence)
  - ligand SMILES                   (canonical)
  - pK label                        (-log10 Kd/Ki)

SOURCES / decisions (documented for faithfulness):
  * Sequence + pocket mask are derived FROM THE STRUCTURE (self-consistent indexing):
      {pid}_protein.pdb  -> ordered residue list (chain,resnum,icode) -> 1-letter seq.
      {pid}_pocket.pdb   -> the set of residue keys that are IN the pocket.
    The pocket.pdb was produced upstream as the binding-site residues near the ligand
    (matches the profsa/dsmbind pocket = residues within ~10 A of the ligand). We simply
    mark residue i of the full sequence as pocket==1 iff its (chain,resnum,icode) key
    appears in the pocket.pdb. This guarantees the marker is perfectly aligned with the
    ESM per-residue embedding (which is computed on this exact structure-derived sequence).
  * SMILES comes from LP_PDBBind.csv `smiles`; if a pid is absent there (misato), fall
    back to canonical SMILES of the structure ligand SDF/mol2.
  * pK label comes from LP_PDBBind.csv `value` (same source as every other baseline).
  * Long sequences (> MAX_LEN=1000, ~6% of pids, e.g. dimers/large complexes) are cropped
    to a MAX_LEN window chosen to cover the MOST pocket residues (greedy best window),
    so ESM-2 (max ~1022 tokens) stays in budget while keeping the binding site. Pocket
    residues outside the window are simply dropped from the marker (rare).

ESM cache: embeddings are keyed by a hash of the FINAL (possibly cropped) sequence, so
identical proteins across pids share one cached float16 [L,1280] tensor on disk.
"""
import os
import sys
import json
import hashlib
import argparse
from collections import Counter

import numpy as np
import pandas as pd

# gemmi/torch import-order is irrelevant here (no gemmi); import torch late for esm.
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HA_ROOT = os.path.dirname(HERE)                       # base/honestaffinity/
REPO = os.path.dirname(os.path.dirname(HA_ROOT))      # VoxBind/
CACHE_DIR = os.path.join(HA_ROOT, "cache")
ESM_DIR = os.path.join(CACHE_DIR, "esm")              # per-sequence float16 embeddings

LP = os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "raw", "LP_PDBBind.csv")
STRUCT_BASES = [
    os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "structures", "pbpp-2020"),
    os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "structures", "misato_qm_built"),
]

MAX_LEN = 1000  # ESM-2 handles ~1022 tokens; leave headroom for BOS/EOS.

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


# --------------------------------------------------------------------------- IO
def resolve(pid):
    for b in STRUCT_BASES:
        d = os.path.join(b, pid)
        prot = os.path.join(d, f"{pid}_protein.pdb")
        poc = os.path.join(d, f"{pid}_pocket.pdb")
        sdf = os.path.join(d, f"{pid}_ligand.sdf")
        mol2 = os.path.join(d, f"{pid}_ligand.mol2")
        if os.path.exists(prot) and os.path.exists(poc):
            return prot, poc, sdf, mol2
    return None, None, None, None


def parse_residues(pdb):
    """Ordered unique protein residues as [(chain,resnum,icode), resname], first-seen order."""
    res, seen = [], set()
    with open(pdb) as f:
        for l in f:
            if l[:4] != "ATOM":
                continue
            key = (l[21], l[22:26].strip(), l[26])
            if key in seen:
                continue
            seen.add(key)
            res.append((key, l[17:20].strip()))
    return res


def pocket_keys(pdb):
    keys = set()
    with open(pdb) as f:
        for l in f:
            if l[:4] != "ATOM" and l[:6] != "HETATM":
                continue
            keys.add((l[21], l[22:26].strip(), l[26]))
    return keys


def best_window(mask, max_len):
    """Return (start, end) window of length <= max_len covering the most pocket residues.
    Slides a max_len window; ties broken toward the earliest window centered on pocket."""
    n = len(mask)
    if n <= max_len:
        return 0, n
    pre = np.cumsum([0] + list(mask))  # prefix sums of pocket indicator
    best_cnt, best_start = -1, 0
    for s in range(0, n - max_len + 1):
        cnt = pre[s + max_len] - pre[s]
        if cnt > best_cnt:
            best_cnt, best_start = cnt, s
    return best_start, best_start + max_len


def build_record(pid, pk_map, smi_map):
    prot_pdb, poc_pdb, sdf, mol2 = resolve(pid)
    if prot_pdb is None:
        return None, "no_structure"
    residues = parse_residues(prot_pdb)
    if len(residues) == 0:
        return None, "empty_protein"
    pkeys = pocket_keys(poc_pdb)
    seq = "".join(AA3TO1.get(rn, "X") for _, rn in residues)
    mask = np.array([1 if k in pkeys else 0 for k, _ in residues], dtype=np.int8)

    # crop long sequences to a pocket-covering window
    if len(seq) > MAX_LEN:
        s, e = best_window(mask, MAX_LEN)
        seq = seq[s:e]
        mask = mask[s:e]

    if mask.sum() == 0:
        return None, "empty_pocket"

    # SMILES: prefer LP_PDBBind, fall back to structure ligand
    smi = smi_map.get(pid)
    if not smi:
        smi = ligand_smiles(sdf, mol2)
    if not smi:
        return None, "no_smiles"

    pK = pk_map.get(pid)
    if pK is None or not np.isfinite(pK):
        return None, "no_label"

    return {
        "pid": pid,
        "seq": seq,
        "pocket_mask": mask.tolist(),
        "smiles": smi,
        "pK": float(pK),
    }, "ok"


def ligand_smiles(sdf, mol2):
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        return None
    m = None
    if sdf and os.path.exists(sdf):
        m = next(iter(Chem.SDMolSupplier(sdf, sanitize=True, removeHs=True)), None)
        if m is None:
            m = next(iter(Chem.SDMolSupplier(sdf, sanitize=False, removeHs=True)), None)
    if m is None and mol2 and os.path.exists(mol2):
        m = Chem.MolFromMol2File(mol2, sanitize=True) or Chem.MolFromMol2File(mol2, sanitize=False)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return None


# ------------------------------------------------------------------- ESM cache
def seq_hash(seq):
    return hashlib.sha1(seq.encode()).hexdigest()[:16]


def esm_path(seq):
    return os.path.join(ESM_DIR, f"{seq_hash(seq)}.pt")


def cache_esm_embeddings(seqs, gpu=0, batch_tokens=8000):
    """Compute + cache ESM-2-650M per-residue embeddings (float16 [L,1280]) for the given
    unique sequences that are not already cached. Returns count computed."""
    import esm
    os.makedirs(ESM_DIR, exist_ok=True)
    todo = [s for s in set(seqs) if not os.path.exists(esm_path(s))]
    if not todo:
        print(f"[esm] all {len(set(seqs))} unique sequences already cached")
        return 0
    print(f"[esm] loading esm2_t33_650M_UR50D (this downloads ~2.5GB on first run)...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = alphabet.get_batch_converter()
    dev = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval().half() if dev.type == "cuda" else model.eval()

    # length-sorted, token-budgeted batching to bound memory
    todo.sort(key=len)
    i, done = 0, 0
    with torch.no_grad():
        while i < len(todo):
            batch, tok = [], 0
            while i < len(todo):
                L = len(todo[i])
                if batch and tok + L > batch_tokens:
                    break
                batch.append(todo[i])
                tok += L
                i += 1
                if L > batch_tokens:  # a single over-budget seq goes alone
                    break
            data = [(f"s{j}", s) for j, s in enumerate(batch)]
            _, _, toks = bc(data)
            toks = toks.to(dev)
            out = model(toks, repr_layers=[33], return_contacts=False)
            reps = out["representations"][33]  # [B, Lmax+2, 1280]
            for j, s in enumerate(batch):
                emb = reps[j, 1:len(s) + 1].float().cpu().half()  # strip BOS/EOS
                torch.save(emb, esm_path(s))
                done += 1
            if done % 200 < len(batch):
                print(f"[esm]   cached {done}/{len(todo)}", flush=True)
    print(f"[esm] done: cached {done} new sequences")
    return done


# ------------------------------------------------------------------------ main
def load_lp():
    lp = pd.read_csv(LP).rename(columns={"Unnamed: 0": "pid", "value": "pK"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    pk_map = dict(zip(lp["pid"], lp["pK"]))
    smi_map = dict(zip(lp["pid"], lp["smiles"]))
    return pk_map, smi_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True,
                    help="split name, e.g. lp_edrscc_v2_cl123")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_esm", action="store_true", help="build records only, skip ESM cache")
    args = ap.parse_args()

    split_csv = os.path.join(REPO, "voxbind", "splits", f"{args.split}.csv")
    sp = pd.read_csv(split_csv)
    sp["pid"] = sp["pid"].astype(str).str.lower()
    pids = list(sp["pid"])
    split_of = dict(zip(sp["pid"], sp["split"]))
    if args.limit:
        pids = pids[:args.limit]

    pk_map, smi_map = load_lp()

    records, fails = [], {}
    for k, pid in enumerate(pids):
        rec, status = build_record(pid, pk_map, smi_map)
        if rec is None:
            fails[pid] = status
            continue
        rec["split"] = split_of[pid]
        records.append(rec)
        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{len(pids)} built={len(records)}", flush=True)

    from collections import Counter as C
    counts = Counter(r["split"] for r in records)
    print(f"\n[{args.split}] built {len(records)} / {len(pids)} | "
          f"train={counts.get('train',0)} val={counts.get('val',0)} test={counts.get('test',0)}")
    print(f"  failures {len(fails)}: {dict(C(fails.values()))}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    rec_path = os.path.join(CACHE_DIR, f"records_{args.split}.json")
    with open(rec_path, "w") as f:
        json.dump({"split": args.split, "records": records, "fails": fails,
                   "max_len": MAX_LEN}, f)
    print(f"  wrote {rec_path}")

    if not args.no_esm:
        n = cache_esm_embeddings([r["seq"] for r in records], gpu=args.gpu)
        print(f"[{args.split}] ESM cache updated (+{n})")


if __name__ == "__main__":
    main()
