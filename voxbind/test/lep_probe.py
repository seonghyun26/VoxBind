"""lep_probe.py — frozen-encoder probe on ATOM3D Ligand Efficacy Prediction (LEP).

The LEP experiment from CheapNet (OpenReview A1HhtITVEi). Each example is one small
molecule shown in TWO protein conformations — an ACTIVE-state and an INACTIVE-state
complex — and the binary task is whether the molecule ACTIVATES the protein (label 1)
or not (0). Split = split-by-protein. Metrics = AUROC / AUPRC (mean±std over seeds).

This mirrors the LP-PDBBind frozen-encoder probe (dataset/01c_pdbbind_probe.py), but:
  * voxels are built ON THE FLY from the ATOM3D atom tables (there is no precomputed
    voxel cache and no electron-density map for these docked poses);
  * BOTH the active and inactive complexes are encoded; the probe input is the
    concatenation [f_active ‖ f_inactive] (mirrors CheapNet using both states).

Voxelisation matches EXACTLY the atomblob convention the 100M encoders were trained on
(voxbind/dataset/legacy/01b_pdbbind_preprocess.py): per-element vdW blob radii for both
ligand (7 ch: C,O,N,S,F,Cl,P) and pocket (4 ch: C,O,N,S), centered on the ligand
heavy-atom geometric centroid, 64³ grid @ 0.25 Å, cubes_around 8. Pocket = protein
atoms within 6 Å of any ligand atom (CheapNet's distance filter).

Conditions:
  coords  — C, coords-only 100M encoder (model_zoo/coords_100m_v2_mask075, e49), 11 ch.
  cdg     — C+D+G, density+gradmag 100M encoder (model_zoo/champion_100m_v2_mask075,
            e49), 13 ch. LEP has NO electron density → the density+gradmag channels are
            ZERO-FILLED (a transfer/capacity probe of the fuller encoder, NOT a real
            "density helps" test — flagged as such in the writeup).

Run (voxbind env, GPU 2 or 3):
    cd /home/shpark/prj-denovo/VoxBind/voxbind
    CUDA_VISIBLE_DEVICES=2 python test/lep_probe.py --conditions coords cdg --seeds 3

Outputs:
    dataset/data/lep/features/lep_{condition}_e49.pt   (per-structure 640-D cache)
    dataset/data/lep/lep_probe_results.json            (AUROC/AUPRC per condition)
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn

REPO = Path("/home/shpark/prj-denovo/VoxBind")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "voxbind"))

from voxbind.voxelizer import Voxelizer                                   # noqa: E402
from voxbind.constants import ELEMENTS_HASH_CROSSDOCKED, RADIUS_PER_ATOM  # noqa: E402

# Reuse the exact frozen-encoder loader from the LP-PDBBind probe.
_spec = importlib.util.spec_from_file_location(
    "p01c", str(REPO / "voxbind" / "dataset" / "01c_pdbbind_probe.py"))
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)

# ── voxelisation constants (mirror legacy 01b_pdbbind_preprocess.py) ──────────
LIG_CH_ELEMS = ["C", "O", "N", "S", "F", "Cl", "P"]   # channels 0..6
POC_CH_ELEMS = ["C", "O", "N", "S"]                   # channels 0..3
N_LIG_CH, N_POC_CH = 7, 4
GRID_DIM, RESOLUTION, CUBES_AROUND = 64, 0.25, 8
POCKET_CUTOFF = 6.0                                   # Å, CheapNet distance filter
HYDROGENS = {"H", "D", "T"}
_VDW = RADIUS_PER_ATOM["MOL"]
_E2CH = ELEMENTS_HASH_CROSSDOCKED

DATA_DIR = REPO / "voxbind" / "dataset" / "data" / "lep"
FEAT_DIR = DATA_DIR / "features"
RESULTS_JSON = DATA_DIR / "lep_probe_results.json"

CONDITIONS = {
    "coords": dict(exp="model_zoo/coords_100m_v2_mask075", epoch=49,
                   n_in=11, density=False, label="C"),
    "cdg":    dict(exp="model_zoo/champion_100m_v2_mask075", epoch=49,
                   n_in=13, density=True, label="C+D+G (zero-density)"),
}


def _norm_elem(e: str) -> str:
    e = str(e).strip()
    return (e[0].upper() + e[1:].lower()) if e else ""


def _channel_and_radius(elem, ch_elems):
    """Map an element symbol → (channel_idx, vdw_radius) for the given channel set,
    or (None, None) if the element is not one of the voxel channels."""
    e = _norm_elem(elem)
    ch = _E2CH.get(e)
    if ch is None or ch >= len(ch_elems):
        return None, None
    return ch, _VDW.get(e, 1.8)


def build_atomblob(struct: dict, voxelizer: Voxelizer, device: str) -> "torch.Tensor | None":
    """ATOM3D atom table (elem/xyz/chain) → (11, G, G, G) atomblob voxel, or None.

    Ligand = chain 'L'; pocket = protein atoms within POCKET_CUTOFF Å of any ligand
    atom. Centered on the ligand heavy-atom geometric centroid (all heavy elements,
    matching CENTER_SOURCE), coords clamped to ±25 Å, per-element vdW blob radii.
    """
    elem = np.asarray(struct["elem"])
    xyz = np.asarray(struct["xyz"], dtype=np.float32)
    chain = np.asarray(struct["chain"]).astype(str)

    is_lig = chain == "L"
    if not is_lig.any():
        return None
    lig_xyz_all = xyz[is_lig]
    lig_elem = elem[is_lig]

    # Ligand heavy-atom centroid = grid center (includes unsupported heavies, e.g. Br/I).
    heavy = np.array([_norm_elem(e) not in HYDROGENS for e in lig_elem])
    if not heavy.any():
        return None
    center = lig_xyz_all[heavy].mean(axis=0)

    # Pocket = protein atoms within cutoff of ANY ligand atom.
    prot_mask = ~is_lig
    prot_xyz, prot_elem = xyz[prot_mask], elem[prot_mask]
    if prot_xyz.shape[0]:
        d2 = ((prot_xyz[:, None, :] - lig_xyz_all[None, :, :]) ** 2).sum(-1)
        near = d2.min(axis=1) <= POCKET_CUTOFF ** 2
        prot_xyz, prot_elem = prot_xyz[near], prot_elem[near]

    def _pack(xyz_a, elem_a, ch_elems):
        cs, chs, rs = [], [], []
        for p, e in zip(xyz_a, elem_a):
            ch, rad = _channel_and_radius(e, ch_elems)
            if ch is None:
                continue
            cs.append(p); chs.append(ch); rs.append(rad)
        if not cs:
            return (torch.zeros(1, 0, 3), torch.zeros(1, 0), torch.zeros(1, 0))
        c = torch.tensor(np.asarray(cs, dtype=np.float32) - center).clamp(-25, 25)
        return (c.unsqueeze(0),
                torch.tensor(chs, dtype=torch.float32).unsqueeze(0),
                torch.tensor(rs, dtype=torch.float32).unsqueeze(0))

    lc, lch, lr = _pack(lig_xyz_all, lig_elem, LIG_CH_ELEMS)
    pc, pch, prr = _pack(prot_xyz, prot_elem, POC_CH_ELEMS)
    if lc.shape[1] == 0:
        return None

    v_lig = voxelizer({"coords": lc, "atoms_channel": lch, "radius": lr}, num_channels=N_LIG_CH)
    v_poc = voxelizer({"coords": pc, "atoms_channel": pch, "radius": prr}, num_channels=N_POC_CH)
    return torch.cat([v_lig, v_poc], dim=1).squeeze(0)   # (11, G, G, G) on device


@torch.no_grad()
def extract_features(condition: str, complexes: dict, device: str) -> dict:
    """Encode every active & inactive structure with the frozen encoder → 640-D.

    Returns {split: {id: {'active': np[D], 'inactive': np[D], 'label': int}}}.
    Cached to disk; recomputed only if the cache is missing.
    """
    cfg = CONDITIONS[condition]
    cache = FEAT_DIR / f"lep_{condition}_e{cfg['epoch']}.pt"
    if cache.exists():
        print(f"[{condition}] loading cached features {cache.name}")
        return torch.load(cache, weights_only=False)

    exp_dir = REPO / "voxbind" / cfg["exp"]
    print(f"[{condition}] loading encoder {exp_dir.name} e{cfg['epoch']} (n_in={cfg['n_in']})")
    encoder = pr.load_encoder(exp_dir, cfg["epoch"], device)
    voxelizer = Voxelizer(grid_dim=GRID_DIM, resolution=RESOLUTION,
                          cubes_around=CUBES_AROUND, device=device).to(device)

    def encode(struct):
        blob = build_atomblob(struct, voxelizer, device)   # (11,G,G,G) or None
        if blob is None:
            return None
        x = blob.unsqueeze(0)                              # (1,11,G,G,G)
        if cfg["density"]:                                 # zero-fill density+gradmag
            pad = x.new_zeros(1, cfg["n_in"] - 11, GRID_DIM, GRID_DIM, GRID_DIM)
            x = torch.cat([x, pad], dim=1)                 # (1,13,G,G,G)
        tok = encoder.forward_features(x.float())          # (1,T,D)
        return tok.mean(dim=1).squeeze(0).cpu().numpy()    # (D,)

    out, n_skip = {}, 0
    for split, rows in complexes.items():
        feats = {}
        for i, r in enumerate(rows):
            fa, fi = encode(r["active"]), encode(r["inactive"])
            if fa is None or fi is None:
                n_skip += 1
                continue
            feats[r["id"] + f"__{i}"] = {"active": fa, "inactive": fi, "label": r["label"]}
        out[split] = feats
        print(f"  [{split}] encoded {len(feats)}/{len(rows)}")
    if n_skip:
        print(f"  skipped {n_skip} structures (no ligand/pocket atoms)")

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    print(f"  cached → {cache}")
    return out


# ── metrics (voxbind env has no sklearn) ─────────────────────────────────────
def roc_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average-rank tie correction
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision[y_sorted == 1].sum()) / n_pos)


class MLP(nn.Module):
    def __init__(self, d_in, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _matrix(feats):
    ids = list(feats)
    X = np.stack([np.concatenate([feats[i]["active"], feats[i]["inactive"]]) for i in ids])
    y = np.array([feats[i]["label"] for i in ids], dtype=np.float32)
    return X.astype(np.float32), y


def train_probe(features, seed, device, hidden=128, dropout=0.2, lr=1e-3,
                weight_decay=1e-4, max_epochs=200, patience=30, batch_size=64):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = _matrix(features["train"])
    Xva, yva = _matrix(features["val"])
    Xte, yte = _matrix(features["test"])
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

    dev = torch.device(device)
    Xtr_t = torch.tensor(Xtr, device=dev); ytr_t = torch.tensor(ytr, device=dev)
    Xva_t = torch.tensor(Xva, device=dev); Xte_t = torch.tensor(Xte, device=dev)

    model = MLP(Xtr.shape[1], hidden, dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()

    best_val, best_test, wait = -1.0, None, 0
    n = Xtr.shape[0]
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for k in range(0, n, batch_size):
            idx = perm[k:k + batch_size]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = torch.sigmoid(model(Xva_t)).cpu().numpy()
            te = torch.sigmoid(model(Xte_t)).cpu().numpy()
        v_auroc = roc_auc(yva, va)
        if v_auroc > best_val:
            best_val = v_auroc
            best_test = dict(auroc=roc_auc(yte, te), auprc=average_precision(yte, te),
                             val_auroc=v_auroc, val_auprc=average_precision(yva, va))
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["coords", "cdg"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    complexes = torch.load(DATA_DIR / "lep_complexes.pt", weights_only=False)
    for split, rows in complexes.items():
        pos = sum(r["label"] for r in rows)
        print(f"[data] {split}: n={len(rows)} pos={pos} neg={len(rows) - pos}")

    results = {}
    if RESULTS_JSON.exists():
        results = json.load(open(RESULTS_JSON))

    for cond in args.conditions:
        feats = extract_features(cond, complexes, args.device)
        seed_metrics = [train_probe(feats, s, args.device) for s in range(args.seeds)]
        agg = {}
        for k in ("auroc", "auprc", "val_auroc", "val_auprc"):
            vals = np.array([m[k] for m in seed_metrics], dtype=np.float64)
            agg[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
        agg["label"] = CONDITIONS[cond]["label"]
        agg["n_test"] = len(feats["test"])
        agg["seeds"] = args.seeds
        agg["per_seed"] = [{k: m[k] for k in ("auroc", "auprc")} for m in seed_metrics]
        results[cond] = agg
        print(f"\n=== {cond} ({agg['label']}) ===")
        print(f"  test AUROC {agg['auroc']['mean']:.4f} ± {agg['auroc']['std']:.4f}")
        print(f"  test AUPRC {agg['auprc']['mean']:.4f} ± {agg['auprc']['std']:.4f}\n")

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(RESULTS_JSON, "w"), indent=2)
    print(f"wrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
