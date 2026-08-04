#!/usr/bin/env python3
"""Quick intermediate read on the cached two-tower tokens: 1 seed, per-epoch val rho vs champion 0.644."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
torch.set_num_threads(2)
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
import importlib.util
_spec = importlib.util.spec_from_file_location("probe01c", str(ROOT / "dataset" / "01c_pdbbind_probe.py"))
_p = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_p)
from voxbind.models.twotower_head import CrossAttnAffinityHead

CACHE = ROOT / "dataset" / "data" / "pdbbind" / "features" / "twotower_tokens" / "260803_twotower_pocket__260803_twotower_ligand_e49.pt"
device = "cuda:4"
tok = torch.load(CACHE, weights_only=False); print(f"loaded {len(tok)} pairs", flush=True)
split_map, _ = _p.load_frozen_split_map("lp_edrscc_v2")
lp = _p.load_lp_index(_p.LP_CSV)
y = {p: v for p, v in zip(lp["pdb_id"], lp["pK"]) if pd.notna(v)}
have = set(tok) & set(y)
tr = [p for p in have if split_map.get(p) == "train"]
va = [p for p in have if split_map.get(p) == "val"]
te = [p for p in have if split_map.get(p) == "test"]
dim = next(iter(tok.values()))[0].shape[-1]
print(f"train {len(tr)} val {len(va)} test {len(te)} dim {dim}", flush=True)
torch.manual_seed(0)
head = CrossAttnAffinityHead(dim=dim, grid_p=8, n_layers=2, n_heads=8, use_rope=True).to(device)
opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-2)
def batch(pids):
    P = torch.stack([tok[p][0] for p in pids]).float().to(device)
    L = torch.stack([tok[p][1] for p in pids]).float().to(device)
    Y = torch.tensor([y[p] for p in pids], dtype=torch.float32, device=device)
    return P, L, Y
def predict(pids):
    head.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(pids), 64):
            P, L, _ = batch(pids[s:s+64]); out.append(head(P, L).float().cpu())
    return torch.cat(out).numpy()
vy = np.array([y[p] for p in va]); ty = np.array([y[p] for p in te])
best_v = -1
for ep in range(300):
    head.train(); perm = [tr[i] for i in torch.randperm(len(tr))]
    for s in range(0, len(perm), 32):
        P, L, Y = batch(perm[s:s+32])
        loss = nn.functional.mse_loss(head(P, L), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    vp = predict(va); vr = spearmanr(vp, vy).correlation
    if vr > best_v:
        best_v = vr; tp = predict(te)
        tr_ = spearmanr(tp, ty).correlation; trr = pearsonr(tp, ty)[0]
        trmse = float(np.sqrt(((tp - ty)**2).mean()))
    if ep % 5 == 0 or ep < 5:
        print(f"ep{ep:3d} val_rho={vr:.4f} | best_val={best_v:.4f} test_rho={tr_:.4f} r={trr:.4f} rmse={trmse:.3f} (champ 0.644)", flush=True)
print(f"DONE best_val={best_v:.4f} test_rho={tr_:.4f} test_r={trr:.4f} test_rmse={trmse:.3f}", flush=True)
