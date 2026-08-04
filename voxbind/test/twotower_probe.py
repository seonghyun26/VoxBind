#!/usr/bin/env python3
"""Two-tower affinity probe: frozen pocket + ligand towers → cross-attention interaction head → pK.

Extracts per-patch tokens from each frozen tower (pocket = 4 pocket atoms + LIGAND-MASKED
apo-like density + gradmag; ligand = 7 ligand atoms), caches them, then trains
CrossAttnAffinityHead (bidirectional pocket↔ligand cross-attention + 3D RoPE) on lp_edrscc_v2.
Reports Pearson r / Spearman ρ / RMSE (3 seeds), vs the single-encoder champion 0.644.

    python test/twotower_probe.py --pocket_exp 260803_twotower_pocket \
        --ligand_exp 260803_twotower_ligand --epoch 49 --seeds 3
"""
import argparse, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
PDBBIND = ROOT / "dataset" / "data" / "pdbbind"
ATOMS = PDBBIND / "voxels_ligvdw" / "atoms"
DENS = PDBBIND / "voxels_v5" / "density"
CACHE = PDBBIND / "features" / "twotower_tokens"

# import helpers from the digit-named probe module via importlib
_spec = importlib.util.spec_from_file_location("probe01c", str(ROOT / "dataset" / "01c_pdbbind_probe.py"))
_p = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_p)
from voxbind.models.mae_ops import (
    gradient_magnitude3d, per_sample_zscore, region_keep_masks,
)
from voxbind.models.twotower_head import CrossAttnAffinityHead


def _tower_spec(cfg) -> dict:
    """Extract the input-construction knobs from a tower's cfg.yaml (source of truth)."""
    input_mode = (cfg.get("input_mode", None)
                  or OmegaConf.select(cfg, "model.input_mode", default=None))
    with_gradmag = cfg.get("with_gradmag", None)
    if with_gradmag is None:
        with_gradmag = OmegaConf.select(cfg, "model.with_gradmag", default=False)
    return dict(
        input_mode      = str(input_mode),
        with_gradmag    = bool(with_gradmag),
        pocket_ed_mask  = str(OmegaConf.select(cfg, "mae.pocket_ed_mask", default="ligand_footprint")),
        mask_as_channel = bool(OmegaConf.select(cfg, "mae.mask_as_channel", default=False)),
        thresh          = float(OmegaConf.select(cfg, "mae.lig_mask_thresh", default=0.1)),
    )


def _build_tower(a: torch.Tensor, d: torch.Tensor, spec: dict) -> torch.Tensor:
    """Build one tower's input tensor, mirroring train_density's assembly EXACTLY.

    a: (11,G,G,G) atom channels [7 ligand, 4 pocket]; d: (1,G,G,G) 2Fo-Fc density.
    Channel order matches pretraining: [ atoms, density, (gradmag), (mask) ].
    """
    lig, poc = a[:7], a[7:11]
    mode = spec["input_mode"]
    if mode == "ligand":
        return lig
    if mode == "pocket_density":
        atoms = poc
    elif mode == "ligand_density":
        atoms = lig
    else:
        raise ValueError(f"two-tower probe: unsupported input_mode={mode!r}")

    lig_occ = lig.sum(0, keepdim=True)                           # (1,G,G,G)
    poc_occ = poc.sum(0, keepdim=True)
    keep_p, keep_l = region_keep_masks(lig_occ, poc_occ, spec["pocket_ed_mask"], spec["thresh"])
    keep = keep_p if mode == "pocket_density" else keep_l
    d_masked = d * keep                                          # rho_P or rho_L
    parts = [atoms, d_masked]
    if spec["with_gradmag"]:
        g = per_sample_zscore(gradient_magnitude3d(d_masked.unsqueeze(0))).squeeze(0)
        parts.append(g)
    if spec["mask_as_channel"]:
        parts.append(keep.to(d_masked.dtype))                   # trailing region-mask channel
    return torch.cat(parts, dim=0)


@torch.no_grad()
def build_inputs(pid, pspec, lspec):
    """(pocket_input, ligand_input) for one complex, or None if missing."""
    fa, fd = ATOMS / f"{pid}.npy", DENS / f"{pid}.npy"
    if not (fa.exists() and fd.exists()):
        return None
    a = torch.from_numpy(np.load(fa).astype(np.float32))          # (11,G,G,G)
    d = torch.from_numpy(np.load(fd).astype(np.float32)).unsqueeze(0)  # (1,G,G,G)
    pocket = _build_tower(a, d, pspec)
    ligand = _build_tower(a, d, lspec)
    return pocket, ligand


@torch.no_grad()
def extract_tokens(pids, pocket_enc, ligand_enc, pspec, lspec, device, bsz=32):
    """Run both towers → {pid: (pocket_tokens, ligand_tokens)} on CPU (fp16)."""
    out, buf = {}, []
    def flush(items):
        ps = torch.stack([x[1][0] for x in items]).to(device)
        ls = torch.stack([x[1][1] for x in items]).to(device)
        pt = _p.forward_tokens(pocket_enc, ps).half().cpu()
        lt = _p.forward_tokens(ligand_enc, ls).half().cpu()
        for i, (pid, _) in enumerate(items):
            out[pid] = (pt[i], lt[i])
    for pid in pids:
        io = build_inputs(pid, pspec, lspec)
        if io is None:
            continue
        buf.append((pid, io))
        if len(buf) >= bsz:
            flush(buf); buf = []
    if buf:
        flush(buf)
    return out


def train_head(tok, y, tr, va, te, dim, device, seed, epochs=300, lr=3e-4, wd=1e-2):
    torch.manual_seed(seed)
    head = CrossAttnAffinityHead(dim=dim, grid_p=8, n_layers=2, n_heads=8, use_rope=True).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    def batch(pids):
        P = torch.stack([tok[p][0] for p in pids]).float().to(device)
        L = torch.stack([tok[p][1] for p in pids]).float().to(device)
        Y = torch.tensor([y[p] for p in pids], dtype=torch.float32, device=device)
        return P, L, Y
    best_val, best_test, best_pred = -1, None, None
    for ep in range(epochs):
        head.train(); perm = [tr[i] for i in torch.randperm(len(tr))]
        for s in range(0, len(perm), 32):
            P, L, Y = batch(perm[s:s + 32])
            loss = nn.functional.mse_loss(head(P, L), Y)
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        def predict(pids):                                    # chunked eval → no OOM
            out = []
            with torch.no_grad():
                for s in range(0, len(pids), 64):
                    P, L, _ = batch(pids[s:s + 64])
                    out.append(head(P, L).float().cpu())
            return torch.cat(out).numpy()
        vp = predict(va); vy = np.array([y[p] for p in va])
        vr = spearmanr(vp, vy).correlation
        if vr > best_val:
            best_val, best_pred = vr, predict(te)
    ty = np.array([y[p] for p in te])
    return dict(val_rho=best_val,
                test_rho=spearmanr(best_pred, ty).correlation,
                test_r=pearsonr(best_pred, ty)[0],
                test_rmse=float(np.sqrt(((best_pred - ty) ** 2).mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pocket_exp", required=True); ap.add_argument("--ligand_exp", required=True)
    ap.add_argument("--epoch", type=int, default=49); ap.add_argument("--split", default="lp_edrscc_v2")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--gpu", default="0")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"

    split_map, _ = _p.load_frozen_split_map(args.split)
    lp = _p.load_lp_index(_p.LP_CSV)
    y = {p: v for p, v in zip(lp["pdb_id"], lp["pK"]) if pd.notna(v)}
    pids = [p for p in split_map if p in y]

    cache_f = CACHE / f"{args.pocket_exp}__{args.ligand_exp}_e{args.epoch}.pt"
    if cache_f.exists() and not args.refresh:
        tok = torch.load(cache_f, weights_only=False); print(f"loaded {len(tok)} cached token pairs")
    else:
        pcfg = OmegaConf.load(ROOT / "exps" / args.pocket_exp / "cfg.yaml")
        lcfg = OmegaConf.load(ROOT / "exps" / args.ligand_exp / "cfg.yaml")
        pspec, lspec = _tower_spec(pcfg), _tower_spec(lcfg)
        print(f"pocket tower: {pspec}")
        print(f"ligand tower: {lspec}")
        pocket_enc = _p.load_encoder(ROOT / "exps" / args.pocket_exp, args.epoch, device, cfg=pcfg)
        ligand_enc = _p.load_encoder(ROOT / "exps" / args.ligand_exp, args.epoch, device, cfg=lcfg)
        print("extracting tokens...")
        tok = extract_tokens(pids, pocket_enc, ligand_enc, pspec, lspec, device)
        CACHE.mkdir(parents=True, exist_ok=True); torch.save(tok, cache_f)
        print(f"cached {len(tok)} token pairs → {cache_f}")

    have = set(tok)
    tr = [p for p in pids if split_map[p] == "train" and p in have]
    va = [p for p in pids if split_map[p] == "val" and p in have]
    te = [p for p in pids if split_map[p] == "test" and p in have]
    dim = next(iter(tok.values()))[0].shape[-1]
    print(f"split (with tokens): train {len(tr)} val {len(va)} test {len(te)} | dim {dim}")

    rows = [train_head(tok, y, tr, va, te, dim, device, seed=s) for s in range(args.seeds)]
    df = pd.DataFrame(rows)
    print("\n=== two-tower cross-attention probe (vs champion 0.644) ===")
    for k in ("test_rho", "test_r", "test_rmse", "val_rho"):
        print(f"  {k:10s}: {df[k].mean():.4f} ± {df[k].std():.4f}")


if __name__ == "__main__":
    main()
