#!/usr/bin/env python3
"""Two-tower affinity probe: frozen pocket + ligand towers → cross-attention interaction head → pK.

Each tower's input construction is read from its own cfg.yaml (input_mode / pocket_ed_mask /
mask_as_channel / with_gradmag), so pocket_density, ligand_density, protein_vdw vs ligand_footprint
masking, and the mask-as-channel variants all probe correctly without code changes.

Efficient two-stage pipeline:
  1) extract_tokens — DataLoader workers stream raw voxels off disk (pinned) while both frozen
     towers assemble inputs + encode on GPU in bulk; per-patch tokens cached to disk (fp16).
  2) train_head — tokens are staged ONCE (GPU-resident, or pinned host w/ --stage_cpu) and
     CrossAttnAffinityHead (bidirectional pocket↔ligand cross-attention + 3D RoPE) trains by
     on-device index gather — no per-minibatch CPU stack / H2D.
Reports Pearson r / Spearman ρ / RMSE (mean±std over seeds) on lp_edrscc_v2, vs champion 0.644.

    python test/twotower_probe.py --pocket_exp 260806_tt_pocket_protein_vdw_mc \
        --ligand_exp 260806_tt_ligdens_protein_vdw_mc --epoch 49 --seeds 3
"""
import argparse, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(4)
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


class _RawVoxDS(torch.utils.data.Dataset):
    """Streams raw voxels off disk for one complex: (pid, atoms(11,G,G,G), density(1,G,G,G)),
    both fp32. Disk I/O (np.load) is the ONLY CPU work here — all masking / gradmag / encoding
    runs on GPU in extract_tokens — so this parallelizes cleanly across DataLoader workers and
    keeps the GPU fed (fixes the old serial per-pid CPU np.load+gradmag GPU-starver)."""
    def __init__(self, pids):
        self.pids = [p for p in pids
                     if (ATOMS / f"{p}.npy").exists() and (DENS / f"{p}.npy").exists()]

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        pid = self.pids[i]
        a = torch.from_numpy(np.load(ATOMS / f"{pid}.npy").astype(np.float32))       # (11,G,G,G)
        d = torch.from_numpy(np.load(DENS / f"{pid}.npy").astype(np.float32))[None]   # (1,G,G,G)
        return pid, a, d


def _build_tower_batch(a: torch.Tensor, d: torch.Tensor, spec: dict) -> torch.Tensor:
    """Batched, on-GPU tower-input assembly — mirrors train_density's assembly EXACTLY.

    a: (B,11,G,G,G) atoms [7 ligand, 4 pocket]; d: (B,1,G,G,G) 2Fo-Fc density.
    Channel order matches pretraining: [ atoms, density, (gradmag), (mask) ].
    """
    lig, poc = a[:, :7], a[:, 7:11]
    mode = spec["input_mode"]
    if mode == "ligand":
        return lig
    if mode == "pocket":
        return poc                                                  # coords-only pocket (4 atoms)
    if mode not in ("pocket_density", "ligand_density"):
        raise ValueError(f"two-tower probe: unsupported input_mode={mode!r}")
    atoms = poc if mode == "pocket_density" else lig
    lig_occ = lig.sum(1, keepdim=True)
    poc_occ = poc.sum(1, keepdim=True)
    keep_p, keep_l = region_keep_masks(lig_occ, poc_occ, spec["pocket_ed_mask"], spec["thresh"])
    keep = keep_p if mode == "pocket_density" else keep_l
    d_masked = d * keep                                              # rho_P or rho_L
    parts = [atoms, d_masked]
    if spec["with_gradmag"]:
        parts.append(per_sample_zscore(gradient_magnitude3d(d_masked)))
    if spec["mask_as_channel"]:
        parts.append(keep.to(d_masked.dtype))                       # trailing region-mask channel
    return torch.cat(parts, dim=1)


@torch.no_grad()
def extract_tokens(pids, pocket_enc, ligand_enc, pspec, lspec, device, bsz=48, workers=8):
    """Bulk GPU token extraction. DataLoader workers stream raw voxels off disk (pinned,
    async H2D); both towers assemble their inputs + encode on GPU per batch. Returns
    {pid: (pocket_tok, ligand_tok)} as fp16 CPU tensors."""
    from torch.utils.data import DataLoader
    # pin_memory=False: workers parallelize the disk load; a one-time extraction doesn't need
    # pinned async H2D, and pinning to a specific device tripped a CUDA "invalid argument" here.
    loader = DataLoader(_RawVoxDS(pids), batch_size=bsz, num_workers=workers,
                        pin_memory=False, shuffle=False, drop_last=False)
    out = {}
    for pids_b, a_b, d_b in loader:
        a_b = a_b.to(device, non_blocking=True)
        d_b = d_b.to(device, non_blocking=True)
        # forward_features returns (B, nG·N, D) for channel_group encoders; POOL groups → (B, N, D)
        # per-patch tokens on the shared 8³ grid, which is what CrossAttnAffinityHead's RoPE
        # (grid_p=8 → 512 positions) expects — and 1/nG the memory.
        pt = pocket_enc._pool_groups(_p.forward_tokens(pocket_enc, _build_tower_batch(a_b, d_b, pspec))).half().cpu()
        lt = ligand_enc._pool_groups(_p.forward_tokens(ligand_enc, _build_tower_batch(a_b, d_b, lspec))).half().cpu()
        for j, pid in enumerate(pids_b):
            out[pid] = (pt[j], lt[j])
    return out


def stage_tokens(tok, y, pids, stage_dev):
    """Stack a split's cached tokens into contiguous tensors ONCE (shared across seeds).

    Returns (P, L, Y): pocket/ligand tokens (N,T,D) fp16 + labels (N,) fp32, resident on
    `stage_dev`. Staging once here removes the per-minibatch CPU stack + H2D the old loop did
    every batch of every epoch of every seed. On a free GPU the whole split set (~6 GB fp16)
    lives on-device; pass --stage_cpu to keep it pinned on the host instead (per-batch async H2D)."""
    P = torch.stack([tok[p][0] for p in pids])                # (N,T,D) fp16, CPU
    L = torch.stack([tok[p][1] for p in pids])
    Y = torch.tensor([y[p] for p in pids], dtype=torch.float32)
    if stage_dev == "cpu":
        return P.pin_memory(), L.pin_memory(), Y.pin_memory()
    return P.to(stage_dev), L.to(stage_dev), Y.to(stage_dev)


def train_head(data, device, seed, epochs=300, lr=3e-4, wd=1e-2, bsz=32, n_layers=2, n_heads=8):
    """Train CrossAttnAffinityHead on pre-staged tokens. `data` = {split: (P,L,Y)} from
    stage_tokens(); minibatches are gathered by index on the staging device (no CPU stack)."""
    Ptr, Ltr, Ytr = data["train"]
    Pva, Lva, _ = data["val"]
    Pte, Lte, _ = data["test"]
    dim = Ptr.shape[-1]
    ntr = Ptr.shape[0]
    torch.manual_seed(seed)
    head = CrossAttnAffinityHead(dim=dim, grid_p=8, n_layers=n_layers, n_heads=n_heads, use_rope=True).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)

    def fetch(P, L, idx):                                      # gather + (async) H2D + upcast
        return (P[idx].to(device, non_blocking=True).float(),
                L[idx].to(device, non_blocking=True).float())

    @torch.no_grad()
    def predict(P, L):                                         # chunked eval → no OOM
        head.eval(); out = []
        for s in range(0, P.shape[0], 256):
            idx = torch.arange(s, min(s + 256, P.shape[0]), device=P.device)
            pb, lb = fetch(P, L, idx)
            out.append(head(pb, lb).float().cpu())
        return torch.cat(out).numpy()

    yva = data["val"][2].cpu().numpy()
    yte = data["test"][2].cpu().numpy()
    best_val, best_pred = -1.0, None
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(ntr, device=Ptr.device)
        for s in range(0, ntr, bsz):
            idx = perm[s:s + bsz]
            pb, lb = fetch(Ptr, Ltr, idx)
            yb = Ytr[idx].to(device, non_blocking=True)
            loss = nn.functional.mse_loss(head(pb, lb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        vr = spearmanr(predict(Pva, Lva), yva).correlation
        if vr > best_val:
            best_val, best_pred = vr, predict(Pte, Lte)
    return dict(val_rho=float(best_val),
                test_rho=float(spearmanr(best_pred, yte).correlation),
                test_r=float(pearsonr(best_pred, yte)[0]),
                test_rmse=float(np.sqrt(((best_pred - yte) ** 2).mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pocket_exp", required=True); ap.add_argument("--ligand_exp", required=True)
    ap.add_argument("--epoch", type=int, default=49); ap.add_argument("--split", default="lp_edrscc_v2")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--gpu", default="0")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--extract_only", action="store_true",
                    help="Extract+cache tokens then exit — run ONCE before fanning out seed "
                         "workers (else concurrent workers all redo the extraction).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Run ONLY this single seed (process-parallel seed fan-out); writes "
                         "result to --out and skips wandb. Omit to run range(--seeds) in-process.")
    ap.add_argument("--out", default=None,
                    help="Write result JSON here (single-seed row, or the aggregate mean/std).")
    ap.add_argument("--n_layers", type=int, default=2, help="CrossAttnAffinityHead interaction layers.")
    ap.add_argument("--n_heads", type=int, default=8, help="CrossAttnAffinityHead attention heads.")
    ap.add_argument("--stage_cpu", action="store_true",
                    help="Keep staged tokens pinned on host (per-batch async H2D) instead of "
                         "resident on GPU — use if the probe GPU is shared/tight (~6 GB fp16).")
    ap.add_argument("--no_wandb", action="store_true",
                    help="Disable wandb logging (default: log to binding-affinity).")
    ap.add_argument("--wandb_project", default="binding-affinity")
    ap.add_argument("--wandb_tags", default=None, help="Comma-separated extra wandb tags.")
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

    if args.extract_only:
        print(f"[extract_only] tokens cached at {cache_f} — exiting"); return

    have = set(tok)
    tr = [p for p in pids if split_map[p] == "train" and p in have]
    va = [p for p in pids if split_map[p] == "val" and p in have]
    te = [p for p in pids if split_map[p] == "test" and p in have]
    dim = next(iter(tok.values()))[0].shape[-1]
    print(f"split (with tokens): train {len(tr)} val {len(va)} test {len(te)} | dim {dim}")

    # Stage tokens ONCE (shared across all seeds) → GPU-resident (or pinned host w/ --stage_cpu).
    stage_dev = "cpu" if args.stage_cpu else device
    data = {sp: stage_tokens(tok, y, ids, stage_dev)
            for sp, ids in (("train", tr), ("val", va), ("test", te))}
    gb = sum(t.element_size() * t.nelement()
             for s in data.values() for t in s[:2]) / 1e9
    print(f"staged tokens on {stage_dev} ({gb:.1f} GB fp16)")

    # Single-seed worker (parallel fan-out): train one seed, write JSON, no wandb.
    if args.seed is not None:
        m = train_head(data, device, seed=args.seed, n_layers=args.n_layers, n_heads=args.n_heads)
        print(f"seed={args.seed}  val_rho={m['val_rho']:.4f}  test_rho={m['test_rho']:.4f}  "
              f"test_r={m['test_r']:.4f}  test_rmse={m['test_rmse']:.4f}")
        if args.out:
            import json
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({"seed": args.seed, **m}))
            print(f"-> {args.out}")
        return

    wb = None
    if not args.no_wandb:
        try:
            import wandb as _wandb
            _extra = [t for t in (args.wandb_tags or "").split(",") if t]
            wb = _wandb.init(
                project=args.wandb_project, entity="eddy26", job_type="probe", reinit=True,
                name=f"twotower_{args.pocket_exp}+{args.ligand_exp}_e{args.epoch}",
                tags=["probe", "twotower", f"split:{args.split}", f"epoch:{args.epoch}", *_extra],
                config=dict(kind="twotower_crossattn", pocket_exp=args.pocket_exp,
                            ligand_exp=args.ligand_exp, epoch=args.epoch,
                            split=args.split, seeds=args.seeds, dim=dim),
            )
        except Exception as e:
            print(f"[wandb] init failed ({e!r}); continuing without wandb"); wb = None

    rows = []
    for s in range(args.seeds):
        m = train_head(data, device, seed=s, n_layers=args.n_layers, n_heads=args.n_heads)
        rows.append(m)
        print(f"  seed={s}  val_rho={m['val_rho']:.4f}  test_rho={m['test_rho']:.4f}  "
              f"test_r={m['test_r']:.4f}  test_rmse={m['test_rmse']:.4f}")
        if wb is not None:
            wb.log({"seed/val_rho": m["val_rho"], "seed/test_rho": m["test_rho"],
                    "seed/test_r": m["test_r"], "seed/test_rmse": m["test_rmse"], "seed": s})
    df = pd.DataFrame(rows)
    print("\n=== two-tower cross-attention probe (vs champion 0.644) ===")
    for k in ("test_rho", "test_r", "test_rmse", "val_rho"):
        print(f"  {k:10s}: {df[k].mean():.4f} ± {df[k].std():.4f}")
    if args.out:
        import json
        agg = {"n": len(rows)}
        for k in ("test_rho", "test_r", "test_rmse", "val_rho"):
            agg[f"{k}_mean"] = float(df[k].mean()); agg[f"{k}_std"] = float(df[k].std())
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(agg, indent=2)); print(f"-> {args.out}")
    if wb is not None:
        wb.summary.update({f"{k}_mean": float(df[k].mean()) for k in
                           ("test_rho", "test_r", "test_rmse", "val_rho")})
        wb.summary.update({f"{k}_std": float(df[k].std()) for k in
                           ("test_rho", "test_r", "test_rmse", "val_rho")})
        wb.finish()


if __name__ == "__main__":
    main()
