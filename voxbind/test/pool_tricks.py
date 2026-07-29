"""pool_tricks.py — does a smarter token→vector AGGREGATION beat plain mean-pool?

The cached probe feature is `tokens.mean(dim=1)` — a flat average over all ~512 patch
tokens. For affinity that dilutes the ligand/interface hotspot with bulk pocket + empty
space. Here we extract several poolings from the SAME frozen tokens and score each with
BOTH readouts (MLP2 + TabPFN), matched lp_edrscc_v2, NO leak (train-only fit).

Poolings (per complex, over per-patch tokens (npatch, D)):
  mean  — baseline (= current cache)
  max   — per-dim max token   (binding hotspot: strongest-activating patch)
  std   — per-dim token spread (distributional info mean discards)
  lig   — ligand-occupancy-weighted mean  (interface emphasis)
  poc   — pocket-occupancy-weighted mean

Representations tested = concatenations of the above (z-scored on train).

Usage:
  python test/pool_tricks.py --encoder champion --device cuda:0 --learners mlp tabpfn
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
VOX = HERE.parent
PROBE_PY = VOX / "dataset" / "01c_pdbbind_probe.py"

ENCODERS = {
    "champion": (VOX / "exps/260705_ar_cvit_100m_v2_mask075", 49),
    "t21":      (VOX / "exps/260725_ar_cvit_100m_v3_m095", 49),
}
COND = "atomblob_density_gradmag"
VOXV = "v5"


def load_probe():
    spec = importlib.util.spec_from_file_location("probe01c", PROBE_PY)
    m = importlib.util.module_from_spec(spec); sys.modules["probe01c"] = m
    spec.loader.exec_module(m); return m


def extract(P, exp_dir, epoch, device):
    """Return {pool_name: {pid: vec}} for mean/max/std/lig/poc."""
    cfg = OmegaConf.load(exp_dir / "cfg.yaml")
    spec = P.infer_feature_spec(COND, cfg, "auto")
    n_lig = int(cfg.model.get("n_channels_ligand", 7))
    n_poc = int(cfg.model.get("n_channels_pocket", 4))
    vox_dir = P.voxel_dir_for(VOXV); atom_dir = P.atom_dir_for(vox_dir, spec.atom_source)
    dens_dir = vox_dir / "density"
    enc = P.load_encoder(exp_dir, epoch, device, cfg=cfg)
    n_in = enc.n_in_channels
    smap, scheme = P.load_frozen_split_map("lp_edrscc_v2")
    pids = sorted(smap.keys())
    ds = P._ExtractDataset(pids, COND, n_in, atom_dir, dens_dir, spec.input_mode, spec.with_gradmag, None)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=P._extract_collate)
    G = int(cfg.vox.grid_dim); pe = 8; npatch = (G // pe) ** 3
    out = {k: {} for k in ("mean", "max", "std", "lig", "poc")}
    for bpids, x, errs in loader:
        if x is None:
            continue
        x = x.to(device); B = x.shape[0]
        with torch.no_grad():
            tok = P.encode_tokens(enc, x)                          # (B, nG*npatch, D)
        nG = tok.shape[1] // npatch
        tok = tok.reshape(B, nG, npatch, tok.shape[2]).mean(1)     # (B, npatch, D)
        occ = x.reshape(B, n_in, G // pe, pe, G // pe, pe, G // pe, pe).sum(dim=(3, 5, 7))
        occ = occ.reshape(B, n_in, npatch)
        lig_w = occ[:, :n_lig].sum(1); poc_w = occ[:, n_lig:n_lig + n_poc].sum(1)

        def wpool(w):
            w = w.clamp(min=0); s = w.sum(1, keepdim=True)
            wn = torch.where(s > 1e-6, w / s.clamp(min=1e-6), torch.full_like(w, 1.0 / npatch))
            return (wn.unsqueeze(-1) * tok).sum(1)

        gm = tok.mean(1); gx = tok.max(1).values; gs = tok.std(1)
        gl = wpool(lig_w); gp = wpool(poc_w)
        for i, pid in enumerate(bpids):
            out["mean"][pid] = gm[i].cpu().clone(); out["max"][pid] = gx[i].cpu().clone()
            out["std"][pid] = gs[i].cpu().clone();  out["lig"][pid] = gl[i].cpu().clone()
            out["poc"][pid] = gp[i].cpu().clone()
    print(f"  extracted {len(out['mean'])} complexes (D={tok.shape[2]}, npatch={npatch}, nG={nG})")
    return out, smap


def metrics(pred, y):
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    return (float(pearsonr(pred, y).statistic), float(spearmanr(pred, y).statistic),
            float(np.sqrt(((pred - y) ** 2).mean())))


def build_data(P, pools, parts, smap, zscore=False):
    """Assemble a features dict {pid: concat} then P.build_dataset.

    zscore=False reproduces the canonical probe (raw LayerNorm-scale features → MLP 0.644
    on mean). zscore=True is needed only when concatenating scale-mismatched blocks
    (max/std) so no block dominates; it costs the raw-tuned MLP ~0.05 but is invariant to
    TabPFN. Convex-combination pools (mean/lig/poc) are already ~unit scale → keep raw.
    """
    pids = list(pools["mean"].keys())
    feats = {}
    for pid in pids:
        feats[pid] = torch.from_numpy(
            np.concatenate([pools[p][pid].numpy() for p in parts]).astype(np.float32))
    lp_df = P.load_lp_index(P.LP_CSV)
    data = P.build_dataset(feats, lp_df, drop_covalent=True, cl1_only=False,
                           target_map=None, split_map=smap)
    if zscore:
        mu = data["train"]["X"].mean(0, keepdims=True)
        sd = data["train"]["X"].std(0, keepdims=True) + 1e-6
        for s in ("train", "val", "test"):
            data[s]["X"] = ((data[s]["X"] - mu) / sd).astype(np.float32)
    return data


def eval_mlp(P, data, device, seeds=3):
    rows = [P.train_one(data, seed=s, device=device, max_epochs=200, patience=30,
                        batch_size=64, lr=1e-3, weight_decay=1e-4, hidden=128, dropout=0.1)
            for s in range(seeds)]
    def mean(k): return float(np.mean([r[k] for r in rows]))
    return dict(val_rho=mean("best_val_spearman"), val_r=mean("val_pearson"),
                test_rho=mean("test_spearman"), test_r=mean("test_pearson"),
                test_rmse=mean("test_rmse"))


def eval_tabpfn(data, device, seeds=(0, 1, 2)):
    from tabpfn import TabPFNRegressor
    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    res = {}
    for split in ("val", "test"):
        preds = []
        for s in seeds:
            reg = TabPFNRegressor(device=device, random_state=s); reg.fit(Xtr, ytr)
            preds.append(reg.predict(data[split]["X"]))
        r, rho, rmse = metrics(np.mean(preds, 0), data[split]["y"])
        res[f"{split}_r"], res[f"{split}_rho"] = r, rho
        if split == "test":
            res["test_rmse"] = rmse
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="champion", choices=list(ENCODERS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--learners", nargs="+", default=["mlp", "tabpfn"], choices=["mlp", "tabpfn"])
    args = ap.parse_args()

    P = load_probe()
    exp_dir, epoch = ENCODERS[args.encoder]
    print(f"=== pool tricks · {args.encoder} ({exp_dir.name} e{epoch}) ===")
    cache = HERE / f"pool_feats_{args.encoder}.pt"
    if cache.exists():
        blob = torch.load(cache, weights_only=False)
        pools, smap = blob["pools"], blob["smap"]
        print(f"  loaded cached pools ({len(pools['mean'])} complexes) from {cache.name}")
    else:
        pools, smap = extract(P, exp_dir, epoch, args.device)
        torch.save({"pools": pools, "smap": smap}, cache)

    REPS = {
        "mean            ": ["mean"],
        "max             ": ["max"],
        "std             ": ["std"],
        "mean+max        ": ["mean", "max"],
        "mean+std        ": ["mean", "std"],
        "mean+max+std    ": ["mean", "max", "std"],
        "lig+poc         ": ["lig", "poc"],
        "mean+lig        ": ["mean", "lig"],
        "mean+max+lig    ": ["mean", "max", "lig"],
    }
    for name, parts in REPS.items():
        zscore = any(p in ("max", "std") for p in parts)   # raw for convex pools; z-score only scale-mixed
        data = build_data(P, pools, parts, smap, zscore=zscore)
        dim = data["train"]["X"].shape[1]
        print(f"\n[{name}] dim={dim}{'  (z-scored)' if zscore else '  (raw)'}")
        if "mlp" in args.learners:
            m = eval_mlp(P, data, args.device)
            print(f"   MLP    val ρ={m['val_rho']:.4f}  |  test ρ={m['test_rho']:.4f} r={m['test_r']:.4f} rmse={m['test_rmse']:.4f}")
        if "tabpfn" in args.learners:
            t = eval_tabpfn(data, args.device)
            print(f"   TabPFN val ρ={t['val_rho']:.4f}  |  test ρ={t['test_rho']:.4f} r={t['test_r']:.4f} rmse={t['test_rmse']:.4f}")


if __name__ == "__main__":
    main()
