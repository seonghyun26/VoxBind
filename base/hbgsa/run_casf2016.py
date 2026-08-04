"""run_casf2016.py — train HBGSA on the STANDARD lp_edrscc_v2 split, evaluate on CASF-2016.

Correct HonestAffinity protocol (matches GET/EGNN/ProFSA):
- Train on the STANDARD lp_edrscc_v2 manifest (train=3850 / val=817) — the 90 CASF
  pids that overlap v2-train STAY in train (they are genuinely trained on → memorized).
- After each seed trains, run a SEPARATE inference pass over ALL 214 CASF pids
  (voxbind/splits/casf2016_eval.csv), including the 90 train members.
- Compute metrics: leaky = all 214 (memorization included), nontrain = 124 with in_v2train=0.
- Write base/_casf/HBGSA.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
REPO_DIR  = Path(__file__).resolve().parent.parent.parent   # VoxBind/
HBGSA_DIR = Path(__file__).resolve().parent
SRC_DIR   = HBGSA_DIR / "src"
OUT_DIR   = REPO_DIR / "base" / "_casf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_SPLIT_CSV, SPLITS_DIR
CASF_CSV        = REPO_DIR / "voxbind" / "splits" / "casf2016_eval.csv"
CASF_TEST_SPLIT = SPLITS_DIR / "casf2016_testonly_split.csv"   # all 214 CASF pids → test


# ── build a CASF-only "test" split so build_manifest resolves those 214 rows ─
def build_casf_test_split():
    casf = pd.read_csv(CASF_CSV)
    casf["pid"] = casf["pid"].str.lower()
    df = pd.DataFrame({"pid": casf["pid"], "split": "test"})
    df.to_csv(CASF_TEST_SPLIT, index=False)
    print(f"[casf-split] written → {CASF_TEST_SPLIT}  ({len(df)} CASF pids as test)")
    return df


# ── build H-bond cache for any pid in a manifest that lacks it ──────────────
def ensure_hbond_cache(manifest, label=""):
    from hbonds import build_cache, HB_CACHE
    missing = manifest[[not (HB_CACHE / f"{r.pdb_id}.npz").exists()
                        for _, r in manifest.iterrows()]]
    if len(missing) == 0:
        print(f"[hbonds{(' '+label) if label else ''}] all {len(manifest)} pids cached")
        return
    print(f"[hbonds{(' '+label) if label else ''}] building {len(missing)} missing: "
          f"{missing.pdb_id.tolist()[:20]}")
    build_cache(missing.reset_index(drop=True), overwrite=False)


# ── main train + inference driver ───────────────────────────────────────────
def train_and_predict(seeds=(0, 1, 2), device="cuda", tag="casf2016_hbgsa_3p06m"):
    import argparse
    import time
    import torch
    from torch.utils.data import DataLoader
    from scipy.stats import pearsonr, spearmanr

    from manifest import build_manifest
    from featurize import build_smiles_vocab
    from dataset import HBGSADataset, collate
    from model import HBGSA, hybrid_loss
    from train import set_seed, move, predict

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n[train] device={device_obj}  tag={tag}  seeds={seeds}")

    # ── STANDARD v2 manifest (train=3850 / val=817 / test=1320) ─────────────
    manifest = build_manifest(split_csv=DEFAULT_SPLIT_CSV, cl1_only=False, verbose=True)
    print(f"[train] STANDARD v2 manifest: "
          f"train={int((manifest.new_split=='train').sum())}  "
          f"val={int((manifest.new_split=='val').sum())}  "
          f"test={int((manifest.new_split=='test').sum())}")
    ensure_hbond_cache(manifest[manifest.new_split.isin(['train', 'val'])], label="v2 train/val")

    # ── CASF-214 inference manifest (all 214 as 'test') ─────────────────────
    casf_manifest = build_manifest(split_csv=CASF_TEST_SPLIT, cl1_only=False,
                                   require_smiles=True, verbose=True)
    casf_test = casf_manifest[casf_manifest.new_split == "test"].reset_index(drop=True)
    print(f"[train] CASF inference manifest: {len(casf_test)} complexes resolved")
    ensure_hbond_cache(casf_test, label="CASF-214")

    # ── SMILES vocab from the STANDARD v2 TRAIN split (tokens must match training) ──
    vocab = build_smiles_vocab(manifest[manifest.new_split == "train"].smiles)
    print(f"[train] SMILES vocab size: {len(vocab)}")

    # 3.06M model args (train.py defaults)
    args = argparse.Namespace(
        batch_size=64, num_workers=4,
        d_model=128, emb_dim=128, n_layers=2, n_heads=4,
        gcn_hidden=128, pocket_hidden=128, head_hidden=128,
        conv_channels=213, conv_dilations_tuple=(1, 2, 4, 8), convs_per_block=2,
        lr=1e-3, weight_decay=1e-5, epochs=150, patience=20,
    )

    results_dir = HBGSA_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    all_preds = {}

    # CASF inference loader (shared across seeds; deterministic order)
    casf_loader = DataLoader(HBGSADataset(casf_test, vocab), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate,
                             num_workers=args.num_workers, pin_memory=True)

    for seed in seeds:
        print(f"\n{'='*60}\n[seed{seed}] training on STANDARD v2 (train=3850)\n{'='*60}", flush=True)
        set_seed(seed)
        sub = {s: manifest[manifest.new_split == s] for s in ("train", "val")}
        loaders = {
            s: DataLoader(HBGSADataset(df, vocab), batch_size=args.batch_size,
                          shuffle=(s == "train"), collate_fn=collate,
                          num_workers=args.num_workers, pin_memory=True, drop_last=(s == "train"))
            for s, df in sub.items()
        }
        # standardize target on TRAIN (calibrates SmoothL1; correlation is affine-invariant)
        y_mean = float(sub["train"].pK.mean())
        y_std  = float(sub["train"].pK.std()) or 1.0

        model = HBGSA(
            smiles_vocab_size=len(vocab),
            seq_d_model=args.d_model, smi_d_model=args.d_model, emb_dim=args.emb_dim,
            n_layers=args.n_layers, n_heads=args.n_heads, gcn_hidden=args.gcn_hidden,
            pocket_hidden=args.pocket_hidden, head_hidden=args.head_hidden,
            conv_channels=args.conv_channels, conv_dilations=args.conv_dilations_tuple,
            convs_per_block=args.convs_per_block,
        ).to(device_obj)
        n_param = sum(p.numel() for p in model.parameters())
        print(f"[seed{seed}] params: {n_param:,} ({n_param/1e6:.2f}M)", flush=True)

        opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=6)

        best_val, best_state, bad = -1e9, None, 0
        for ep in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            for batch in loaders["train"]:
                batch = move(batch, device_obj)
                target = (batch["y"] - y_mean) / y_std
                loss, _, _ = hybrid_loss(model(batch), target)
                opt.zero_grad(); loss.backward(); opt.step()
            _, yv, pv = predict(model, loaders["val"], device_obj, y_mean, y_std)
            val_s = float(spearmanr(yv, pv)[0])
            val_r = float(pearsonr(yv, pv)[0])
            sched.step(val_s)
            if val_s > best_val:
                best_val, bad = val_s, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(f"  seed{seed} ep{ep:3d}  val_ρ={val_s:.4f}  val_r={val_r:.4f}  "
                  f"best_ρ={best_val:.4f}  ({time.time()-t0:.1f}s)", flush=True)
            if bad >= args.patience:
                print(f"  seed{seed} early-stop ep{ep}", flush=True)
                break

        # ── SEPARATE inference over all 214 CASF pids (best val model) ──────
        model.load_state_dict(best_state)
        pids, y_lp, pred = predict(model, casf_loader, device_obj, y_mean, y_std)
        preds = {"pdb_id": pids.tolist(), "y": y_lp.tolist(), "pred": pred.tolist()}
        pred_path = results_dir / f"preds_{tag}_seed{seed}.json"
        pred_path.write_text(json.dumps(preds))
        print(f"[seed{seed}] best_val_ρ={best_val:.4f}  → CASF inference: "
              f"{len(pids)} predictions  saved → {pred_path}", flush=True)
        all_preds[seed] = preds

    return all_preds


# ── compute metrics and write JSON ──────────────────────────────────────────
def compute_and_write_json(all_preds, seeds=(0, 1, 2), tag="casf2016_hbgsa_3p06m"):
    from scipy.stats import pearsonr, spearmanr

    casf = pd.read_csv(CASF_CSV)
    casf["pid"] = casf["pid"].str.lower()
    casf_pk       = dict(zip(casf["pid"], casf["pK"]))
    nontrain_pids = set(casf[casf["in_v2train"] == 0]["pid"])
    leaky_pids    = set(casf["pid"])

    print(f"\n[metrics] total CASF={len(leaky_pids)}  nontrain={len(nontrain_pids)}")

    seed_metrics = {"leaky": [], "nontrain": []}
    for seed in seeds:
        preds    = all_preds[seed]
        pid_arr  = np.array([p.lower() for p in preds["pdb_id"]])
        pred_arr = np.array(preds["pred"])
        casf_y   = np.array([casf_pk.get(p, np.nan) for p in pid_arr])   # CASF canonical pK

        leaky_mask = np.array([p in leaky_pids for p in pid_arr])
        nt_mask    = np.array([p in nontrain_pids for p in pid_arr])
        for subset_name, mask in [("leaky", leaky_mask), ("nontrain", nt_mask)]:
            ym, pm = casf_y[mask], pred_arr[mask]
            ok = ~np.isnan(ym)
            ym, pm = ym[ok], pm[ok]
            r    = float(pearsonr(ym, pm)[0])
            rho  = float(spearmanr(ym, pm)[0])
            rmse = float(np.sqrt(np.mean((ym - pm) ** 2)))
            print(f"  seed{seed} {subset_name:8s} n={len(ym)}  "
                  f"r={r:.4f}  ρ={rho:.4f}  rmse={rmse:.4f}")
            seed_metrics[subset_name].append({"r": r, "rho": rho, "rmse": rmse, "n": len(ym)})

    n_leaky, n_nontrain = seed_metrics["leaky"][0]["n"], seed_metrics["nontrain"][0]["n"]
    if n_leaky != 214:
        print(f"WARNING: leaky n={n_leaky} != 214")
    if n_nontrain != 124:
        print(f"WARNING: nontrain n={n_nontrain} != 124")

    def agg(vals, key):
        arr = np.array([v[key] for v in vals])
        return {"mean": round(float(arr.mean()), 4), "std": round(float(arr.std(ddof=0)), 4)}

    result = {
        "model": "HBGSA",
        "train": "lp_edrscc_v2 train",
        "tag":   tag,
        "seeds": list(seeds),
        "leaky": {
            "pearson":  agg(seed_metrics["leaky"], "r"),
            "spearman": agg(seed_metrics["leaky"], "rho"),
            "rmse":     agg(seed_metrics["leaky"], "rmse"),
            "n": n_leaky,
        },
        "nontrain": {
            "pearson":  agg(seed_metrics["nontrain"], "r"),
            "spearman": agg(seed_metrics["nontrain"], "rho"),
            "rmse":     agg(seed_metrics["nontrain"], "rmse"),
            "n": n_nontrain,
        },
    }
    out_path = OUT_DIR / "HBGSA.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[done] written → {out_path}")
    print(json.dumps(result, indent=2))
    return result


# ── main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse as _ap
    ap = _ap.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="casf2016_hbgsa_3p06m")
    ap.add_argument("--skip_train", action="store_true",
                    help="reload saved preds and recompute metrics only")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    build_casf_test_split()

    if args.skip_train:
        print("[skip_train] reloading preds from disk…")
        all_preds = {}
        for seed in seeds:
            p = HBGSA_DIR / "results" / f"preds_{args.tag}_seed{seed}.json"
            all_preds[seed] = json.loads(p.read_text())
            print(f"  seed{seed}: {len(all_preds[seed]['pdb_id'])} pids")
    else:
        all_preds = train_and_predict(seeds=seeds, device=args.device, tag=args.tag)

    compute_and_write_json(all_preds, seeds=seeds, tag=args.tag)
