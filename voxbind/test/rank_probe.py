#!/usr/bin/env python
"""Ranking-aware probe heads on the FROZEN champion features (lp_edrscc_v2).

The eval metric is Spearman ρ, but the probe trains with MSE. This tests whether a
loss that DIRECTLY targets ordering beats MSE / MSE+corr on ρ. All heads share the
same MLP2 + full-batch train loop + early-stop-on-val-ρ, so mse/mse+corr recomputed
here are the apples-to-apples baselines (should land ~0.644 / 0.647).

Heads:
  mse          F.mse_loss                                  (calibrated; RMSE meaningful)
  mse+corr     mse − Pearson(pred,y)   [current winner]    (calibrated)
  rank         RankNet: BCE over all ordered pairs         (ordering only → RMSE meaningless)
  mse+rank     mse + λ·rank                                (calibrated + ordering)
  softsp       1 − Pearson(softrank(pred), softrank(y))    (differentiable Spearman)
  mse+softsp   mse + λ·softsp

Run on a FREE gpu (per_group trains on 0-3):  CUDA_VISIBLE_DEVICES=4 python test/rank_probe.py
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from ablation_probe import build, load_pK, split_map, MLP2, HP, DEVICE, P

torch.set_num_threads(2)
CHAMP = P + "260705_ar_cvit_100m_v2_mask075.pt"
LAM = 1.0                # mse+rank / mse+softsp mixing weight
TAU = 0.1                # soft-rank temperature
HEADS = ["mse", "mse+corr", "rank", "mse+rank", "softsp", "mse+softsp"]


def pearson_t(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-8)


def rank_loss(pred, y):
    dp = pred.unsqueeze(1) - pred.unsqueeze(0)          # (n,n)
    dy = y.unsqueeze(1) - y.unsqueeze(0)
    tgt = (dy > 0).float()
    mask = dy.abs() > 1e-6
    return F.binary_cross_entropy_with_logits(dp[mask], tgt[mask])


def soft_rank(s, tau=TAU):
    d = (s.unsqueeze(0) - s.unsqueeze(1)) / tau         # d[j,i] = (s_j - s_i)/tau
    return torch.sigmoid(d).sum(0)                      # approx rank of each element


def soft_spearman_loss(pred, y):
    rp = soft_rank(pred)
    ry = soft_rank(y).detach()
    return 1.0 - pearson_t(rp, ry)


def loss_fn(head, pred, y):
    if head == "mse":         return F.mse_loss(pred, y)
    if head == "mse+corr":    return F.mse_loss(pred, y) - pearson_t(pred, y)
    if head == "rank":        return rank_loss(pred, y)
    if head == "mse+rank":    return F.mse_loss(pred, y) + LAM * rank_loss(pred, y)
    if head == "softsp":      return soft_spearman_loss(pred, y)
    if head == "mse+softsp":  return F.mse_loss(pred, y) + LAM * soft_spearman_loss(pred, y)
    raise ValueError(head)


def to_t(a):
    return torch.tensor(a, dtype=torch.float32, device=DEVICE)


def train_one(data, head, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = to_t(data["train"]["X"]), to_t(data["train"]["y"])
    Xva, yva = to_t(data["val"]["X"]),   data["val"]["y"]
    Xte, yte = to_t(data["test"]["X"]),  data["test"]["y"]
    n, bs = Xtr.shape[0], HP["bs"]                       # MINIBATCH — matches ablation_probe
    m = MLP2(Xtr.shape[1], HP["hidden"], HP["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    best_vs, best_state, since = -1e9, None, 0
    for ep in range(HP["max_epochs"]):
        m.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if idx.numel() < 4:      # ranking losses need ≥2 comparable pairs
                continue
            opt.zero_grad()
            L = loss_fn(head, m(Xtr[idx]), ytr[idx])
            L.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = m(Xva).cpu().numpy()
        vs = spearmanr(pv, yva).statistic
        if vs > best_vs:
            best_vs, since = vs, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            since += 1
            if since >= HP["patience"]:
                break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        pt = m(Xte).cpu().numpy()
    return dict(val_rho=float(best_vs), test_r=float(pearsonr(pt, yte).statistic),
                test_rho=float(spearmanr(pt, yte).statistic),
                test_rmse=float(np.sqrt(((pt - yte) ** 2).mean())))


def main():
    pK, sm = load_pK(), split_map()
    data, nte = build([CHAMP], pK, sm)
    print(f"champion features · n_test={nte} · λ={LAM} τ={TAU}\n")
    rows = []
    for head in HEADS:
        rs = [train_one(data, head, s) for s in range(5)]
        agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        agg["test_rho_std"] = float(np.std([r["test_rho"] for r in rs]))
        agg["head"] = head
        rows.append(agg)
        cal = "" if head.startswith("mse") else "  (RMSE meaningless — ordering only)"
        print(f"  {head:<12} test_ρ={agg['test_rho']:.4f}±{agg['test_rho_std']:.4f}  "
              f"val_ρ={agg['val_rho']:.4f}  test_r={agg['test_r']:.4f}  rmse={agg['test_rmse']:.4f}{cal}")
    best = max(rows, key=lambda r: r["test_rho"])
    print(f"\n  BEST test_ρ: {best['head']} = {best['test_rho']:.4f} "
          f"(vs mse+corr {[r for r in rows if r['head']=='mse+corr'][0]['test_rho']:.4f})")
    json.dump(rows, open("../notebook/html/rank_probe.json", "w"), indent=1)
    print("  wrote notebook/html/rank_probe.json")


if __name__ == "__main__":
    main()
