"""Predicted-vs-actual H-bond-count scatter, one panel per Table 2 experiment.

The probe (dataset/01c_pdbbind_probe.py) only persists summary metrics, so we
regenerate per-complex predictions here: reuse the probe module's feature caches,
target map, splits (build_dataset) and MLP2 head, retrain each condition (seed 0,
identical hyperparameters to `probe`), and capture the test-set predictions.

Encoders are the exact Table 2 rows (verified against probe_results_e99_v5_hbonds*.csv):
  density_gradmag                       -> Density + gradmag only (ViT)
  atomblob                              -> Coordinates only (ViT)
  atomblob_density_gradmag (v5)         -> Coords + density + gradmag (ViT)
  atomblob_density_gradmag (v5, rope3d) -> Coords + density + gradmag (RoPE-3D)
All four share one 835-complex test set (4-way feature intersection ∩ H-bond label).

Out: notebook/html/260611/hbond_pred_scatter.{png,b64.txt}  (per-panel ρ/r are
seed-0, so within ~σ of Table 2's 3-seed means).
"""
import base64
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROBE_PY = ROOT / "voxbind/dataset/01c_pdbbind_probe.py"
FEAT_DIR = ROOT / "voxbind/dataset/data/pdbbind/features"
OUT_PNG = ROOT / "notebook/html/260611/hbond_pred_scatter.png"
OUT_B64 = ROOT / "notebook/html/260611/hbond_pred_scatter.b64.txt"
DEVICE = "cpu"

# ── import the probe module (reuse its loaders / splits / head verbatim) ─────
spec = importlib.util.spec_from_file_location("probe", PROBE_PY)
probe = importlib.util.module_from_spec(spec)
sys.modules["probe"] = probe
spec.loader.exec_module(probe)

# label, feature-cache filename   (Table 2 row order)
PANELS = [
    ("Density + gradmag only (ViT)",        "density_gradmag_e99_v5.pt"),
    ("Coordinates only (ViT)",              "atomblob_e99_v5.pt"),
    ("Coords + density + gradmag (ViT)",    "atomblob_density_gradmag_e99_v5.pt"),
    ("Coords + density + gradmag (RoPE-3D)", "atomblob_density_gradmag_e99_v5_rope3d.pt"),
]

lp_df = probe.load_lp_index(probe.LP_CSV)
target_map = probe.load_target_map("hbonds")

feats = {}
for _, fn in PANELS:
    feats[fn] = torch.load(FEAT_DIR / fn, weights_only=False)["features"]
shared = set.intersection(*(set(f.keys()) for f in feats.values()))
print(f"4-way shared pids: {len(shared):,}")


def train_capture(data, seed=0, max_epochs=200, patience=30, batch_size=64,
                  lr=1e-3, weight_decay=1e-4, hidden=128, dropout=0.1):
    """Mirror of probe.train_one, but also returns (y_true, y_pred) on test."""
    torch.manual_seed(seed); np.random.seed(seed)
    t = lambda s: torch.from_numpy(data[s]["X"]).to(DEVICE)
    y = lambda s: torch.from_numpy(data[s]["y"]).to(DEVICE)
    Xtr, ytr, Xva, yva, Xte, yte = t("train"), y("train"), t("val"), y("val"), t("test"), y("test")
    model = probe.MLP2(Xtr.shape[1], hidden=hidden, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    n = Xtr.shape[0]; best_val = -np.inf; best_state = None; since = 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad(); loss = loss_fn(model(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vs = spearmanr(model(Xva).cpu().numpy(), yva.cpu().numpy()).statistic
        if vs > best_val:
            best_val = vs; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; since = 0
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred_te = model(Xte).cpu().numpy()
    return yte.cpu().numpy(), pred_te


results = []
for label, fn in PANELS:
    f = {p: v for p, v in feats[fn].items() if p in shared}
    data = probe.build_dataset(f, lp_df, drop_covalent=True, cl1_only=False, target_map=target_map)
    y_true, y_pred = train_capture(data)
    rho = spearmanr(y_pred, y_true).statistic
    r = pearsonr(y_pred, y_true).statistic
    rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
    results.append((label, y_true, y_pred, rho, r, rmse))
    print(f"{label:38s} n={len(y_true)}  rho={rho:.3f}  r={r:.3f}  rmse={rmse:.3f}")

# ── plot 2×2 predicted-vs-actual ─────────────────────────────────────────────
INK, INK_SOFT, PT = "#1c2433", "#5b6678", "#2e7d5b"
allv = np.concatenate([np.concatenate([yt, yp]) for _, yt, yp, *_ in results])
lo, hi = float(np.floor(allv.min())), float(np.ceil(allv.max()))

rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.4), dpi=200)
fig.patch.set_facecolor("white")
for ax, (label, y_true, y_pred, rho, r, rmse) in zip(axes.ravel(), results):
    ax.set_facecolor("white")
    jit = y_true + rng.uniform(-0.28, 0.28, size=y_true.shape)   # ints overlap → jitter x
    ax.scatter(jit, y_pred, s=13, c=PT, alpha=0.28, edgecolors="none", zorder=3)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.1, color=INK_SOFT, alpha=0.8, zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.text(0.04, 0.96, f"$\\rho$ = {rho:.3f}\n$r$ = {r:.3f}\nRMSE = {rmse:.2f}\nn = {len(y_true)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.2, color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d7dee5", lw=0.8, alpha=0.9))
    ax.set_title(label, fontsize=10.5, color=INK, pad=6)
    ax.set_xlabel("Actual H-bond count", fontsize=9.5, color=INK_SOFT)
    ax.set_ylabel("Predicted H-bond count", fontsize=9.5, color=INK_SOFT)
    ax.tick_params(labelsize=8.5, colors=INK_SOFT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9d2dc")
    ax.grid(True, color="#eef2f5", lw=0.7, zorder=0); ax.set_axisbelow(True)

fig.suptitle("Table 2 · Predicted vs. actual H-bond count (test set, seed 0)",
             fontsize=13, color=INK, y=0.995)
fig.text(0.5, 0.965, "Dashed line = ideal (y = x); x jittered ±0.28 since counts are integers.",
         ha="center", va="center", fontsize=8.8, color=INK_SOFT)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT_PNG, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.12)
print("saved", OUT_PNG)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.12)
OUT_B64.write_text(base64.b64encode(buf.getvalue()).decode("ascii"))
print("done")
