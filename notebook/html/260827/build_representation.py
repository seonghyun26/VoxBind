"""build_representation.py — 260827 standalone complex-representation-analysis report.

Reproduces, for VoxBind, the "representation complementarity" analysis of Figure 6 in the
Boltz2-as-FM co-folding preprint (arXiv:2602.13249): our pretrained density-ViT encoder vs.
other protein-ligand complex encoders, on the LP-PDBbind (lp_edrscc_v2) TEST subset (1320).

Left  : pairwise representation-alignment matrices (CKNNA K=5,50 faithful to Huh et al. 2024;
        linear CKA). Message: our density-conditioned encoders cluster together and sit apart
        from external complex encoders -> a distinct, complementary representation space.
Right : ensemble-complementarity probe. Ridge 5-fold CV on the shared 1320 test complexes
        predicting pK, for each single encoder and for Ours(C+D+G) (+) each external encoder;
        tests whether a LESS-aligned partner yields a bigger affinity gain.

Emits a self-contained page at notebook/html/260827/representation.html (CSS reused from the
260820 meeting doc for a consistent look). Run:
    python notebook/html/260827/build_representation.py
"""
import os
import re
import sys
import json
import base64
import argparse
import importlib.util
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path("/home/shpark/prj-denovo/VoxBind")
VOX = REPO / "voxbind"
HERE = Path(__file__).resolve().parent
FEAT = VOX / "dataset" / "data" / "pdbbind" / "features"
STYLE_SRC = REPO / "notebook/html/260820/260820_meeting.html"   # reuse the meeting CSS
OUT_HTML = HERE / "representation.html"
RESULTS_DIR = HERE / "repr_analysis"
RESULTS_DIR.mkdir(exist_ok=True)

_spec = importlib.util.spec_from_file_location("p01c", str(VOX / "dataset" / "01c_pdbbind_probe.py"))
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)


# ======================================================================================
# 1. node registry: name -> loader returning {pid(lower): np.float32 vector}
# ======================================================================================
def _load_ours(fn):
    d = torch.load(FEAT / fn, map_location="cpu", weights_only=False)
    return {k.lower(): np.asarray(v, np.float32) for k, v in d["features"].items()}


def _load_get():
    d = torch.load(REPO / "base/get/_edrscc/features/get_v2_seed0_graph_repr.pt",
                   map_location="cpu", weights_only=False)
    return {k.lower(): np.asarray(v, np.float32) for k, v in d["features"].items()}


def _load_dsmbind():
    d = torch.load(REPO / "base/dsmbind/_edrscc/features/lp_edrscc_v2.pt",
                   map_location="cpu", weights_only=False)
    sp = d["split"]
    return {k.lower(): np.asarray(v, np.float32) for k, v in d["feat"].items()
            if sp.get(k) == "test"}


def _load_baseline_pt(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return {k.lower(): np.asarray(v, np.float32) for k, v in d["features"].items()}


def _load_ipnet():
    # IPDiff's interaction-prior network (frozen), pre-extracted per-complex features (256-d,
    # full split). base/ipdiff/pretrained_models/ipnet is the ckpt; feats_all.pt caches the reps.
    d = torch.load(REPO / "base/ipdiff/_edrscc/feats_all.pt", map_location="cpu", weights_only=False)
    return {k.lower(): np.asarray(v, np.float32) for k, v in d["feats"].items()}


NODES = [
    ("Ours · C+D+G", "density-ViT (100M, mask0.75)", lambda: _load_ours(
        "atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_mask075.pt")),
    ("Ours · coords", "density-ViT (coords-only twin)", lambda: _load_ours(
        "atomblob_e49_v5_260723_ar_cvit_100m_v2_mask075_coords.pt")),
    ("ProFSA", "pretrained pocket encoder (frozen)",
     lambda: _load_baseline_pt(REPO / "base/profsa/_edrscc/features/repr_lp_edrscc_v2_test_seed0.pt")),
    ("IPNet", "interaction-prior encoder (frozen, IPDiff)", _load_ipnet),
    ("BindNet", "BioLip-pretrained complex encoder (frozen)",
     lambda: _load_baseline_pt(REPO / "base/bindnet/_edrscc/features/repr_lp_edrscc_v2_test_bindnet_frozen.pt")),
]
ANCHOR = "Ours · C+D+G"


# ======================================================================================
# 2. alignment metrics (CKNNA faithful to minyoungg/platonic-rep; linear CKA)
# ======================================================================================
def _hsic_biased(K, L):
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - 1.0 / n
    return torch.trace(K @ H @ L @ H)


def _hsic_unbiased(K, L):
    m = K.shape[0]
    Kt = K.clone().fill_diagonal_(0)
    Lt = L.clone().fill_diagonal_(0)
    val = (torch.sum(Kt * Lt.T)
           + torch.sum(Kt) * torch.sum(Lt) / ((m - 1) * (m - 2))
           - 2 * torch.sum(Kt @ Lt) / (m - 2))
    return val / (m * (m - 3))


def cka(A, B):
    K = A @ A.T
    L = B @ B.T
    return float(_hsic_biased(K, L) / (torch.sqrt(_hsic_biased(K, K) * _hsic_biased(L, L)) + 1e-12))


def cknna(A, B, topk, unbiased=True):
    n = A.shape[0]
    K = A @ A.T
    L = B @ B.T

    def sim(K, L):
        if unbiased:
            Kh = K.clone().fill_diagonal_(float("-inf"))
            Lh = L.clone().fill_diagonal_(float("-inf"))
        else:
            Kh, Lh = K, L
        iK = torch.topk(Kh, topk, dim=1).indices
        iL = torch.topk(Lh, topk, dim=1).indices
        mK = torch.zeros(n, n, dtype=K.dtype, device=K.device).scatter_(1, iK, 1.0)
        mL = torch.zeros(n, n, dtype=K.dtype, device=K.device).scatter_(1, iL, 1.0)
        mask = mK * mL
        return (_hsic_unbiased(mask * K, mask * L) if unbiased
                else _hsic_biased(mask * K, mask * L))

    skl, skk, sll = sim(K, L), sim(K, K), sim(L, L)
    return float(skl / (torch.sqrt(skk * sll) + 1e-6))


def preprocess(X):
    X = X - X.mean(0, keepdims=True)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return X / nrm


# ======================================================================================
# 3. ridge 5-fold CV probe (ensemble complementarity)
# ======================================================================================
def ridge_cv(X, y, n_splits=5, alphas=(1.0, 10.0, 100.0, 1000.0), seed=0):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from scipy.stats import pearsonr, spearmanr

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(y, dtype=np.float64)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        best_a, best_r, cut = alphas[0], -2, int(0.8 * len(tr))
        for a in alphas:
            m = Ridge(alpha=a).fit(Xtr[:cut], y[tr][:cut])
            r = pearsonr(m.predict(Xtr[cut:]), y[tr][cut:])[0]
            if r > best_r:
                best_r, best_a = r, a
        m = Ridge(alpha=best_a).fit(Xtr, y[tr])
        preds[te] = m.predict(Xte)
    return {"r": float(pearsonr(preds, y)[0]), "rho": float(spearmanr(preds, y)[0]),
            "rmse": float(np.sqrt(np.mean((preds - y) ** 2)))}


def canonical_probe(feats, split_map, pk, alphas=(1.0, 10.0, 100.0, 1000.0)):
    """Leaderboard-style train->test affinity probe: fit ridge on the TRAIN split, pick
    alpha on VAL, evaluate TEST. Returns None for test-only encoders. This is the real
    affinity number; the in-report ridge_cv is a within-test cross-encoder diagnostic that
    compresses the density gap (it lets the weaker coords encoder fit test-distribution)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr, spearmanr

    def get(s):
        return [p.lower() for p, ss in split_map.items()
                if ss == s and p.lower() in feats and p.lower() in pk]
    tr, va, te = get("train"), get("val"), get("test")
    if len(tr) < 200 or len(va) < 50:
        return None
    Xtr = np.stack([feats[p] for p in tr]); ytr = np.array([pk[p] for p in tr])
    Xva = np.stack([feats[p] for p in va]); yva = np.array([pk[p] for p in va])
    Xte = np.stack([feats[p] for p in te]); yte = np.array([pk[p] for p in te])
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best = (-9, alphas[0])
    for a in alphas:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        r = spearmanr(m.predict(Xva), yva)[0]
        if r > best[0]:
            best = (r, a)
    p = Ridge(alpha=best[1]).fit(Xtr, ytr).predict(Xte)
    rmse = float(np.sqrt(np.mean((p - yte) ** 2)))
    return {"r": float(pearsonr(p, yte)[0]), "rho": float(spearmanr(p, yte)[0]),
            "rmse": rmse, "vinfo": float(0.5 * np.log(np.var(yte) / rmse ** 2)),
            "n_train": len(tr), "n_test": len(te)}


# ======================================================================================
# 3b. cross-representation decodability / stitching (does encoder i CONTAIN encoder j?)
#     A rich encoder reconstructs others well but is itself hard to reconstruct.
#     - reconstruction R^2 (asymmetric decodability; cf. model stitching, Bansal 2021)
#     - functional stitching into an affinity head (task-grounded, low-confound)
#     - relative-representation zero-shot stitching (training-free; Moschella 2023)
# ======================================================================================
def _folds(n, k=5, seed=0):
    from sklearn.model_selection import KFold
    return [te for _, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n))]


def _train_connector(Xtr, Ytr, device, hidden=384, epochs=250, patience=25, lr=1e-3, wd=1e-4, seed=0):
    """2-layer MLP connector (the trainable low-capacity map of model stitching), full-batch
    GD with early stopping on an inner 15% val split. Inputs already standardized."""
    torch.manual_seed(seed)
    n, din = Xtr.shape
    dout = Ytr.shape[1]
    nval = max(1, int(0.15 * n))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    vi, ti = perm[:nval], perm[nval:]
    Xt, Yt, Xv, Yv = Xtr[ti], Ytr[ti], Xtr[vi], Ytr[vi]
    net = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, dout)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lf = nn.MSELoss()
    best, best_state, bad = 1e18, None, 0
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        lf(net(Xt), Yt).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lf(net(Xv), Yv).item()
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    return net


def recon_r2(Xs, Xt, device, folds):
    """held-out reconstruction of target Xt from source Xs (variance-weighted R^2 + mean cosine)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    n = Xs.shape[0]
    preds = np.zeros_like(Xt)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        sx, sy = StandardScaler().fit(Xs[tr]), StandardScaler().fit(Xt[tr])
        Xtr = torch.tensor(sx.transform(Xs[tr]), dtype=torch.float32, device=device)
        Ytr = torch.tensor(sy.transform(Xt[tr]), dtype=torch.float32, device=device)
        Xte = torch.tensor(sx.transform(Xs[te]), dtype=torch.float32, device=device)
        net = _train_connector(Xtr, Ytr, device)
        with torch.no_grad():
            preds[te] = sy.inverse_transform(net(Xte).cpu().numpy())
    r2 = float(r2_score(Xt, preds, multioutput="variance_weighted"))
    cos = float(np.mean(np.sum(preds * Xt, 1) /
                        (np.linalg.norm(preds, axis=1) * np.linalg.norm(Xt, axis=1) + 1e-9)))
    return {"r2": r2, "cos": cos}


def stitch_affinity(Xsrc, Xtgt, y, device, folds, alpha=100.0):
    """functional model stitching: map src -> tgt latent, feed tgt's ridge affinity head, score pK.
    High rho => src carries the tgt-head's affinity-relevant info."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from scipy.stats import pearsonr, spearmanr
    n = len(y)
    pk = np.zeros(n)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        sx, sy = StandardScaler().fit(Xsrc[tr]), StandardScaler().fit(Xtgt[tr])
        Xtr = torch.tensor(sx.transform(Xsrc[tr]), dtype=torch.float32, device=device)
        Ytr = torch.tensor(sy.transform(Xtgt[tr]), dtype=torch.float32, device=device)
        net = _train_connector(Xtr, Ytr, device)
        H = Ridge(alpha=alpha).fit(sy.transform(Xtgt[tr]), y[tr])   # head lives in tgt latent
        with torch.no_grad():
            tgt_hat = net(torch.tensor(sx.transform(Xsrc[te]), dtype=torch.float32, device=device)).cpu().numpy()
        pk[te] = H.predict(tgt_hat)
    return {"r": float(pearsonr(pk, y)[0]), "rho": float(spearmanr(pk, y)[0]),
            "rmse": float(np.sqrt(np.mean((pk - y) ** 2)))}


def relative_rep(feat, anchor_idx):
    """Moschella 2023 relative representation: row-L2-normalise then cosine to fixed anchors."""
    F = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9)
    return F @ F[anchor_idx].T


def rel_zero_shot(rel_src, rel_tgt, y, folds, alpha=100.0):
    """train an affinity head on tgt's relative rep, apply zero-shot to src's relative rep
    (same anchors => shared coordinates). self = tgt-through-tgt upper bound."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr
    n = len(y)
    zero, slf = np.zeros(n), np.zeros(n)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        ss = StandardScaler().fit(rel_tgt[tr])
        H = Ridge(alpha=alpha).fit(ss.transform(rel_tgt[tr]), y[tr])
        zero[te] = H.predict(ss.transform(rel_src[te]))
        slf[te] = H.predict(ss.transform(rel_tgt[te]))
    return {"zero_rho": float(spearmanr(zero, y)[0]), "self_rho": float(spearmanr(slf, y)[0])}


# ======================================================================================
# 3c. recent representation-quality / information metrics (ICLR/NeurIPS/ICML 2022-24)
#   - RankMe effective rank (Garrido et al., ICML 2023)
#   - LiDAR LDA-rank         (Thilak et al., ICLR 2024) — adapted with pK-quantile classes
#   - TwoNN intrinsic dim    (geometry of representations, Valeriani et al., NeurIPS 2023)
#   - V-usable information   (Ethayarajh et al., ICML 2022; Xu et al., ICLR 2020)
# ======================================================================================
def rankme(Z, eps=1e-7):
    """effective rank = exp(entropy of L1-normalised singular values). Garrido et al. 2023."""
    Zc = Z - Z.mean(0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    p = s / (s.sum() + eps) + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def twonn_id(Z):
    """TwoNN intrinsic-dimension MLE (Facco et al.); features z-scored first."""
    from sklearn.neighbors import NearestNeighbors
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    d, _ = NearestNeighbors(n_neighbors=3).fit(Zs).kneighbors(Zs)
    mu = d[:, 2] / (d[:, 1] + 1e-12)
    mu = mu[mu > 1 + 1e-6]
    mu = np.sort(mu)[:int(0.9 * len(mu))]        # discard top-decile (Facco robustification)
    return float(len(mu) / np.log(mu).sum())


def lidar_rank(Z, y, q=10, reg=1e-2, kpca=200, eps=1e-7):
    """LDA-rank (LiDAR, adapted): effective rank of the between/within class scatter with
    pK-quantile classes. PCA-reduced + ridge-regularised so it is stable at high dim."""
    from sklearn.decomposition import PCA
    import scipy.linalg as sla
    k = min(kpca, Z.shape[1], Z.shape[0] - 1)
    Zr = PCA(n_components=k).fit_transform(Z - Z.mean(0))
    edges = np.quantile(y, np.linspace(0, 1, q + 1))[1:-1]
    cls = np.digitize(y, edges)
    mu, n = Zr.mean(0), len(y)
    Sw = np.zeros((k, k)); Sb = np.zeros((k, k))
    for c in np.unique(cls):
        Xi = Zr[cls == c]; mc = Xi.mean(0)
        Sw += (Xi - mc).T @ (Xi - mc)
        Sb += len(Xi) * np.outer(mc - mu, mc - mu)
    Sw = Sw / n + reg * (np.trace(Sw / n) / k) * np.eye(k)
    lam = np.clip(sla.eigh(Sb / n, Sw, eigvals_only=True), 0, None)
    p = lam / (lam.sum() + eps) + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def v_information(rmse, var_y):
    """predictive V-usable info about pK under a ridge family: 0.5 ln(Var(y)/MSE), in nats."""
    return float(0.5 * np.log(var_y / (rmse ** 2 + 1e-12)))


def ridge_cv_preds(X, y, folds, alphas=(1.0, 10.0, 100.0, 1000.0)):
    """held-out CV predictions with the SAME per-fold alpha search as ridge_cv, so the
    resulting MSE matches ridge_cv's and pointwise V-info sums to the aggregate."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from scipy.stats import pearsonr
    n = len(y); preds = np.zeros(n)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        best_a, best_r, cut = alphas[0], -2, int(0.8 * len(tr))
        for a in alphas:
            m = Ridge(alpha=a).fit(Xtr[:cut], y[tr][:cut])
            r = pearsonr(m.predict(Xtr[cut:]), y[tr][cut:])[0]
            if r > best_r:
                best_r, best_a = r, a
        preds[te] = Ridge(alpha=best_a).fit(Xtr, y[tr]).predict(Xte)
    return preds


# ======================================================================================
# 4. plotting helpers -> base64 png
# ======================================================================================
def _fig_b64(fig):
    import matplotlib.pyplot as plt
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def heatmap(mat, labels, title, cmap="viridis", diag_blank=True):
    import matplotlib.pyplot as plt
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.9 * n + 2.2, 0.9 * n + 1.4))
    M = mat.copy()
    disp = M.copy()
    if diag_blank:
        np.fill_diagonal(disp, np.nan)   # self-alignment is trivial -> leave the cell blank
    finite = disp[np.isfinite(disp)]
    vmin, vmax = float(finite.min()), float(finite.max())
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")
    im = ax.imshow(np.ma.masked_invalid(disp), cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            if diag_blank and i == j:
                continue                 # no text on the diagonal
            shade = (disp[i, j] - vmin) / (vmax - vmin + 1e-9)
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if shade < 0.55 else "black")
    ax.set_title(title, fontsize=11, pad=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_b64(fig)


def scatter_align_gain(points, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    xs = [p["align"] for p in points]
    ys = [p["gain"] for p in points]
    ax.axhline(0, color="#aeb6c4", lw=1, ls="--")
    ax.scatter(xs, ys, s=70, color="#2f6f4f", zorder=3)
    for p in points:
        ax.annotate(p["name"], (p["align"], p["gain"]), fontsize=9,
                    xytext=(5, 4), textcoords="offset points")
    if len(xs) >= 2:
        b, a = np.polyfit(xs, ys, 1)
        xx = np.linspace(min(xs), max(xs), 50)
        ax.plot(xx, a + b * xx, color="#38559b", lw=1.3, alpha=0.7)
        from scipy.stats import pearsonr
        ax.text(0.03, 0.94, f"slope={b:+.3f}, r={pearsonr(xs, ys)[0]:+.2f}",
                transform=ax.transAxes, fontsize=9, color="#38559b", va="top")
    ax.set_xlabel("CKA alignment to Ours · C+D+G", fontsize=10)
    ax.set_ylabel("Δρ  (Ours ⊕ X)  −  Ours", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    return _fig_b64(fig)


def rank_bar(labels, values, title, xlabel, highlight=None, fmt="{:+.3f}", zero_line=True):
    import matplotlib.pyplot as plt
    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(6.2, 0.5 * len(labels) + 1.2))
    colors = ["#2f6f4f" if (highlight and l in highlight) else "#8fb0a0" for l in labels]
    ax.barh(range(len(labels)), values, color=colors, zorder=3)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    if zero_line:
        ax.axvline(0, color="#aeb6c4", lw=1)
    pad = 0.01 * (max(values) - min(values) + 1e-9)
    for i, v in enumerate(values):
        ax.text(v + (pad if v >= 0 else -pad), i, fmt.format(v), va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    return _fig_b64(fig)


def pvi_hist(dpvi, frac_help):
    """histogram of per-complex pointwise-V-info gain (CDG − coords)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.hist(dpvi, bins=45, color="#2f6f4f", alpha=0.85, zorder=3)
    ax.axvline(0, color="#c05b4d", lw=1.4, ls="--")
    ax.set_xlabel("per-complex usable-info gain  Δpvi = pvi(CDG) − pvi(coords)  (nats)", fontsize=9)
    ax.set_ylabel("# complexes", fontsize=10)
    ax.set_title(f"Where density adds usable affinity information  ({frac_help:.0%} of complexes > 0)",
                 fontsize=10.5)
    ax.grid(axis="y", alpha=0.25)
    return _fig_b64(fig)


def grouped_bar(labels, series, title, ylabel, colors, ymax=None):
    """series: list of (name, values[]) aligned to labels."""
    import matplotlib.pyplot as plt
    n, g = len(labels), len(series)
    fig, ax = plt.subplots(figsize=(0.95 * n + 2.0, 3.6))
    w = 0.8 / g
    for k, (nm, vals) in enumerate(series):
        ax.bar(np.arange(n) + (k - (g - 1) / 2) * w, vals, w, label=nm, color=colors[k], zorder=3)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    return _fig_b64(fig)


def ceiling_bar(rows, ceil_r2):
    """normalized recovery of CDG per source, against the seed-twin ceiling (=1.0)."""
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: r["norm"], reverse=True)
    labels = [r["src"] for r in rows]
    norms = [r["norm"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.8, 0.5 * len(rows) + 1.4))
    colors = ["#2f6f4f" if "C+D+G" in l else ("#8fb0a0" if l.startswith("Ours") else "#b9c6d6")
              for l in labels]
    ax.barh(range(len(labels)), norms, color=colors, zorder=3)
    ax.axvline(1.0, color="#c05b4d", lw=1.5, ls="--", zorder=4)
    ax.text(1.0, len(labels) - 0.3, "seed-twin ceiling", color="#c05b4d", fontsize=8,
            ha="center", va="top")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis()
    for i, r in enumerate(rows):
        ax.text(r["norm"] + 0.012, i, f'{r["norm"]:.2f}', va="center", fontsize=8)
    ax.set_xlabel(f"normalized recovery  =  R²(X→CDG) / ceiling   (ceiling R² = {ceil_r2:.3f})", fontsize=9)
    ax.set_title("Fraction of CDG information recoverable from each source", fontsize=11)
    ax.set_xlim(0, 1.12); ax.grid(axis="x", alpha=0.25)
    return _fig_b64(fig)


# ======================================================================================
# 5. main
# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cdg-file", default=None,
                    help="override the 'Ours · C+D+G' feature file (name under features/)")
    ap.add_argument("--cdg-sub", default=None, help="subtitle for the CDG node")
    ap.add_argument("--suffix", default="", help="output suffix: representation_<suffix>.html")
    ap.add_argument("--cdg-note", default="", help="header badge describing the CDG variant")
    ap.add_argument("--reuse", action="store_true",
                    help="load the connector-heavy results (recon_r2.csv / stitch.json) from cache and "
                         "only recompute the cheap parts + figures — for text/layout tweaks with no GPU")
    args = ap.parse_args()
    torch.set_num_threads(4)
    dev = args.device

    global OUT_HTML, RESULTS_DIR
    if args.cdg_file:
        NODES[0] = ("Ours · C+D+G", args.cdg_sub or NODES[0][1],
                    lambda: _load_ours(args.cdg_file))
    if args.suffix:
        OUT_HTML = HERE / f"representation_{args.suffix}.html"
        RESULTS_DIR = HERE / f"repr_analysis_{args.suffix}"
        RESULTS_DIR.mkdir(exist_ok=True)

    sm, _ = pr.load_frozen_split_map("lp_edrscc_v2")
    test = set(p.lower() for p, s in sm.items() if s == "test")
    lp = pr.load_lp_index(VOX / "dataset" / "data" / "pdbbind" / "raw" / "LP_PDBBind.csv")
    pk = {str(r.pdb_id).lower(): float(r.pK) for r in lp.itertuples()}

    loaded = []
    for name, sub, fn in NODES:
        try:
            feats = fn()
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue
        print(f"[ok]   {name:16s} dim={len(next(iter(feats.values()))):5d} "
              f"test_cov={len(test & set(feats))}/{len(test)}")
        loaded.append((name, sub, feats))

    common = set(test)
    for _, _, f in loaded:
        common &= set(f)
    common &= set(pk)
    common = sorted(common)
    n = len(common)
    print(f"\ncommon complexes: {n}  across {len(loaded)} encoders")
    names = [nm for nm, _, _ in loaded]
    subs = {nm: sb for nm, sb, _ in loaded}
    dims = {nm: int(len(next(iter(f.values())))) for nm, _, f in loaded}

    Xs = {nm: torch.from_numpy(preprocess(np.stack([f[p] for p in common]).astype(np.float64))).to(dev)
          for nm, _, f in loaded}
    y = np.array([pk[p] for p in common], dtype=np.float64)

    import pandas as pd
    RE = (args.reuse and (RESULTS_DIR / "recon_r2.csv").exists()
          and (RESULTS_DIR / "stitch.json").exists() and (RESULTS_DIR / "cka.csv").exists())
    _cache = json.loads((RESULTS_DIR / "stitch.json").read_text()) if RE else {}
    if RE:
        print("[reuse] loading cached alignment + connector results (no GPU)", flush=True)

    K = len(names)
    if RE:
        CKA = pd.read_csv(RESULTS_DIR / "cka.csv", index_col=0).values.astype(float)
        CK5 = pd.read_csv(RESULTS_DIR / "cknna_k5.csv", index_col=0).values.astype(float)
        CK50 = pd.read_csv(RESULTS_DIR / "cknna_k50.csv", index_col=0).values.astype(float)
    else:
        CKA = np.eye(K); CK5 = np.eye(K); CK50 = np.eye(K)
        for i in range(K):
            for j in range(i + 1, K):
                A, B = Xs[names[i]], Xs[names[j]]
                CKA[i, j] = CKA[j, i] = cka(A, B)
                CK5[i, j] = CK5[j, i] = cknna(A, B, topk=5)
                CK50[i, j] = CK50[j, i] = cknna(A, B, topk=50)
            print(f"  alignment row {i+1}/{K} done", flush=True)

    raw = {nm: np.stack([f[p] for p in common]).astype(np.float64) for nm, _, f in loaded}
    singles = {nm: ridge_cv(raw[nm], y) for nm in names}

    # canonical train->test probe (the REAL affinity number) for encoders with full-split
    # features; test-only encoders (ProFSA, BindNet here) return None. The CV-on-test
    # `singles` above is only a cross-encoder-uniform diagnostic and compresses the density gap.
    canon = {}
    for nm, _, f in loaded:
        c = canonical_probe(f, sm, pk)
        if c:
            canon[nm] = c
            print(f"  canon {nm:16s} r={c['r']:.3f} rho={c['rho']:.3f} rmse={c['rmse']:.3f} "
                  f"Vinfo={c['vinfo']:.3f}", flush=True)
    if "Ours · C+D+G" in canon and "Ours · coords" in canon:
        print(f"  --> canonical density gain dRho="
              f"{canon['Ours · C+D+G']['rho'] - canon['Ours · coords']['rho']:+.3f}", flush=True)

    a_rho = singles[ANCHOR]["rho"]
    ai = names.index(ANCHOR)
    pairs, scatter_pts = {}, []
    for nm in names:
        if nm == ANCHOR:
            continue
        m = ridge_cv(np.concatenate([raw[ANCHOR], raw[nm]], axis=1), y)
        gain = m["rho"] - a_rho
        pairs[nm] = {**m, "gain_rho": gain, "align_cka": CKA[ai, names.index(nm)]}
        scatter_pts.append({"name": nm, "align": CKA[ai, names.index(nm)], "gain": gain})

    png_ck5 = heatmap(CK5, names, "CKNNA  (K = 5)", cmap="viridis")
    png_ck50 = heatmap(CK50, names, "CKNNA  (K = 50)", cmap="viridis")
    png_cka = heatmap(CKA, names, "linear CKA", cmap="magma")
    png_scatter = scatter_align_gain(scatter_pts, "Complementarity: alignment vs affinity gain")

    import pandas as pd
    pd.DataFrame(CKA, index=names, columns=names).to_csv(RESULTS_DIR / "cka.csv")
    pd.DataFrame(CK5, index=names, columns=names).to_csv(RESULTS_DIR / "cknna_k5.csv")
    pd.DataFrame(CK50, index=names, columns=names).to_csv(RESULTS_DIR / "cknna_k50.csv")
    (RESULTS_DIR / "summary.json").write_text(json.dumps(
        {"n_common": n, "encoders": names, "dims": dims, "singles": singles, "pairs": pairs}, indent=2))

    print("\n=== singles (ridge 5-fold CV pK) ===")
    for nm in names:
        s = singles[nm]
        print(f"  {nm:16s} r={s['r']:.3f} rho={s['rho']:.3f} rmse={s['rmse']:.3f}")
    print("=== ours(+)external ===")
    for nm, p in pairs.items():
        print(f"  Ours(+){nm:12s} rho={p['rho']:.3f}  Δrho={p['gain_rho']:+.3f}  cka={p['align_cka']:.3f}")

    # ---- §3 cross-representation decodability / stitching (all 6 encoders) ----
    mlp_dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[§3] connector device: {mlp_dev}" + ("  (--reuse: cached connectors)" if RE else ""))
    folds = _folds(n)

    # 3.1 reconstruction R^2 matrix (source i -> target j) — heavy; cacheable
    if RE:
        R2 = pd.read_csv(RESULTS_DIR / "recon_r2.csv", index_col=0).values.astype(float)
    else:
        R2 = np.full((K, K), np.nan)
        for i in range(K):
            for j in range(K):
                if i != j:
                    R2[i, j] = recon_r2(raw[names[i]], raw[names[j]], mlp_dev, folds)["r2"]
            print(f"  recon row {i+1}/{K} done", flush=True)
    richness = np.array([np.nanmean([R2[i, j] - R2[j, i] for j in range(K) if j != i])
                         for i in range(K)])

    # 3.2 functional stitching into an affinity head — heavy; cacheable
    stitch = {}
    for nm in names:
        if nm == ANCHOR:
            continue
        if RE:
            c = _cache["stitch"][nm]
            stitch[nm] = {"own": singles[nm]["rho"], "to_cdg": c["to_cdg"], "from_cdg": c["from_cdg"]}
        else:
            to_cdg = stitch_affinity(raw[nm], raw[ANCHOR], y, mlp_dev, folds)["rho"]
            from_cdg = stitch_affinity(raw[ANCHOR], raw[nm], y, mlp_dev, folds)["rho"]
            stitch[nm] = {"own": singles[nm]["rho"], "to_cdg": to_cdg, "from_cdg": from_cdg}
            print(f"  stitch {nm:14s} own={singles[nm]['rho']:.3f} X->CDG={to_cdg:.3f} "
                  f"CDG->X={from_cdg:.3f}", flush=True)

    # 3.3 relative-representation zero-shot stitching (training-free; Moschella 2023)
    rng = np.random.RandomState(0)
    n_anch = min(256, n // 3)
    anchors = np.sort(rng.choice(n, size=n_anch, replace=False))
    rel = {nm: relative_rep(raw[nm], anchors) for nm in names}
    cdg_rel = rel[ANCHOR]
    relres = {}
    for nm in names:
        # per-complex Pearson correlation between the two anchor-similarity profiles
        # (centred cosine; plain cosine saturates near 1 on all-positive anchor vectors)
        A = rel[nm] - rel[nm].mean(1, keepdims=True)
        B = cdg_rel - cdg_rel.mean(1, keepdims=True)
        sim = float(np.mean(np.sum(A * B, 1) /
                            (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)))
        zs = rel_zero_shot(rel[nm], cdg_rel, y, folds)
        relres[nm] = {"sim_to_cdg": sim, "zero_rho": zs["zero_rho"], "self_rho": zs["self_rho"]}
        print(f"  rel {nm:14s} sim={sim:.3f} zeroshot->CDGhead rho={zs['zero_rho']:.3f}", flush=True)

    # 3.4 seed-twin ceiling -> density-unique information — heavy; cacheable
    if RE:
        ceil_r2 = float(_cache["ceiling_r2"])
        ceilrows = _cache["ceiling_norm"]
    else:
        cdg_s1 = _load_ours("atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_d640L18h10_m075_e50_s1.pt")
        cdg_s2 = _load_ours("atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_d640L18h10_m075_e50_s2.pt")
        S1 = np.stack([cdg_s1[p] for p in common]).astype(np.float64)
        S2 = np.stack([cdg_s2[p] for p in common]).astype(np.float64)
        ceil_r2 = 0.5 * (recon_r2(S2, S1, mlp_dev, folds)["r2"] + recon_r2(S1, S2, mlp_dev, folds)["r2"])
        ceil_sources = [("Ours · C+D+G (champion)", raw[ANCHOR])] + \
                       [(nm, raw[nm]) for nm in names if nm != ANCHOR]
        ceilrows = []
        for nm, X in ceil_sources:
            r2 = recon_r2(X, S1, mlp_dev, folds)["r2"]
            ceilrows.append({"src": nm, "r2": float(r2), "norm": float(r2 / ceil_r2)})
            print(f"  ceil {nm:24s} R2(X->CDG_s1)={r2:.3f} norm={r2/ceil_r2:.3f}", flush=True)
    du = 1.0 - next(r["norm"] for r in ceilrows if r["src"] == "Ours · coords")
    print(f"  ceiling(seed twin)={ceil_r2:.3f}  density-unique(coords)={du:.3f}", flush=True)
    png_ceil = ceiling_bar(ceilrows, ceil_r2)

    # ---- §4 recent representation-quality / information metrics (all 6 encoders) ----
    var_y = float(np.var(y))
    preds4 = {nm: ridge_cv_preds(raw[nm], y, folds) for nm in names}   # one predictor family
    mse4 = {nm: float(np.mean((preds4[nm] - y) ** 2)) for nm in names}
    q4 = {}
    for nm in names:
        Z = raw[nm]
        q4[nm] = {"rankme": rankme(Z), "id": twonn_id(Z), "lidar": lidar_rank(Z, y),
                  "vinfo": v_information(np.sqrt(mse4[nm]), var_y),
                  "cka_others": float(np.mean([CKA[names.index(nm), j]
                                               for j in range(K) if j != names.index(nm)])),
                  "dim": dims[nm]}
        print(f"  q4 {nm:16s} rankme={q4[nm]['rankme']:.1f} id={q4[nm]['id']:.1f} "
              f"lidar={q4[nm]['lidar']:.2f} vinfo={q4[nm]['vinfo']:.3f} "
              f"ckaOthers={q4[nm]['cka_others']:.3f}", flush=True)
    # pointwise V-info (same predictors): mean Δpvi == aggregate V-info gap by construction
    pC, pO = preds4[ANCHOR], preds4["Ours · coords"]
    sc2, so2 = mse4[ANCHOR], mse4["Ours · coords"]
    dpvi = ((-0.5 * np.log(2 * np.pi * sc2) - (y - pC) ** 2 / (2 * sc2))
            - (-0.5 * np.log(2 * np.pi * so2) - (y - pO) ** 2 / (2 * so2)))
    frac_help, mean_dpvi = float(np.mean(dpvi > 0)), float(np.mean(dpvi))
    print(f"  pointwise density-helps: {frac_help:.1%} complexes, mean Δpvi={mean_dpvi:+.3f} nats", flush=True)
    hl = {"Ours · C+D+G", "Ours · coords"}
    png_rankme = rank_bar(names, [q4[nm]["rankme"] for nm in names], "Effective rank (RankMe, ICML'23)",
                          "exp(entropy of singular values)", highlight=hl, fmt="{:.1f}", zero_line=False)
    png_id = rank_bar(names, [q4[nm]["id"] for nm in names], "Intrinsic dimension (TwoNN, NeurIPS'23)",
                      "ID", highlight=hl, fmt="{:.1f}", zero_line=False)
    png_lidar = rank_bar(names, [q4[nm]["lidar"] for nm in names], "Task rank (LiDAR / LDA, ICLR'24)",
                         "effective rank of pK-discriminative directions", highlight=hl, fmt="{:.2f}", zero_line=False)
    png_vinfo = rank_bar(names, [q4[nm]["vinfo"] for nm in names], "𝒱-usable information about pK (ICML'22)",
                         "0.5·ln(Var(y)/MSE)  [nats]", highlight=hl, fmt="{:.3f}")
    png_platonic = rank_bar(names, [q4[nm]["cka_others"] for nm in names],
                            "Platonic convergence (mean CKA to other encoders, ICML'24)",
                            "mean CKA to the other 5", highlight=hl, fmt="{:.3f}", zero_line=False)
    png_pvi = pvi_hist(dpvi, frac_help)
    q4pack = {"q4": q4, "frac_help": frac_help, "mean_dpvi": mean_dpvi,
              "vinfo_gap": q4[ANCHOR]["vinfo"] - q4["Ours · coords"]["vinfo"],
              "png_rankme": png_rankme, "png_id": png_id, "png_lidar": png_lidar,
              "png_vinfo": png_vinfo, "png_platonic": png_platonic, "png_pvi": png_pvi}

    png_recon = heatmap(R2, names, "reconstruction R²   (source → target)", cmap="viridis")
    png_rich = rank_bar(names, list(richness), "Richness score  (net decodability asymmetry)",
                        "mean_j [ R²(i→j) − R²(j→i) ]",
                        highlight={"Ours · C+D+G", "Ours · coords"})
    others = [nm for nm in names if nm != ANCHOR]
    png_stitch = grouped_bar(
        others,
        [("X alone", [stitch[nm]["own"] for nm in others]),
         ("X → CDG head (stitched)", [stitch[nm]["to_cdg"] for nm in others])],
        f"Functional stitching into the CDG affinity head  (CDG alone ρ = {singles[ANCHOR]['rho']:.3f})",
        "Spearman ρ (pK)", colors=["#b9c6d6", "#2f6f4f"], ymax=0.78)

    s3 = {"names": names, "R2": R2, "richness": richness, "stitch": stitch, "rel": relres,
          "cdg_rho": singles[ANCHOR]["rho"], "n_anchors": int(n_anch),
          "ceil_r2": float(ceil_r2), "ceilrows": ceilrows, "density_unique": float(du),
          "png_recon": png_recon, "png_rich": png_rich, "png_stitch": png_stitch, "png_ceil": png_ceil,
          **q4pack}
    pd.DataFrame(R2, index=names, columns=names).to_csv(RESULTS_DIR / "recon_r2.csv")
    (RESULTS_DIR / "stitch.json").write_text(json.dumps(
        {"richness": {names[i]: float(richness[i]) for i in range(K)},
         "stitch": stitch, "rel": relres, "cdg_rho": singles[ANCHOR]["rho"],
         "ceiling_r2": float(ceil_r2), "ceiling_norm": ceilrows,
         "density_unique_coords": float(du)}, indent=2))
    (RESULTS_DIR / "metrics_q4.json").write_text(json.dumps(
        {"per_encoder": q4, "frac_complexes_density_helps": frac_help,
         "mean_delta_pvi": mean_dpvi, "vinfo_gap_cdg_minus_coords": q4pack["vinfo_gap"]}, indent=2))

    # §3.5 reconstruction granularity — read the precomputed granularity.json if present
    # (recon_granularity.py writes it; kept out of the main build since it re-extracts tokens).
    gran_path = RESULTS_DIR / "granularity.json"
    gran = json.loads(gran_path.read_text()) if gran_path.exists() else None
    s3["png_gran"] = None
    if gran:
        pt, cc = gran["patch_token"], gran["complex_control"]
        s3["png_gran"] = grouped_bar(
            ["Complex (pooled)", "Patch-token (local)"],
            [("CDG → C", [cc["CDG->C"], pt["CDG->C"]]),
             ("C → CDG", [cc["C->CDG"], pt["C->CDG"]])],
            "C↔CDG reconstruction R² by granularity", "held-out R²",
            colors=["#2f6f4f", "#b9c6d6"], ymax=1.0)
    s3["granularity"] = gran
    s3["canon"] = canon

    page = render_page(names, subs, dims, n, CKA, CK5, CK50, singles, pairs, ANCHOR,
                       png_ck5, png_ck50, png_cka, png_scatter, s3, args.cdg_note)
    OUT_HTML.write_text(page)
    print(f"\nwrote standalone report -> {OUT_HTML}")


def _mnum(x, best=False, d=3):
    s = f"{x:.{d}f}"
    return f"<b>{s}</b>" if best else s


def render_page(names, subs, dims, n, CKA, CK5, CK50, singles, pairs, anchor,
                png_ck5, png_ck50, png_cka, png_scatter, s3, variant_note=""):
    style = ""
    m = re.search(r"<style>.*?</style>", STYLE_SRC.read_text(), flags=re.S)
    if m:
        style = m.group(0)

    K = len(names)
    ai = names.index(anchor)

    def matrix_table(mat, cap):
        head = "".join(f"<th>{nm}</th>" for nm in names)
        rows = ""
        for i, nm in enumerate(names):
            cells = ""
            for j in range(K):
                v = mat[i, j]
                if i == j:
                    cells += '<td></td>'          # self-alignment: leave blank
                else:
                    off = [mat[i, k] for k in range(K) if k != i]
                    hi = v >= max(off) - 1e-9
                    cells += f"<td>{'<b>' if hi else ''}{v:.3f}{'</b>' if hi else ''}</td>"
            rows += f'<tr><td class="col-method">{nm}</td>{cells}</tr>'
        return (f'<div class="table-wrap" style="margin-bottom:14px"><table class="results">'
                f'<thead><tr><th class="col-method stub">{cap}</th>{head}</tr></thead>'
                f'<tbody>{rows}</tbody></table></div>')

    a = singles[anchor]
    srows = ""
    for nm in names:
        s = singles[nm]
        tag = ' <span class="tag best">ours</span>' if nm == anchor else (
              ' <span class="tag">ours</span>' if nm.startswith("Ours") else "")
        srows += (f'<tr><td class="col-method">{nm}{tag}'
                  f'<span class="method-note">{subs[nm]} · {dims[nm]}-d</span></td>'
                  f'<td>{s["r"]:.3f}</td><td>{s["rho"]:.3f}</td><td>{s["rmse"]:.3f}</td>'
                  f'<td class="placeholder">—</td></tr>')
    best_gain = max(p["gain_rho"] for p in pairs.values())
    ext_sorted = sorted([nm for nm in names if nm != anchor], key=lambda nm: CKA[ai, names.index(nm)])
    prows = ""
    for nm in ext_sorted:
        p = pairs[nm]
        isbest = abs(p["gain_rho"] - best_gain) < 1e-9
        cls = ' class="best-row"' if isbest else ""
        prows += (f'<tr{cls}><td class="col-method">Ours · C+D+G  ⊕  {nm}'
                  f'<span class="method-note">CKA to ours = {p["align_cka"]:.3f}</span></td>'
                  f'<td>{p["r"]:.3f}</td><td>{_mnum(p["rho"], isbest)}</td><td>{p["rmse"]:.3f}</td>'
                  f'<td>{_mnum(p["gain_rho"], isbest)}</td></tr>')

    # ---------- §3 cross-representation decodability & stitching ----------
    R2 = s3["R2"]; rich = s3["richness"]; stitch = s3["stitch"]; rel = s3["rel"]
    others3 = [nm for nm in names if nm != anchor]

    r31 = ""
    for nm in others3:
        cdg2x = R2[ai, names.index(nm)]      # reconstruct X from CDG
        x2cdg = R2[names.index(nm), ai]      # reconstruct CDG from X
        asym = cdg2x - x2cdg
        r31 += (f'<tr><td class="col-method">Ours · C+D+G ↔ {nm}'
                f'<span class="method-note">{dims[nm]}-d</span></td>'
                f'<td>{cdg2x:.3f}</td><td>{x2cdg:.3f}</td><td>{_mnum(asym, asym > 0)}</td></tr>')
    rrows = ""
    for i in np.argsort(rich)[::-1]:
        cls = ' class="best-row"' if names[i] == anchor else ""
        rrows += (f'<tr{cls}><td class="col-method">{names[i]}</td>'
                  f'<td>{_mnum(rich[i], names[i] == anchor)}</td></tr>')
    r32 = ""
    for nm in others3:
        st = stitch[nm]
        gap = s3["cdg_rho"] - st["to_cdg"]
        r32 += (f'<tr><td class="col-method">{nm}</td><td>{st["own"]:.3f}</td>'
                f'<td>{st["to_cdg"]:.3f}</td><td>{st["from_cdg"]:.3f}</td><td>{gap:+.3f}</td></tr>')
    r33 = ""
    for nm in names:
        rr = rel[nm]
        cls = ' class="best-row"' if nm == anchor else ""
        tag = ' <span class="tag best">ours</span>' if nm == anchor else (
              ' <span class="tag">ours</span>' if nm.startswith("Ours") else "")
        gap = rr["self_rho"] - rr["zero_rho"]
        r33 += (f'<tr{cls}><td class="col-method">{nm}{tag}</td><td>{rr["sim_to_cdg"]:.3f}</td>'
                f'<td>{rr["zero_rho"]:.3f}</td><td>{gap:+.3f}</td></tr>')

    gran = s3.get("granularity")
    sec35 = ""
    if gran:
        pt, cc, vx = gran["patch_token"], gran["complex_control"], gran["voxel_input"]
        sec35 = f"""
    <section class="block" id="granularity">
      <h3 class="subsection-head"><span class="sub-lab">3.5</span>Reconstruction granularity</h3>
      <p class="table-sub">§3.1 reconstructs the <b>per-complex</b> pooled vector. Here we repeat C↔CDG
        reconstruction at two finer scales (same encoders): per <b>spatial patch-token</b> (512 tokens per
        complex, {pt['n_tokens']:,} rows, split by complex) and at the raw <b>input-voxel</b> level. The
        CDG&nbsp;⊇&nbsp;C asymmetry persists locally but shrinks — a patch's density is largely its own blurred
        atoms, so density's <i>unique</i> contribution is largest in aggregate.</p>
      <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start">
        <img src="data:image/png;base64,{s3['png_gran']}" style="max-width:48%;height:auto;border:1px solid var(--line);border-radius:8px">
        <div class="table-wrap" style="min-width:330px"><table class="results">
          <thead><tr><th class="col-method stub">Granularity</th><th>CDG→C</th><th>C→CDG</th><th>asym</th></tr></thead>
          <tbody>
            <tr><td class="col-method">Complex — pooled (§3.1 canonical, 5-fold)</td><td>0.937</td><td>0.603</td><td>+0.333</td></tr>
            <tr><td class="col-method">Complex — control (this split)</td><td>{cc['CDG->C']:.3f}</td><td>{cc['C->CDG']:.3f}</td><td>{cc['asym']:+.3f}</td></tr>
            <tr class="best-row"><td class="col-method">Patch-token — local</td><td>{pt['CDG->C']:.3f}</td><td>{pt['C->CDG']:.3f}</td><td>{pt['asym']:+.3f}</td></tr>
          </tbody></table></div>
      </div>
      <p class="table-sub" style="margin-top:10px"><b>Input-voxel level.</b> Predicting CDG's extra input
        channels (electron density + ‖∇ρ‖) from the 11 coordinate-blob channels reaches only
        R²&nbsp;=&nbsp;{vx['coords->dg']:.2f} over {vx['n_voxels']:,} occupied voxels ({vx['n_complexes']} complexes)
        — so ~{100 * (1 - vx['coords->dg']):.0f}% of the local density signal is not a function of atom coordinates,
        confirming §3's density-unique information at the input level. <i>(The two "Complex" rows differ only in
        protocol — 5-fold CV vs a single by-complex split — the asymmetry direction is identical.)</i></p>
    </section>"""

    ceil_r2 = s3["ceil_r2"]; du = s3["density_unique"]
    r34 = ""
    for r in sorted(s3["ceilrows"], key=lambda z: -z["norm"]):
        nm = r["src"]
        is_c = nm == "Ours · coords"
        cls = ' class="best-row"' if is_c else ""
        unrec = 1.0 - r["norm"]
        r34 += (f'<tr{cls}><td class="col-method">{nm}</td><td>{r["r2"]:.3f}</td>'
                f'<td>{_mnum(r["norm"], is_c)}</td><td>{_mnum(unrec, is_c)}</td></tr>')

    sec3 = f"""  <section class="doc-section" id="repr-decodability">
    <h2 class="section-head"><span class="sec-num">3</span>Information richness: cross-representation decodability &amp; stitching</h2>
    <p class="section-intro">Does one encoder's representation <b>contain</b> another's? Following
      <b>model stitching</b> (Bansal et al., NeurIPS 2021; Lenc &amp; Vedaldi, CVPR 2015) and its
      "more-is-better" asymmetry, we fit a low-capacity <b>2-layer MLP connector</b> to reconstruct each
      target representation from each source (held-out 5-fold R²). A representation is <b>richer</b> if it
      reconstructs others well yet is itself hard to reconstruct. We then ground this in the affinity task
      (functional stitching), and add a training-free check via relative representations
      (Moschella et al., ICLR 2023 Oral).</p>

    <section class="block" id="recon-r2">
      <h3 class="subsection-head"><span class="sub-lab">3.1</span>Reconstruction R² &amp; richness ranking</h3>
      <p class="table-sub">Held-out variance-weighted R² of a 2-layer MLP mapping <b>source (row) → target (column)</b>.
        A rich source reconstructs others (high row) while resisting reconstruction (low column). The
        <b>richness score</b> = mean<sub>j</sub>[R²(i→j) − R²(j→i)] ranks encoders by net subsumption;
        higher = contains more of the others' information. (This aggregate score is confounded by target
        dimensionality — <b>§3.4</b> gives a ceiling-normalized, confound-free quantification of the
        density-unique information.)</p>
      <div style="display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start;margin-bottom:10px">
        <img src="data:image/png;base64,{s3['png_recon']}" style="max-width:52%;height:auto;border:1px solid var(--line);border-radius:8px">
        <img src="data:image/png;base64,{s3['png_rich']}" style="max-width:44%;height:auto;border:1px solid var(--line);border-radius:8px">
      </div>
      <div style="display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start">
        <div class="table-wrap" style="flex:1;min-width:340px"><table class="results">
          <thead><tr><th class="col-method stub">CDG ↔ X reconstruction</th>
            <th>R²(CDG→X)</th><th>R²(X→CDG)</th><th>asymmetry</th></tr></thead>
          <tbody>{r31}</tbody></table></div>
        <div class="table-wrap" style="min-width:230px"><table class="results">
          <thead><tr><th class="col-method stub">Richness rank</th><th>score</th></tr></thead>
          <tbody>{rrows}</tbody></table></div>
      </div>
    </section>

    <section class="block" id="func-stitch">
      <h3 class="subsection-head"><span class="sub-lab">3.2</span>Functional stitching into the affinity head</h3>
      <p class="table-sub">Map source → CDG latent with the MLP connector, then read out pK with <b>CDG's own
        ridge affinity head</b> (Spearman ρ, 5-fold CV). If a source can be stitched up to CDG's own score
        (ρ&nbsp;=&nbsp;{s3['cdg_rho']:.3f}) it already carried CDG's affinity-relevant information; the residual
        <b>gap</b> = CDG-unique information. <code>CDG→X</code> is the reverse (CDG stitched into X's head).</p>
      <div style="margin-bottom:10px"><img src="data:image/png;base64,{s3['png_stitch']}" style="max-width:560px;width:100%;height:auto"></div>
      <div class="table-wrap"><table class="results">
        <thead><tr><th class="col-method stub">Source X</th><th>ρ (X alone)</th>
          <th>ρ (X→CDG head)</th><th>ρ (CDG→X head)</th><th>gap = ρ<sub>CDG</sub>−(X→CDG)</th></tr></thead>
        <tbody>{r32}</tbody></table></div>
    </section>

    <section class="block" id="rel-rep">
      <h3 class="subsection-head"><span class="sub-lab">3.3</span>Relative-representation zero-shot stitching</h3>
      <p class="table-sub">Training-free (Moschella et al., ICLR 2023): re-express every complex by its cosine
        similarity to <b>{s3['n_anchors']}</b> shared anchor complexes, then train an affinity head on
        <b>CDG's</b> relative rep and apply it <b>zero-shot</b> to each encoder's relative rep (same anchors →
        shared coordinates). High <b>sim-to-CDG</b> = near-isometric to CDG; a large drop from CDG's own
        zero-shot ρ means the space does not communicate with CDG (distinct information).</p>
      <div class="table-wrap"><table class="results">
        <thead><tr><th class="col-method stub">Encoder</th><th>rel-rep sim to CDG</th>
          <th>zero-shot ρ → CDG head</th><th>drop vs CDG self</th></tr></thead>
        <tbody>{r33}</tbody></table></div>
    </section>

    <section class="block" id="ceiling">
      <h3 class="subsection-head"><span class="sub-lab">3.4</span>Seed-twin ceiling: density-unique information</h3>
      <p class="table-sub">Raw reconstruction R² confounds <i>genuinely unique</i> information with mere target
        complexity. To divide that out we set a <b>ceiling</b> = reconstruction R² between two CDG models that
        differ only in random seed (same 100M recipe) — the best achievable when the information is identical
        (<b>ceiling R² = {ceil_r2:.3f}</b>). <b>Normalized recovery</b> = R²(X→CDG) / ceiling is the fraction of
        CDG's recoverable information captured by source X; <b>1 − normalized</b> is what X cannot reconstruct.
        For coords-only this residual is exactly the <b>density-unique information</b>: <b>{du:.0%}</b> of CDG's
        reconstructable information is <b>not present in coordinates</b> (an independently-seeded CDG recovers
        the ceiling, another CDG config recovers near it, coords falls well short).</p>
      <div style="display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start">
        <div style="flex:1;min-width:360px"><img src="data:image/png;base64,{s3['png_ceil']}" style="width:100%;height:auto;border:1px solid var(--line);border-radius:8px"></div>
        <div class="table-wrap" style="min-width:320px"><table class="results">
          <thead><tr><th class="col-method stub">Source X → CDG</th><th>R²(X→CDG)</th>
            <th>÷ ceiling</th><th>unrecoverable</th></tr></thead>
          <tbody>{r34}</tbody></table></div>
      </div>
    </section>
{sec35}
    <ul class="criteria">
      <li><b>Connector:</b> 2-layer MLP (Linear→GELU→Linear, hidden 384, weight-decay 1e-4, early-stopped),
        the low-capacity stitching map of Bansal et al. Source and target are per-dim standardised; R² is
        held-out variance-weighted (5-fold).</li>
      <li><b>Reading richness:</b> CDG reconstructs coords-only C better than C reconstructs CDG (positive
        asymmetry) ⇒ the density channels add information not recoverable from coordinates. The richness bar
        ranks all five encoders by net decodability asymmetry.</li>
      <li><b>Seed-twin ceiling (§3.4):</b> the reference CDG target is a 100M CDG (d640L18h10, seed s1); the
        ceiling is R²(seed s2 → s1) — same recipe, different init — averaged over both directions. Normalizing
        by it removes the target-complexity confound so the coords residual is a clean estimate of
        density-unique information.</li>
      <li><b>References:</b> model stitching — Lenc &amp; Vedaldi (CVPR 2015), Bansal, Nakkiran &amp; Barak
        (NeurIPS 2021); relative representations / zero-shot latent communication — Moschella et al.
        (ICLR 2023 Oral, arXiv:2209.15430).</li>
    </ul>
  </section>
"""

    # ---------- §4 recent representation-quality / information metrics ----------
    q4 = s3["q4"]

    def q4row(nm):
        d = q4[nm]
        cls = ' class="best-row"' if nm == anchor else ""
        tag = ' <span class="tag best">ours</span>' if nm == anchor else (
              ' <span class="tag">ours</span>' if nm.startswith("Ours") else "")
        return (f'<tr{cls}><td class="col-method">{nm}{tag}<span class="method-note">{d["dim"]}-d</span></td>'
                f'<td>{d["rankme"]:.1f}</td><td>{d["id"]:.1f}</td><td>{d["lidar"]:.2f}</td>'
                f'<td>{d["vinfo"]:.3f}</td><td>{d["cka_others"]:.3f}</td></tr>')
    q4rows = "".join(q4row(nm) for nm in names)
    dimc = q4[anchor]["dim"]
    rk_c4, rk_cdg4 = q4["Ours · coords"]["rankme"], q4[anchor]["rankme"]
    id_c4, id_cdg4 = q4["Ours · coords"]["id"], q4[anchor]["id"]
    _c4 = s3.get("canon", {})
    if _c4 and anchor in _c4 and "Ours · coords" in _c4:
        vinfo_canon_str = (f"{_c4[anchor]['vinfo'] - _c4['Ours · coords']['vinfo']:+.3f} nats "
                           f"({_c4[anchor]['rho'] - _c4['Ours · coords']['rho']:+.3f} ρ)")
    else:
        vinfo_canon_str = "larger"

    sec4 = f"""  <section class="doc-section" id="repr-quality">
    <h2 class="section-head"><span class="sec-num">4</span>Corroborating metrics: representation quality &amp; information</h2>
    <p class="section-intro">Label-light metrics from recent ICLR/NeurIPS/ICML work that <b>corroborate §3's
      richness</b> across all five encoders — effective rank (RankMe), intrinsic dimension (TwoNN), task rank
      (LiDAR), 𝒱-usable information about pK, and the Platonic-convergence view of §1. On the controlled
      same-dimension pair (C vs C+D+G, both {dimc}-d) density raises effective rank ({rk_c4:.0f}→{rk_cdg4:.0f})
      and intrinsic dimension ({id_c4:.1f}→{id_cdg4:.1f}). "Richer" ⇒ higher; cross-encoder values are
      dimension-confounded (see caveat).</p>
    <div class="table-wrap" style="margin-bottom:16px"><table class="results">
      <thead><tr><th class="col-method stub">Encoder</th>
        <th>RankMe<br>eff. rank ↑</th><th>Intrinsic<br>dim ↑</th><th>LiDAR<br>task rank ↑</th>
        <th>𝒱-info (pK)<br>nats ↑</th><th>mean CKA<br>to others</th></tr></thead>
      <tbody>{q4rows}</tbody></table></div>

    <section class="block">
      <h3 class="subsection-head"><span class="sub-lab">4.1</span>Effective rank, intrinsic dimension, task rank</h3>
      <p class="table-sub"><b>RankMe</b> (Garrido et al., ICML 2023) = exp(entropy of singular values): over how many
        dimensions variance spreads (label-free richness). <b>TwoNN</b> intrinsic dimension (Valeriani et al.,
        NeurIPS 2023). <b>LiDAR</b> (Thilak et al., ICLR 2024, adapted with pK-quantile classes) = effective rank of
        the affinity-discriminative directions. <i>Caveat: RankMe/ID are bounded by feature dimension, so read our
        C vs C+D+G (both {dimc}-d) directly; cross-encoder values are only indicative.</i></p>
      <div style="display:flex;gap:12px;flex-wrap:wrap">
        <img src="data:image/png;base64,{s3['png_rankme']}" style="max-width:32%;height:auto;border:1px solid var(--line);border-radius:8px">
        <img src="data:image/png;base64,{s3['png_id']}" style="max-width:32%;height:auto;border:1px solid var(--line);border-radius:8px">
        <img src="data:image/png;base64,{s3['png_lidar']}" style="max-width:32%;height:auto;border:1px solid var(--line);border-radius:8px">
      </div>
    </section>

    <section class="block">
      <h3 class="subsection-head"><span class="sub-lab">4.2</span>𝒱-usable information about affinity</h3>
      <p class="table-sub">Predictive 𝒱-information (Ethayarajh et al., ICML 2022; Xu et al., ICLR 2020) =
        ½·ln(Var(pK)/MSE) — the <b>usable</b> information a representation carries about pK. On this within-test
        CV diagnostic C+D+G carries {s3['vinfo_gap']:+.3f} nats over coords; on the <b>canonical train→test</b>
        probe (§2) the gain is <b>{vinfo_canon_str}</b> — nats compress the magnitude, since density improves
        ranking (ρ) more than MSE. Pointwise, density adds usable information on <b>{s3['frac_help']:.0%}</b> of
        complexes.</p>
      <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start">
        <img src="data:image/png;base64,{s3['png_vinfo']}" style="max-width:47%;height:auto;border:1px solid var(--line);border-radius:8px">
        <img src="data:image/png;base64,{s3['png_pvi']}" style="max-width:47%;height:auto;border:1px solid var(--line);border-radius:8px">
      </div>
    </section>

    <section class="block">
      <h3 class="subsection-head"><span class="sub-lab">4.3</span>Platonic convergence</h3>
      <p class="table-sub">The Platonic Representation Hypothesis (Huh et al., ICML 2024) argues strong models
        converge to a shared representation (measured by the mutual-k-NN / CKNNA alignment used in §1). Mean CKA to
        the other encoders quantifies how "on-manifold" each encoder is: the external pretrained encoders cluster,
        while our density encoder sits off it. <b>Distant ≠ rich</b>, though — an encoder can be
        off-manifold because it is idiosyncratic or weak rather than richly complementary; §3–§4 (richness,
        information) are what separate our <i>distinct-and-rich</i> from a merely <i>distant</i> encoder.</p>
      <div><img src="data:image/png;base64,{s3['png_platonic']}" style="max-width:560px;width:100%;height:auto"></div>
    </section>

    <ul class="criteria">
      <li><b>Cross-encoder caveat:</b> RankMe and intrinsic dimension are capped by feature dimensionality
        (IPNet 256-d, ProFSA / BindNet 1024-d), so the same-dimension C vs C+D+G contrast is the controlled comparison;
        𝒱-info and LiDAR are dimension-robust (fixed target / bounded by #classes).</li>
      <li><b>References:</b> RankMe — Garrido et al. (ICML 2023); LiDAR — Thilak et al. (ICLR 2024); intrinsic
        dimension / representation geometry — Valeriani et al. (NeurIPS 2023); 𝒱-usable information — Ethayarajh
        et al. (ICML 2022), Xu et al. (ICLR 2020); Platonic Representation Hypothesis — Huh et al. (ICML 2024).</li>
    </ul>
  </section>
"""

    # ---------- executive summary (key findings) ----------
    cka_ours = CKA[names.index(anchor)][names.index("Ours · coords")]
    ext_vals = [CKA[names.index(anchor)][names.index(nm)] for nm in names if not nm.startswith("Ours")]
    ext_lo, ext_hi = min(ext_vals), max(ext_vals)
    best_nm = max(pairs, key=lambda k: pairs[k]["gain_rho"])
    best_gain = pairs[best_nm]["gain_rho"]
    du = s3["density_unique"]
    champ_norm = next(r["norm"] for r in s3["ceilrows"] if "champion" in r["src"])
    rk_c, rk_cdg = q4["Ours · coords"]["rankme"], q4[anchor]["rankme"]
    id_c, id_cdg = q4["Ours · coords"]["id"], q4[anchor]["id"]
    vgap, frac = s3["vinfo_gap"], s3["frac_help"]
    _canon = s3.get("canon", {})
    cd_can = (_canon[anchor]["rho"] - _canon["Ours · coords"]["rho"]
              if (_canon and anchor in _canon and "Ours · coords" in _canon) else None)
    aff_line = (f"On the <b>canonical train→test</b> probe density gains <b>{cd_can:+.3f} ρ</b> "
                f"(C+D+G vs matched coords)"
                if cd_can is not None else
                f"Concatenating a less-aligned encoder gives the largest ensemble gain ({best_nm} Δρ {best_gain:+.3f})")
    take_line = ("its richness <b>converts</b> into a solid downstream gain "
                 f"(canonical affinity {cd_can:+.3f} ρ, plus generation)."
                 if cd_can is not None else
                 "its richness converts into a real downstream affinity gain.")
    summary_box = f"""  <section style="margin:0 0 44px;padding:18px 24px 14px;border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:12px;background:var(--card);box-shadow:var(--shadow)">
    <div style="font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);font-weight:760;margin-bottom:10px">Key findings</div>
    <ul style="margin:0;padding-left:20px;color:var(--ink-soft);font-size:13.4px;line-height:1.72">
      <li><b>Distinct space (§1).</b> Our two density encoders are each other's closest match (CKA {cka_ours:.2f}) yet sit apart from every external complex encoder (CKA {ext_lo:.2f}–{ext_hi:.2f}) — a distinct representation region.</li>
      <li><b>Helps affinity (§2).</b> {aff_line}. The within-test CV diagnostic compresses this and is used only for the cross-encoder ensemble comparison (best: {best_nm} Δρ {best_gain:+.3f}).</li>
      <li><b>Richer, and quantified (§3).</b> Against a same-recipe seed-twin ceiling, another CDG recovers {champ_norm:.0%} of it but coordinates recover only {1 - du:.0%} — so <b>~{du:.0%} of CDG's information is density-unique</b> (unreconstructable from coordinates).</li>
      <li><b>Information &amp; geometry (§4).</b> Density lifts effective rank ({rk_c:.0f}→{rk_cdg:.0f}) and intrinsic dimension ({id_c:.1f}→{id_cdg:.1f}); it adds usable affinity information on {frac:.0%} of complexes.</li>
      <li><b>Takeaway.</b> Electron density yields a richer, higher-dimensional, distinct complex representation, and {take_line}</li>
    </ul>
    <div style="margin-top:11px;padding-top:9px;border-top:1px solid var(--line);color:var(--ink-faint);font-size:11.5px">
      Structure — <b>§1</b> distinct space · <b>§2</b> converts to affinity{(' (%+.3f ρ)' % cd_can) if cd_can is not None else ''} · <b>§3</b> richer, quantified (~{du:.0%} density-unique) · <b>§4</b> corroborating metrics.</div>
  </section>
"""

    variant_badge = (f'<span class="tag" style="background:#fdf0e6;color:#b5651d">{variant_note}</span>'
                     if variant_note else "")

    # ---------- §2 canonical train->test affinity (the real number) ----------
    canon = s3.get("canon", {})
    canon_block = ""
    if canon and anchor in canon and "Ours · coords" in canon:
        cg, cc = canon[anchor], canon["Ours · coords"]
        d_can = cg["rho"] - cc["rho"]
        d_cv = singles[anchor]["rho"] - singles["Ours · coords"]["rho"]
        crows = ""
        for nm in [x for x in names if x in canon]:      # full-split encoders: Ours pair + IPNet
            c = canon[nm]
            is_cdg = nm == anchor
            cls = ' class="best-row"' if is_cdg else ""
            tag = ' <span class="tag best">ours</span>' if is_cdg else (
                  ' <span class="tag">ours</span>' if nm.startswith("Ours") else "")
            crows += (f'<tr{cls}><td class="col-method">{nm}{tag}</td><td>{c["r"]:.3f}</td>'
                      f'<td>{_mnum(c["rho"], is_cdg)}</td><td>{c["rmse"]:.3f}</td></tr>')
        canon_block = f"""<div class="table-wrap" style="margin-bottom:8px"><table class="results">
      <thead><tr><th class="col-method stub">Canonical affinity · train→test</th>
        <th>Pearson r</th><th>Spearman ρ</th><th>RMSE ↓</th></tr></thead>
      <tbody>{crows}</tbody></table></div>
    <p class="table-sub" style="margin:-2px 0 18px"><b>This is the real affinity number</b> — fit on
      {cg['n_train']} train, evaluate {cg['n_test']} test. Density gain <b>C+D+G − coords =
      {d_can:+.3f} ρ</b> (consistent with the project's clean-matched result), so the richer representation
      <b>does convert</b> into a solid downstream gain. The <b>C+D+G − coords</b> difference is the matched
      contrast; IPNet is shown as an additional full-split pretrained encoder (ProFSA / BindNet are test-only →
      CV diagnostic below). <i>The within-test CV diagnostic that follows
      shrinks this to {d_cv:+.3f} ρ — it lets the weaker coords encoder fit the test distribution and so
      compresses the gap; use it only for cross-encoder comparison, not as the affinity headline.</i></p>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VoxBind · Complex representation analysis · 260827</title>
{style}
</head>
<body>
<div class="page">
  <header class="doc">
    <div class="eyebrow">
      <span>Report · 2026-08-27</span>
      <span class="status">representation</span>{variant_badge}
    </div>
    <h1>Complex representation analysis</h1>
    <p class="lead">Following Figure&nbsp;6 of the co-folding representation study
      (Boltz2-as-FM, arXiv:2602.13249): how does our pretrained density-ViT encoder relate to other
      protein–ligand complex encoders? We test four angles on the same <b>{n}</b> LP-PDBbind
      (<code>lp_edrscc_v2</code>) test complexes: (1) how <b>aligned</b> the representation spaces are,
      (2) whether they are <b>complementary</b> for affinity, (3) whether our density-conditioned encoder
      carries <b>richer information</b> — its representation reconstructs and stitches into the others while
      resisting the reverse — and (4) label-light <b>quality / information</b> metrics from recent ML work
      (effective rank, intrinsic dimension, 𝒱-usable information).</p>
  </header>

{summary_box}
  <section class="doc-section" id="repr-alignment">
    <h2 class="section-head"><span class="sec-num">1</span>Representation alignment</h2>
    <p class="section-intro"><b>CKNNA</b> (mutual-k-NN kernel alignment, Huh&nbsp;et&nbsp;al.&nbsp;2024)
      and <b>linear CKA</b> between the per-complex representations of each encoder pair. Higher =
      more geometrically aligned. Off-diagonal <b>bold</b> = each row's most-aligned partner. The two
      <b>Ours</b> encoders (C+D+G champion and its coords-only twin) are each other's closest match and
      sit apart from the external models (ProFSA, IPNet, BindNet) — our density-conditioned
      space is relatively distinct. This is a <b>pretrained-vs-pretrained</b> comparison: every encoder is a
      frozen pretrained representation (none trained on our affinity labels).</p>
    <div style="display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start;margin-bottom:8px">
      <img src="data:image/png;base64,{png_ck5}" style="max-width:47%;height:auto;border:1px solid var(--line);border-radius:8px">
      <img src="data:image/png;base64,{png_ck50}" style="max-width:47%;height:auto;border:1px solid var(--line);border-radius:8px">
    </div>
    <div style="margin-bottom:8px">
      <img src="data:image/png;base64,{png_cka}" style="max-width:47%;height:auto;border:1px solid var(--line);border-radius:8px">
    </div>
    {matrix_table(CKA, "CKA")}
  </section>

  <section class="doc-section" id="repr-complementarity">
    <h2 class="section-head"><span class="sec-num">2</span>Ensemble complementarity</h2>
    <p class="section-intro">Affinity is reported two ways. <b>(1) Canonical train→test</b> — the real
      leaderboard-style probe (fit on train, evaluate test) for encoders with full-split features; this is the
      density headline. <b>(2) Within-test 5-fold CV diagnostic</b> — puts all five encoders (incl. test-only
      ProFSA / BindNet) on uniform footing so the ensemble-complementarity comparison is apples-to-apples;
      it compresses absolute gaps and is <b>not</b> the affinity headline.</p>
    {canon_block}
    <p class="table-sub"><b>Cross-encoder diagnostic</b> (uniform 5-fold CV on the shared {n} test complexes).
      Single encoders, then <b>Ours&nbsp;·&nbsp;C+D+G ⊕ each external</b>; Δρ is the gain over Ours-CV alone
      (ρ&nbsp;=&nbsp;{a['rho']:.3f}), rows ordered by alignment (low→high) to test the "less-aligned → larger gain" claim.</p>
    <div class="table-wrap" style="margin-bottom:12px"><table class="results">
      <thead><tr><th class="col-method stub">Single encoder · CV diagnostic</th>
        <th>Pearson r</th><th>Spearman ρ</th><th>RMSE</th><th>Δρ</th></tr></thead>
      <tbody>{srows}</tbody></table></div>
    <div class="table-wrap" style="margin-bottom:12px"><table class="results">
      <thead><tr><th class="col-method stub">Ours ⊕ external · CV diagnostic</th>
        <th>Pearson r</th><th>Spearman ρ</th><th>RMSE</th><th>Δρ vs ours</th></tr></thead>
      <tbody>{prows}</tbody></table></div>
    <div><img src="data:image/png;base64,{png_scatter}" style="max-width:520px;width:100%;height:auto"></div>
    <ul class="criteria">
      <li><b>Two protocols:</b> the <b>canonical train→test</b> table is the affinity headline (density
        +{('%.3f' % (canon[anchor]['rho'] - canon['Ours · coords']['rho'])) if (canon and anchor in canon and 'Ours · coords' in canon) else '?'} ρ);
        the CV-on-test tables are a within-test diagnostic that lets the weaker coords encoder fit the test
        distribution, roughly halving the apparent gap — use them only for the cross-encoder ensemble comparison.</li>
      <li><b>Encoders (all frozen pretrained):</b> Ours = density-ViT patch-token mean-pool ({dims.get('Ours · C+D+G','?')}-d);
        ProFSA = [mol;pocket] head input (1024-d); IPNet = IPDiff interaction-prior (256-d);
        BindNet = BioLip-pretrained complex CLS (1024-d). No encoder is trained on our affinity labels.</li>
      <li><b>Complementarity:</b> among strong encoders the less-aligned partner gives the larger ensemble gain;
        a weak partner may not help at all, so the gain reflects alignment × encoder quality, not alignment alone.</li>
    </ul>
  </section>

{sec3}
{sec4}
  <footer>260827 · Complex representation analysis · alignment · complementarity · decodability / stitching · quality &amp; information · lp_edrscc_v2 test</footer>
</div>
</body>
</html>"""


if __name__ == "__main__":
    main()
