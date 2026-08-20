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


NODES = [
    ("Ours · C+D+G", "density-ViT (100M, mask0.75)", lambda: _load_ours(
        "atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_mask075.pt")),
    ("Ours · coords", "density-ViT (coords-only twin)", lambda: _load_ours(
        "atomblob_e49_v5_260723_ar_cvit_100m_v2_mask075_coords.pt")),
    ("ProFSA", "pretrained pocket encoder (frozen)",
     lambda: _load_baseline_pt(REPO / "base/profsa/_edrscc/features/repr_lp_edrscc_v2_test_seed0.pt")),
    ("GET", "E(3)-equivariant, supervised", _load_get),
    ("AEV-PLIG", "GATv2 + AEV, supervised",
     lambda: _load_baseline_pt(REPO / "base/aevplig/_edrscc/features/repr_lp_edrscc_v2_test_seed0.pt")),
    ("DSMBind", "unsupervised energy (zero-shot)", _load_dsmbind),
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


# ======================================================================================
# 5. main
# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(4)
    dev = args.device

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

    K = len(names)
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

    page = render_page(names, subs, dims, n, CKA, CK5, CK50, singles, pairs, ANCHOR,
                       png_ck5, png_ck50, png_cka, png_scatter)
    OUT_HTML.write_text(page)
    print(f"\nwrote standalone report -> {OUT_HTML}")


def _mnum(x, best=False, d=3):
    s = f"{x:.{d}f}"
    return f"<b>{s}</b>" if best else s


def render_page(names, subs, dims, n, CKA, CK5, CK50, singles, pairs, anchor,
                png_ck5, png_ck50, png_cka, png_scatter):
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
      <span class="status">representation</span>
    </div>
    <h1>Complex representation analysis</h1>
    <p class="lead">Following Figure&nbsp;6 of the co-folding representation study
      (Boltz2-as-FM, arXiv:2602.13249): does our pretrained density-ViT encoder learn a
      <b>distinct, complementary</b> protein–ligand complex-representation space relative to
      other complex encoders? All encoders are compared on the same <b>{n}</b> LP-PDBbind
      (<code>lp_edrscc_v2</code>) test complexes with a uniform protocol.</p>
  </header>

  <section class="doc-section" id="repr-alignment">
    <h2 class="section-head"><span class="sec-num">1</span>Representation alignment</h2>
    <p class="section-intro"><b>CKNNA</b> (mutual-k-NN kernel alignment, Huh&nbsp;et&nbsp;al.&nbsp;2024)
      and <b>linear CKA</b> between the per-complex representations of each encoder pair. Higher =
      more geometrically aligned. Off-diagonal <b>bold</b> = each row's most-aligned partner. The two
      <b>Ours</b> encoders (C+D+G champion and its coords-only twin) are each other's closest match and
      sit apart from the external models (ProFSA, GET, AEV-PLIG, DSMBind) — our density-conditioned
      space is relatively distinct. HBGSA / CheapNet are omitted (no persisted checkpoint to tap).</p>
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
    <p class="section-intro">Ridge 5-fold CV predicting pK on the shared {n} test complexes, from each
      single encoder and from <b>Ours&nbsp;·&nbsp;C+D+G concatenated with each external encoder</b>. Δρ is
      the Spearman gain over Ours alone (ρ&nbsp;=&nbsp;{a['rho']:.3f}). Rows are ordered by alignment to ours
      (low → high); the paper's claim is that <b>less-aligned</b> partners give the <b>larger</b> gain.</p>
    <div class="table-wrap" style="margin-bottom:12px"><table class="results">
      <thead><tr><th class="col-method stub">Single encoder</th>
        <th>Pearson r</th><th>Spearman ρ</th><th>RMSE</th><th>Δρ</th></tr></thead>
      <tbody>{srows}</tbody></table></div>
    <div class="table-wrap" style="margin-bottom:12px"><table class="results">
      <thead><tr><th class="col-method stub">Ours ⊕ external</th>
        <th>Pearson r</th><th>Spearman ρ</th><th>RMSE</th><th>Δρ vs ours</th></tr></thead>
      <tbody>{prows}</tbody></table></div>
    <div><img src="data:image/png;base64,{png_scatter}" style="max-width:520px;width:100%;height:auto"></div>
    <ul class="criteria">
      <li><b>CKNNA / CKA:</b> features are column-centred then row-L2-normalised; CKNNA follows the
        <code>minyoungg/platonic-rep</code> reference (unbiased HSIC on the mutual-k-NN masked kernel).</li>
      <li><b>Encoders:</b> Ours = frozen density-ViT patch-token mean-pool ({dims.get('Ours · C+D+G','?')}-d);
        ProFSA = frozen [mol;pocket] head input; GET / AEV-PLIG = pooled graph vector (seed0, supervised);
        DSMBind = mean-pooled all-atom encoder + energy (zero-shot).</li>
      <li><b>Caveat:</b> GET and AEV-PLIG are supervised on the LP train split, so their reps are already
        affinity-tuned; ProFSA (frozen) and DSMBind (zero-shot) are cleaner complementarity tests. The probe
        is a representation diagnostic (uniform CV on shared test complexes), not the train→test leaderboard
        number. The alignment→gain trend holds among strong encoders; the weakest (DSMBind) does not help,
        so the gain reflects alignment × encoder quality, not alignment alone.</li>
    </ul>
  </section>

  <footer>260827 · Complex representation analysis · CKNNA / CKA + ensemble complementarity · lp_edrscc_v2 test</footer>
</div>
</body>
</html>"""


if __name__ == "__main__":
    main()
