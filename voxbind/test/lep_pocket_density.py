"""lep_pocket_density.py — LEP probe with REAL receptor electron density (pocket-only).

Follow-up to test/lep_probe.py. The question: LEP ligands are Glide-DOCKED (no ligand
density), but the active/inactive RECEPTORS are real PDB depositions. So we CAN fetch the
receptor's experimental 2Fo-Fc map and feed genuine POCKET density to the C+D+G encoder,
instead of zero-filling. Does real pocket-conformation density help activator classification?

Pipeline (reuses the tested v5 resample machinery):
  * per unique X-ray structure: match receptor atoms (LEP frame ↔ deposited PDB) by
    (resseq, atom-name) → rigid Kabsch transform (LEP → crystal frame; verified RMSD ~0.3 Å);
    fetch PDB-REDO MTZ (FWT/PHWT) → gemmi FFT → 2Fo-Fc grid in the crystal frame.
  * per example side: sample the map onto the 64³ box at the ligand-centered pose via
    resample_box(center, R, t) (SAME local grid as the atomblob → co-registered), normalize
    with the champion's PLINDER-v2 recipe (arcsinh/0.5 + z-score), compute gradmag.
  * assemble [atomblob(11) ‖ density ‖ gradmag] (13 ch) → frozen champion encoder → 640-D.

Restricted to examples where BOTH conformers are X-ray (train 239 / val 47 / test 62); coords
and zero-fill C+D+G are re-evaluated on the SAME subset for a fair comparison.

Conditions:
  coords_sub     — C encoder, both-X-ray subset (from cached lep_coords features)
  cdg_zero_sub   — C+D+G, density zero-filled, same subset (from cached lep_cdg features)
  cdg_recdens    — C+D+G, real receptor 2Fo-Fc density (full map)
  cdg_pocdens    — C+D+G, real density with the ligand region masked to background (pocket-only)

Run (voxbind env, GPU 1):
    cd /home/shpark/prj-denovo/VoxBind/voxbind
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib CUDA_VISIBLE_DEVICES=1 python test/lep_pocket_density.py
"""
import gemmi  # before torch

import importlib.util
import json
import os
import pickle
import sys
import urllib.request

import numpy as np
import scipy.ndimage
import torch

REPO = "/home/shpark/prj-denovo/VoxBind"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "voxbind"))

from voxbind.voxelizer import Voxelizer                                    # noqa: E402
from voxbind.models.mae_ops import gradient_magnitude3d, per_sample_zscore # noqa: E402

# reuse lep_probe helpers (build_atomblob, train_probe, metrics, encoder specs)
_lp = importlib.util.spec_from_file_location("lp", os.path.join(REPO, "voxbind/test/lep_probe.py"))
lp = importlib.util.module_from_spec(_lp); _lp.loader.exec_module(lp)
pr = lp.pr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G, RES = 64, 0.25
_OFFS = (np.arange(G) - G * 0.5) * RES
POCKET_MASK_R = 2.5                                    # Å; mask density within this of ligand
# champion PLINDER-v2 density recipe (matches the encoder's training input)
S, MU, SIG = 0.5, -0.015437547297403289, 0.5084830820835629
BG = float((np.arcsinh(0.0 / S) - MU) / SIG)           # normalized value of empty density

DATA = os.path.join(REPO, "voxbind/dataset/data/lep")
CIF_DIR = os.path.join(DATA, "cif"); MTZ_DIR = os.path.join(DATA, "redo")
os.makedirs(CIF_DIR, exist_ok=True); os.makedirs(MTZ_DIR, exist_ok=True)
XRAY = {'2I4J','3KMR','3KMZ','3R2A','3R5M','4BUO','4DJH','4DKL','4EOQ','4EOR','4GBR','4MQT',
        '4Z9G','4ZUD','5A8E','5C1M','5CXV','5G53','5N2S','5T04','5TUD','5U09','5V54','5V56',
        '5YC8','5ZKP','5ZKQ','6AK3','6B73','6BQG','6BQH','6C5Q','6DO1','6DS0','6E67','6GT3',
        '6IBL','6M9T'}


def kabsch(P, Q):
    """Rigid transform mapping P onto Q (P @ R.T + t ≈ Q)."""
    P0, Q0 = P.mean(0), Q.mean(0)
    U, S_, Vt = np.linalg.svd((P - P0).T @ (Q - Q0))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = Q0 - P0 @ R.T
    rmsd = float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean()))
    return R, t, rmsd


def robust_kabsch(P, Q, thresh=1.5, iters=6):
    """Outlier-rejecting Kabsch (handles multi-copy structures where naive (resseq,name)
    matching mixes NCS copies). Iteratively refit on inliers < thresh Å."""
    P, Q = np.asarray(P), np.asarray(Q)
    keep = np.ones(len(P), bool)
    R, t, rmsd = kabsch(P, Q)
    for _ in range(iters):
        resid = np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1))
        new = resid < thresh
        if new.sum() < 50 or new.sum() == keep.sum():
            keep = new; break
        keep = new
        R, t, rmsd = kabsch(P[keep], Q[keep])
    R, t, rmsd = kabsch(P[keep], Q[keep])
    return R, t, rmsd, int(keep.sum())


def resample_box(arr, fT, nu, nv, nw, center, Rk, tk):
    """Sample raw map onto a 64³ box centered at `center`, mapped LEP→crystal via (Rk,tk)."""
    gx, gy, gz = np.meshgrid(_OFFS, _OFFS, _OFFS, indexing="ij")
    o = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1).astype(np.float64)
    cart = (center.reshape(1, 3) + o) @ Rk.T + tk.reshape(1, 3)
    frac = cart @ fT
    ci = (frac[:, 0] * nu) % nu; cj = (frac[:, 1] * nv) % nv; ck = (frac[:, 2] * nw) % nw
    d = scipy.ndimage.map_coordinates(arr, [ci, cj, ck], order=1, mode="wrap", prefilter=False)
    return d.reshape(G, G, G)


def fetch(url, path, timeout=60):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def deposited_heavy(pdb):
    p = fetch(f"https://files.rcsb.org/download/{pdb}.cif", os.path.join(CIF_DIR, f"{pdb}.cif"))
    st = gemmi.read_structure(p); st.setup_entities()
    bychain = {}
    for ch in st[0]:
        dd = bychain.setdefault(ch.name, {})
        for res in ch:
            for at in res:
                if at.element.name == 'H':
                    continue
                dd.setdefault((res.seqid.num, at.name), [at.pos.x, at.pos.y, at.pos.z])
    return bychain


def grid_from_mtz(pdb):
    p = fetch(f"https://pdb-redo.eu/db/{pdb.lower()}/{pdb.lower()}_final.mtz",
              os.path.join(MTZ_DIR, f"{pdb}.mtz"))
    mtz = gemmi.read_mtz_file(p)
    grid = mtz.transform_f_phi_to_map('FWT', 'PHWT', sample_rate=2.5)
    arr = np.array(grid, dtype=np.float32)
    orth = np.array(grid.unit_cell.orth.mat.tolist())
    fT = np.linalg.inv(orth).T
    return arr, fT, arr.shape[0], arr.shape[1], arr.shape[2]


def build_structure_cache(recept):
    """Per X-ray structure → (R, t, rmsd, grid-tuple). Skips structures that fail."""
    cache, fails = {}, []
    for pdb in sorted(recept):
        try:
            r = recept[pdb]
            heavy = r['element'] != 'H'
            lep_xyz = r['xyz'][heavy]
            lep_key = list(zip(r['residue'][heavy].astype(int), r['name'][heavy]))
            dep = deposited_heavy(pdb)
            # robust fit per deposited chain; keep the chain with the lowest inlier RMSD
            best = None                                   # (R, t, rmsd, n_inlier)
            for ch, dd in dep.items():
                P, Q = [], []
                for k, xyz in zip(lep_key, lep_xyz):
                    if k in dd:
                        P.append(xyz); Q.append(dd[k])
                if len(P) < 50:
                    continue
                R, t, rmsd, nin = robust_kabsch(np.array(P), np.array(Q))
                if nin >= 100 and (best is None or rmsd < best[2]):
                    best = (R, t, rmsd, nin)
            if best is None or best[2] > 1.0:             # alignment quality gate
                fails.append(pdb)
                print(f"  {pdb}: SKIP (best rmsd={best[2]:.2f} nin={best[3]})" if best
                      else f"  {pdb}: SKIP (no chain matched)")
                continue
            R, t, rmsd, nin = best
            grid = grid_from_mtz(pdb)
            cache[pdb] = dict(R=R, t=t, rmsd=rmsd, grid=grid)
            print(f"  {pdb}: inliers {nin} rmsd={rmsd:.2f} grid={grid[2]}x{grid[3]}x{grid[4]}")
        except Exception as e:
            fails.append(pdb); print(f"  {pdb}: FAIL {type(e).__name__} {str(e)[:50]}")
    print(f"structure cache: {len(cache)}/{len(recept)} ok, fails={fails}")
    return cache


def lig_centroid(struct):
    elem = np.asarray(struct["elem"]); xyz = np.asarray(struct["xyz"], np.float32)
    is_lig = np.asarray(struct["chain"]).astype(str) == "L"
    lig = xyz[is_lig]; le = elem[is_lig]
    heavy = np.array([lp._norm_elem(e) not in lp.HYDROGENS for e in le])
    return lig[heavy].mean(0), lig[heavy]                # centroid (LEP frame), heavy lig xyz


def density_channels(struct, sc, mask_ligand):
    """Real receptor density + gradmag (64³ each), normalized like training. mask_ligand →
    zero the density/gradmag within POCKET_MASK_R of ligand heavy atoms (pocket-only)."""
    center, lig_heavy = lig_centroid(struct)
    arr, fT, nu, nv, nw = sc["grid"]
    raw = resample_box(arr, fT, nu, nv, nw, center.astype(float), sc["R"], sc["t"])
    dens = ((np.arcsinh(raw / S) - MU) / SIG).astype(np.float32)
    dt = torch.from_numpy(dens)[None, None]
    grad = per_sample_zscore(gradient_magnitude3d(dt)).squeeze().numpy().astype(np.float32)
    if mask_ligand:
        local = lig_heavy - center                       # ligand positions in local grid frame
        idx = np.round(local / RES + G * 0.5).astype(int)
        idx = idx[np.all((idx >= 0) & (idx < G), 1)]
        m = np.zeros((G, G, G), bool)
        rr = int(np.ceil(POCKET_MASK_R / RES))
        for x, y, z in idx:
            xs, ys, zs = (slice(max(0, x-rr), min(G, x+rr+1)),
                          slice(max(0, y-rr), min(G, y+rr+1)),
                          slice(max(0, z-rr), min(G, z+rr+1)))
            m[xs, ys, zs] = True
        dens = dens.copy(); grad = grad.copy()
        dens[m] = BG; grad[m] = 0.0
    return dens, grad


@torch.no_grad()
def extract(examples, struct_cache, encoder, voxelizer, mask_ligand):
    """{id: {active:640, inactive:640, label}} for both-X-ray examples with valid maps."""
    out = {}
    for r in examples:
        pa, pi = r["id"].split("__")[1].upper(), r["id"].split("__")[2].upper()
        if pa not in struct_cache or pi not in struct_cache:
            continue
        feats = {}
        ok = True
        for side, pdb in (("active", pa), ("inactive", pi)):
            blob = lp.build_atomblob(r[side], voxelizer, DEVICE)          # (11,G,G,G)
            if blob is None:
                ok = False; break
            dens, grad = density_channels(r[side], struct_cache[pdb], mask_ligand)
            d = torch.from_numpy(dens).to(DEVICE)[None]
            g = torch.from_numpy(grad).to(DEVICE)[None]
            x = torch.cat([blob, d, g], 0)[None].float()                  # (1,13,G,G,G)
            tok = encoder.forward_features(x)
            feats[side] = tok.mean(1).squeeze(0).cpu().numpy()
        if ok:
            out[r["id"]] = {"active": feats["active"], "inactive": feats["inactive"],
                            "label": r["label"]}
    return out


def subset_cached(cache_name, keep_ids):
    """Load a lep_probe feature cache and keep only ids (with a split tag) in keep_ids."""
    full = torch.load(os.path.join(DATA, "features", cache_name), weights_only=False)
    out = {}
    for split, feats in full.items():
        out[split] = {k: v for k, v in feats.items()
                      if k.rsplit("__", 1)[0] in keep_ids[split]}
    return out


def main():
    complexes = torch.load(os.path.join(DATA, "lep_complexes.pt"), weights_only=False)
    recept = pickle.load(open(os.path.join(DATA, "xray_receptor_atoms.pkl"), "rb"))

    print("building per-structure transform + map cache ...")
    sc = build_structure_cache(recept)

    both = {s: [r for r in complexes[s]
                if r["id"].split("__")[1].upper() in sc and r["id"].split("__")[2].upper() in sc]
            for s in ("train", "val", "test")}
    for s in both:
        print(f"[{s}] both-X-ray with valid maps: {len(both[s])}/{len(complexes[s])}")
    keep_ids = {s: {r["id"] for r in both[s]} for s in both}

    print("\nloading champion (C+D+G) encoder ...")
    from pathlib import Path
    cfg = lp.CONDITIONS["cdg"]
    encoder = pr.load_encoder(Path(REPO) / "voxbind" / cfg["exp"], cfg["epoch"], DEVICE)
    vox = Voxelizer(grid_dim=G, resolution=RES, cubes_around=8, device=DEVICE).to(DEVICE)

    conditions = {}
    for name, mask in (("cdg_recdens", False), ("cdg_pocdens", True)):
        cache_f = os.path.join(DATA, "features", f"lep_{name}_e49.pt")
        if os.path.exists(cache_f):
            conditions[name] = torch.load(cache_f, weights_only=False)
        else:
            print(f"\nextracting {name} (mask_ligand={mask}) ...")
            conditions[name] = {s: extract(both[s], sc, encoder, vox, mask) for s in both}
            torch.save(conditions[name], cache_f)
        for s in both:
            print(f"  [{name}/{s}] {len(conditions[name][s])}")

    # coords + zero-fill on the SAME subset (from lep_probe caches)
    conditions["coords_sub"] = subset_cached("lep_coords_e49.pt", keep_ids)
    conditions["cdg_zero_sub"] = subset_cached("lep_cdg_e49.pt", keep_ids)

    print("\n=== probes on the both-X-ray subset (3 seeds) ===")
    results = {}
    for name, feats in conditions.items():
        ms = [lp.train_probe(feats, s, DEVICE) for s in range(3)]
        ag = {}
        for k in ("auroc", "auprc"):
            v = np.array([m[k] for m in ms], float)
            ag[k] = [float(np.nanmean(v)), float(np.nanstd(v))]
        ag["n_test"] = len(feats["test"])
        results[name] = ag
        print(f"  {name:14s} AUROC {ag['auroc'][0]:.4f}±{ag['auroc'][1]:.4f}  "
              f"AUPRC {ag['auprc'][0]:.4f}±{ag['auprc'][1]:.4f}  n_test={ag['n_test']}")

    json.dump(results, open(os.path.join(DATA, "lep_pocket_density_results.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(DATA, 'lep_pocket_density_results.json')}")


if __name__ == "__main__":
    main()
