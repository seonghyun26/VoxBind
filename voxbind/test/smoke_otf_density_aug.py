"""smoke_otf_density_aug.py — does on-the-fly density cropping WITH augmentation work?

Tests the proposed pretraining change: instead of cropping a fixed 64³ density grid
and then shifting/rotating it (zero-fill at exposed edges), fold the augmentation
(R, t) into the SAMPLING of the full CCP4 map:

    x_map = center + R.T · (o − t)        # o = output-box offsets, in Å, centered at 0

Decisive correctness check = CO-REGISTRATION: the resampled density must still sit on
the ligand atoms after BOTH are augmented by the same (R, t). If the geometry/convention
were wrong, density would no longer overlap the atoms → low density-at-atoms z-score.

Also compares against the current zero-fill path (crop → _rotate_density → _translate_density)
to confirm (a) interior agreement and (b) the zero-fill artifact is removed.

Run (gemmi needs the conda libstdc++):
    cd /home/shpark/prj-denovo/VoxBind/voxbind
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python test/smoke_otf_density_aug.py
"""

import gemmi  # noqa: F401  — MUST precede torch (CXXABI_1.3.15 / silent _load_grid failure)

import os
import random
import sys

import numpy as np
import scipy.ndimage

REPO = "/home/shpark/prj-denovo/VoxBind"
sys.path.insert(0, REPO)

import torch  # noqa: E402

from voxbind.dataset.crossdocked_xray import (  # noqa: E402
    _load_grid, _crop_density, _rotate_density, _translate_density, normalize_crop,
)
from voxbind.utils.dataset_utils import random_rot_matrix  # noqa: E402

G, RES = 64, 0.25
CCP4_DIR = os.path.join(REPO, "voxbind/dataset/data/ccp4")
DATA = os.path.join(REPO, "voxbind/dataset/data/data_train.pt")
N_SAMPLES = 6
DELTA = 1.0  # Å, matches DatasetCrossDockedXray default


def pdbid(pocket_id: str) -> str:
    return pocket_id.split("/")[1].split("_")[0].lower()


def crop_density_aug(arr_norm, frac_T, nu, nv, nw, center, R, t):
    """On-the-fly crop with augmentation folded into the sampling coordinates.

    Sample the full periodic map at  center + R.T·(o − t)  for every output offset o.
    Row-vector form:  src_offset = (o − t) @ R   (so R=I, t=0 reproduces _crop_density).
    """
    offs = (np.arange(G, dtype=np.float32) - G * 0.5) * RES
    gx, gy, gz = np.meshgrid(offs, offs, offs, indexing="ij")
    o = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)        # (G³, 3) Å offsets
    Rn = R.numpy().astype(np.float32)
    tn = t.numpy().reshape(1, 3).astype(np.float32)
    src = (o - tn) @ Rn + center.reshape(1, 3).astype(np.float32)     # map-frame Cartesian
    frac = src @ frac_T
    ci = (frac[:, 0] * nu) % nu
    cj = (frac[:, 1] * nv) % nv
    ck = (frac[:, 2] * nw) % nw
    d = scipy.ndimage.map_coordinates(arr_norm, [ci, cj, ck], order=1, mode="wrap",
                                      prefilter=False)
    return d.reshape(G, G, G).astype(np.float32)


def density_at_atoms(crop, offsets_ang):
    """Mean density (z) at the voxels nearest the recentered atom offsets (Å)."""
    idx = np.round(offsets_ang / RES + G * 0.5).astype(int)
    inb = np.all((idx >= 0) & (idx < G), axis=1)
    idx = idx[inb]
    if len(idx) == 0:
        return float("nan"), 0
    return float(crop[idx[:, 0], idx[:, 1], idx[:, 2]].mean()), len(idx)


def main():
    assert os.path.isdir(CCP4_DIR), CCP4_DIR
    data = torch.load(DATA, weights_only=False)
    random.Random(1234).shuffle(data)
    data = data[: len(data) - 100]
    data = [(p, l) for p, l in data if l["max_len"] <= 30]

    rng = random.Random(0)
    rng.shuffle(data)

    rows = []
    n_ok = 0
    for pocket_, ligand_ in data:
        if len(rows) >= N_SAMPLES:
            break
        pid = pdbid(pocket_["id"])
        path = os.path.join(CCP4_DIR, f"{pid}.ccp4")
        if not os.path.exists(path):
            continue
        try:
            grid = _load_grid(__import__("pathlib").Path(path))
            if grid is None:
                print(f"  [skip] {pid}: _load_grid returned None")
                continue
            arr_norm, frac_T, nu, nv, nw = grid

            mask = ligand_["atoms_channel"] < 7
            lig = ligand_["coords"][mask].numpy().astype(np.float32)
            center = lig.mean(axis=0)                 # ligand centroid (map Cartesian frame)
            p_lig = lig - center                      # recentered ligand offsets (Å)
            # pocket = real crystal receptor → sits in real density (good abs. co-reg probe;
            # the ligand is often cross-docked in v5, so its density is mismatched by design).
            pmask = pocket_["atoms_channel"] < 4
            p_poc = (pocket_["coords"][pmask].numpy().astype(np.float32) - center)

            # ── aug-OFF reference: crop at centroid, no transform ──────────────
            d_off = normalize_crop(_crop_density(arr_norm, frac_T, nu, nv, nw, center))
            zL_off, _ = density_at_atoms(d_off, p_lig)
            zP_off, nP = density_at_atoms(d_off, p_poc)

            # ── augmentation (same R,t for atoms + density) ───────────────────
            R = random_rot_matrix()
            t = (torch.rand((1, 3)) - 0.5) * 2 * DELTA
            Rt = R.numpy().T
            tn = t.numpy().reshape(1, 3)
            pL_aug = p_lig @ Rt + tn                                   # p' = p @ R.T + t
            pP_aug = p_poc @ Rt + tn

            # NEW: fold (R,t) into the map sampling (real density everywhere)
            d_new = normalize_crop(crop_density_aug(arr_norm, frac_T, nu, nv, nw, center, R, t))
            zL_new, _ = density_at_atoms(d_new, pL_aug)
            zP_new, _ = density_at_atoms(d_new, pP_aug)

            # OLD: crop@center → normalize → rotate → translate (zero-fill)
            d_old = _translate_density(_rotate_density(d_off.copy(), R), t)
            zP_old, _ = density_at_atoms(d_old, pP_aug)

            # diagnostics
            c = slice(12, 52)                                          # central 40³
            a, b = d_new[c, c, c].ravel(), d_old[c, c, c].ravel()
            corr = float(np.corrcoef(a, b)[0, 1])
            zf_new = float((d_new == 0).mean())
            zf_old = float((d_old == 0).mean())
            finite = bool(np.isfinite(d_new).all())

            rows.append((pid, nP, zP_off, zP_new, zP_old, zL_off, zL_new, corr, zf_new, zf_old, finite))
            n_ok += 1
        except Exception as e:
            print(f"  [ERROR] {pid}: {type(e).__name__}: {e}")

    if not rows:
        print("NO SAMPLES PROCESSED"); sys.exit(1)

    print(f"\non-the-fly density aug smoke test  (n={len(rows)}, δ_trans={DELTA}Å, full 360° rot)")
    print("z@POCKET = abs. co-registration probe (real crystal receptor); z@LIG low by design (v5 cross-docked)\n")
    hdr = (f"{'pdb':>5} {'#poc':>4} {'zPoc_off':>9} {'zPoc_NEW':>9} {'zPoc_OLD':>9} "
           f"{'zLig_off':>9} {'zLig_NEW':>9} {'corr(N,O)':>9} {'zero%_NEW':>9} {'zero%_OLD':>9} {'finite':>6}")
    print(hdr); print("-" * len(hdr))
    for pid, nP, zPo, zPn, zPold, zLo, zLn, corr, zfn, zfo, fin in rows:
        print(f"{pid:>5} {nP:>4} {zPo:>9.3f} {zPn:>9.3f} {zPold:>9.3f} {zLo:>9.3f} {zLn:>9.3f} "
              f"{corr:>9.3f} {100*zfn:>8.2f}% {100*zfo:>8.2f}% {str(fin):>6}")

    zP_off = float(np.mean([r[2] for r in rows]))
    zP_new = float(np.mean([r[3] for r in rows]))
    zP_old = float(np.mean([r[4] for r in rows]))
    corr_m = float(np.mean([r[7] for r in rows]))
    zfn_m = float(np.mean([r[8] for r in rows]))
    zfo_m = float(np.mean([r[9] for r in rows]))
    all_finite = all(r[10] for r in rows)

    print("\nmeans:")
    print(f"  density@POCKET z  : aug-off {zP_off:.3f} | NEW(fold-in) {zP_new:.3f} | OLD(zero-fill) {zP_old:.3f}")
    print(f"  interior corr(N,O): {corr_m:.3f}")
    print(f"  exact-zero frac   : NEW {100*zfn_m:.2f}%  vs  OLD {100*zfo_m:.2f}%")

    # PASS criteria (pocket = decisive absolute co-registration signal)
    checks = {
        "all crops finite":                  all_finite,
        "pocket co-registered (z>0.5)":      zP_new > 0.5,
        "co-reg preserved under aug (~off)":  zP_new > 0.85 * zP_off,
        "interior matches OLD (>0.95)":       corr_m > 0.95,
        "NEW removes zero-fill (<1%, <OLD)":  zfn_m < 0.01 and zfn_m < zfo_m,
    }
    print("\nchecks:")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    ok = all(checks.values())
    print(f"\n{'== SMOKE TEST PASSED ==' if ok else '== SMOKE TEST FAILED =='}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
