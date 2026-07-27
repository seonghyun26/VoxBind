"""pose_dedup.py — DENSITY-AWARE pose dedup (v2.3 candidate).

Standard dedup (v2.1) collapses every (PDB, ligand-CCD) group to ONE crop, discarding the
~13% of copies that are genuinely different poses (measured: aligned density-crop correlation
0.86 for symmetry copies vs ~0.40 for >1A-RMSD copies; cross-ligand baseline ~0.005).

Density-aware dedup instead keeps one representative per DENSITY cluster inside a group:
greedily keep a copy iff its aligned density-crop correlation with every already-kept copy is
below `tau` (i.e. it is NOT a near-duplicate of anything kept). Symmetry copies (corr≈1) are
dropped; distinct poses (corr<tau) are retained. tau≈0.85 drops symmetry/near-identical while
keeping alternate conformations.

Reuses the production resampler (crossdocked_density._load_raw_grid / _resample_density) so the
crops are exactly what the model ingests. Copy B is expressed in copy A's frame via the ligand
Kabsch rotation before correlating, so pure re-orientation is not counted as a difference.
"""
import numpy as np


def _kabsch(A0, B0):
    """Rotation R with B0 ≈ A0 @ R.T (both pre-centred), + rmsd."""
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    Dd = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, Dd]) @ U.T
    return R, float(np.sqrt(((B0 - A0 @ R.T) ** 2).sum(1).mean()))


def _corr(a, b):
    a = a.ravel(); b = b.ravel()
    sa, sb = a.std(), b.std()
    if sa < 1e-8 or sb < 1e-8:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def pose_dedup_group(coords_list, resample_fn, tau=0.85, cap=16):
    """Greedy density-aware keep-set for one (PDB, ligand) group.

    coords_list : list of (N,3) ligand heavy-atom coords (map/deposited frame).
    resample_fn : center_cart, R_aug -> G³ density crop (partial of _resample_density bound to
                  this PDB's map). R_aug=None => axis-aligned crop at center.
    Returns the indices (into coords_list) that are KEPT.
    """
    copies = coords_list[:cap]
    if len(copies) <= 1:
        return list(range(len(coords_list)))
    centers = [c.mean(0) for c in copies]
    kept = [0]
    kept_crops = {0: resample_fn(centers[0], None)}
    for i in range(1, len(copies)):
        max_c = 0.0
        for j in kept:
            R, _ = _kabsch(copies[i] - centers[i], copies[j] - centers[j])
            crop_i_in_j = resample_fn(centers[i], R.T)      # copy i expressed in copy j's frame
            max_c = max(max_c, _corr(crop_i_in_j, kept_crops[j]))
            if max_c >= tau:
                break
        if max_c < tau:
            kept.append(i)
            kept_crops[i] = resample_fn(centers[i], None)
    return kept
