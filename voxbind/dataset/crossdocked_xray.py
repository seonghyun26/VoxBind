"""crossdocked_xray.py — CrossDocked2020 dataset with X-ray electron density

Extends DatasetCrossdocked to optionally include a cropped 2Fo-Fc electron
density map as an additional input channel.

Design
------
- CCP4 maps (downloaded by dataset/00a_data_density_download.py) are loaded lazily
  and cached per worker process (one OrderedDict per DataLoader worker).
- The density box is cropped at the ligand centroid (PDB Cartesian frame),
  matching the VoxBind 64³ × 0.25 Å grid convention exactly.
- When data augmentation is active, the same rotation matrix used for atom
  coordinates is also applied to the density volume via
  scipy.ndimage.affine_transform (linear interpolation, zero-padding).
- Translation augmentation is NOT applied to the density (shift ≤ ±1 Å is
  negligible relative to the 16 Å box; the model is expected to be robust).
- When use_xray=False, or when no map is available for a given PDB ID, the
  returned "xray_density" field is None so the model can fall back gracefully.

Returned batch fields
---------------------
    "pocket"       : same as DatasetCrossdocked
    "ligand"       : same as DatasetCrossdocked
    "xray_density" : torch.Tensor (64, 64, 64) float32, z-score normalised
                     OR None if the map is unavailable / use_xray=False

Usage
-----
    from voxbind.dataset.crossdocked_xray import DatasetCrossDockedXray

    dset = DatasetCrossDockedXray(
        data_dir   = "dataset/data",
        ccp4_dir   = "dataset/data/ccp4",
        split      = "train",
        use_xray   = True,
        cache_size = 32,   # CCP4 maps kept per worker process
    )
"""

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import Dataset

from voxbind.constants import ELEMENTS_HASH_CROSSDOCKED, RADIUS_PER_ATOM
from voxbind.utils.dataset_utils import (
    filter_atoms_by_distance, pad, recenter_structures,
    random_rot_matrix,
)

# Grid constants (match voxbind/configs/vox/default.yaml)
_GRID_DIM = 64
_RESOLUTION = 0.25  # Å per voxel
_HALF_EXTENT = _GRID_DIM * _RESOLUTION / 2  # 8.0 Å


# ── Density helpers ─────────────────────────────────────────────────────────────

def _load_grid(ccp4_path: Path):
    """Load a CCP4 map with gemmi and return normalised array + fractionalization matrix.

    Returns (arr_norm, frac_matrix_T, nu, nv, nw) or None on failure.
    """
    try:
        import gemmi  # imported here so gemmi is an optional dependency
    except ImportError:
        return None

    try:
        m = gemmi.read_ccp4_map(str(ccp4_path))
        m.setup(float("nan"))   # populate grid from file data (required before array access)
        grid = m.grid
        cell = grid.unit_cell

        arr = np.array(grid, dtype=np.float32)   # shape: (nu, nv, nw)
        nu, nv, nw = arr.shape

        sigma = arr.std()
        if not np.isfinite(sigma) or sigma < 1e-10:
            return None
        arr_norm = (arr - arr.mean()) / sigma

        # Fractionalization matrix: invert the orthogonalization matrix (always available)
        # cell.orth.mat converts fractional → Cartesian; its inverse goes Cartesian → fractional
        orth_mat = np.array(cell.orth.mat.tolist())   # (3, 3)
        frac_M = np.linalg.inv(orth_mat)              # fractionalization: f = frac_M @ x
        return arr_norm, frac_M.T, nu, nv, nw
    except Exception as e:
        print(f"  [warn] _load_grid failed for {ccp4_path.name}: {e}")
        return None


def _crop_density(
    arr_norm: np.ndarray,
    frac_matrix_T: np.ndarray,
    nu: int,
    nv: int,
    nw: int,
    center_cart: np.ndarray,
    G: int = _GRID_DIM,
    res: float = _RESOLUTION,
    transform: Optional[tuple] = None,
) -> np.ndarray:
    """Trilinear interpolation of CCP4 grid onto a G³ box centred at center_cart.

    Periodic boundary conditions are handled via scipy.ndimage.map_coordinates
    with mode='wrap', which is correct for crystallographic unit-cell maps.

    Parameters
    ----------
    arr_norm      : (nu, nv, nw) float32 z-score normalised density
    frac_matrix_T : (3, 3) fractionalization matrix transposed (x @ frac_matrix_T = f)
    center_cart   : (3,) Cartesian Å position of the ligand centroid
    G, res        : grid dimension and resolution (Å/voxel)
    transform     : optional (R, t) rigid frame correction, applied as
                    x_dep = x_cd @ R.T + t before the map is sampled. Recovered
                    per structure by residue-matched Kabsch vs the deposited PDB
                    (see dataset/00d_align_xray_density.py). None = no correction.

    Returns
    -------
    (G, G, G) float32 density array
    """
    # Cartesian offsets for each voxel in the output box (relative to centre)
    # voxel i → offset = (i - G/2) * res
    offsets = (np.arange(G, dtype=np.float32) - G * 0.5) * res

    # Build (G³, 3) array of Cartesian coordinates
    gx, gy, gz = np.meshgrid(
        offsets + center_cart[0],
        offsets + center_cart[1],
        offsets + center_cart[2],
        indexing="ij",
    )
    cart_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (G³, 3)

    # Optional rigid frame correction: CrossDocked frame → deposited (map) frame.
    # CrossDocked2020 superposes each pocket's receptors onto a shared reference
    # frame that differs from the deposited crystal frame the CCP4 map lives in;
    # without this the crop would sample the wrong region. transform = (R, t).
    if transform is not None:
        R, t = transform
        cart_pts = cart_pts @ np.asarray(R, dtype=cart_pts.dtype).T \
            + np.asarray(t, dtype=cart_pts.dtype)

    # Cartesian → fractional  (row vectors × M.T)
    frac_pts = cart_pts @ frac_matrix_T  # (G³, 3)

    # Fractional → CCP4 grid indices (continuous, wrap-periodic)
    ci = (frac_pts[:, 0] * nu) % nu
    cj = (frac_pts[:, 1] * nv) % nv
    ck = (frac_pts[:, 2] * nw) % nw

    # Trilinear interpolation with crystallographic periodic boundary
    density = scipy.ndimage.map_coordinates(
        arr_norm, [ci, cj, ck], order=1, mode="wrap", prefilter=False
    )
    return density.reshape(G, G, G).astype(np.float32)


def _rotate_density(
    density: np.ndarray,
    rot_matrix: torch.Tensor,
    G: int = _GRID_DIM,
) -> np.ndarray:
    """Apply 3D rotation to a voxel density array.

    Uses scipy.ndimage.affine_transform to rotate the volume by the same
    rotation matrix that was applied to the atom coordinates.

    For atom coords:    coords_new = coords @ R.T   (right-multiply)
    For voxel volume:   output[R @ x] = input[x]
    In ndimage terms:   matrix = R.T  (maps output voxel → input voxel)

    Parameters
    ----------
    density    : (G, G, G) float32 density array
    rot_matrix : (3, 3) float32 rotation matrix (same as used in rotate_coords)

    Returns
    -------
    (G, G, G) float32 rotated density
    """
    R = rot_matrix.numpy()          # (3, 3)
    c = np.full(3, G * 0.5)        # grid centre in voxel coords
    # offset so rotation is about the grid centre: input = R.T @ (output - c) + c
    offset = c - R.T @ c

    rotated = scipy.ndimage.affine_transform(
        density, R.T, offset=offset,
        order=1, mode="constant", cval=0.0,
    )
    return rotated.astype(np.float32)


def _translate_density(
    density: np.ndarray,
    translation: torch.Tensor,
    res: float = _RESOLUTION,
) -> np.ndarray:
    """Sub-voxel shift of a density volume by `translation` Å.

    Mirrors `translate_coords`' additive shift on atom coords: when atom
    positions move by +translation Å, the voxel that represents each atom's
    position shifts by +translation/res voxels — so the density volume must
    shift by the same amount to stay co-aligned with the atoms.
    """
    shift_voxels = (translation.detach().cpu().numpy().reshape(3) / res).tolist()
    shifted = scipy.ndimage.shift(
        density, shift=shift_voxels, order=1, mode="constant", cval=0.0,
    )
    return shifted.astype(np.float32)


# ── Local normalization ─────────────────────────────────────────────────────────

def normalize_crop(crop: np.ndarray) -> np.ndarray:
    """Apply local normalization to a single 64³ density crop.

    Steps:
      1. Compute local mean and std.
      2. Clip at ±3σ to suppress outlier electron-density peaks
         (heavy atoms, crystal contacts, refinement artifacts).
      3. Re-z-score using the pre-clip statistics → output mean ≈ 0, std ≈ 1.

    Using pre-clip statistics for the final division is intentional:
    the clipped voxels land at exactly ±3 in the output, which is
    deterministic and consistent across samples.

    Parameters
    ----------
    crop : (G, G, G) float32 or float16 array (globally z-scored by _load_grid)

    Returns
    -------
    (G, G, G) float32 array, locally z-score normalised
    """
    crop = crop.astype(np.float32)
    mu = crop.mean()
    sigma = crop.std()
    if not np.isfinite(sigma) or sigma < 1e-10:
        return crop - mu   # flat / empty crop → return all-zeros
    crop_clipped = np.clip(crop, mu - 3.0 * sigma, mu + 3.0 * sigma)
    return (crop_clipped - mu) / sigma


# ── Dataset class ───────────────────────────────────────────────────────────────

class DatasetCrossDockedXray(Dataset):
    """CrossDocked2020 dataset augmented with X-ray 2Fo-Fc electron density.

    Two density loading modes (mutually exclusive, crops_dir takes priority):

    crops_dir (preferred for training)
        Load precomputed float16 numpy arrays produced by dataset/00b_data_density_preprocess.py.
        Fast — no CCP4 I/O at training time. Rotation is still applied on-the-fly.

    ccp4_dir (on-the-fly, no preprocessing required)
        Load and interpolate CCP4 maps lazily with a per-worker LRU cache.
        Slower but requires only the raw downloaded .map files.

    Parameters
    ----------
    data_dir      : directory containing data_train.pt / data_test.pt
    crops_dir     : directory of precomputed crops (output of dataset/00b_data_density_preprocess.py).
                    When set, ccp4_dir is ignored.
    ccp4_dir      : directory containing {pdb_id}.map CCP4 files (on-the-fly mode)
    split         : "train", "val", or "test"
    use_xray      : if False, behaves identically to DatasetCrossdocked
    cache_size    : CCP4 grids kept in per-worker memory (on-the-fly mode only)
    normalize     : if True (default), apply local ±3σ clip + z-score to each
                    density crop via normalize_crop(). Set False when crops_dir
                    already contains pre-normalized crops.
    aug           : apply random rotation + translation augmentation
    small         : load only the first 500 samples (debug)
    ligand_radius : Gaussian blob radius for ligand atoms
    pocket_radius : Gaussian blob radius for pocket atoms (-1 = element-wise)
    max_len       : discard ligands with max coordinate extent > max_len Å
    delta_translate : max translation magnitude for augmentation (Å)
    verbose       : print filtering stats
    """

    def __init__(
        self,
        data_dir: str = "dataset/data",
        crops_dir: str = "",
        ccp4_dir: str = "dataset/data/ccp4",
        split: str = "train",
        use_xray: bool = True,
        cache_size: int = 32,
        aug: bool = True,
        small: bool = False,
        ligand_radius: float = 0.5,
        pocket_radius: float = -1,
        max_len: int = 30,
        delta_translate: float = 1.0,
        normalize: bool = True,
        verbose: bool = False,
        subset_n: Optional[int] = None,
        subset_xray_only: bool = False,
        subset_val_n: Optional[int] = None,
    ):
        assert split in ("train", "val", "test")

        self.data_dir = data_dir
        self.ccp4_dir = Path(ccp4_dir)
        self.split = split
        self.use_xray = use_xray
        self.cache_size = cache_size
        self.aug = aug
        self.small = small
        self.ligand_radius = ligand_radius
        self.pocket_radius = pocket_radius
        self.max_len = max_len
        self.delta = delta_translate
        self.normalize = normalize
        self.verbose = verbose

        # When subset_val_n is set, the val split is carved from the TRAIN
        # slice's x-ray pool — held out after the train subset (so train and
        # val are disjoint) and reusing the train crops. `phys_split` is the
        # split whose data + crops are physically read from disk; `split`
        # stays "val" for the DataLoader / augmentation semantics.
        self._val_from_train_pool = (split == "val" and subset_val_n is not None)
        phys_split = "train" if self._val_from_train_pool else split

        # Precomputed crops mode: load float16 .npy files produced by dataset/00b_data_density_preprocess.py
        _crops_path = Path(crops_dir) if crops_dir else None
        if use_xray and _crops_path and (_crops_path / phys_split).exists():
            self._crops_dir = _crops_path / phys_split
            avail_path = _crops_path / f"{phys_split}_available.npy"
            self._crops_available = np.load(str(avail_path)) if avail_path.exists() else None
            self._use_crops = True
        else:
            self._crops_dir = None
            self._crops_available = None
            self._use_crops = False

        # Per-worker LRU cache for loaded CCP4 grids (on-the-fly mode)
        self._grid_cache: OrderedDict = OrderedDict()

        # Precomputed lookup table for per-element pocket radii
        self.pocket_radius_lut: Optional[torch.Tensor] = None
        if self.pocket_radius <= 0:
            elements = list(ELEMENTS_HASH_CROSSDOCKED.keys())
            self.pocket_radius_lut = torch.tensor(
                [RADIUS_PER_ATOM["MOL"][e] for e in elements],
                dtype=torch.float32,
            )

        # Load data splits (same logic as DatasetCrossdocked). When the val
        # split is carved from the train pool, phys_split == "train" so it
        # reads the train slice of data_train.pt.
        import os
        if phys_split in ("train", "val"):
            import random as _random
            data = torch.load(os.path.join(data_dir, "data_train.pt"), weights_only=False)
            _random.Random(1234).shuffle(data)
            val_sz = 100
            self.data = data[: len(data) - val_sz] if phys_split == "train" else data[len(data) - val_sz :]
        else:
            self.data = torch.load(os.path.join(data_dir, "data_test.pt"), weights_only=False)

        self.data = self.data[:500] if small else self.data
        self._filter_by_size(max_len)

        # Set of available PDB IDs (to avoid file-system hits per sample)
        if use_xray and self.ccp4_dir.exists():
            self._available_pdbs = {p.stem for p in self.ccp4_dir.glob("*.map")}
        else:
            self._available_pdbs = set()

        # Optional subset selection. Train split: keep the first subset_n
        # samples (optionally x-ray-only). Val split with subset_val_n set
        # (_val_from_train_pool): keep subset_val_n samples held out from the
        # SAME x-ray pool, starting after the train subset — so train and val
        # are disjoint and val reuses the train crops. Stored as
        # new_index → original_index so __getitem__ loads crops by position.
        self._subset_indices: Optional[list] = None
        if (split == "train" and subset_n is not None) or self._val_from_train_pool:
            n_total = len(self.data)
            if subset_xray_only:
                if self._crops_available is None:
                    raise RuntimeError(
                        "subset_xray_only=True requires precomputed crops with "
                        f"{phys_split}_available.npy in crops_dir={crops_dir}"
                    )
                avail = self._crops_available
                if len(avail) != n_total:
                    raise RuntimeError(
                        f"crops_available length ({len(avail)}) != filtered dataset "
                        f"size ({n_total}); preprocessing is misaligned."
                    )
                pool = [i for i in range(n_total) if bool(avail[i])]
            else:
                pool = list(range(n_total))

            if self._val_from_train_pool:
                # held-out val: skip the first subset_n (train) pool entries
                start = subset_n or 0
                kept = pool[start: start + subset_val_n]
                want, label = subset_val_n, "subset_val_n"
            else:
                kept = pool[:subset_n]
                want, label = subset_n, "subset_n"

            if len(kept) < want:
                raise RuntimeError(
                    f"requested {label}={want} but only {len(kept)} samples match "
                    f"(subset_xray_only={subset_xray_only}, n_total={n_total}, split={split})"
                )
            self._subset_indices = kept

        if verbose:
            mode = "crops" if self._use_crops else "on-the-fly CCP4"
            avail_n = (int(self._crops_available.sum())
                       if self._crops_available is not None
                       else len(self._available_pdbs))
            print(
                f"DatasetCrossDockedXray | split={split} | n={len(self.data):,} | "
                f"use_xray={use_xray} | mode={mode} | available={avail_n:,}"
            )

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _filter_by_size(self, max_len: int) -> None:
        filtered = [(p, l) for p, l in self.data if l["max_len"] <= max_len]
        if self.verbose:
            print(f"| filter_by_size: {len(self.data):,} → {len(filtered):,}")
        self.data = filtered

    def _get_pdb_id(self, pocket_id: str) -> str:
        """Extract lowercase PDB ID from a pocket sample ID string."""
        # Format: "crossdocked/{PDBID}_{chain}_rec_..."
        return pocket_id.split("/")[1].split("_")[0].lower()

    def _get_grid(self, pdb_id: str):
        """Return cached CCP4 grid or load from disk (LRU eviction)."""
        if pdb_id in self._grid_cache:
            # Move to end (most recently used)
            self._grid_cache.move_to_end(pdb_id)
            return self._grid_cache[pdb_id]

        ccp4_path = self.ccp4_dir / f"{pdb_id}.map"
        result = _load_grid(ccp4_path) if ccp4_path.exists() else None

        # Evict oldest entry if cache is full
        if len(self._grid_cache) >= self.cache_size:
            self._grid_cache.popitem(last=False)

        self._grid_cache[pdb_id] = result
        return result

    # ── Public API ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        if self._subset_indices is not None:
            return len(self._subset_indices)
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        if self._subset_indices is not None:
            index = self._subset_indices[index]
        pocket_, ligand_ = self.data[index]

        # ── Ligand ────────────────────────────────────────────────────────────
        mask = ligand_["atoms_channel"] < 7
        ligand = {
            "id": ligand_["id"],
            "coords": ligand_["coords"][mask],
            "atoms_channel": ligand_["atoms_channel"][mask],
            "radius": torch.full_like(
                ligand_["atoms_channel"][mask], self.ligand_radius, dtype=torch.float32
            ),
        }
        # Ligand centroid in PDB Cartesian frame (used for density crop)
        center_coords = ligand["coords"].mean(dim=0)

        # ── Pocket ────────────────────────────────────────────────────────────
        mask = pocket_["atoms_channel"] < 4
        poc_ch = pocket_["atoms_channel"][mask]
        if self.pocket_radius > 0:
            radius = torch.full_like(poc_ch, self.pocket_radius, dtype=torch.float32)
        else:
            radius = self.pocket_radius_lut[poc_ch.long()]
        pocket = {
            "id": pocket_["id"],
            "coords": pocket_["coords"][mask],
            "atoms_channel": poc_ch,
            "radius": radius,
            "center_coords": center_coords,
        }

        # ── Centre both structures on the ligand centroid ─────────────────────
        ligand, pocket = recenter_structures(ligand, pocket, center_coords)

        # ── Augmentation (rotation then translation) ──────────────────────────
        # Translation is inlined (vs. translate_coords) so we can capture the
        # noise vector and apply the same shift to the density volume below.
        rot_matrix: Optional[torch.Tensor] = None
        trans_noise: Optional[torch.Tensor] = None
        if self.aug:
            rot_matrix = random_rot_matrix()
            ligand["coords"] = ligand["coords"].reshape(-1, 3) @ rot_matrix.T
            pocket["coords"] = pocket["coords"].reshape(-1, 3) @ rot_matrix.T
            trans_noise = (torch.rand((1, 3), dtype=ligand["coords"].dtype) - 0.5) * 2 * self.delta
            pocket["coords"].add_(trans_noise)
            ligand["coords"].add_(trans_noise)

        # ── Box / pad ─────────────────────────────────────────────────────────
        ligand, pocket = filter_atoms_by_distance(ligand, pocket)
        ligand, pocket = pad(ligand, pocket)

        # ── X-ray density ─────────────────────────────────────────────────────
        xray_density: Optional[torch.Tensor] = None
        if self.use_xray:
            if self._use_crops:
                # Fast path: load precomputed float16 numpy crop
                if self._crops_available is None or self._crops_available[index]:
                    crop_path = self._crops_dir / f"{index:06d}.npy"
                    if crop_path.exists():
                        density_np = np.load(str(crop_path)).astype(np.float32)
                        if self.normalize:
                            density_np = normalize_crop(density_np)
                        if rot_matrix is not None:
                            density_np = _rotate_density(density_np, rot_matrix)
                        if trans_noise is not None:
                            density_np = _translate_density(density_np, trans_noise)
                        xray_density = torch.from_numpy(density_np)
            else:
                # On-the-fly path: load CCP4, crop, interpolate
                pdb_id = self._get_pdb_id(pocket_["id"])
                if pdb_id in self._available_pdbs:
                    grid = self._get_grid(pdb_id)
                    if grid is not None:
                        arr_norm, frac_T, nu, nv, nw = grid
                        center_np = center_coords.numpy()
                        density_np = _crop_density(arr_norm, frac_T, nu, nv, nw, center_np)
                        if self.normalize:
                            density_np = normalize_crop(density_np)
                        if rot_matrix is not None:
                            density_np = _rotate_density(density_np, rot_matrix)
                        if trans_noise is not None:
                            density_np = _translate_density(density_np, trans_noise)
                        xray_density = torch.from_numpy(density_np)

        # Always return a tensor (zeros when unavailable) so PyTorch default
        # collate_fn can stack the batch without issue.
        if xray_density is None:
            xray_density = torch.zeros(_GRID_DIM, _GRID_DIM, _GRID_DIM)
            xray_available = torch.tensor(False)
        else:
            xray_available = torch.tensor(True)

        return {
            "pocket": pocket,
            "ligand": ligand,
            "xray_density": xray_density,    # (G, G, G) float32
            "xray_available": xray_available, # scalar bool
        }
