"""crossdocked_density.py — on-the-fly RESAMPLE density loader for pretraining.

Runtime twin of crossdocked_xray.py, but it consumes the new dataset/build pipeline's
`s3 --resample` outputs ({split}_manifest.npz + resample.json) and crops the density from
the FULL CCP4 map at the AUGMENTED pose **each step** — a single trilinear interpolation
with no edge zero-fill, replacing crossdocked_xray's lossy crop-then-_rotate_density path.

It SUBCLASSES DatasetCrossDockedXray so all the validated, augmentation-sensitive machinery
(tuple loading, size filter, radius LUTs, recenter, the rot+trans aug applied to the coords,
filter_atoms_by_distance/pad, gradmag) is reused unchanged; only the density branch is new.
The original crossdocked_xray.py is left untouched.

Inputs (from dataset/build `s3 --resample`, i.e. `00b --mode resample`):
    <resample_dir>/resample.json          grid + normalization recipe + ccp4 path + sampling note
    <resample_dir>/{split}_manifest.npz   pdb_id, centroid, R, t, ok   (positional, dataset order)
plus the tuples (`data_file`) and the retained FULL CCP4 maps.

Co-alignment: the SAME per-sample (R_aug, t_aug) drives both the atom coords (p → p @ R_aug.T
+ t_aug) and the density resample; an output voxel offset o maps to map position
    cart = (centroid + (o − t_aug) @ R_aug) @ R_kabsch.T + t_kabsch
i.e. the inverse of the coord aug composed with the Kabsch crop→map frame correction.
With aug off this reduces exactly to crossdocked_xray._crop_density(transform=(R,t)).

Usage
-----
    from voxbind.dataset.crossdocked_density import DatasetCrossDockedDensity
    dset = DatasetCrossDockedDensity(
        resample_dir="dataset/data/xray_resample_v5",   # output of s3 --resample
        data_dir="dataset/data", split="train",
        data_file="data_train.pt", subset_n=78428, return_gradmag=True,
    )
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.ndimage
import torch

from voxbind.dataset.crossdocked_xray import (
    DatasetCrossDockedXray,
    _GRID_DIM,
    _RESOLUTION,
)
from voxbind.models.density_mae import gradient_magnitude3d, per_sample_zscore
from voxbind.utils.dataset_utils import (
    filter_atoms_by_distance,
    pad,
    random_rot_matrix,
    recenter_structures,
)


# ── raw grid + augmented-offset resampler ───────────────────────────────────────

def _load_raw_grid(ccp4_path: Path):
    """Like crossdocked_xray._load_grid but returns the RAW (un-z-scored) array.

    The resample recipe (v5 arcsinh+z, etc.) is fit on RAW map values by 00b poolnorm,
    so the resampled crop must be normalized from the raw map — not the per-map z-scored
    array _load_grid returns. Returns (arr, frac_matrix_T, nu, nv, nw) or None.
    """
    try:
        import gemmi
    except ImportError:
        return None
    try:
        m = gemmi.read_ccp4_map(str(ccp4_path))
        m.setup(float("nan"))
        grid = m.grid
        arr = np.array(grid, dtype=np.float32)
        nu, nv, nw = arr.shape
        orth_mat = np.array(grid.unit_cell.orth.mat.tolist())  # fractional → Cartesian
        frac_M = np.linalg.inv(orth_mat)                       # Cartesian → fractional
        return arr, frac_M.T, nu, nv, nw
    except Exception as e:
        print(f"  [warn] _load_raw_grid failed for {ccp4_path.name}: {e}")
        return None


def _resample_density(
    arr: np.ndarray,
    frac_matrix_T: np.ndarray,
    nu: int,
    nv: int,
    nw: int,
    center_cart: np.ndarray,
    R_kab: Optional[np.ndarray] = None,
    t_kab: Optional[np.ndarray] = None,
    R_aug: Optional[np.ndarray] = None,
    t_aug: Optional[np.ndarray] = None,
    G: int = _GRID_DIM,
    res: float = _RESOLUTION,
) -> np.ndarray:
    """Crop a G³ box from the full map at the AUGMENTED pose.

    Output voxel offset o (Å, grid-centred) maps to map position
        cart = (center_cart + (o − t_aug) @ R_aug) @ R_kab.T + t_kab
    `(o − t_aug) @ R_aug` is the inverse of the coord aug (coords: p → p @ R_aug.T + t_aug),
    so the density co-aligns with the augmented atoms; the Kabsch (R_kab, t_kab) then maps
    the CrossDocked crop frame → the deposited map frame. R_aug/t_aug=None ⇒ no aug ⇒
    identical sampling to crossdocked_xray._crop_density(transform=(R_kab, t_kab)).
    """
    off = (np.arange(G, dtype=np.float32) - G * 0.5) * res
    gx, gy, gz = np.meshgrid(off, off, off, indexing="ij")
    o = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (G³, 3) centred offsets (Å)

    if R_aug is not None:
        if t_aug is not None:
            o = o - np.asarray(t_aug, dtype=o.dtype).reshape(1, 3)
        o = o @ np.asarray(R_aug, dtype=o.dtype)          # row-vec @ R_aug ≡ R_aug.T @ col-vec

    cart = o + np.asarray(center_cart, dtype=o.dtype).reshape(1, 3)
    if R_kab is not None:
        cart = cart @ np.asarray(R_kab, dtype=o.dtype).T + np.asarray(t_kab, dtype=o.dtype).reshape(1, 3)

    frac = cart @ frac_matrix_T
    ci = (frac[:, 0] * nu) % nu
    cj = (frac[:, 1] * nv) % nv
    ck = (frac[:, 2] * nw) % nw
    dens = scipy.ndimage.map_coordinates(arr, [ci, cj, ck], order=1, mode="wrap", prefilter=False)
    return dens.reshape(G, G, G).astype(np.float32)


def _apply_recipe_norm(raw_crop: np.ndarray, norm: dict) -> np.ndarray:
    """Apply the baked, dataset-wide normalization recipe to a raw crop.

    Mirrors 00b poolnorm exactly (the recipe carries the pre-fit constants), so a resampled
    crop lands on the same scale as the frozen v{2..5} crops. v5 (arcsinh+z) is canonical.
    """
    x = raw_crop.astype(np.float32)
    scheme = (norm.get("scheme") or "").lower()
    if "arcsinh" in scheme:                                       # v5: (arcsinh(x/s) − μ_a)/σ_a
        s = float(norm["arcsinh_scale"])
        return (np.arcsinh(x / s) - float(norm["mu_a"])) / float(norm["sigma_a"])
    if "max-abs" in scheme or "max_abs" in norm:                  # v3: x / max_abs  → [−1, 1]
        return x / float(norm["max_abs"])
    if "clip" in scheme:                                          # v4: (clip(x) − μ_c)/σ_c
        return (np.clip(x, float(norm["clip_lo_raw"]), float(norm["clip_hi_raw"]))
                - float(norm["mu_clipped"])) / float(norm["sigma_clipped"])
    return (x - float(norm["mu"])) / float(norm["sigma"])         # v2: z-score


# ── Dataset class ────────────────────────────────────────────────────────────────

class DatasetCrossDockedDensity(DatasetCrossDockedXray):
    """Pretraining dataset that resamples density from the full map at the augmented pose.

    Same returned dict as DatasetCrossDockedXray ("pocket", "ligand", "xray_density",
    "xray_available", optional "xray_gradmag"). Availability + the per-sample crop frame
    come from the resample manifest (the `ok`, `centroid`, Kabsch `R`/`t` columns).
    """

    def __init__(
        self,
        *,
        resample_dir: str,
        data_dir: str = "dataset/data",
        split: str = "train",
        aug: bool = True,
        ligand_radius: float = 0.5,
        pocket_radius: float = -1,
        max_len: int = 30,
        delta_translate: float = 1.0,
        subset_n: Optional[int] = None,
        subset_xray_only: bool = True,
        subset_val_n: Optional[int] = None,
        return_gradmag: bool = False,
        data_file: str = "data_train.pt",
        small: bool = False,
        cache_size: int = 32,
        verbose: bool = False,
    ):
        # Parent loads tuples + size filter + radius LUTs and the per-worker grid cache.
        # Disable the parent's frozen-crop subset/normalize: we select via the resample
        # manifest's `ok` and normalize via the recipe, not via crops_dir.
        super().__init__(
            data_dir=data_dir, crops_dir="", ccp4_dir="", split=split, use_xray=True,
            cache_size=cache_size, aug=aug, small=small,
            ligand_radius=ligand_radius, pocket_radius=pocket_radius,
            max_len=max_len, delta_translate=delta_translate, normalize=False,
            verbose=False, subset_n=None, subset_xray_only=False, subset_val_n=subset_val_n,
            return_gradmag=return_gradmag, gradmag_crops_dir="", data_file=data_file,
        )

        rd = Path(resample_dir)
        recipe = json.loads((rd / "resample.json").read_text())
        if recipe.get("kind") != "resample_manifest":
            raise RuntimeError(f"{rd/'resample.json'} is not a resample manifest recipe.")
        self._recipe = recipe
        self._recipe_norm = recipe["normalization"]
        self.ccp4_dir = Path(recipe["ccp4_dir"])
        self._ccp4_ext = recipe.get("ccp4_ext", ".ccp4")

        # phys_split mirrors the parent: a val split carved from the train tail reads train.
        phys_split = "train" if (split == "val" and subset_val_n is not None) else split
        man = np.load(rd / f"{phys_split}_manifest.npz", allow_pickle=True)
        self._man_pdb = man["pdb_id"].astype(str)
        self._man_centroid = man["centroid"].astype(np.float32)
        self._man_R = man["R"].astype(np.float32)
        self._man_t = man["t"].astype(np.float32)
        self._man_ok = man["ok"].astype(bool)

        n_total = len(self.data)
        if len(self._man_ok) != n_total and not self.small:
            raise RuntimeError(
                f"resample manifest length ({len(self._man_ok)}) != filtered dataset size "
                f"({n_total}); s3 --resample and the loader disagree (same data_file + max_len?)."
            )

        # Subset selection using the manifest `ok` (replaces the parent's crop-based path).
        self._val_from_train_pool = (split == "val" and subset_val_n is not None)
        if (split == "train" and subset_n is not None) or self._val_from_train_pool:
            pool = ([i for i in range(n_total) if bool(self._man_ok[i])]
                    if subset_xray_only else list(range(n_total)))
            if self._val_from_train_pool:
                start = subset_n or 0
                kept = pool[start: start + subset_val_n]
                want, label = subset_val_n, "subset_val_n"
            else:
                kept = pool[:subset_n]
                want, label = subset_n, "subset_n"
            if len(kept) < want:
                raise RuntimeError(
                    f"requested {label}={want} but only {len(kept)} resample-available samples "
                    f"match (subset_xray_only={subset_xray_only}, n_total={n_total}, split={split})."
                )
            self._subset_indices = kept

        # Raw-grid LRU (separate from the parent's z-scored .map cache).
        self._raw_grid_cache: OrderedDict = OrderedDict()

        if verbose:
            print(
                f"DatasetCrossDockedDensity | split={split} | n={len(self):,} | "
                f"resample={rd.name} | norm={self._recipe_norm.get('scheme')} | "
                f"maps={self.ccp4_dir} | available={int(self._man_ok.sum()):,}"
            )

    def _get_raw_grid(self, pdb_id: str):
        """Cached RAW CCP4 grid (LRU), loaded from the recipe's ccp4_dir / ext."""
        if pdb_id in self._raw_grid_cache:
            self._raw_grid_cache.move_to_end(pdb_id)
            return self._raw_grid_cache[pdb_id]
        path = self.ccp4_dir / f"{pdb_id}{self._ccp4_ext}"
        result = _load_raw_grid(path) if path.exists() else None
        if len(self._raw_grid_cache) >= self.cache_size:
            self._raw_grid_cache.popitem(last=False)
        self._raw_grid_cache[pdb_id] = result
        return result

    def __getitem__(self, index: int) -> dict:
        if self._subset_indices is not None:
            index = self._subset_indices[index]
        pocket_, ligand_ = self.data[index]

        # ── Ligand / pocket (identical to crossdocked_xray) ──────────────────────
        mask = ligand_["atoms_channel"] < 7
        if self.ligand_radius > 0:
            lig_radius = torch.full_like(
                ligand_["atoms_channel"][mask], self.ligand_radius, dtype=torch.float32)
        else:
            lig_radius = self.ligand_radius_lut[ligand_["atoms_channel"][mask].long()]
        ligand = {
            "id": ligand_["id"],
            "coords": ligand_["coords"][mask],
            "atoms_channel": ligand_["atoms_channel"][mask],
            "radius": lig_radius,
        }
        center_coords = ligand["coords"].mean(dim=0)  # crop center == atom-recenter center

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

        ligand, pocket = recenter_structures(ligand, pocket, center_coords)

        # ── Augmentation (the SAME R_aug/t_aug drive atoms AND the density resample) ──
        rot_matrix: Optional[torch.Tensor] = None
        trans_noise: Optional[torch.Tensor] = None
        if self.aug:
            rot_matrix = random_rot_matrix()
            ligand["coords"] = ligand["coords"].reshape(-1, 3) @ rot_matrix.T
            pocket["coords"] = pocket["coords"].reshape(-1, 3) @ rot_matrix.T
            trans_noise = (torch.rand((1, 3), dtype=ligand["coords"].dtype) - 0.5) * 2 * self.delta
            pocket["coords"].add_(trans_noise)
            ligand["coords"].add_(trans_noise)

        ligand, pocket = filter_atoms_by_distance(ligand, pocket)
        ligand, pocket = pad(ligand, pocket)

        # ── X-ray density: RESAMPLE from the full map at the augmented pose ───────
        xray_density: Optional[torch.Tensor] = None
        if self.use_xray and bool(self._man_ok[index]):
            pdb_id = self._man_pdb[index]
            # Safety: the positional manifest row must line up with this sample.
            assert pdb_id == self._get_pdb_id(pocket_["id"]), (
                f"manifest/data misalignment at {index}: "
                f"{pdb_id} != {self._get_pdb_id(pocket_['id'])}"
            )
            grid = self._get_raw_grid(pdb_id)
            if grid is not None:
                arr, frac_T, nu, nv, nw = grid
                R_aug = rot_matrix.numpy() if rot_matrix is not None else None
                t_aug = trans_noise.reshape(3).numpy() if trans_noise is not None else None
                dens = _resample_density(
                    arr, frac_T, nu, nv, nw,
                    center_cart=center_coords.numpy(),
                    R_kab=self._man_R[index], t_kab=self._man_t[index],
                    R_aug=R_aug, t_aug=t_aug,
                )
                dens = _apply_recipe_norm(dens, self._recipe_norm)
                xray_density = torch.from_numpy(dens.astype(np.float32))

        if xray_density is None:
            xray_density = torch.zeros(_GRID_DIM, _GRID_DIM, _GRID_DIM)
            xray_available = torch.tensor(False)
        else:
            xray_available = torch.tensor(True)

        out = {
            "pocket": pocket,
            "ligand": ligand,
            "xray_density": xray_density,
            "xray_available": xray_available,
        }

        # gradmag ‖∇ρ‖ derived from the final (resampled) density, per-sample z-scored.
        if self.return_gradmag:
            if bool(xray_available):
                g = gradient_magnitude3d(
                    xray_density.view(1, 1, _GRID_DIM, _GRID_DIM, _GRID_DIM))
                out["xray_gradmag"] = per_sample_zscore(g).view(_GRID_DIM, _GRID_DIM, _GRID_DIM)
            else:
                out["xray_gradmag"] = torch.zeros(_GRID_DIM, _GRID_DIM, _GRID_DIM)

        return out
