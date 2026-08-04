"""crossdocked_density_box.py — fast precomputed-box density loader.

Drop-in twin of DatasetCrossDockedDensity that reads a precomputed 96³ canonical-pose
density box (dataset/plinder/05_make_boxes.py) instead of the full ccp4 map, then resamples
the 64³ at the AUGMENTED pose from that box. Identical aug semantics — the SAME per-sample
(R_aug, t_aug) drive the atoms and the box resample — but no full-map read/parse per step,
so the GPU stops starving (~40% → ~85% util).

Equivalence: with aug OFF, the 64³ is the box's central 64³ = exactly OTF's canonical 64³
(integer offsets → no interpolation). With aug ON there is one extra trilinear interp
(box → rotated 64³) vs OTF's single map→64³; rotated-in voxels beyond the 96³ box zero-fill
(8-voxel / 2 Å margin per side covers moderate rotations).

Select via dset.box_path in the config (see dataset/__init__._make_dataset).
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.ndimage
import torch

from voxbind.dataset.crossdocked_density import (
    DatasetCrossDockedDensity,
    _apply_recipe_norm,
    _GRID_DIM,
    _RESOLUTION,
    gradient_magnitude3d,
    per_sample_zscore,
    random_rot_matrix,
    recenter_structures,
    filter_atoms_by_distance,
    pad,
)


def _resample_from_box(box, R_aug, t_aug, G_out=_GRID_DIM, G_box=96, res=_RESOLUTION):
    """Crop G_out³ from the centroid-centered canonical box at the augmented pose.

    Mirrors crossdocked_density._resample_density's offset transform EXACTLY (so the
    augmentation co-aligns with the atoms), but indexes the box (crop frame) instead of the
    full map — no Kabsch / fractional step. Zero-fills (mode=constant) voxels rotated past
    the box edge. With R_aug=None this returns the box's central G_out³ verbatim.
    """
    off = (np.arange(G_out, dtype=np.float32) - G_out * 0.5) * res
    gx, gy, gz = np.meshgrid(off, off, off, indexing="ij")
    o = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)      # (G_out³, 3)
    if R_aug is not None:
        if t_aug is not None:
            o = o - np.asarray(t_aug, dtype=o.dtype).reshape(1, 3)
        o = o @ np.asarray(R_aug, dtype=o.dtype)                    # == _resample_density
    idx = o / res + G_box * 0.5                                     # crop-frame Å → box voxel
    dens = scipy.ndimage.map_coordinates(
        box, [idx[:, 0], idx[:, 1], idx[:, 2]], order=1, mode="constant", cval=0.0, prefilter=False)
    return dens.reshape(G_out, G_out, G_out).astype(np.float32)


class DatasetCrossDockedDensityBox(DatasetCrossDockedDensity):
    """DatasetCrossDockedDensity, but density comes from a precomputed 96³ box memmap."""

    def __init__(self, *, box_path: str, resample_dir_diff: str = "", box_path_diff: str = "", **kwargs):
        super().__init__(**kwargs)
        meta = json.loads((Path(box_path).with_name(Path(box_path).stem + "_meta.json")).read_text())
        self._g_box = int(meta["g_box"])
        self._box_n = int(meta["n"])
        self._box_path = box_path
        # storage dtype: 'float16' (raw) or 'uint8' (affine-quantised raw over [quant_lo, quant_hi],
        # dequantised on read). uint8 halves the memmap so a 116³ box fits RAM page cache.
        self._box_dtype = np.dtype(meta.get("dtype", "float16"))
        self._qlo = meta.get("quant_lo"); self._qhi = meta.get("quant_hi")
        self._box = None  # opened lazily per worker (fork-safe)

        # ── Optional 2nd density source: mFo-DFc difference-map box (SAME pose manifest) ──
        # resample_dir_diff carries the diff normalization recipe; its box{g}.dat is read too
        # (box_path_diff overrides the path). Emits xray_diff_density / xray_diff_gradmag.
        self._box_diff = None
        self._recipe_norm_diff = None
        if resample_dir_diff:
            rdd = Path(resample_dir_diff)
            self._recipe_norm_diff = json.loads((rdd / "resample.json").read_text())["normalization"]
            bpd = box_path_diff or str(rdd / Path(box_path).name)   # same box{g}.dat filename
            meta_d = json.loads(Path(bpd).with_name(Path(bpd).stem + "_meta.json").read_text())
            self._box_diff_path = bpd
            self._box_diff_dtype = np.dtype(meta_d.get("dtype", "float16"))
            self._box_diff_qlo = meta_d.get("quant_lo"); self._box_diff_qhi = meta_d.get("quant_hi")
            self._box_diff_g = int(meta_d["g_box"])
            self._box_diff_n = int(meta_d["n"])

    def _boxes(self):
        if self._box is None:
            self._box = np.memmap(self._box_path, dtype=self._box_dtype, mode="r",
                                  shape=(self._box_n, self._g_box, self._g_box, self._g_box))
        return self._box

    def _read_box(self, index: int) -> np.ndarray:
        """row → float32 raw-density box (dequantise uint8; float16 passes through)."""
        raw = self._boxes()[index]
        if self._box_dtype == np.uint8:
            return self._qlo + (raw.astype(np.float32) / 255.0) * (self._qhi - self._qlo)
        return np.asarray(raw, dtype=np.float32)

    def _boxes_diff(self):
        if self._box_diff is None:
            self._box_diff = np.memmap(self._box_diff_path, dtype=self._box_diff_dtype, mode="r",
                                       shape=(self._box_diff_n, self._box_diff_g, self._box_diff_g, self._box_diff_g))
        return self._box_diff

    def _read_box_diff(self, index: int) -> np.ndarray:
        """row → float32 raw mFo-DFc box (dequantise uint8; float16 passes through)."""
        raw = self._boxes_diff()[index]
        if self._box_diff_dtype == np.uint8:
            return self._box_diff_qlo + (raw.astype(np.float32) / 255.0) * (self._box_diff_qhi - self._box_diff_qlo)
        return np.asarray(raw, dtype=np.float32)

    def __getitem__(self, index: int) -> dict:
        if self._subset_indices is not None:
            index = self._subset_indices[index]
        pocket_, ligand_ = self.data[index]

        # ── Ligand / pocket (identical to DatasetCrossDockedDensity) ──────────────
        mask = ligand_["atoms_channel"] < self.n_lig_ch
        if "radius" in ligand_:
            lig_radius = ligand_["radius"][mask].to(torch.float32)
        elif self.ligand_radius > 0:
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
        center_coords = ligand["coords"].mean(dim=0)

        mask = pocket_["atoms_channel"] < self.n_poc_ch
        poc_ch = pocket_["atoms_channel"][mask]
        if "radius" in pocket_:
            radius = pocket_["radius"][mask].to(torch.float32)
        elif self.pocket_radius > 0:
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

        # ── Augmentation (the SAME R_aug/t_aug drive atoms AND the box resample) ──
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

        # ── X-ray density: RESAMPLE 64³ from the precomputed 96³ box at the aug pose ──
        xray_density: Optional[torch.Tensor] = None
        if self.use_xray and bool(self._man_ok[index]):
            box = self._read_box(index)
            R_aug = rot_matrix.numpy() if rot_matrix is not None else None
            t_aug = trans_noise.reshape(3).numpy() if trans_noise is not None else None
            dens = _resample_from_box(box, R_aug, t_aug, G_out=_GRID_DIM, G_box=self._g_box, res=_RESOLUTION)
            if not getattr(self, "raw_density", False):
                dens = _apply_recipe_norm(dens, self._recipe_norm)   # skip → raw (box-clipped to quant range)
            xray_density = torch.from_numpy(dens.astype(np.float32))

        if xray_density is None:
            xray_density = torch.zeros(_GRID_DIM, _GRID_DIM, _GRID_DIM)
            xray_available = torch.tensor(False)
        else:
            xray_available = torch.tensor(True)

        if (self.gradmag_as_density and bool(xray_available)):
            g = gradient_magnitude3d(xray_density.view(1, 1, _GRID_DIM, _GRID_DIM, _GRID_DIM))
            xray_density = per_sample_zscore(g).view(_GRID_DIM, _GRID_DIM, _GRID_DIM)

        out = {
            "pocket": pocket,
            "ligand": ligand,
            "xray_density": xray_density,
            "xray_available": xray_available,
        }

        if self.return_gradmag:
            if bool(xray_available):
                g = gradient_magnitude3d(
                    xray_density.view(1, 1, _GRID_DIM, _GRID_DIM, _GRID_DIM))
                out["xray_gradmag"] = per_sample_zscore(g).view(_GRID_DIM, _GRID_DIM, _GRID_DIM)
            else:
                out["xray_gradmag"] = torch.zeros(_GRID_DIM, _GRID_DIM, _GRID_DIM)

        # ── 2nd density source: mFo-DFc difference map, resampled from its own box at the
        #    SAME augmented pose (rot_matrix/trans_noise), own recipe-norm + own gradmag. ──
        if self._recipe_norm_diff is not None:
            if bool(xray_available):
                R_aug = rot_matrix.numpy() if rot_matrix is not None else None
                t_aug = trans_noise.reshape(3).numpy() if trans_noise is not None else None
                boxd = self._read_box_diff(index)
                dd = _resample_from_box(boxd, R_aug, t_aug, G_out=_GRID_DIM,
                                        G_box=self._box_diff_g, res=_RESOLUTION)
                dd = torch.from_numpy(_apply_recipe_norm(dd, self._recipe_norm_diff).astype(np.float32))
                out["xray_diff_density"] = dd
                out["xray_diff_gradmag"] = per_sample_zscore(
                    gradient_magnitude3d(dd.view(1, 1, _GRID_DIM, _GRID_DIM, _GRID_DIM))
                ).view(_GRID_DIM, _GRID_DIM, _GRID_DIM)
            else:
                out["xray_diff_density"] = torch.zeros(_GRID_DIM, _GRID_DIM, _GRID_DIM)
                out["xray_diff_gradmag"] = torch.zeros(_GRID_DIM, _GRID_DIM, _GRID_DIM)

        return out
