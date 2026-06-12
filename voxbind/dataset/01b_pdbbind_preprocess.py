"""01b_pdbbind_preprocess.py — Preprocess PDBbind v2020 into training voxels.

Consolidated Phase 4 / 4b entry point. Two subcommands:

    cd voxbind
    python dataset/01b_pdbbind_preprocess.py voxelize   # atoms (+ density) → voxels/  (v1)
    python dataset/01b_pdbbind_preprocess.py poolnorm    # pocket-pool density → voxels_v2/, voxels_v3/, voxels_v4/

──────────────────────────────────────────────────────────────────────────────
voxelize — voxelize PDBbind v2020 refined complexes (atoms + v1 density crop)
──────────────────────────────────────────────────────────────────────────────
For each refined complex with has_struct=True in index.csv:
  1. Parse heavy atoms from {pid}_pocket.pdb (channels 0..3 = C,O,N,S)
     and from {pid}_ligand.sdf  (channels 0..6 = C,O,N,S,F,Cl,P).
  2. Compute the ligand heavy-atom geometric centre ("COM" in the existing
     voxel pipeline terminology).
  3. Recentre both atom sets to the ligand centre. Clamp to ±25 Å
     (matches voxbind.utils.dataset_utils.filter_atoms_by_distance).
  4. Voxelize with voxbind.voxelizer.Voxelizer (pyuul, 64³, 0.25 Å):
       - ligand : 7 channels, uniform 0.5 Å Gaussian radius
       - pocket : 4 channels, element-wise vdW radii
     and concatenate → (11, 64, 64, 64).  This is exactly the tensor
     `atomblob` encoder consumes during training.
  5. If a 2Fo-Fc CCP4 map is available (from `density` acquisition):
       - Load + globally z-score via _load_grid (gemmi).
       - Crop a 64³ × 0.25 Å box at the ligand centre via _crop_density
         with transform=None (PDBbind structures live in the deposited
         crystal frame, so no Kabsch alignment is needed).
       - Locally ±3σ-clip + re-z-score via normalize_crop.
     → (64, 64, 64) density, ready as the trailing channel for `atomblob_density`.

Centring on the ligand centre matches Beyond Atoms §3.1 ("both [grids] centered
on the ligand's center of mass") and the existing VoxBind training pipeline.
As in CrossDocked, this is implemented as an unweighted heavy-atom coordinate
mean, not a mass-weighted chemistry centre of mass.

    python dataset/01b_pdbbind_preprocess.py voxelize
    python dataset/01b_pdbbind_preprocess.py voxelize --max_complexes 20   # smoke test
    python dataset/01b_pdbbind_preprocess.py voxelize --no_density         # atoms only
    python dataset/01b_pdbbind_preprocess.py voxelize --device cpu         # if no GPU
  → voxels/atoms/{pid}.npy  voxels/density/{pid}.npy  voxels/availability.csv

──────────────────────────────────────────────────────────────────────────────
poolnorm — pocket-pool density normalisations (v2: z-score, v3: max-abs, v4: clip+z)
──────────────────────────────────────────────────────────────────────────────
Produces FOUR new density-crop variants from the raw CCP4 maps in
dataset/data/pdbbind/ccp4/, normalised against the pool of all pocket crops
(not per-map, not per-crop):

  v2 — pocket-pool z-score
       x' = (x − μ_pool) / σ_pool
       μ_pool, σ_pool computed over EVERY voxel of EVERY pocket crop concatenated.

  v3 — pocket-pool symmetric max-abs
       x' = x / max_abs_pool
       max_abs_pool = max(|x|) across the same pool. Output strict [−1, +1].

  v4 — pocket-pool clip + z-score
       x' = (clip(x, [CLIP_LO, CLIP_HI]) − μ_clip) / σ_clip
       Clip raw values to a fixed window (default −0.578 / +1.647, the CrossDocked
       pool P0.1%/P99.9%; valid here since the PDBbind raw pool range matches),
       then z-score over the CLIPPED pool (μ_clip, σ_clip computed directly from
       the clipped raw crops — no v2 round-trip). Bounds v2's bright metal/heavy-
       atom tail to a ≈ +5σ ceiling while keeping atom signal at unit scale that
       v3's max-abs collapses to ~0.01. Override the window with --clip_lo/--clip_hi.

  v5 — pocket-pool arcsinh soft-squash + z-score (NO clip)
       x' = (arcsinh(x / s) − μ_a) / σ_a
       Instead of clipping, squash the bright-peak tail SMOOTHLY: arcsinh(x/s) is
       ~linear for |x|<<s and ~log for |x|>>s, so the near-zero bulk is preserved
       and the +30 metal peaks compress to ≈+9 (s=0.5) with NO clip and NO pile-up
       spike — every voxel kept. Contrast: plain z-score (v2) leaves that peak at
       +92. Tune the squash with --v5_arcsinh_scale (smaller = harder).

All four schemes:
  - SKIP the per-map z-score that `_load_grid` does in v1 (use raw CCP4 values)
  - SKIP the per-crop ±3σ clip that `normalize_crop` does in v1
  - apply ONE dataset-wide statistic derived from all pocket crops

Atom voxels (11, 64, 64, 64) are NOT affected — v2/v3/v4/v5 reuse the existing
voxels/atoms/ directory via symlinks.

    python dataset/01b_pdbbind_preprocess.py poolnorm
  → voxels_v2/density/{pid}.npy  voxels_v2/atoms→../voxels/atoms  voxels_v2/stats.json
    voxels_v3/density/{pid}.npy  voxels_v3/atoms→../voxels/atoms  voxels_v3/stats.json
    voxels_v4/density/{pid}.npy  voxels_v4/atoms→../voxels/atoms  voxels_v4/stats.json
    voxels_v5/density/{pid}.npy  voxels_v5/atoms→../voxels/atoms  voxels_v5/stats.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Import gemmi BEFORE torch so its libstdc++.so.6 is loaded first. If torch
# (or PyUUL's CUDA libs) load an older libstdc++ ahead of gemmi, gemmi's C++
# extension fails with ImportError("... CXXABI_1.3.15 not found ...") inside
# every grid load and density crops silently return None.
# (see feedback_gemmi_torch_import_order memory)
import gemmi  # noqa: F401  (kept first to fix shared-library load order)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from voxbind.constants import ELEMENTS_HASH_CROSSDOCKED, RADIUS_PER_ATOM
from voxbind.voxelizer import Voxelizer
from voxbind.dataset.crossdocked_xray import _crop_density, normalize_crop


# ── Config ─────────────────────────────────────────────────────────────────────

PDBBIND_DIR  = Path(__file__).parent / "data" / "pdbbind"
INDEX_CSV    = PDBBIND_DIR / "index.csv"
STRUCT_DIR   = PDBBIND_DIR / "structures" / "pbpp-2020"
CCP4_DIR     = PDBBIND_DIR / "ccp4"
VOX_DIR      = PDBBIND_DIR / "voxels"
ATOMS_DIR_V1 = VOX_DIR / "atoms"
V2_DIR       = PDBBIND_DIR / "voxels_v2"
V3_DIR       = PDBBIND_DIR / "voxels_v3"
V4_DIR       = PDBBIND_DIR / "voxels_v4"
V5_DIR       = PDBBIND_DIR / "voxels_v5"

N_LIG_CH = 7   # C,O,N,S,F,Cl,P
N_POC_CH = 4   # C,O,N,S
LIGAND_CHANNEL_ELEMENTS = ["C", "O", "N", "S", "F", "Cl", "P"]
POCKET_CHANNEL_ELEMENTS = ["C", "O", "N", "S"]
LIGAND_SUPPORTED_HEAVY = set(LIGAND_CHANNEL_ELEMENTS)
POCKET_SUPPORTED_HEAVY = set(POCKET_CHANNEL_ELEMENTS)
GRID_DIM     = 64
RESOLUTION   = 0.25
LIGAND_RAD   = 0.5
CUBES_AROUND = 8
VOXELIZE_METADATA_VERSION = 2
CENTER_SOURCE = "ligand_sdf_all_non_h_geometric_centroid"

# v4 clip thresholds (raw 2Fo-Fc units): CrossDocked pool P0.1%/P99.9% (see 00b).
# Asymmetric — the positive tail is atom-peak signal, the negative tail is noise
# floor. Valid for PDBbind: its raw pool range (≈[-2.30, +30.30], voxels_v2/
# stats.json) matches CrossDocked's (≈[-2.62, +31.34]). Override via --clip_lo/_hi.
CLIP_LO = -0.578
CLIP_HI = +1.647

# v5 soft-squashes the bright-peak tail with arcsinh (NO clip → no pile-up spike),
# then z-scores. arcsinh(x/s) is ~linear for |x|<<s and ~log for |x|>>s, so the
# near-zero bulk is preserved and the +30 metal peaks compress smoothly (s=0.5 →
# raw +30 maps to ≈+9 after z-score, vs +92 for a plain z-score / v2). Every voxel
# is kept. Tune with --v5_arcsinh_scale (smaller s = harder tail squash).
V5_ARCSINH_SCALE = 0.5

DEFAULT_V5_REFERENCE_STATS = (
    Path(__file__).parent / "data" / "xray_crops_aligned_v5" / "stats.json"
)

# Element → channel index lookup (matches voxbind.constants.ELEMENTS_HASH_CROSSDOCKED)
_E2CH = ELEMENTS_HASH_CROSSDOCKED   # {"C":0, "O":1, "N":2, "S":3, "F":4, "Cl":5, "P":6, "H":7}
# Element-wise vdW radii (Å), shared by pocket atoms and — when --ligand_vdw is
# set — ligand atoms (mirrors crossdocked_xray's pocket / ligand_radius<=0 path).
_VDW_RADIUS_LUT = torch.tensor(
    [RADIUS_PER_ATOM["MOL"][e] for e in _E2CH.keys()],
    dtype=torch.float32,
)

# Hydrogen isotope labels to exclude from the ligand centering point. The voxel
# channels still use only C/O/N/S/F/Cl/P, but centering should not shift when a
# PDBbind ligand contains unsupported heavy atoms such as Br or I.
_HYDROGEN_ELEMS = {"H", "D", "T"}


# ═══════════════════════════════════════════════════════════════════════════════
# voxelize — atoms (11ch) + optional v1 density crop
# ═══════════════════════════════════════════════════════════════════════════════

def _load_grid(ccp4_path: Path):
    """Load a CCP4 map → (arr_norm, frac_M_T, nu, nv, nw) or None.

    v1 loader: applies a per-map global z-score before cropping.
    (See `load_raw_grid` for the un-normalised variant used by poolnorm.)
    """
    try:
        m = gemmi.read_ccp4_map(str(ccp4_path))
        m.setup(float("nan"))
        grid = m.grid
        cell = grid.unit_cell
        arr = np.array(grid, dtype=np.float32)
        nu, nv, nw = arr.shape
        sigma = arr.std()
        if not np.isfinite(sigma) or sigma < 1e-10:
            return None
        arr_norm = (arr - arr.mean()) / sigma
        orth_mat = np.array(cell.orth.mat.tolist())
        frac_M = np.linalg.inv(orth_mat)
        return arr_norm, frac_M.T, nu, nv, nw
    except Exception:
        return None


def _channel_of(element: str, max_ch: int):
    ch = _E2CH.get(element)
    if ch is None or ch >= max_ch:
        return None
    return ch


def _normalise_element(element: str) -> str:
    """Return a canonical element symbol, e.g. 'CL'/'cl' → 'Cl'."""
    element = element.strip()
    if not element:
        return ""
    return element[0].upper() + element[1:].lower()


def _is_heavy_element(element: str) -> bool:
    """True for non-hydrogen atom symbols, including unsupported channels."""
    element = _normalise_element(element)
    return bool(element) and element not in _HYDROGEN_ELEMS


def expected_voxelize_metadata(args: argparse.Namespace) -> dict:
    """Metadata that defines whether an atom/density voxel cache is reusable."""
    return {
        "metadata_version": VOXELIZE_METADATA_VERSION,
        "kind": "pdbbind_voxelize",
        "center_source": CENTER_SOURCE,
        "grid_dim": GRID_DIM,
        "resolution": RESOLUTION,
        "cubes_around": CUBES_AROUND,
        "ligand_channels": LIGAND_CHANNEL_ELEMENTS,
        "pocket_channels": POCKET_CHANNEL_ELEMENTS,
        "element_filter": args.element_filter,
        "ligand_vdw": bool(args.ligand_vdw),
        "ligand_radius": None if args.ligand_vdw else LIGAND_RAD,
        "pocket_radius": "element_vdw",
        "density_enabled": not bool(args.no_density),
        "density_mode": None if args.no_density else "v1_global_zscore_crop_clip_zscore",
        "index_csv": str(Path(args.index_csv).expanduser().resolve()),
        "struct_dir": str(Path(args.struct_dir).expanduser().resolve()),
        "ccp4_dir": None if args.no_density else str(Path(args.ccp4_dir).expanduser().resolve()),
    }


def validate_voxelize_cache(out_dir: Path, expected: dict, *, overwrite: bool,
                            allow_stale_cache: bool) -> None:
    """Fail fast when existing .npy caches were produced by a different recipe."""
    meta_path = out_dir / "metadata.json"
    has_cache = any((out_dir / name).exists() and any((out_dir / name).glob("*.npy"))
                    for name in ("atoms", "density"))
    if not has_cache:
        return
    if overwrite:
        return
    if not meta_path.exists():
        msg = (
            f"{out_dir} contains cached voxels but no metadata.json. "
            "Re-run with --overwrite to rebuild with the current all-heavy "
            "ligand centre, or pass --allow_stale_cache to reuse legacy files."
        )
        if allow_stale_cache:
            print(f"  [warn] {msg}")
            return
        raise RuntimeError(msg)

    actual = json.loads(meta_path.read_text())
    mismatches = [
        f"{k}: got {actual.get(k)!r}, expected {v!r}"
        for k, v in expected.items()
        if actual.get(k) != v
    ]
    if mismatches and not allow_stale_cache:
        msg = "\n    ".join(mismatches[:12])
        raise RuntimeError(
            f"{out_dir} cache metadata does not match this voxelize command:\n"
            f"    {msg}\n"
            "Re-run with --overwrite, or pass --allow_stale_cache if this is intentional."
        )
    if mismatches:
        print(f"  [warn] stale cache metadata in {meta_path}; reusing because --allow_stale_cache was set")


def _unsupported_heavy(elements: set[str], supported: set[str]) -> list[str]:
    return sorted(e for e in elements if _is_heavy_element(e) and e not in supported)


def parse_pocket_pdb(path: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Heavy-atom coords + 4-channel atom channels for a pocket PDB file."""
    coords, channels, heavy_elements = [], [], set()
    with path.open() as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            el = _normalise_element(line[76:78])
            if not el:
                # Fall back: first letter of atom name (cols 13–16)
                el = _normalise_element(''.join(c for c in line[12:16] if c.isalpha())[:1])
            if _is_heavy_element(el):
                heavy_elements.add(el)
            ch = _channel_of(el, N_POC_CH)
            if ch is None:
                continue
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            channels.append(ch)
    return (
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(channels, dtype=torch.long),
        _unsupported_heavy(heavy_elements, POCKET_SUPPORTED_HEAVY),
    )


def parse_ligand_sdf(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Supported ligand voxels plus all-heavy-atom centroid from SDF V2000.

    Returns
    -------
    coords, channels
        Only atoms represented by the 7-channel CrossDocked ligand vocabulary
        (C/O/N/S/F/Cl/P), used for voxelization.
    all_heavy_centroid
        Geometric centroid of every non-H ligand atom in the SDF. This is used
        as the PDBbind grid center so unsupported heavy atoms (Br/I/metals) do
        not move the atom and density crops relative to the physical ligand.
    """
    coords, channels, all_heavy_coords, heavy_elements = [], [], [], set()
    with path.open() as f:
        lines = f.readlines()
    n_atoms = int(lines[3][:3])
    for i in range(n_atoms):
        line = lines[4 + i]
        el = _normalise_element(line[31:34])
        if _is_heavy_element(el):
            heavy_elements.add(el)
            all_heavy_coords.append([float(line[0:10]), float(line[10:20]), float(line[20:30])])
        ch = _channel_of(el, N_LIG_CH)
        if ch is None:
            continue
        coords.append([float(line[0:10]), float(line[10:20]), float(line[20:30])])
        channels.append(ch)
    if not all_heavy_coords:
        raise ValueError(f"{path} contains no non-H ligand atoms")
    return (
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(channels, dtype=torch.long),
        torch.tensor(all_heavy_coords, dtype=torch.float32).mean(dim=0),
        _unsupported_heavy(heavy_elements, LIGAND_SUPPORTED_HEAVY),
    )


def voxelize_complex(
    lig_xyz: torch.Tensor, lig_ch: torch.Tensor,
    poc_xyz: torch.Tensor, poc_ch: torch.Tensor,
    voxelizer: Voxelizer,
    device: str,
    ligand_vdw: bool = False,
    ligand_center: "torch.Tensor | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ((11,G,G,G) atom voxels, (3,) ligand centroid in deposited frame).

    ligand_vdw=True gives ligand atoms element-wise vdW radii (instead of the
    uniform LIGAND_RAD blob), matching an encoder trained with ligand_radius<=0.
    """
    # Heavy-atom centroid in deposited frame (used later for density cropping).
    lig_com = ligand_center if ligand_center is not None else lig_xyz.mean(dim=0)

    # Recentre to the ligand centre, clamp to ±25 Å (mirrors filter_atoms_by_distance).
    lig_xyz_c = (lig_xyz - lig_com).clamp(-25, 25)
    poc_xyz_c = (poc_xyz - lig_com).clamp(-25, 25)

    if ligand_vdw:
        lig_rad = _VDW_RADIUS_LUT[lig_ch]                # element-wise vdW (ligand_radius<=0 path)
    else:
        lig_rad = torch.full_like(lig_ch, LIGAND_RAD, dtype=torch.float32)
    poc_rad = _VDW_RADIUS_LUT[poc_ch]

    # Voxelizer expects (B, N, *) shapes; B=1 here.
    lig_dict = {
        "coords":        lig_xyz_c.unsqueeze(0),
        "atoms_channel": lig_ch.float().unsqueeze(0),
        "radius":        lig_rad.unsqueeze(0),
    }
    poc_dict = {
        "coords":        poc_xyz_c.unsqueeze(0),
        "atoms_channel": poc_ch.float().unsqueeze(0),
        "radius":        poc_rad.unsqueeze(0),
    }

    v_lig = voxelizer(lig_dict, num_channels=N_LIG_CH)   # (1, 7, G, G, G)
    v_poc = voxelizer(poc_dict, num_channels=N_POC_CH)   # (1, 4, G, G, G)
    v_atom = torch.cat([v_lig, v_poc], dim=1)             # (1, 11, G, G, G)
    return v_atom.squeeze(0).cpu().numpy(), lig_com.numpy()


def crop_density_for(pid: str, lig_com: np.ndarray, ccp4_dir: Path):
    """(G,G,G) locally-normalised density crop, or None if CCP4 unavailable/invalid."""
    p = ccp4_dir / f"{pid}.ccp4"
    if not p.exists():
        return None
    g = _load_grid(p)
    if g is None:
        return None
    arr_norm, frac_T, nu, nv, nw = g
    try:
        crop = _crop_density(arr_norm, frac_T, nu, nv, nw, lig_com,
                              G=GRID_DIM, res=RESOLUTION, transform=None)
    except Exception:
        return None
    return normalize_crop(crop) if crop is not None else None


def should_filter_complex(element_filter: str,
                          lig_unsupported: list[str],
                          poc_unsupported: list[str]) -> str | None:
    """Return a skip reason when the chosen VoxBind element filter excludes it."""
    if element_filter in ("ligand", "all") and lig_unsupported:
        return f"unsupported_ligand={','.join(lig_unsupported)}"
    if element_filter == "all" and poc_unsupported:
        return f"unsupported_pocket={','.join(poc_unsupported)}"
    return None


def remove_cached_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists() or path.is_symlink():
            path.unlink()


def run_voxelize(args: argparse.Namespace) -> None:
    index_csv  = Path(args.index_csv)
    struct_dir = Path(args.struct_dir)
    ccp4_dir   = Path(args.ccp4_dir)
    out_dir    = Path(args.out_dir)
    atom_dir   = out_dir / "atoms"
    dens_dir   = out_dir / "density"
    avail_csv  = out_dir / "availability.csv"

    atom_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_density:
        dens_dir.mkdir(parents=True, exist_ok=True)

    print("=== PDBbind v2020 — voxelization ===")
    print(f"  index_csv  : {index_csv}")
    print(f"  struct_dir : {struct_dir}")
    print(f"  ccp4_dir   : {ccp4_dir}")
    print(f"  out_dir    : {out_dir}")
    print(f"  device     : {args.device}")
    print(f"  no_density : {args.no_density}")
    print(f"  ligand_vdw : {args.ligand_vdw}")
    print(f"  elem_filter: {args.element_filter}")
    print(f"  overwrite  : {args.overwrite}")
    print(f"  cache_check: {'warn only' if args.allow_stale_cache else 'strict'}")

    expected_meta = expected_voxelize_metadata(args)
    validate_voxelize_cache(
        out_dir,
        expected_meta,
        overwrite=args.overwrite,
        allow_stale_cache=args.allow_stale_cache,
    )

    df = pd.read_csv(index_csv)
    df = df[df["has_struct"].astype(bool)].reset_index(drop=True)
    if args.max_complexes:
        df = df.head(args.max_complexes)
    print(f"  complexes  : {len(df):,}")

    voxelizer = Voxelizer(
        grid_dim=GRID_DIM,
        resolution=RESOLUTION,
        radius=LIGAND_RAD,
        cubes_around=CUBES_AROUND,
        device=args.device,
        backend="pyuul",
    )

    rows: list[dict] = []
    n_skipped = n_filtered = n_err = 0
    err_log: list[tuple[str, str]] = []
    filter_log: list[tuple[str, str]] = []

    pbar = tqdm(df.iterrows(), total=len(df), desc="voxelize", unit="cplx")
    for _, row in pbar:
        pid = row["pdb_id"]
        atom_path = atom_dir / f"{pid}.npy"
        dens_path = dens_dir / f"{pid}.npy"

        wants_atom = args.overwrite or not atom_path.exists()
        wants_dens = (not args.no_density) and (args.overwrite or not dens_path.exists())

        # Already cached — record availability and move on.
        if not (wants_atom or wants_dens):
            rows.append({
                "pdb_id": pid,
                "has_atoms":   atom_path.exists(),
                "has_density": dens_path.exists() if not args.no_density else False,
                "n_lig": -1, "n_poc": -1,
                "filtered": False,
                "filter_reason": "",
                "unsupported_ligand_elements": "",
                "unsupported_pocket_elements": "",
            })
            n_skipped += 1
            continue

        try:
            cdir = struct_dir / pid
            lig_xyz, lig_ch, lig_center, lig_unsupported = parse_ligand_sdf(cdir / f"{pid}_ligand.sdf")
            poc_xyz, poc_ch, poc_unsupported = parse_pocket_pdb(cdir / f"{pid}_pocket.pdb")
            filter_reason = should_filter_complex(
                args.element_filter,
                lig_unsupported,
                poc_unsupported,
            )
            if filter_reason:
                remove_cached_outputs(atom_path, dens_path)
                n_filtered += 1
                filter_log.append((pid, filter_reason))
                rows.append({
                    "pdb_id": pid,
                    "has_atoms": False,
                    "has_density": False,
                    "n_lig": int(lig_xyz.shape[0]),
                    "n_poc": int(poc_xyz.shape[0]),
                    "filtered": True,
                    "filter_reason": filter_reason,
                    "unsupported_ligand_elements": ",".join(lig_unsupported),
                    "unsupported_pocket_elements": ",".join(poc_unsupported),
                })
                pbar.set_postfix(ok=len(rows) - n_filtered - n_err,
                                  filtered=n_filtered, err=n_err, refresh=False)
                continue

            n_lig, n_poc = int(lig_xyz.shape[0]), int(poc_xyz.shape[0])
            if n_lig == 0 or n_poc == 0:
                raise ValueError(f"empty ligand({n_lig}) or pocket({n_poc})")

            v_atom, lig_com = voxelize_complex(
                lig_xyz, lig_ch, poc_xyz, poc_ch, voxelizer, args.device,
                ligand_vdw=args.ligand_vdw,
                ligand_center=lig_center,
            )
            if wants_atom:
                np.save(str(atom_path), v_atom.astype(np.float16))

            if wants_dens:
                v_dens = crop_density_for(pid, lig_com, ccp4_dir)
                if v_dens is not None:
                    np.save(str(dens_path), v_dens.astype(np.float16))

            rows.append({
                "pdb_id": pid,
                "has_atoms":   atom_path.exists(),
                "has_density": dens_path.exists() if not args.no_density else False,
                "n_lig": n_lig, "n_poc": n_poc,
                "filtered": False,
                "filter_reason": "",
                "unsupported_ligand_elements": ",".join(lig_unsupported),
                "unsupported_pocket_elements": ",".join(poc_unsupported),
            })
            pbar.set_postfix(ok=len(rows) - n_err, err=n_err, refresh=False)

        except Exception as e:
            n_err += 1
            err_log.append((pid, repr(e)[:160]))
            rows.append({
                "pdb_id": pid,
                "has_atoms":   atom_path.exists(),
                "has_density": dens_path.exists() if not args.no_density else False,
                "n_lig": -1, "n_poc": -1,
                "filtered": False,
                "filter_reason": "",
                "unsupported_ligand_elements": "",
                "unsupported_pocket_elements": "",
            })

    # ── Summary + availability CSV ────────────────────────────────────────────
    avail = pd.DataFrame(rows)
    avail.to_csv(avail_csv, index=False)
    (out_dir / "metadata.json").write_text(json.dumps(expected_meta, indent=2) + "\n")

    n_atom = int(avail["has_atoms"].sum())
    n_dens = int(avail["has_density"].sum())

    print()
    print("─" * 64)
    print(f"  Atom voxels written : {n_atom:,} / {len(avail):,}")
    if not args.no_density:
        print(f"  Density crops       : {n_dens:,} / {len(avail):,}")
    print(f"  Already cached      : {n_skipped:,}")
    print(f"  Filtered complexes  : {n_filtered:,}")
    print(f"  Errors              : {n_err:,}")
    if filter_log:
        filter_path = out_dir / "voxelize_filtered.txt"
        filter_path.write_text("\n".join(f"{p}\t{m}" for p, m in filter_log) + "\n")
        print(f"  Filter log          : {filter_path}")
    if err_log:
        err_path = out_dir / "voxelize_errors.txt"
        err_path.write_text("\n".join(f"{p}\t{m}" for p, m in err_log) + "\n")
        print(f"  Error log           : {err_path}")
    print(f"  availability CSV    : {avail_csv}")
    print(f"  metadata JSON       : {out_dir / 'metadata.json'}")


# ═══════════════════════════════════════════════════════════════════════════════
# poolnorm — pocket-pool dataset-wide density normalisations (v2, v3)
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw_grid(ccp4_path: Path):
    """Return (arr_raw, frac_M_T, nu, nv, nw) or None — RAW CCP4 values.

    Unlike `_load_grid`, this does NOT apply the per-map z-score; poolnorm needs
    raw values to compute a single dataset-wide statistic.
    """
    try:
        m = gemmi.read_ccp4_map(str(ccp4_path))
        m.setup(float("nan"))
        grid = m.grid
        cell = grid.unit_cell
        arr = np.array(grid, dtype=np.float32)
        nu, nv, nw = arr.shape
        if not np.isfinite(arr).all():
            return None
        orth_mat = np.array(cell.orth.mat.tolist())
        frac_M = np.linalg.inv(orth_mat)
        return arr, frac_M.T, nu, nv, nw
    except Exception:
        return None


def parse_ligand_com(sdf_path: Path) -> "np.ndarray | None":
    """All-heavy-atom geometric centroid from a ligand SDF V2000 file."""
    try:
        with sdf_path.open() as f:
            lines = f.readlines()
        n_atoms = int(lines[3][:3])
        coords = []
        for i in range(n_atoms):
            line = lines[4 + i]
            el = _normalise_element(line[31:34])
            if not _is_heavy_element(el):
                continue
            coords.append([float(line[0:10]), float(line[10:20]), float(line[20:30])])
        if not coords:
            return None
        return np.array(coords, dtype=np.float32).mean(axis=0)
    except Exception:
        return None


def unsupported_ligand_sdf(sdf_path: Path) -> list[str]:
    """Unsupported non-H ligand elements relative to the 7 VoxBind ligand channels."""
    with sdf_path.open() as f:
        lines = f.readlines()
    n_atoms = int(lines[3][:3])
    elems = {
        _normalise_element(lines[4 + i][31:34])
        for i in range(n_atoms)
    }
    return _unsupported_heavy(elems, LIGAND_SUPPORTED_HEAVY)


def unsupported_pocket_pdb(pdb_path: Path) -> list[str]:
    """Unsupported non-H pocket elements relative to the 4 VoxBind pocket channels."""
    elems = set()
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            el = _normalise_element(line[76:78])
            if not el:
                el = _normalise_element(''.join(c for c in line[12:16] if c.isalpha())[:1])
            elems.add(el)
    return _unsupported_heavy(elems, POCKET_SUPPORTED_HEAVY)


def _read_v5_reference_stats(path: Path) -> dict:
    """Load CrossDocked-style v5 stats for reference-normalising PDBbind crops."""
    if not path.exists():
        raise FileNotFoundError(f"missing v5 reference stats: {path}")
    stats = json.loads(path.read_text())
    missing = [k for k in ("mu_a", "sigma_a") if k not in stats]
    if missing:
        raise ValueError(f"{path} is missing required v5 stat key(s): {missing}")
    sigma = float(stats["sigma_a"])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"invalid sigma_a in {path}: {sigma}")
    out = {
        "path": str(path),
        "arcsinh_scale": float(stats.get("arcsinh_scale", V5_ARCSINH_SCALE)),
        "mu_a": float(stats["mu_a"]),
        "sigma_a": sigma,
    }
    return out


def run_poolnorm(args: argparse.Namespace) -> None:
    index_csv  = Path(args.index_csv)
    struct_dir = Path(args.struct_dir)
    ccp4_dir   = Path(args.ccp4_dir)
    atoms_dir  = Path(args.atoms_dir).resolve()
    v2_dir     = Path(args.v2_dir)
    v3_dir     = Path(args.v3_dir)
    v4_dir     = Path(args.v4_dir)
    v5_dir     = Path(args.v5_dir)

    v2_dens = v2_dir / "density"; v2_dens.mkdir(parents=True, exist_ok=True)
    v3_dens = v3_dir / "density"; v3_dens.mkdir(parents=True, exist_ok=True)
    v4_dens = v4_dir / "density"; v4_dens.mkdir(parents=True, exist_ok=True)
    v5_dens = v5_dir / "density"; v5_dens.mkdir(parents=True, exist_ok=True)

    # Optional versioned ‖∇ρ‖ channel saved next to each density version (drop-in
    # gradmag source for the probe's --noise_voxels_dir). per_sample_zscore(‖∇·‖)
    # of the FINAL normalized density — exactly as the training/probe pipeline derives it.
    gmag_roots, gmag_dens, _gmag = {}, {}, None
    if args.save_gradmag:
        from voxbind.models.density_mae import gradient_magnitude3d, per_sample_zscore
        gmag_roots = {v: d.with_name(d.name + "_gradmag")
                      for v, d in (("v2", v2_dir), ("v3", v3_dir), ("v4", v4_dir), ("v5", v5_dir))}
        gmag_dens = {v: r / "density" for v, r in gmag_roots.items()}
        for d in gmag_dens.values():
            d.mkdir(parents=True, exist_ok=True)

        def _gmag(c16):
            g = gradient_magnitude3d(torch.from_numpy(c16.astype(np.float32))
                                     .view(1, 1, GRID_DIM, GRID_DIM, GRID_DIM))
            return per_sample_zscore(g).view(GRID_DIM, GRID_DIM, GRID_DIM).numpy().astype(np.float16)

    # Symlink atoms (v2/v3/v4/v5 reuse v1's atom voxels)
    for parent in (v2_dir, v3_dir, v4_dir, v5_dir):
        link = parent / "atoms"
        if not link.exists() and not link.is_symlink():
            os.symlink(atoms_dir, link)
            print(f"  [symlink] {link} → {atoms_dir}")

    print("=== PDBbind v2/v3/v4/v5 density normalisations (pocket dataset-wide) ===")
    print(f"  index_csv : {index_csv}")
    print(f"  ccp4_dir  : {ccp4_dir}")
    print(f"  v2_dir    : {v2_dir}")
    print(f"  v3_dir    : {v3_dir}")
    print(f"  v4_dir    : {v4_dir}")
    print(f"  v5_dir    : {v5_dir}")
    print(f"  v4 clip   : [{args.clip_lo:+.3f}, {args.clip_hi:+.3f}]  (raw 2Fo-Fc units)")
    print(f"  v5 norm   : arcsinh(x / {args.v5_arcsinh_scale}) → z-score  (soft-squash, no clip)")
    print(f"  v5 stats  : {args.v5_stats_source}")
    print(f"  elem_filter: {args.element_filter}")

    df = pd.read_csv(index_csv)
    df = df[df["has_struct"].astype(bool)].reset_index(drop=True)

    # Only complexes that have a CCP4 map on disk
    pids = [pid for pid in df["pdb_id"].tolist()
            if (ccp4_dir / f"{pid}.ccp4").exists()]
    if args.max_complexes:
        pids = pids[:args.max_complexes]
    print(f"  complexes : {len(pids):,}")

    # ── Pass 1: crop each pocket from RAW CCP4, accumulate dataset-wide stats ─
    crops: dict[str, np.ndarray] = {}    # pid → float16 (G, G, G)
    sum_v = 0.0
    sum_v2 = 0.0
    sum_cv = 0.0     # clipped sum   → v4 μ_clip
    sum_cv2 = 0.0    # clipped sum²  → v4 σ_clip
    n_voxels = 0
    min_v = +np.inf
    max_v = -np.inf

    n_skipped = 0
    skip_log: list[tuple[str, str]] = []
    pbar = tqdm(pids, desc="pass 1: crop + accumulate", unit="cplx")
    for pid in pbar:
        g = load_raw_grid(ccp4_dir / f"{pid}.ccp4")
        if g is None:
            n_skipped += 1; skip_log.append((pid, "load_raw_grid")); continue
        arr, frac_T, nu, nv, nw = g

        lig_path = struct_dir / pid / f"{pid}_ligand.sdf"
        poc_path = struct_dir / pid / f"{pid}_pocket.pdb"
        try:
            lig_unsupported = unsupported_ligand_sdf(lig_path) \
                if args.element_filter in ("ligand", "all") else []
            poc_unsupported = unsupported_pocket_pdb(poc_path) \
                if args.element_filter == "all" else []
        except Exception as e:
            n_skipped += 1; skip_log.append((pid, f"element_filter:{e!r}")); continue
        filter_reason = should_filter_complex(args.element_filter, lig_unsupported, poc_unsupported)
        if filter_reason:
            n_skipped += 1; skip_log.append((pid, filter_reason)); continue

        lig_com = parse_ligand_com(lig_path)
        if lig_com is None:
            n_skipped += 1; skip_log.append((pid, "parse_ligand_com")); continue

        try:
            crop = _crop_density(arr, frac_T, nu, nv, nw, lig_com,
                                  G=GRID_DIM, res=RESOLUTION, transform=None)
        except Exception as e:
            n_skipped += 1; skip_log.append((pid, f"crop:{e!r}")); continue

        # Accumulate dataset-wide stats in float64 to avoid precision loss.
        c64 = crop.astype(np.float64)
        sum_v   += c64.sum()
        sum_v2  += (c64 * c64).sum()
        n_voxels += c64.size
        min_v = min(min_v, float(c64.min()))
        max_v = max(max_v, float(c64.max()))

        # v4: same pool, accumulated over the CLIPPED values → μ_clip, σ_clip.
        cc64 = np.clip(c64, args.clip_lo, args.clip_hi)
        sum_cv  += cc64.sum()
        sum_cv2 += (cc64 * cc64).sum()

        crops[pid] = crop.astype(np.float16)        # save mem for pass 2

    if n_voxels == 0:
        print("[error] no crops produced; aborting.")
        sys.exit(1)

    mu_pool      = sum_v / n_voxels
    var_pool     = sum_v2 / n_voxels - mu_pool * mu_pool
    sigma_pool   = float(np.sqrt(max(var_pool, 0.0)))
    max_abs_pool = max(abs(min_v), abs(max_v))

    mu_clip      = sum_cv / n_voxels
    var_clip     = sum_cv2 / n_voxels - mu_clip * mu_clip
    sigma_clip   = float(np.sqrt(max(var_clip, 0.0)))

    v5_ref = None
    s5 = args.v5_arcsinh_scale
    if args.v5_stats_source == "reference":
        ref_path = Path(args.v5_reference_stats_json)
        v5_ref = _read_v5_reference_stats(ref_path)
        s5 = v5_ref["arcsinh_scale"]

    # v5: arcsinh soft-squash (NO clip → no pile-up spike), then z-score over the
    # squashed pool. arcsinh(x/s) is ~linear for |x|<<s and ~log for |x|>>s, so the
    # near-zero bulk is preserved and the bright metal-peak tail compresses smoothly
    # (raw +30 → ~+9 vs +92 for plain z-score). Every voxel is used. s = scale.
    sum_a5 = sum_a2_5 = 0.0
    for _c in crops.values():
        _a = np.arcsinh(_c.astype(np.float64) / s5)
        sum_a5 += _a.sum(); sum_a2_5 += (_a * _a).sum()
    mu_a5    = sum_a5 / n_voxels
    sigma_a5 = float(np.sqrt(max(sum_a2_5 / n_voxels - mu_a5 * mu_a5, 0.0)))
    if v5_ref is None:
        mu_a5_norm = mu_a5
        sigma_a5_norm = sigma_a5
    else:
        mu_a5_norm = v5_ref["mu_a"]
        sigma_a5_norm = v5_ref["sigma_a"]

    print(f"\n── pocket-pool dataset-wide stats ─────────────────────────────")
    print(f"  n_crops          : {len(crops):,}")
    print(f"  n_voxels (total) : {n_voxels:,}")
    print(f"  raw min / max    : [{min_v:+.4f}, {max_v:+.4f}]")
    print(f"  μ_pool           : {mu_pool:+.6f}")
    print(f"  σ_pool           : {sigma_pool:.6f}")
    print(f"  max_abs_pool     : {max_abs_pool:.6f}")
    print(f"  v4 clip          : [{args.clip_lo:+.3f}, {args.clip_hi:+.3f}]")
    print(f"  μ_clip           : {mu_clip:+.6f}")
    print(f"  σ_clip           : {sigma_clip:.6f}")
    print(f"  v5 arcsinh s={s5} : μ_a_fit {mu_a5:+.6f}  σ_a_fit {sigma_a5:.6f}")
    if v5_ref is not None:
        print(f"  v5 reference      : μ_a {mu_a5_norm:+.6f}  σ_a {sigma_a5_norm:.6f}  ({v5_ref['path']})")
    print(f"  skipped          : {n_skipped:,}")

    stats_v2 = {
        "metadata_version": VOXELIZE_METADATA_VERSION,
        "center_source":   CENTER_SOURCE,
        "grid_dim":        GRID_DIM,
        "resolution":      RESOLUTION,
        "element_filter":  args.element_filter,
        "scheme":         "pocket-pool z-score",
        "formula":        "x' = (x - mu) / sigma",
        "mu":             mu_pool,
        "sigma":          sigma_pool,
        "n_crops":        len(crops),
        "n_voxels_total": n_voxels,
        "raw_min":        min_v,
        "raw_max":        max_v,
    }
    stats_v3 = {
        "metadata_version": VOXELIZE_METADATA_VERSION,
        "center_source":   CENTER_SOURCE,
        "grid_dim":        GRID_DIM,
        "resolution":      RESOLUTION,
        "element_filter":  args.element_filter,
        "scheme":         "pocket-pool symmetric max-abs",
        "formula":        "x' = x / max_abs",
        "max_abs":        max_abs_pool,
        "n_crops":        len(crops),
        "n_voxels_total": n_voxels,
        "raw_min":        min_v,
        "raw_max":        max_v,
    }
    stats_v4 = {
        "metadata_version": VOXELIZE_METADATA_VERSION,
        "center_source":   CENTER_SOURCE,
        "grid_dim":        GRID_DIM,
        "resolution":      RESOLUTION,
        "element_filter":  args.element_filter,
        "scheme":         "pocket-pool clip + z-score",
        "formula":        "x' = (clip(x, [clip_lo, clip_hi]) - mu_clip) / sigma_clip",
        "clip_lo_raw":    args.clip_lo,
        "clip_hi_raw":    args.clip_hi,
        "mu_clip":        mu_clip,
        "sigma_clip":     sigma_clip,
        "n_crops":        len(crops),
        "n_voxels_total": n_voxels,
        "raw_min":        min_v,
        "raw_max":        max_v,
    }
    stats_v5 = {
        "metadata_version": VOXELIZE_METADATA_VERSION,
        "center_source":   CENTER_SOURCE,
        "grid_dim":        GRID_DIM,
        "resolution":      RESOLUTION,
        "element_filter":  args.element_filter,
        "scheme":         "pocket-pool arcsinh soft-squash + z-score (no clip)",
        "formula":        "x' = (arcsinh(x / s) - mu_a) / sigma_a",
        "arcsinh_scale":  s5,
        "stats_source":   args.v5_stats_source,
        "reference_stats_json": v5_ref["path"] if v5_ref is not None else None,
        "mu_a":           mu_a5_norm,
        "sigma_a":        sigma_a5_norm,
        "pdbbind_fit_mu_a":    mu_a5,
        "pdbbind_fit_sigma_a": sigma_a5,
        "n_crops":        len(crops),
        "n_voxels_total": n_voxels,
        "raw_min":        min_v,
        "raw_max":        max_v,
    }
    (v2_dir / "stats.json").write_text(json.dumps(stats_v2, indent=2))
    (v3_dir / "stats.json").write_text(json.dumps(stats_v3, indent=2))
    (v4_dir / "stats.json").write_text(json.dumps(stats_v4, indent=2))
    (v5_dir / "stats.json").write_text(json.dumps(stats_v5, indent=2))
    for v, root in gmag_roots.items():
        (root / "stats.json").write_text(json.dumps({
            "metadata_version": VOXELIZE_METADATA_VERSION,
            "scheme":          f"gradient magnitude ‖∇ρ‖ of PDBbind {v} density",
            "formula":         f"x' = per_sample_zscore(‖∇(density_{v})‖)",
            "density_version": v, "gradmag": True,
            "grid_dim": GRID_DIM, "resolution": RESOLUTION, "n_crops": len(crops),
        }, indent=2))

    # ── Pass 2: apply v2, v3, v4 normalisations, save to disk ─────────────────
    print(f"\n── pass 2: apply normalisations + save ────────────────────────")
    for pid, crop16 in tqdm(crops.items(), desc="pass 2: normalise + save", unit="cplx"):
        c = crop16.astype(np.float32)
        c_v2 = ((c - mu_pool) / sigma_pool).astype(np.float16)
        c_v3 = (c / max_abs_pool).astype(np.float16)
        c_v4 = ((np.clip(c, args.clip_lo, args.clip_hi) - mu_clip) / sigma_clip).astype(np.float16)
        c_v5 = ((np.arcsinh(c / s5) - mu_a5_norm) / sigma_a5_norm).astype(np.float16)
        np.save(str(v2_dens / f"{pid}.npy"), c_v2)
        np.save(str(v3_dens / f"{pid}.npy"), c_v3)
        np.save(str(v4_dens / f"{pid}.npy"), c_v4)
        np.save(str(v5_dens / f"{pid}.npy"), c_v5)
        if gmag_dens:
            np.save(str(gmag_dens["v2"] / f"{pid}.npy"), _gmag(c_v2))
            np.save(str(gmag_dens["v3"] / f"{pid}.npy"), _gmag(c_v3))
            np.save(str(gmag_dens["v4"] / f"{pid}.npy"), _gmag(c_v4))
            np.save(str(gmag_dens["v5"] / f"{pid}.npy"), _gmag(c_v5))

    if skip_log:
        skip_path = PDBBIND_DIR / "voxels_poolnorm_skip_log.txt"
        skip_path.write_text("\n".join(f"{p}\t{m}" for p, m in skip_log) + "\n")
        print(f"  skip log         : {skip_path}")

    print(f"\n  v2 crops written : {len(crops):,} → {v2_dens}")
    print(f"  v3 crops written : {len(crops):,} → {v3_dens}")
    print(f"  v4 crops written : {len(crops):,} → {v4_dens}")
    print(f"  v5 crops written : {len(crops):,} → {v5_dens}")
    print(f"  stats jsons      : v2 | v3 | v4 | v5  (each <dir>/stats.json)")
    print("\n  Next:  python dataset/01c_pdbbind_probe.py features --voxel_version v2  (or v3, v4, v5)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess PDBbind v2020 into voxels (voxelize | poolnorm)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser(
        "voxelize",
        help="Voxelize refined set (atoms + optional v1 density crop)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pv.add_argument("--index_csv",  default=str(INDEX_CSV))
    pv.add_argument("--struct_dir", default=str(STRUCT_DIR))
    pv.add_argument("--ccp4_dir",   default=str(CCP4_DIR))
    pv.add_argument("--out_dir",    default=str(VOX_DIR))
    pv.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Voxelizer device (cuda recommended)")
    pv.add_argument("--max_complexes", type=int, default=0,
                    help="Limit to first N complexes (0 = all). Smoke testing.")
    pv.add_argument("--no_density", action="store_true",
                    help="Skip CCP4 crop (atom voxels only)")
    pv.add_argument("--ligand_vdw", action="store_true",
                    help="Element-wise vdW ligand radii (mirrors crossdocked_xray "
                         "ligand_radius<=0) instead of the uniform 0.5 A blob. Match "
                         "to a gradmag/ligvdw encoder; write to a separate --out_dir.")
    pv.add_argument("--element_filter", choices=["none", "ligand", "all"], default="none",
                    help="Drop complexes with unsupported non-H atom types. none = keep "
                         "and ignore unsupported channels; ligand = drop Br/I/etc in "
                         "the ligand only; all = also drop unsupported pocket metals/ions.")
    pv.add_argument("--overwrite",  action="store_true",
                    help="Recompute outputs even if .npy already exists")
    pv.add_argument("--allow_stale_cache", action="store_true",
                    help="Warn instead of failing when existing voxel caches have "
                         "missing/mismatched metadata. Use only for intentional "
                         "legacy-cache reuse.")
    pv.set_defaults(func=run_voxelize)

    pp = sub.add_parser(
        "poolnorm",
        help="Pocket-pool density normalisations (v2: z-score, v3: max-abs, v4: clip+z, v5: arcsinh+z)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pp.add_argument("--index_csv",  default=str(INDEX_CSV))
    pp.add_argument("--struct_dir", default=str(STRUCT_DIR))
    pp.add_argument("--ccp4_dir",   default=str(CCP4_DIR))
    pp.add_argument("--atoms_dir",  default=str(ATOMS_DIR_V1),
                    help="Existing v1 atom voxels to symlink from v2/v3/v4")
    pp.add_argument("--v2_dir",     default=str(V2_DIR))
    pp.add_argument("--v3_dir",     default=str(V3_DIR))
    pp.add_argument("--v4_dir",     default=str(V4_DIR))
    pp.add_argument("--v5_dir",     default=str(V5_DIR))
    pp.add_argument("--clip_lo", type=float, default=CLIP_LO,
                    help="v4 lower clip threshold (raw 2Fo-Fc units)")
    pp.add_argument("--clip_hi", type=float, default=CLIP_HI,
                    help="v4 upper clip threshold (raw 2Fo-Fc units)")
    pp.add_argument("--v5_arcsinh_scale", type=float, default=V5_ARCSINH_SCALE,
                    help="v5 arcsinh scale s in arcsinh(x/s) (smaller = harder tail squash)")
    pp.add_argument("--v5_stats_source", choices=["pdbbind", "reference"], default="pdbbind",
                    help="pdbbind = fit v5 mu/sigma on PDBbind crops; reference = "
                         "reuse stats from --v5_reference_stats_json, typically the "
                         "CrossDocked v5 stats used for pretraining")
    pp.add_argument("--v5_reference_stats_json", default=str(DEFAULT_V5_REFERENCE_STATS),
                    help="stats.json with CrossDocked-style v5 keys "
                         "(arcsinh_scale, mu_a, sigma_a) used when "
                         "--v5_stats_source=reference")
    pp.add_argument("--element_filter", choices=["none", "ligand", "all"], default="none",
                    help="Apply the same unsupported-element filter while building "
                         "density normalisation crops. Use the same value as voxelize.")
    pp.add_argument("--max_complexes", type=int, default=0,
                    help="Limit to first N (0 = all). Smoke testing.")
    pp.add_argument("--save_gradmag", action="store_true",
                    help="Also save a versioned ‖∇ρ‖ channel next to each density version: "
                         "{v_dir}_gradmag/density/{pid}.npy = per-sample-z-scored ‖∇(density)‖, "
                         "+ stats.json. Drop-in for the probe's --noise_voxels_dir.")
    pp.set_defaults(func=run_poolnorm)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
