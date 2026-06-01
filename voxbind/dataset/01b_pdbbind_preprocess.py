"""01b_pdbbind_preprocess.py — Preprocess PDBbind v2020 into training voxels.

Consolidated Phase 4 / 4b entry point. Two subcommands:

    cd voxbind
    python dataset/01b_pdbbind_preprocess.py voxelize   # atoms (+ density) → voxels/  (v1)
    python dataset/01b_pdbbind_preprocess.py poolnorm    # pocket-pool density → voxels_v2/, voxels_v3/

──────────────────────────────────────────────────────────────────────────────
voxelize — voxelize PDBbind v2020 refined complexes (atoms + v1 density crop)
──────────────────────────────────────────────────────────────────────────────
For each refined complex with has_struct=True in index.csv:
  1. Parse heavy atoms from {pid}_pocket.pdb (channels 0..3 = C,O,N,S)
     and from {pid}_ligand.sdf  (channels 0..6 = C,O,N,S,F,Cl,P).
  2. Compute the ligand heavy-atom centre of mass (COM).
  3. Recentre both atom sets to the ligand COM. Clamp to ±25 Å
     (matches voxbind.utils.dataset_utils.filter_atoms_by_distance).
  4. Voxelize with voxbind.voxelizer.Voxelizer (pyuul, 64³, 0.25 Å):
       - ligand : 7 channels, uniform 0.5 Å Gaussian radius
       - pocket : 4 channels, element-wise vdW radii
     and concatenate → (11, 64, 64, 64).  This is exactly the tensor
     `atomblob` encoder consumes during training.
  5. If a 2Fo-Fc CCP4 map is available (from `density` acquisition):
       - Load + globally z-score via _load_grid (gemmi).
       - Crop a 64³ × 0.25 Å box at the ligand COM via _crop_density
         with transform=None (PDBbind structures live in the deposited
         crystal frame, so no Kabsch alignment is needed).
       - Locally ±3σ-clip + re-z-score via normalize_crop.
     → (64, 64, 64) density, ready as the trailing channel for `atomblob_density`.

Centring on the ligand COM matches Beyond Atoms §3.1 ("both [grids] centered on
the ligand's center of mass") and the existing VoxBind training pipeline.

    python dataset/01b_pdbbind_preprocess.py voxelize
    python dataset/01b_pdbbind_preprocess.py voxelize --max_complexes 20   # smoke test
    python dataset/01b_pdbbind_preprocess.py voxelize --no_density         # atoms only
    python dataset/01b_pdbbind_preprocess.py voxelize --device cpu         # if no GPU
  → voxels/atoms/{pid}.npy  voxels/density/{pid}.npy  voxels/availability.csv

──────────────────────────────────────────────────────────────────────────────
poolnorm — pocket-pool density normalisations (v2: z-score, v3: max-abs)
──────────────────────────────────────────────────────────────────────────────
Produces TWO new density-crop variants from the raw CCP4 maps in
dataset/data/pdbbind/ccp4/, normalised against the pool of all pocket crops
(not per-map, not per-crop):

  v2 — pocket-pool z-score
       x' = (x − μ_pool) / σ_pool
       μ_pool, σ_pool computed over EVERY voxel of EVERY pocket crop concatenated.

  v3 — pocket-pool symmetric max-abs
       x' = x / max_abs_pool
       max_abs_pool = max(|x|) across the same pool. Output strict [−1, +1].

Both schemes:
  - SKIP the per-map z-score that `_load_grid` does in v1 (use raw CCP4 values)
  - SKIP the per-crop ±3σ clip that `normalize_crop` does in v1
  - apply ONE dataset-wide statistic derived from all pocket crops

Atom voxels (11, 64, 64, 64) are NOT affected — v2 and v3 reuse the existing
voxels/atoms/ directory via symlinks.

    python dataset/01b_pdbbind_preprocess.py poolnorm
  → voxels_v2/density/{pid}.npy  voxels_v2/atoms→../voxels/atoms  voxels_v2/stats.json
    voxels_v3/density/{pid}.npy  voxels_v3/atoms→../voxels/atoms  voxels_v3/stats.json
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

N_LIG_CH = 7   # C,O,N,S,F,Cl,P
N_POC_CH = 4   # C,O,N,S
GRID_DIM     = 64
RESOLUTION   = 0.25
LIGAND_RAD   = 0.5
CUBES_AROUND = 8

# Element → channel index lookup (matches voxbind.constants.ELEMENTS_HASH_CROSSDOCKED)
_E2CH = ELEMENTS_HASH_CROSSDOCKED   # {"C":0, "O":1, "N":2, "S":3, "F":4, "Cl":5, "P":6, "H":7}
_POCKET_RADIUS_LUT = torch.tensor(
    [RADIUS_PER_ATOM["MOL"][e] for e in _E2CH.keys()],
    dtype=torch.float32,
)

# Channel-set of ligand heavy atoms used to compute the ligand COM. Mirrors
# ELEMENTS_HASH_CROSSDOCKED channels 0-6 (C, O, N, S, F, Cl, P).
_LIG_HEAVY_ELEMS = {"C", "O", "N", "S", "F", "Cl", "P"}


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


def parse_pocket_pdb(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Heavy-atom coords + 4-channel atom channels for a pocket PDB file."""
    coords, channels = [], []
    with path.open() as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            el = line[76:78].strip().capitalize()
            if not el:
                # Fall back: first letter of atom name (cols 13–16)
                el = ''.join(c for c in line[12:16] if c.isalpha())[:1].upper()
            ch = _channel_of(el, N_POC_CH)
            if ch is None:
                continue
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            channels.append(ch)
    return (torch.tensor(coords, dtype=torch.float32),
            torch.tensor(channels, dtype=torch.long))


def parse_ligand_sdf(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Heavy-atom coords + 7-channel atom channels for a ligand SDF V2000 file."""
    coords, channels = [], []
    with path.open() as f:
        lines = f.readlines()
    n_atoms = int(lines[3][:3])
    for i in range(n_atoms):
        line = lines[4 + i]
        el = line[31:34].strip()
        ch = _channel_of(el, N_LIG_CH)
        if ch is None:
            continue
        coords.append([float(line[0:10]), float(line[10:20]), float(line[20:30])])
        channels.append(ch)
    return (torch.tensor(coords, dtype=torch.float32),
            torch.tensor(channels, dtype=torch.long))


def voxelize_complex(
    lig_xyz: torch.Tensor, lig_ch: torch.Tensor,
    poc_xyz: torch.Tensor, poc_ch: torch.Tensor,
    voxelizer: Voxelizer,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ((11,G,G,G) atom voxels, (3,) ligand COM in deposited frame)."""
    # Heavy-atom COM in deposited frame (used later for density cropping).
    lig_com = lig_xyz.mean(dim=0)

    # Recentre to ligand COM, clamp to ±25 Å (mirrors filter_atoms_by_distance).
    lig_xyz_c = (lig_xyz - lig_com).clamp(-25, 25)
    poc_xyz_c = (poc_xyz - lig_com).clamp(-25, 25)

    lig_rad = torch.full_like(lig_ch, LIGAND_RAD, dtype=torch.float32)
    poc_rad = _POCKET_RADIUS_LUT[poc_ch]

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
    print(f"  overwrite  : {args.overwrite}")

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
    n_skipped = n_err = 0
    err_log: list[tuple[str, str]] = []

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
            })
            n_skipped += 1
            continue

        try:
            cdir = struct_dir / pid
            lig_xyz, lig_ch = parse_ligand_sdf(cdir / f"{pid}_ligand.sdf")
            poc_xyz, poc_ch = parse_pocket_pdb(cdir / f"{pid}_pocket.pdb")
            n_lig, n_poc = int(lig_xyz.shape[0]), int(poc_xyz.shape[0])
            if n_lig == 0 or n_poc == 0:
                raise ValueError(f"empty ligand({n_lig}) or pocket({n_poc})")

            v_atom, lig_com = voxelize_complex(
                lig_xyz, lig_ch, poc_xyz, poc_ch, voxelizer, args.device,
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
            })

    # ── Summary + availability CSV ────────────────────────────────────────────
    avail = pd.DataFrame(rows)
    avail.to_csv(avail_csv, index=False)

    n_atom = int(avail["has_atoms"].sum())
    n_dens = int(avail["has_density"].sum())

    print()
    print("─" * 64)
    print(f"  Atom voxels written : {n_atom:,} / {len(avail):,}")
    if not args.no_density:
        print(f"  Density crops       : {n_dens:,} / {len(avail):,}")
    print(f"  Already cached      : {n_skipped:,}")
    print(f"  Errors              : {n_err:,}")
    if err_log:
        err_path = out_dir / "voxelize_errors.txt"
        err_path.write_text("\n".join(f"{p}\t{m}" for p, m in err_log) + "\n")
        print(f"  Error log           : {err_path}")
    print(f"  availability CSV    : {avail_csv}")


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
    """Heavy-atom centre of mass from a ligand SDF V2000 file."""
    try:
        with sdf_path.open() as f:
            lines = f.readlines()
        n_atoms = int(lines[3][:3])
        coords = []
        for i in range(n_atoms):
            line = lines[4 + i]
            el = line[31:34].strip()
            if el not in _LIG_HEAVY_ELEMS:
                continue
            coords.append([float(line[0:10]), float(line[10:20]), float(line[20:30])])
        if not coords:
            return None
        return np.array(coords, dtype=np.float32).mean(axis=0)
    except Exception:
        return None


def run_poolnorm(args: argparse.Namespace) -> None:
    index_csv  = Path(args.index_csv)
    struct_dir = Path(args.struct_dir)
    ccp4_dir   = Path(args.ccp4_dir)
    atoms_dir  = Path(args.atoms_dir).resolve()
    v2_dir     = Path(args.v2_dir)
    v3_dir     = Path(args.v3_dir)

    v2_dens = v2_dir / "density"; v2_dens.mkdir(parents=True, exist_ok=True)
    v3_dens = v3_dir / "density"; v3_dens.mkdir(parents=True, exist_ok=True)

    # Symlink atoms (v2/v3 reuse v1's atom voxels)
    for parent in (v2_dir, v3_dir):
        link = parent / "atoms"
        if not link.exists() and not link.is_symlink():
            os.symlink(atoms_dir, link)
            print(f"  [symlink] {link} → {atoms_dir}")

    print("=== PDBbind v2/v3 density normalisations (pocket dataset-wide) ===")
    print(f"  index_csv : {index_csv}")
    print(f"  ccp4_dir  : {ccp4_dir}")
    print(f"  v2_dir    : {v2_dir}")
    print(f"  v3_dir    : {v3_dir}")

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

        lig_com = parse_ligand_com(struct_dir / pid / f"{pid}_ligand.sdf")
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

        crops[pid] = crop.astype(np.float16)        # save mem for pass 2

    if n_voxels == 0:
        print("[error] no crops produced; aborting.")
        sys.exit(1)

    mu_pool      = sum_v / n_voxels
    var_pool     = sum_v2 / n_voxels - mu_pool * mu_pool
    sigma_pool   = float(np.sqrt(max(var_pool, 0.0)))
    max_abs_pool = max(abs(min_v), abs(max_v))

    print(f"\n── pocket-pool dataset-wide stats ─────────────────────────────")
    print(f"  n_crops          : {len(crops):,}")
    print(f"  n_voxels (total) : {n_voxels:,}")
    print(f"  raw min / max    : [{min_v:+.4f}, {max_v:+.4f}]")
    print(f"  μ_pool           : {mu_pool:+.6f}")
    print(f"  σ_pool           : {sigma_pool:.6f}")
    print(f"  max_abs_pool     : {max_abs_pool:.6f}")
    print(f"  skipped          : {n_skipped:,}")

    stats_v2 = {
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
        "scheme":         "pocket-pool symmetric max-abs",
        "formula":        "x' = x / max_abs",
        "max_abs":        max_abs_pool,
        "n_crops":        len(crops),
        "n_voxels_total": n_voxels,
        "raw_min":        min_v,
        "raw_max":        max_v,
    }
    (v2_dir / "stats.json").write_text(json.dumps(stats_v2, indent=2))
    (v3_dir / "stats.json").write_text(json.dumps(stats_v3, indent=2))

    # ── Pass 2: apply v2 and v3 normalisations, save to disk ──────────────────
    print(f"\n── pass 2: apply normalisations + save ────────────────────────")
    for pid, crop16 in tqdm(crops.items(), desc="pass 2: normalise + save", unit="cplx"):
        c = crop16.astype(np.float32)
        c_v2 = ((c - mu_pool) / sigma_pool).astype(np.float16)
        c_v3 = (c / max_abs_pool).astype(np.float16)
        np.save(str(v2_dens / f"{pid}.npy"), c_v2)
        np.save(str(v3_dens / f"{pid}.npy"), c_v3)

    if skip_log:
        skip_path = PDBBIND_DIR / "voxels_v2_v3_skip_log.txt"
        skip_path.write_text("\n".join(f"{p}\t{m}" for p, m in skip_log) + "\n")
        print(f"  skip log         : {skip_path}")

    print(f"\n  v2 crops written : {len(crops):,} → {v2_dens}")
    print(f"  v3 crops written : {len(crops):,} → {v3_dens}")
    print(f"  stats jsons      : {v2_dir / 'stats.json'} | {v3_dir / 'stats.json'}")
    print("\n  Next:  python dataset/01c_pdbbind_probe.py features --voxel_version v2  (or v3)")


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
    pv.add_argument("--overwrite",  action="store_true",
                    help="Recompute outputs even if .npy already exists")
    pv.set_defaults(func=run_voxelize)

    pp = sub.add_parser(
        "poolnorm",
        help="Pocket-pool density normalisations (v2: z-score, v3: max-abs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pp.add_argument("--index_csv",  default=str(INDEX_CSV))
    pp.add_argument("--struct_dir", default=str(STRUCT_DIR))
    pp.add_argument("--ccp4_dir",   default=str(CCP4_DIR))
    pp.add_argument("--atoms_dir",  default=str(ATOMS_DIR_V1),
                    help="Existing v1 atom voxels to symlink from v2/v3")
    pp.add_argument("--v2_dir",     default=str(V2_DIR))
    pp.add_argument("--v3_dir",     default=str(V3_DIR))
    pp.add_argument("--max_complexes", type=int, default=0,
                    help="Limit to first N (0 = all). Smoke testing.")
    pp.set_defaults(func=run_poolnorm)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
