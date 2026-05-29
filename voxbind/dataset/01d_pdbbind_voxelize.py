"""01d_pdbbind_voxelize.py — Phase 4: voxelize PDBbind v2020 refined complexes.

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
  5. If a 2Fo-Fc CCP4 map is available (from 01c):
       - Load + globally z-score via _load_grid (gemmi).
       - Crop a 64³ × 0.25 Å box at the ligand COM via _crop_density
         with transform=None (PDBbind structures live in the deposited
         crystal frame, so no Kabsch alignment is needed).
       - Locally ±3σ-clip + re-z-score via normalize_crop.
     → (64, 64, 64) density, ready as the trailing channel for `atomblob_density`.

Centring on the ligand COM matches Beyond Atoms §3.1 ("both [grids] centered on
the ligand's center of mass") and the existing VoxBind training pipeline.

Usage
-----
    cd voxbind
    python dataset/01d_pdbbind_voxelize.py
    python dataset/01d_pdbbind_voxelize.py --max_complexes 20   # smoke test
    python dataset/01d_pdbbind_voxelize.py --no_density         # atoms only
    python dataset/01d_pdbbind_voxelize.py --device cpu         # if no GPU

Outputs
-------
    dataset/data/pdbbind/voxels/atoms/{pid}.npy        float16 (11, 64, 64, 64)
    dataset/data/pdbbind/voxels/density/{pid}.npy      float16 (64, 64, 64)
    dataset/data/pdbbind/voxels/availability.csv       pid,has_atoms,has_density,n_lig,n_poc
    dataset/data/pdbbind/voxels/voxelize_errors.txt    (if any)
"""

import argparse
import sys
from pathlib import Path

# Import gemmi BEFORE torch so its libstdc++.so.6 is loaded first. If torch
# (or PyUUL's CUDA libs) load an older libstdc++ ahead of gemmi, gemmi's C++
# extension fails with ImportError("... CXXABI_1.3.15 not found ...") inside
# every _load_grid call and density crops silently return None.
import gemmi  # noqa: F401  (kept first to fix shared-library load order)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from voxbind.constants import ELEMENTS_HASH_CROSSDOCKED, RADIUS_PER_ATOM
from voxbind.voxelizer import Voxelizer
from voxbind.dataset.crossdocked_xray import _crop_density, normalize_crop


def _load_grid(ccp4_path: Path):
    """Load a CCP4 map → (arr_norm, frac_M_T, nu, nv, nw) or None."""
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


# ── Config ─────────────────────────────────────────────────────────────────────

PDBBIND_DIR  = Path(__file__).parent / "data" / "pdbbind"
INDEX_CSV    = PDBBIND_DIR / "index.csv"
STRUCT_DIR   = PDBBIND_DIR / "structures" / "pbpp-2020"
CCP4_DIR     = PDBBIND_DIR / "ccp4"
VOX_DIR      = PDBBIND_DIR / "voxels"

N_LIG_CH = 7   # C,O,N,S,F,Cl,P
N_POC_CH = 4   # C,O,N,S
GRID_DIM    = 64
RESOLUTION  = 0.25
LIGAND_RAD  = 0.5
CUBES_AROUND = 8

# Element → channel index lookup (matches voxbind.constants.ELEMENTS_HASH_CROSSDOCKED)
_E2CH = ELEMENTS_HASH_CROSSDOCKED   # {"C":0, "O":1, "N":2, "S":3, "F":4, "Cl":5, "P":6, "H":7}
_POCKET_RADIUS_LUT = torch.tensor(
    [RADIUS_PER_ATOM["MOL"][e] for e in _E2CH.keys()],
    dtype=torch.float32,
)


# ── Parsers (heavy atoms only) ────────────────────────────────────────────────

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


# ── Voxelize one complex ───────────────────────────────────────────────────────

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


# ── CCP4 density crop ──────────────────────────────────────────────────────────

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


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Voxelize PDBbind v2020 refined set (atoms + optional density crop)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--index_csv",  default=str(INDEX_CSV))
    p.add_argument("--struct_dir", default=str(STRUCT_DIR))
    p.add_argument("--ccp4_dir",   default=str(CCP4_DIR))
    p.add_argument("--out_dir",    default=str(VOX_DIR))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Voxelizer device (cuda recommended)")
    p.add_argument("--max_complexes", type=int, default=0,
                   help="Limit to first N complexes (0 = all). Smoke testing.")
    p.add_argument("--no_density", action="store_true",
                   help="Skip CCP4 crop (atom voxels only)")
    p.add_argument("--overwrite",  action="store_true",
                   help="Recompute outputs even if .npy already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()
