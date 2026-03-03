"""
fit_emdb_pocket.py
==================
Fits an EMDB cryo-EM density map to a CrossDocked pocket via rigid-body alignment.

Pipeline
--------
1. Query the EMDB entry to find which PDB model is fitted inside the map.
2. Download both the CrossDocked PDB (X-ray) and the EMDB-fitted PDB (EM model).
3. Sequence-align the two structures chain-by-chain; pick the best-matching pair.
4. Superimpose matched Cα atoms → rotation R, translation t such that
       R @ xray_coords + t  ≈  emdb_pdb_coords
5. Transform the pocket ligand CoM into the EMDB map frame.
6. Load the .map.gz, crop a `crop_dim`³ box (native voxel size) around that point.
7. Z-score normalise and return the crop together with the alignment metadata.

Usage (standalone)
------------------
    conda run -n voxbind python test/fit_emdb_pocket.py
"""

import gzip
import json
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, Superimposer

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent.parent
NB_DATA_DIR  = REPO_ROOT / "notebook" / "data"
EMDB_DIR     = NB_DATA_DIR / "emdb"
PDB_CACHE    = Path(__file__).parent / "data" / "pdb_cache"
PDB_CACHE.mkdir(parents=True, exist_ok=True)

EMDB_ENTRY_URL = "https://www.ebi.ac.uk/emdb/api/entry/{emdb_id}"
RCSB_PDB_URL   = "https://files.rcsb.org/download/{pdb_id}.pdb"

CROP_DIM = 64
HALF     = CROP_DIM // 2

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class FitResult:
    crop:                np.ndarray          # (CROP_DIM, CROP_DIM, CROP_DIM) float32
    pocket_center_emdb:  np.ndarray          # (3,) pocket CoM in EMDB frame (Å)
    crop_origin_vox:     np.ndarray          # (3,) int — corner voxel of crop in map
    voxel_size:          np.ndarray          # (3,) float — Å/voxel
    rotation:            np.ndarray          # (3, 3) rotation matrix R
    translation:         np.ndarray          # (3,) translation t
    rmsd:                float               # Cα superposition RMSD (Å)
    n_aligned:           int                 # number of matched Cα pairs
    xray_pdb_id:         str
    emdb_pdb_id:         str                 # the PDB model fitted in the EMDB entry
    emdb_id:             str
    fallback:            bool = False        # True if superposition was skipped


# ── PDB download & parsing ────────────────────────────────────────────────────

def download_pdb(pdb_id: str) -> Path:
    """Download a PDB file from RCSB, caching locally. Returns path."""
    pdb_id = pdb_id.upper()
    dest = PDB_CACHE / f"{pdb_id}.pdb"
    if dest.exists():
        return dest
    r = requests.get(RCSB_PDB_URL.format(pdb_id=pdb_id), timeout=30)
    r.raise_for_status()
    dest.write_text(r.text)
    return dest


def extract_ca_chains(pdb_path: Path, min_residues: int = 20) -> dict[str, dict]:
    """
    Parse a PDB file and return Cα atoms grouped by chain.

    Returns
    -------
    dict  chain_id → {"seq": str (1-letter), "atoms": list[Atom]}
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("s", str(pdb_path))
    chains = {}
    for model in struct:
        for chain in model:
            residues, atoms = [], []
            for res in chain:
                if res.has_id("CA") and res.resname in AA3TO1:
                    residues.append(AA3TO1[res.resname])
                    atoms.append(res["CA"])
            if len(atoms) >= min_residues:
                chains[chain.id] = {"seq": "".join(residues), "atoms": atoms}
        break  # first model only
    return chains


# ── Sequence-guided superposition ────────────────────────────────────────────

def _make_aligner() -> PairwiseAligner:
    aln = PairwiseAligner()
    aln.mode            = "global"
    aln.match_score     =  2.0
    aln.mismatch_score  = -1.0
    aln.open_gap_score  = -2.0
    aln.extend_gap_score = -0.5
    return aln


def best_superposition(
    xray_chains: dict[str, dict],
    emdb_chains: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray, float, int, str, str]:
    """
    Try every (xray_chain, emdb_chain) pair, align sequences, superimpose Cα atoms.

    Returns (R, t, rmsd, n_aligned, xray_chain_id, emdb_chain_id)
    with the pair that gives the lowest RMSD.
    """
    aligner = _make_aligner()
    best = dict(rmsd=np.inf)

    for xi, xv in xray_chains.items():
        for ei, ev in emdb_chains.items():
            try:
                aln = next(aligner.align(xv["seq"], ev["seq"]))
            except StopIteration:
                continue

            # Collect paired Cα atoms at non-gap positions
            xa, ea = [], []
            xi_pos = ei_pos = 0
            for a, b in zip(*aln):
                a_gap = (a == "-")
                b_gap = (b == "-")
                if not a_gap and not b_gap:
                    xa.append(xv["atoms"][xi_pos])
                    ea.append(ev["atoms"][ei_pos])
                if not a_gap:
                    xi_pos += 1
                if not b_gap:
                    ei_pos += 1

            if len(xa) < 20:
                continue

            sup = Superimposer()
            sup.set_atoms(ea, xa)   # fixed=emdb, mobile=xray → R,t maps xray→emdb

            if sup.rms < best["rmsd"]:
                best = dict(
                    rmsd=sup.rms,
                    R=sup.rotran[0].copy(),
                    t=sup.rotran[1].copy(),
                    n=len(xa),
                    xi=xi, ei=ei,
                )

    if not best.get("R") is not None and "R" not in best:
        raise ValueError("No valid chain pair found for superposition.")

    return (
        best["R"], best["t"], best["rmsd"], best["n"],
        best["xi"], best["ei"],
    )


# ── EMDB helpers ──────────────────────────────────────────────────────────────

def get_emdb_fitted_pdbs(emdb_id: str) -> list[str]:
    """Return PDB IDs of models fitted in this EMDB entry (FULLOVERLAP first)."""
    for attempt in range(3):
        try:
            r = requests.get(EMDB_ENTRY_URL.format(emdb_id=emdb_id), timeout=60)
            break
        except requests.exceptions.Timeout:
            if attempt == 2:
                return []
    if r.status_code != 200:
        return []
    refs = (
        r.json()
        .get("crossreferences", {})
        .get("pdb_list", {})
        .get("pdb_reference", [])
    )
    if isinstance(refs, dict):
        refs = [refs]
    # Prefer FULLOVERLAP relationships
    full = [x["pdb_id"].upper() for x in refs
            if x.get("relationship", {}).get("in_frame") == "FULLOVERLAP"]
    other = [x["pdb_id"].upper() for x in refs
             if x.get("relationship", {}).get("in_frame") != "FULLOVERLAP"]
    return full + other


def load_mrc_gz(gz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a .map.gz → (data_xyz, voxel_size, origin_angstrom).
    data_xyz has shape (NX, NY, NZ) with axes ordered (X, Y, Z).
    """
    with gzip.open(gz_path, "rb") as gz:
        raw = gz.read()
    with tempfile.NamedTemporaryFile(suffix=".map", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        import mrcfile
        with mrcfile.open(tmp_path, mode="r", permissive=True) as mrc:
            vs = np.array(
                [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z],
                dtype=np.float64,
            )
            ox, oy, oz = (float(mrc.header.origin.x),
                          float(mrc.header.origin.y),
                          float(mrc.header.origin.z))
            if ox == 0.0 and oy == 0.0 and oz == 0.0:
                ox = float(mrc.header.nxstart) * vs[0]
                oy = float(mrc.header.nystart) * vs[1]
                oz = float(mrc.header.nzstart) * vs[2]
            origin = np.array([ox, oy, oz], dtype=np.float64)
            mapc   = int(mrc.header.mapc)
            mapr   = int(mrc.header.mapr)
            maps   = int(mrc.header.maps)
            data   = mrc.data.astype(np.float32)
    finally:
        os.unlink(tmp_path)

    # Reorder to (X, Y, Z)
    storage = [maps, mapr, mapc]
    perm = [storage.index(i + 1) for i in range(3)]
    return np.transpose(data, perm), vs, origin


def zscore(data: np.ndarray) -> np.ndarray:
    mu, sd = data.mean(), data.std()
    return np.zeros_like(data) if sd < 1e-10 else ((data - mu) / sd).astype(np.float32)


def crop_at(data_xyz: np.ndarray, vs: np.ndarray, origin: np.ndarray,
            center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop a CROP_DIM³ box from data_xyz centred on `center` (Å).
    Returns (crop, crop_origin_vox) — zero-padded when out of bounds.
    """
    frac  = (center - origin) / vs
    ci    = np.round(frac).astype(int)
    lo    = ci - HALF
    hi    = lo + CROP_DIM
    shape = np.array(data_xyz.shape, dtype=int)
    lo_c  = np.clip(lo, 0, shape)
    hi_c  = np.clip(hi, 0, shape)
    crop  = np.zeros((CROP_DIM, CROP_DIM, CROP_DIM), dtype=np.float32)
    dst   = lo_c - lo
    dhi   = dst + (hi_c - lo_c)
    crop[dst[0]:dhi[0], dst[1]:dhi[1], dst[2]:dhi[2]] = (
        data_xyz[lo_c[0]:hi_c[0], lo_c[1]:hi_c[1], lo_c[2]:hi_c[2]]
    )
    return crop, lo.astype(np.int32)


# ── Main fitting function ─────────────────────────────────────────────────────

def fit_emdb_to_pocket(
    pocket_coords: np.ndarray,
    ligand_coords: np.ndarray,
    xray_pdb_id:   str,
    emdb_id:       str,
    emdb_map_path: Path,
    crop_dim:      int = CROP_DIM,
    verbose:       bool = True,
) -> FitResult:
    """
    Align an EMDB density map to a CrossDocked pocket.

    Parameters
    ----------
    pocket_coords : (N, 3) float — pocket atom coords in PDB (X-ray) frame.
    ligand_coords : (M, 3) float — ligand atom coords in PDB (X-ray) frame.
    xray_pdb_id   : CrossDocked PDB ID, e.g. '3DZH'.
    emdb_id       : EMDB accession, e.g. 'EMD-35526'.
    emdb_map_path : Path to the downloaded .map.gz file.
    crop_dim      : side length of the output crop (default 64).

    Returns
    -------
    FitResult with the aligned, z-scored density crop and all alignment metadata.
    """
    global CROP_DIM, HALF
    CROP_DIM = crop_dim
    HALF     = crop_dim // 2

    pocket_center = ligand_coords.mean(axis=0)   # ligand CoM as crop anchor

    # ── Step 1: find EMDB-fitted PDB models ──────────────────────────────────
    if verbose:
        print(f"[1] Querying EMDB entry {emdb_id} for fitted PDB models …")
    fitted_pdbs = get_emdb_fitted_pdbs(emdb_id)
    if verbose:
        print(f"    Fitted models: {fitted_pdbs or 'none found'}")

    # ── Step 2: try to superimpose ────────────────────────────────────────────
    R, t, rmsd, n_aligned, emdb_pdb_id = None, None, np.inf, 0, ""
    fallback = True

    if fitted_pdbs:
        try:
            if verbose:
                print(f"[2] Downloading PDB structures …")
            xray_path = download_pdb(xray_pdb_id)
            xray_chains = extract_ca_chains(xray_path)
            if verbose:
                print(f"    {xray_pdb_id}: {sum(len(v['atoms']) for v in xray_chains.values())} Cα atoms")

            for candidate in fitted_pdbs[:3]:    # try up to 3 candidates
                try:
                    emdb_path   = download_pdb(candidate)
                    emdb_chains = extract_ca_chains(emdb_path)
                    if verbose:
                        print(f"    {candidate}: {sum(len(v['atoms']) for v in emdb_chains.values())} Cα atoms")
                    R_, t_, rms_, n_, _, _ = best_superposition(xray_chains, emdb_chains)
                    if rms_ < rmsd:
                        R, t, rmsd, n_aligned, emdb_pdb_id = R_, t_, rms_, n_, candidate
                except Exception as e:
                    if verbose:
                        print(f"    [skip] {candidate}: {e}")

        except Exception as e:
            if verbose:
                print(f"    [warn] Superposition failed: {e}")

    if R is not None:
        fallback = False
        pocket_center_emdb = R @ pocket_center + t
        if verbose:
            print(f"[3] Superposition done: RMSD={rmsd:.2f} Å  n={n_aligned}  model={emdb_pdb_id}")
            print(f"    Pocket CoM (X-ray): {pocket_center.round(2)}")
            print(f"    Pocket CoM (EMDB):  {pocket_center_emdb.round(2)}")
    else:
        # Fallback: assume the PDB and EMDB share the same coordinate origin
        R = np.eye(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)
        pocket_center_emdb = pocket_center.copy()
        emdb_pdb_id = fitted_pdbs[0] if fitted_pdbs else ""
        if verbose:
            print("[3] Fallback: no superposition — using raw PDB coordinates.")

    # ── Step 3: load map, crop, normalise ────────────────────────────────────
    if verbose:
        print(f"[4] Loading map {emdb_map_path.name} …")
    data_xyz, vs, origin = load_mrc_gz(emdb_map_path)
    data_norm = zscore(data_xyz)

    if verbose:
        print(f"    Map shape (XYZ): {data_xyz.shape}  voxel: {vs[0]:.3f} Å")

    crop, crop_origin = crop_at(data_norm, vs, origin, pocket_center_emdb)

    coverage = (crop != 0).sum() / crop.size
    if verbose:
        print(f"    Crop coverage: {100*coverage:.1f}%  "
              f"(non-zero voxels / {CROP_DIM}³)")

    return FitResult(
        crop               = crop,
        pocket_center_emdb = pocket_center_emdb.astype(np.float32),
        crop_origin_vox    = crop_origin,
        voxel_size         = vs.astype(np.float32),
        rotation           = R.astype(np.float32),
        translation        = t.astype(np.float32),
        rmsd               = float(rmsd) if not fallback else float("nan"),
        n_aligned          = n_aligned,
        xray_pdb_id        = xray_pdb_id.upper(),
        emdb_pdb_id        = emdb_pdb_id,
        emdb_id            = emdb_id,
        fallback           = fallback,
    )


# ── Convenience: naive crop (no superposition, for comparison) ────────────────

def naive_crop(
    pocket_coords: np.ndarray,
    ligand_coords: np.ndarray,
    emdb_map_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop the EMDB map directly at the ligand CoM in PDB coordinates.
    Returns (crop, voxel_size, crop_origin_vox).
    """
    data_xyz, vs, origin = load_mrc_gz(emdb_map_path)
    data_norm = zscore(data_xyz)
    center = ligand_coords.mean(axis=0)
    crop, crop_origin = crop_at(data_norm, vs, origin, center)
    return crop, vs.astype(np.float32), crop_origin


# ── CLI demo ──────────────────────────────────────────────────────────────────

def main():
    import json, torch

    DATA_DIR     = REPO_ROOT / "voxbind" / "dataset" / "data"
    MAPPING_FILE = NB_DATA_DIR / "crossdocked_pdb_to_emdb.json"

    with open(MAPPING_FILE) as f:
        cd_to_emdb = json.load(f)

    samples = torch.load(DATA_DIR / "data_test.pt", weights_only=False)

    # Find a sample with a downloaded EMDB map
    for pocket_, ligand_ in samples:
        pid      = pocket_["id"].split("/")[1].split("_")[0].upper()
        emdb_ids = [e for e in cd_to_emdb.get(pid, [])
                    if (EMDB_DIR / f"emd_{e.split('-')[-1]}.map.gz").exists()]
        if not emdb_ids:
            continue
        emdb_id  = emdb_ids[0]
        map_path = EMDB_DIR / f"emd_{emdb_id.split('-')[-1]}.map.gz"

        p_mask = pocket_["atoms_channel"].numpy() < 4
        l_mask = ligand_["atoms_channel"].numpy()  < 7
        p_coords = pocket_["coords"].numpy()[p_mask]
        l_coords = ligand_["coords"].numpy()[l_mask]

        print(f"\n{'='*60}")
        print(f"PDB: {pid}  EMDB: {emdb_id}")
        print(f"{'='*60}")

        result = fit_emdb_to_pocket(
            pocket_coords = p_coords,
            ligand_coords = l_coords,
            xray_pdb_id   = pid,
            emdb_id       = emdb_id,
            emdb_map_path = map_path,
            verbose       = True,
        )

        print(f"\nResult:")
        print(f"  crop shape   : {result.crop.shape}")
        print(f"  RMSD         : {result.rmsd:.2f} Å" if not result.fallback else "  RMSD: n/a (fallback)")
        print(f"  n_aligned    : {result.n_aligned}")
        print(f"  emdb_pdb_id  : {result.emdb_pdb_id}")
        print(f"  fallback     : {result.fallback}")
        break


if __name__ == "__main__":
    main()
