"""config.py — paths + hyperparameters for the standalone HBGSA baseline.

This baseline reads the PDBbind affinity labels from LP_PDBBind.csv and the
train/val/test membership from a frozen split manifest under voxbind/splits/
(pid,split). Raw structure files (protein/pocket/ligand) are resolved from the
targetdiff PDBbind-v2020 tree, since this repo keeps only voxels/features under
dataset/data/pdbbind. Reference: HBGSA, arXiv 2604.23115.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# ── locations ────────────────────────────────────────────────────────────────
HBGSA_DIR   = Path(__file__).resolve().parent.parent           # hbgsa_baseline/
REPO_DIR    = HBGSA_DIR.parent                                 # Voxbind/
PDBBIND_DIR = (REPO_DIR / "voxbind" / "dataset" / "data" / "pdbbind").resolve()

LP_CSV      = PDBBIND_DIR / "raw" / "LP_PDBBind.csv"           # pdb_id,smiles,seq,new_split,CL1,covalent,value(=pK)

# Frozen, version-controlled split manifests (pid,split). The split CSV is the
# single source of truth for both membership AND the train/val/test assignment.
SPLITS_DIR        = REPO_DIR / "voxbind" / "splits"
DEFAULT_SPLIT_CSV = SPLITS_DIR / "lp_edrscc_v2.csv"

# Raw protein/pocket/ligand structures (this repo stores only voxels). Resolved
# per-complex across the refined (pbpp-2020) and general PDBbind-v2020 dirs; the
# union covers every lp_edrscc pid.
STRUCT_BASES = [
    Path("/home/shpark/prj-ligand/targetdiff/data/pdbbind_v2020/pocket_10_general/structures/pbpp-2020"),
    Path("/home/shpark/prj-ligand/targetdiff/data/pdbbind_v2020/pocket_10_general/structures"),
    Path("/home/shpark/prj-ligand/targetdiff/data/pdbbind_v2020/extracted_otherPL/v2020-other-PL"),
]

CACHE_DIR   = HBGSA_DIR / "cache"                               # per-complex H-bond graphs
RESULTS_DIR = HBGSA_DIR / "results"
LOGS_DIR    = HBGSA_DIR / "logs"
for _d in (CACHE_DIR, RESULTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=None)
def _resolve_complex_dir(pdb_id: str) -> Path:
    """Directory holding one complex's files, searched across STRUCT_BASES.

    Prefer a base that has the full protein+pocket+ligand set (the pocket branch
    needs {pdb}_pocket.pdb); fall back to protein+ligand, then to bases[0]."""
    cands = [base / pdb_id for base in STRUCT_BASES]
    for d in cands:
        if ((d / f"{pdb_id}_protein.pdb").exists()
                and (d / f"{pdb_id}_pocket.pdb").exists()
                and (d / f"{pdb_id}_ligand.sdf").exists()):
            return d
    for d in cands:
        if (d / f"{pdb_id}_protein.pdb").exists() and (d / f"{pdb_id}_ligand.sdf").exists():
            return d
    return cands[0]


def complex_files(pdb_id: str) -> dict[str, Path]:
    """Absolute paths to the structure files of one PDBbind complex."""
    d = _resolve_complex_dir(pdb_id)
    return {
        "protein":     d / f"{pdb_id}_protein.pdb",
        "pocket":      d / f"{pdb_id}_pocket.pdb",
        "ligand_sdf":  d / f"{pdb_id}_ligand.sdf",
        "ligand_mol2": d / f"{pdb_id}_ligand.mol2",
    }


# ── H-bond detection (paper: PyMOL `distance ... mode=2`) ─────────────────────
HB_DIST_MAX   = 3.5      # donor-acceptor distance cutoff (Å)
HB_ANGLE_MIN  = 120.0    # D-H···A angle cutoff (deg)
HB_POLAR_ELEMS = ("N", "O", "S")
HB_MAX_NODES  = 20       # paper: up to 20 H-bonds as graph nodes
HB_KNN_K      = 5        # dynamic KNN over H-bond midpoints

# ── sequence / pocket / SMILES branches ──────────────────────────────────────
SEQ_MAX_LEN    = 1000    # truncate protein sequence (paper uses full seq; capped for cost)
POCKET_MAX_LEN = 128     # binding-pocket residues (from {pdb}_pocket.pdb)
SMILES_MAX_LEN = 200     # truncate SMILES token stream

# ── model dims ───────────────────────────────────────────────────────────────
EMB_DIM    = 128         # per-branch embedding (paper: each branch → 128)
GCN_HIDDEN = 128

# ── seq / SMILES dilated-conv residual tower (paper v_seq/v_smi front-end) ────
# "Multi-scale dilated convolutions with dilated residual blocks, followed by 1D
# self-attention." conv_channels is the tower width = the main capacity knob;
# the default lands the full model near the paper's 3.06M. Set CONV_DILATIONS=()
# for the attention-only branch (the earlier ~0.77M reproduction).
CONV_CHANNELS   = 213            # tower width (tuned so the full model ≈ 3.06M)
CONV_DILATIONS  = (1, 2, 4, 8)   # one residual block per dilation (multi-scale)
CONVS_PER_BLOCK = 2              # dilated convs per residual block

# ── training ─────────────────────────────────────────────────────────────────
PEARSON_LAMBDA = 50.0    # hybrid loss: SmoothL1 + λ·(1 - Pearson)
