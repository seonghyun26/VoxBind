"""config.py — paths + hyperparameters for the standalone HBGSA baseline.

This baseline is fully self-contained: it reads ONLY the PDBbind data under
voxbind/dataset/data/pdbbind (structures + LP_PDBBind.csv / index.csv) and
imports nothing from the voxbind package. Reference: HBGSA, arXiv 2604.23115.
"""
from __future__ import annotations

from pathlib import Path

# ── locations ────────────────────────────────────────────────────────────────
HBGSA_DIR   = Path(__file__).resolve().parent.parent           # hbgsa_baseline/
PDBBIND_DIR = (HBGSA_DIR.parent / "voxbind" / "dataset" / "data" / "pdbbind").resolve()

STRUCT_DIR  = PDBBIND_DIR / "structures" / "pbpp-2020"          # {pdb}/{pdb}_*.pdb|.sdf
INDEX_CSV   = PDBBIND_DIR / "index.csv"                         # pdb_id,has_struct,new_split,pK,CL1,covalent,...
LP_CSV      = PDBBIND_DIR / "raw" / "LP_PDBBind.csv"            # +smiles,seq

CACHE_DIR   = HBGSA_DIR / "cache"                               # per-complex H-bond graphs
RESULTS_DIR = HBGSA_DIR / "results"
LOGS_DIR    = HBGSA_DIR / "logs"
for _d in (CACHE_DIR, RESULTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def complex_files(pdb_id: str) -> dict[str, Path]:
    """Absolute paths to the structure files of one PDBbind complex."""
    d = STRUCT_DIR / pdb_id
    return {
        "protein": d / f"{pdb_id}_protein.pdb",
        "pocket":  d / f"{pdb_id}_pocket.pdb",
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
