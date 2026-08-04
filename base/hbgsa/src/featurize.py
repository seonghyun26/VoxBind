"""featurize.py — protein-sequence and SMILES featurizers for HBGSA branches.

Protein sequence  → per-residue physicochemical descriptor matrix (validated
                    scales; see note below) for a 1D self-attention branch.
SMILES            → regex atom-level token ids for a 1D self-attention branch.

NOTE on fidelity: HBGSA (arXiv 2604.23115) specifies a "40-dim physicochemical"
per-residue vector (hydrophobicity, charge, polarity, secondary-structure
propensity) but does not publish the exact 40 descriptors. We use a validated,
reproducible descriptor set covering the same property categories rather than
fabricating AAindex values; the branch then projects it to the model width.
"""
from __future__ import annotations

import re

import numpy as np

# ── per-residue physicochemical descriptors (validated sources) ───────────────
from Bio.Data.IUPACData import protein_weights
from Bio.SeqUtils.ProtParamData import Flex, em, hw, ja, kd

AA = "ACDEFGHIKLMNPQRSTVWY"
_CHARGE   = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5}
_POLAR    = set("STNQYCKRHDEW")
_AROMATIC = set("FWYH")


def _aa_descriptor_matrix() -> tuple[np.ndarray, list[str]]:
    """(21, F) z-normalized descriptor matrix; row 20 = unknown 'X' (zeros)."""
    names = ["kd", "hw", "flex", "em", "ja", "charge", "polar", "aromatic", "mw"]
    rows = []
    for a in AA:
        rows.append([
            kd[a], hw[a], Flex[a], em[a], ja[a],
            _CHARGE.get(a, 0.0),
            1.0 if a in _POLAR else 0.0,
            1.0 if a in _AROMATIC else 0.0,
            protein_weights[a],
        ])
    M = np.asarray(rows, dtype=np.float32)                  # (20, F)
    mu, sd = M.mean(0, keepdims=True), M.std(0, keepdims=True) + 1e-6
    M = (M - mu) / sd
    M = np.vstack([M, np.zeros((1, M.shape[1]), np.float32)])  # 'X' → zeros
    return M, names


AA_DESC, AA_DESC_NAMES = _aa_descriptor_matrix()
SEQ_FEAT_DIM = AA_DESC.shape[1]
_AA_IDX = {a: i for i, a in enumerate(AA)}


def encode_sequence(seq: str, max_len: int) -> np.ndarray:
    """Protein sequence → (L, SEQ_FEAT_DIM) physicochemical descriptors (L ≤ max_len)."""
    seq = (seq or "").upper()[:max_len]
    idx = [_AA_IDX.get(c, 20) for c in seq] or [20]
    return AA_DESC[idx]


def pocket_sequence_from_pdb(pdb_path: str) -> str:
    """One-letter residue sequence of a binding-pocket PDB (for the v_pkt branch)."""
    from Bio.PDB import PDBParser, is_aa
    from Bio.SeqUtils import seq1
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("pkt", pdb_path)
    except Exception:
        return ""
    out = []
    for model in structure:                       # first model only
        for chain in model:
            for res in chain:
                if is_aa(res, standard=False):
                    out.append(seq1(res.resname, undef_code="X"))
        break
    return "".join(out)


# ── SMILES regex atom-level tokenizer (Schwaller et al. pattern) ──────────────
_SMILES_RE = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)
PAD_ID, UNK_ID = 0, 1


def smiles_tokens(smiles: str) -> list[str]:
    return _SMILES_RE.findall(smiles or "")


def build_smiles_vocab(smiles_iter) -> dict[str, int]:
    """Vocab from an iterable of SMILES; reserves 0=PAD, 1=UNK."""
    toks = set()
    for s in smiles_iter:
        toks.update(smiles_tokens(s))
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for t in sorted(toks):
        vocab[t] = len(vocab)
    return vocab


def encode_smiles(smiles: str, vocab: dict[str, int], max_len: int) -> np.ndarray:
    ids = [vocab.get(t, UNK_ID) for t in smiles_tokens(smiles)][:max_len]
    if not ids:
        ids = [UNK_ID]
    return np.asarray(ids, dtype=np.int64)


if __name__ == "__main__":
    print("SEQ_FEAT_DIM:", SEQ_FEAT_DIM, "descriptors:", AA_DESC_NAMES)
    print("AA_DESC shape:", AA_DESC.shape)
    s = "CSc1ccccc1[C@H]1CCCN1C(=O)CNC(=O)"
    print("smiles tokens:", smiles_tokens(s)[:20])
    print("seq encode shape:", encode_sequence("MKTAYIAKQR", 1000).shape)
