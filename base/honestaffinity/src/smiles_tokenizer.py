"""smiles_tokenizer.py — fixed 53-token char-level SMILES tokenizer.

SPEC calls for a 53-token learned vocab. The union of SMILES characters across all four
splits' ligands (LP_PDBBind, with structure-SDF fallbacks) is exactly 35 distinct chars.
We use a fixed vocabulary of 53 slots:
    index 0            = <pad>
    index 1            = <unk>
    indices 2 .. 36    = the 35 dataset SMILES characters (data-covering -> 0 UNK)
    indices 37 .. 52   = 16 reserved/spare slots (unused, keep count == 53)
Char-level tokenization: two-char atoms (Cl, Br) enter as their single letters
(C+l, B+r); the multi-scale conv (kernels up to 7) reassembles them contextually.
This gives an exact 53-entry learned embedding table (nn.Embedding(53,256)).
"""

PAD = "<pad>"
UNK = "<unk>"

# 35 characters observed across every split's ligand SMILES (frequency-ordered, fixed).
_CHARS = list("Cc()O@[]H=N12n3+-4SFlP/s5o#\\67.89%0")
assert len(_CHARS) == len(set(_CHARS)) == 35, f"expected 35 unique chars, got {len(set(_CHARS))}"

# reserved spare slots to reach exactly 53 (never emitted by the tokenizer)
_RESERVED = [f"<r{i}>" for i in range(53 - 2 - len(_CHARS))]  # 16 slots

VOCAB = [PAD, UNK] + _CHARS + _RESERVED
assert len(VOCAB) == 53, f"vocab size {len(VOCAB)} != 53"
STOI = {c: i for i, c in enumerate(VOCAB)}
UNK_ID = STOI[UNK]
PAD_ID = STOI[PAD]


def encode(smiles, max_len=200):
    """SMILES string -> list[int] token ids (char-level), truncated to max_len.

    Non-string input (a NaN/None SMILES — one v2 complex has an unresolved ligand SMILES)
    returns [] so the caller's empty-guard maps it to a single PAD token instead of crashing.
    """
    if not isinstance(smiles, str):
        return []
    return [STOI.get(ch, UNK_ID) for ch in smiles[:max_len]]


def vocab_size():
    return len(VOCAB)
