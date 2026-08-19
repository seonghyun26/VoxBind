#!/usr/bin/env python3
"""CPU-only sequence/ligand/affinity analysis of cached complex representations.

The script never instantiates an encoder.  It loads existing ``pid -> vector``
caches on CPU, finds homologous protein pairs with the bundled CPU MMseqs2, and
asks whether representation similarity follows protein sequence, ligand
similarity, and experimental affinity differences.

With no ``--representation`` arguments it compares the four caches that cover
the complete canonical ``lp_edrscc_v2`` cohort:

* VoxBind-C (coordinates only)
* VoxBind-CDG (coordinates + density + density-gradient magnitude)
* IPNet-frozen
* DSMBind-frozen

Additional methods can be supplied as ``--representation NAME=PATH``.  PyTorch
caches with a ``features``, ``feats``, or ``feat`` mapping are auto-detected;
``.npz`` files containing ``pids`` and ``features`` are also supported.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# This is an analysis of precomputed caches.  Hide accelerators before torch is
# imported so an accidental CUDA context cannot be created.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = REPO / "voxbind/dataset/data/pdbbind/raw/LP_PDBBind.csv"
DEFAULT_SPLIT = REPO / "voxbind/splits/lp_edrscc_v2.csv"
DEFAULT_MMSEQS = REPO / "base/_tools/mmseqs/bin/mmseqs"
DEFAULT_OUTPUT = REPO / "voxbind/dataset/data/pdbbind/representation_similarity/current4"
DEFAULT_POCKET_RECORDS = REPO / "base/honestaffinity/cache/records_lp_edrscc_v2.json"
DEFAULT_REPRESENTATIONS = {
    "VoxBind-C": REPO / (
        "voxbind/dataset/data/pdbbind/features/"
        "atomblob_e49_v5_260723_ar_cvit_100m_v2_mask075_coords.pt"
    ),
    "VoxBind-CDG": REPO / (
        "voxbind/dataset/data/pdbbind/features/"
        "atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_mask075.pt"
    ),
    "IPNet-frozen": REPO / "base/ipdiff/_edrscc/feats_all.pt",
    "DSMBind-frozen": REPO / "base/dsmbind/_edrscc/features/lp_edrscc_v2.pt",
}
AFFINITY_BASELINE_REPRESENTATIONS = {
    "CheapNet-seed0": REPO / (
        "base/cheapnet/_edrscc/features/cheapnet_casf_seed0_prehead.pt"
    ),
    "GET-seed0": REPO / "base/get/_edrscc/features/get_v2_seed0_graph_repr.pt",
}

SEQ_BIN_ORDER = ["non_hit", "20-30%", "30-60%", "60-90%", "90-<100%", "exact"]
POCKET_SEQ_BIN_ORDER = ["<20%", "20-40%", "40-60%", "60-80%", "80-<100%", "exact"]
LIGAND_CONTROL_ORDER = ["all_ligands", "ligand_similarity>=0.8"]
CASE_ORDER = [
    "near_seq_near_lig_smooth",
    "near_seq_near_lig_cliff",
    "exact_seq_ligand_shift",
    "near_seq_ligand_shift",
    "remote_seq_near_lig_smooth",
    "remote_seq_near_lig_divergent",
]


@dataclass(frozen=True)
class Representation:
    name: str
    path: Path
    feature_key: str
    vectors: dict[str, np.ndarray]

    @property
    def dim(self) -> int:
        return int(next(iter(self.vectors.values())).size)


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return value or "representation"


def parse_representation_specs(values: list[str]) -> dict[str, Path]:
    if not values:
        return DEFAULT_REPRESENTATIONS.copy()
    specs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"representation must be NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty representation name in {value!r}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (REPO / path).resolve()
        if name in specs:
            raise ValueError(f"duplicate representation name: {name!r}")
        specs[name] = path
    return specs


def _as_vector(value, pid: str, path: Path) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"empty vector for pid={pid!r} in {path}")
    if not np.isfinite(vector).all():
        raise ValueError(f"non-finite vector for pid={pid!r} in {path}")
    return vector


def load_representation(name: str, path: Path) -> Representation:
    if not path.exists():
        raise FileNotFoundError(f"missing representation cache: {path}")

    feature_key = ""
    raw: dict | None = None
    if path.suffix == ".npz":
        bundle = np.load(path, allow_pickle=False)
        if "pids" not in bundle or "features" not in bundle:
            raise ValueError(f"{path}: .npz requires pids and features arrays")
        pids = [str(pid).lower() for pid in bundle["pids"]]
        matrix = np.asarray(bundle["features"], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(pids):
            raise ValueError(f"{path}: invalid feature matrix shape {matrix.shape}")
        raw = dict(zip(pids, matrix))
        feature_key = "features"
    else:
        import torch

        bundle = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(bundle, dict):
            raise ValueError(f"{path}: expected a dict-like cache")
        for key in ("features", "feats", "feat"):
            if isinstance(bundle.get(key), dict):
                raw = bundle[key]
                feature_key = key
                break
        if raw is None and bundle and all(
            hasattr(v, "shape") or isinstance(v, (list, tuple)) for v in bundle.values()
        ):
            raw = bundle
            feature_key = "<root>"
    if raw is None:
        raise ValueError(f"{path}: could not find a pid -> vector mapping")

    vectors = {str(pid).lower(): _as_vector(value, str(pid), path) for pid, value in raw.items()}
    dims = {vector.size for vector in vectors.values()}
    if len(dims) != 1:
        raise ValueError(f"{path}: inconsistent vector dimensions {sorted(dims)}")
    return Representation(name=name, path=path, feature_key=feature_key, vectors=vectors)


def load_metadata(index_path: Path, split_path: Path) -> pd.DataFrame:
    index = pd.read_csv(index_path).rename(columns={"Unnamed: 0": "pid", "value": "pK"})
    required = {"pid", "seq", "smiles", "pK"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"{index_path}: missing columns {sorted(missing)}")
    index["pid"] = index["pid"].astype(str).str.lower()
    index = index.drop_duplicates("pid", keep="first")

    split = pd.read_csv(split_path)
    if not {"pid", "split"}.issubset(split.columns):
        raise ValueError(f"{split_path}: expected pid,split columns")
    split["pid"] = split["pid"].astype(str).str.lower()

    meta = split.merge(index[["pid", "seq", "smiles", "pK"]], on="pid", how="left")
    meta["pK"] = pd.to_numeric(meta["pK"], errors="coerce")
    valid = meta["seq"].notna() & meta["smiles"].notna() & np.isfinite(meta["pK"])
    if not valid.all():
        print(f"[metadata] dropping {(~valid).sum()} rows without seq/smiles/pK", file=sys.stderr)
    meta = meta.loc[valid].copy()
    meta["seq"] = meta["seq"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    meta["smiles"] = meta["smiles"].astype(str)
    return meta.reset_index(drop=True)


def cohort_for(representations: list[Representation], meta: pd.DataFrame) -> pd.DataFrame:
    common = set(meta["pid"])
    for rep in representations:
        common &= set(rep.vectors)
    cohort = meta[meta["pid"].isin(common)].copy().sort_values("pid").reset_index(drop=True)
    if len(cohort) < 2:
        raise ValueError(f"fewer than two common complexes across {len(representations)} caches")
    return cohort


def write_fasta(meta: pd.DataFrame, path: Path) -> None:
    with path.open("w") as handle:
        for row in meta.itertuples(index=False):
            handle.write(f">{row.pid}\n{row.seq}\n")


def run_mmseqs(
    mmseqs: Path,
    fasta: Path,
    output: Path,
    tmp_dir: Path,
    threads: int,
    min_seq_id: float,
    coverage: float,
) -> None:
    if not mmseqs.exists():
        raise FileNotFoundError(f"MMseqs2 binary not found: {mmseqs}")
    tmp_dir.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    cmd = [
        str(mmseqs),
        "easy-search",
        str(fasta),
        str(fasta),
        str(output),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        "0",
        "--alignment-mode",
        "3",
        "--max-seqs",
        "10000",
        "--threads",
        str(threads),
        "--format-output",
        "query,target,fident,alnlen,qcov,tcov,evalue,bits",
        "-v",
        "1",
    ]
    print("[sequence] " + " ".join(cmd[:6]) + " ...")
    subprocess.run(cmd, check=True)


def load_homolog_pairs(path: Path, pid_to_index: dict[str, int], n: int) -> pd.DataFrame:
    columns = ["query", "target", "seq_identity", "aln_len", "qcov", "tcov", "evalue", "bits"]
    hits = pd.read_csv(path, sep="\t", names=columns, header=None)
    hits["query"] = hits["query"].astype(str).str.lower()
    hits["target"] = hits["target"].astype(str).str.lower()
    hits = hits[hits["query"].isin(pid_to_index) & hits["target"].isin(pid_to_index)].copy()
    hits["i"] = hits["query"].map(pid_to_index).astype(np.int32)
    hits["j"] = hits["target"].map(pid_to_index).astype(np.int32)
    hits = hits[hits["i"] != hits["j"]].copy()
    lo = np.minimum(hits["i"].to_numpy(), hits["j"].to_numpy())
    hi = np.maximum(hits["i"].to_numpy(), hits["j"].to_numpy())
    hits["i"], hits["j"] = lo, hi
    for column in ("seq_identity", "qcov", "tcov"):
        values = pd.to_numeric(hits[column], errors="coerce").astype(float)
        if len(values) and values.max() > 1.0:
            values /= 100.0
        hits[column] = values.clip(0.0, 1.0)
    hits["pair_key"] = hits["i"].astype(np.int64) * n + hits["j"].astype(np.int64)
    hits = hits.sort_values(["seq_identity", "bits"], ascending=False).drop_duplicates("pair_key")
    hits["homology_hit"] = True
    hits["control_type"] = "homolog"
    return hits[
        [
            "i",
            "j",
            "pair_key",
            "homology_hit",
            "control_type",
            "seq_identity",
            "aln_len",
            "qcov",
            "tcov",
        ]
    ]


def cap_homolog_pairs(hits: pd.DataFrame, maximum: int, rng: np.random.Generator) -> pd.DataFrame:
    if maximum <= 0 or len(hits) <= maximum:
        return hits
    # Keep all close homologs/activity-cliff candidates.  Sample only the lower
    # identity bulk if the result grows too large.
    close = hits[hits["seq_identity"] >= 0.60]
    lower = hits[hits["seq_identity"] < 0.60]
    allowance = max(maximum - len(close), 0)
    if allowance and len(lower) > allowance:
        lower = lower.iloc[rng.choice(len(lower), size=allowance, replace=False)]
    elif allowance == 0:
        lower = lower.iloc[:0]
    return pd.concat([close, lower], ignore_index=True)


def sample_non_hits(
    count: int,
    n: int,
    forbidden: set[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    if count <= 0:
        return pd.DataFrame(columns=["i", "j", "pair_key", "homology_hit", "control_type"])
    chosen: set[int] = set()
    attempts = 0
    max_attempts = max(count * 50, 10000)
    while len(chosen) < count and attempts < max_attempts:
        batch = min(max((count - len(chosen)) * 3, 1000), 100000)
        left = rng.integers(0, n, size=batch, dtype=np.int64)
        right = rng.integers(0, n, size=batch, dtype=np.int64)
        lo, hi = np.minimum(left, right), np.maximum(left, right)
        for i, j in zip(lo, hi):
            attempts += 1
            if i == j:
                continue
            key = int(i * n + j)
            if key not in forbidden:
                chosen.add(key)
                if len(chosen) >= count:
                    break
    keys = np.fromiter(chosen, dtype=np.int64)
    return pd.DataFrame(
        {
            "i": (keys // n).astype(np.int32),
            "j": (keys % n).astype(np.int32),
            "pair_key": keys,
            "homology_hit": False,
            "control_type": "random_non_hit",
            "seq_identity": np.nan,
            "aln_len": np.nan,
            "qcov": np.nan,
            "tcov": np.nan,
        }
    )


def sample_same_ligand_non_hits(
    meta: pd.DataFrame,
    count: int,
    n: int,
    forbidden: set[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Reservoir-sample exact-ligand pairs absent from the homology search.

    Uniform random protein pairs almost never share a ligand.  This targeted
    control makes the low-sequence/high-ligand quadrant observable without
    enumerating every molecular-fingerprint pair in the cohort.
    """
    if count <= 0:
        return pd.DataFrame(columns=["i", "j", "pair_key", "homology_hit", "control_type"])
    try:
        from rdkit import Chem, RDLogger
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RDKit is required for ligand controls") from exc

    RDLogger.DisableLog("rdApp.*")
    groups: dict[str, list[int]] = {}
    for idx, smiles in enumerate(meta["smiles"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        groups.setdefault(canonical, []).append(idx)

    reservoir: list[int] = []
    n_seen = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        for left_pos, i in enumerate(members[:-1]):
            for j in members[left_pos + 1 :]:
                key = int(i * n + j) if i < j else int(j * n + i)
                if key in forbidden:
                    continue
                n_seen += 1
                if len(reservoir) < count:
                    reservoir.append(key)
                else:
                    replace = int(rng.integers(0, n_seen))
                    if replace < count:
                        reservoir[replace] = key
    keys = np.asarray(reservoir, dtype=np.int64)
    return pd.DataFrame(
        {
            "i": (keys // n).astype(np.int32),
            "j": (keys % n).astype(np.int32),
            "pair_key": keys,
            "homology_hit": False,
            "control_type": "same_ligand_non_hit",
            "seq_identity": np.nan,
            "aln_len": np.nan,
            "qcov": np.nan,
            "tcov": np.nan,
        }
    )


def add_pair_metadata(pairs: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    i = pairs["i"].to_numpy(dtype=np.int64)
    j = pairs["j"].to_numpy(dtype=np.int64)
    pids = meta["pid"].to_numpy()
    seqs = meta["seq"].to_numpy()
    pks = meta["pK"].to_numpy(dtype=float)
    splits = meta["split"].to_numpy()

    pairs = pairs.copy()
    pairs["pid_a"], pairs["pid_b"] = pids[i], pids[j]
    pairs["split_a"], pairs["split_b"] = splits[i], splits[j]
    pairs["pK_a"], pairs["pK_b"] = pks[i], pks[j]
    pairs["affinity_gap"] = np.abs(pks[i] - pks[j])
    pairs["same_sequence"] = np.fromiter((seqs[a] == seqs[b] for a, b in zip(i, j)), dtype=bool)
    pairs.loc[pairs["same_sequence"], "seq_identity"] = 1.0
    pairs["length_ratio"] = np.fromiter(
        (min(len(seqs[a]), len(seqs[b])) / max(len(seqs[a]), len(seqs[b])) for a, b in zip(i, j)),
        dtype=float,
    )
    return pairs


def add_ligand_similarity(pairs: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:  # pragma: no cover - available in the project environment
        raise RuntimeError("RDKit is required for ligand controls") from exc

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    canonical = []
    for smiles in meta["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        fingerprints.append(generator.GetFingerprint(mol) if mol is not None else None)
        canonical.append(Chem.MolToSmiles(mol, canonical=True) if mol is not None else "")

    ii = pairs["i"].to_numpy(dtype=np.int64)
    jj = pairs["j"].to_numpy(dtype=np.int64)
    similarities = np.full(len(pairs), np.nan, dtype=np.float32)
    for row, (i, j) in enumerate(zip(ii, jj)):
        if fingerprints[i] is not None and fingerprints[j] is not None:
            similarities[row] = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
    pairs = pairs.copy()
    pairs["ligand_similarity"] = similarities
    pairs["same_ligand"] = np.fromiter((canonical[i] == canonical[j] for i, j in zip(ii, jj)), dtype=bool)
    return pairs


def load_pocket_sequences(path: Path) -> dict[str, str]:
    """Load structure-derived pocket residues in their parent-sequence order."""
    if not path.exists():
        raise FileNotFoundError(f"missing pocket records: {path}")
    bundle = json.loads(path.read_text())
    records = bundle.get("records") if isinstance(bundle, dict) else bundle
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a records list")

    pockets: dict[str, str] = {}
    for record in records:
        pid = str(record.get("pid", "")).lower()
        sequence = str(record.get("seq", "")).upper()
        mask = record.get("pocket_mask", [])
        if not pid or len(sequence) != len(mask):
            continue
        pocket = "".join(amino_acid for amino_acid, keep in zip(sequence, mask) if keep)
        if pocket:
            pockets[pid] = pocket
    return pockets


def _aligned_identity(sequence_a: str, sequence_b: str, aligner) -> float:
    if sequence_a == sequence_b:
        return 1.0
    alignment = aligner.align(sequence_a, sequence_b)[0]
    coordinates = np.asarray(alignment.coordinates)
    matches = 0
    aligned_columns = 0
    for column in range(coordinates.shape[1] - 1):
        a_start, a_stop = coordinates[0, column : column + 2]
        b_start, b_stop = coordinates[1, column : column + 2]
        a_start, a_stop = int(a_start), int(a_stop)
        b_start, b_stop = int(b_start), int(b_stop)
        a_width, b_width = a_stop - a_start, b_stop - b_start
        aligned_columns += max(a_width, b_width)
        if a_width and b_width:
            matches += sum(
                left == right
                for left, right in zip(sequence_a[a_start:a_stop], sequence_b[b_start:b_stop])
            )
    return matches / aligned_columns if aligned_columns else np.nan


def add_pocket_sequence_similarity(
    pairs: pd.DataFrame,
    meta: pd.DataFrame,
    pocket_sequences: dict[str, str],
) -> pd.DataFrame:
    """Add BLOSUM62-guided global-alignment identity for pocket residues."""
    try:
        from Bio.Align import PairwiseAligner, substitution_matrices
    except ImportError as exc:  # pragma: no cover - available in the project environment
        raise RuntimeError("Biopython is required for pocket-sequence similarity") from exc

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    alphabet = set(str(aligner.substitution_matrix.alphabet))

    def sanitize(sequence: str) -> str:
        return "".join(amino_acid if amino_acid in alphabet else "X" for amino_acid in sequence)

    ordered = [sanitize(pocket_sequences.get(pid, "")) for pid in meta["pid"]]
    ii = pairs["i"].to_numpy(dtype=np.int64)
    jj = pairs["j"].to_numpy(dtype=np.int64)
    identities = np.full(len(pairs), np.nan, dtype=np.float32)
    exact = np.zeros(len(pairs), dtype=bool)
    lengths = np.asarray([len(sequence) for sequence in ordered], dtype=np.int32)
    for row, (i, j) in enumerate(zip(ii, jj)):
        sequence_a, sequence_b = ordered[i], ordered[j]
        if sequence_a and sequence_b:
            identities[row] = _aligned_identity(sequence_a, sequence_b, aligner)
            exact[row] = sequence_a == sequence_b

    pairs = pairs.copy()
    pairs["pocket_seq_identity"] = identities
    pairs["same_pocket_sequence"] = exact
    pairs["pocket_len_a"], pairs["pocket_len_b"] = lengths[ii], lengths[jj]
    pairs["pocket_length_ratio"] = np.divide(
        np.minimum(lengths[ii], lengths[jj]),
        np.maximum(lengths[ii], lengths[jj]),
        out=np.full(len(pairs), np.nan, dtype=np.float32),
        where=np.maximum(lengths[ii], lengths[jj]) > 0,
    )
    return pairs


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def add_representation_metrics(
    pairs: pd.DataFrame,
    meta: pd.DataFrame,
    representations: list[Representation],
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    ii = pairs["i"].to_numpy(dtype=np.int64)
    jj = pairs["j"].to_numpy(dtype=np.int64)
    pids = meta["pid"].tolist()
    columns: dict[str, dict[str, str]] = {}
    pairs = pairs.copy()
    used_slugs: set[str] = set()
    chunk_size = 20000

    for rep in representations:
        matrix = np.stack([rep.vectors[pid] for pid in pids]).astype(np.float32)
        slug = slugify(rep.name)
        if slug in used_slugs:
            raise ValueError(f"representation names collide after slugification: {rep.name!r}")
        used_slugs.add(slug)

        raw = _row_normalize(matrix)
        centered_matrix = matrix - matrix.mean(axis=0, keepdims=True)
        centered = _row_normalize(centered_matrix)
        scale = matrix.std(axis=0, keepdims=True)
        z = (matrix - matrix.mean(axis=0, keepdims=True)) / np.maximum(scale, 1e-6)
        standardized = _row_normalize(z)

        raw_col = f"{slug}__cosine"
        centered_col = f"{slug}__centered_cosine"
        standardized_col = f"{slug}__standardized_cosine"
        zdist_col = f"{slug}__zdist"
        raw_values = np.empty(len(pairs), dtype=np.float32)
        centered_values = np.empty(len(pairs), dtype=np.float32)
        standardized_values = np.empty(len(pairs), dtype=np.float32)
        zdist_values = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), chunk_size):
            stop = min(start + chunk_size, len(pairs))
            left, right = ii[start:stop], jj[start:stop]
            raw_values[start:stop] = np.einsum("ij,ij->i", raw[left], raw[right])
            centered_values[start:stop] = np.einsum(
                "ij,ij->i", centered[left], centered[right]
            )
            standardized_values[start:stop] = np.einsum(
                "ij,ij->i", standardized[left], standardized[right]
            )
            zdist_values[start:stop] = np.sqrt(
                np.mean(np.square(z[left] - z[right]), axis=1)
            )
        pairs[raw_col] = raw_values
        pairs[centered_col] = centered_values
        pairs[standardized_col] = standardized_values
        pairs[zdist_col] = zdist_values
        columns[rep.name] = {
            "raw_cosine": raw_col,
            "centered_cosine": centered_col,
            "similarity": standardized_col,
            "zdist": zdist_col,
        }
    return pairs, columns


def assign_sequence_bins(pairs: pd.DataFrame) -> pd.DataFrame:
    identity = pairs["seq_identity"].to_numpy(dtype=float)
    bins = np.full(len(pairs), "non_hit", dtype=object)
    bins[(identity >= 0.20) & (identity < 0.30)] = "20-30%"
    bins[(identity >= 0.30) & (identity < 0.60)] = "30-60%"
    bins[(identity >= 0.60) & (identity < 0.90)] = "60-90%"
    # MMseqs may report 100% identity on an aligned region even when one full
    # sequence has an insertion/deletion.  Reserve "exact" for full string
    # equality; every non-exact >=90% hit stays in the neighboring bin.
    bins[identity >= 0.90] = "90-<100%"
    bins[pairs["same_sequence"].to_numpy(dtype=bool)] = "exact"
    pairs = pairs.copy()
    pairs["sequence_bin"] = pd.Categorical(bins, categories=SEQ_BIN_ORDER, ordered=True)
    return pairs


def assign_pocket_sequence_bins(pairs: pd.DataFrame) -> pd.DataFrame:
    identity = pairs["pocket_seq_identity"].to_numpy(dtype=float)
    bins = np.full(len(pairs), None, dtype=object)
    bins[(identity >= 0.00) & (identity < 0.20)] = "<20%"
    bins[(identity >= 0.20) & (identity < 0.40)] = "20-40%"
    bins[(identity >= 0.40) & (identity < 0.60)] = "40-60%"
    bins[(identity >= 0.60) & (identity < 0.80)] = "60-80%"
    bins[identity >= 0.80] = "80-<100%"
    bins[pairs["same_pocket_sequence"].to_numpy(dtype=bool)] = "exact"
    pairs = pairs.copy()
    pairs["pocket_sequence_bin"] = pd.Categorical(
        bins, categories=POCKET_SEQ_BIN_ORDER, ordered=True
    )
    return pairs


def classify_cases(pairs: pd.DataFrame) -> pd.DataFrame:
    seq = pairs["seq_identity"]
    lig = pairs["ligand_similarity"]
    gap = pairs["affinity_gap"]
    masks = {
        "near_seq_near_lig_smooth": (seq >= 0.90) & (lig >= 0.80) & (gap <= 0.50),
        "near_seq_near_lig_cliff": (seq >= 0.90) & (lig >= 0.80) & (gap >= 1.00),
        "exact_seq_ligand_shift": pairs["same_sequence"] & (lig <= 0.30),
        "near_seq_ligand_shift": (seq >= 0.90) & (~pairs["same_sequence"]) & (lig <= 0.30),
        "remote_seq_near_lig_smooth": (~pairs["homology_hit"]) & (lig >= 0.80) & (gap <= 0.50),
        "remote_seq_near_lig_divergent": (~pairs["homology_hit"]) & (lig >= 0.80) & (gap >= 1.00),
    }
    frames = []
    for case, mask in masks.items():
        frame = pairs.loc[mask].copy()
        frame.insert(0, "case", case)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["case", *pairs.columns])
    return pd.concat(frames, ignore_index=True)


def _safe_spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return None
    value = float(spearmanr(x[valid], y[valid]).statistic)
    return value if np.isfinite(value) else None


def _lower_similarity_auc(smooth: Iterable[float], divergent: Iterable[float]) -> float | None:
    """AUC for using lower similarity to identify the divergent/cliff group."""
    smooth = np.asarray(list(smooth), dtype=float)
    divergent = np.asarray(list(divergent), dtype=float)
    smooth = smooth[np.isfinite(smooth)]
    divergent = divergent[np.isfinite(divergent)]
    if not len(smooth) or not len(divergent):
        return None
    scores = np.concatenate([-smooth, -divergent])
    ranks = rankdata(scores, method="average")
    n_negative, n_positive = len(smooth), len(divergent)
    positive_rank_sum = ranks[n_negative:].sum()
    auc = (positive_rank_sum - n_positive * (n_positive + 1) / 2) / (
        n_positive * n_negative
    )
    return float(auc)


def build_summaries(
    pairs: pd.DataFrame,
    cases: pd.DataFrame,
    metric_columns: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    seq_rows = []
    pocket_seq_rows = []
    case_rows = []
    diagnostics: dict[str, dict] = {}

    for model, columns in metric_columns.items():
        sim_col = columns["similarity"]
        sequence_sets = {
            # Same-ligand pairs were deliberately oversampled, so exclude them
            # from the marginal curve.  Include them in the explicitly
            # ligand-controlled curve, where that stratification is desired.
            "all_ligands": pairs[pairs["control_type"] != "same_ligand_non_hit"],
            "ligand_similarity>=0.8": pairs[pairs["ligand_similarity"] >= 0.80],
        }
        for ligand_control, sequence_pairs in sequence_sets.items():
            for seq_bin in SEQ_BIN_ORDER:
                values = sequence_pairs.loc[
                    sequence_pairs["sequence_bin"] == seq_bin, sim_col
                ].dropna().to_numpy()
                if len(values):
                    seq_rows.append(
                        {
                            "model": model,
                            "ligand_control": ligand_control,
                            "sequence_bin": seq_bin,
                            "n": len(values),
                            "mean_similarity": float(values.mean()),
                            "median_similarity": float(np.median(values)),
                            "std_similarity": float(values.std()),
                            "sem_similarity": float(values.std() / np.sqrt(len(values))),
                        }
                    )

            for pocket_seq_bin in POCKET_SEQ_BIN_ORDER:
                values = sequence_pairs.loc[
                    sequence_pairs["pocket_sequence_bin"] == pocket_seq_bin, sim_col
                ].dropna().to_numpy()
                if len(values):
                    pocket_seq_rows.append(
                        {
                            "model": model,
                            "ligand_control": ligand_control,
                            "pocket_sequence_bin": pocket_seq_bin,
                            "n": len(values),
                            "mean_similarity": float(values.mean()),
                            "median_similarity": float(np.median(values)),
                            "std_similarity": float(values.std()),
                            "sem_similarity": float(values.std() / np.sqrt(len(values))),
                        }
                    )

        for case in CASE_ORDER:
            values = cases.loc[cases["case"] == case, sim_col].dropna().to_numpy()
            if len(values):
                case_rows.append(
                    {
                        "model": model,
                        "case": case,
                        "n": len(values),
                        "mean_similarity": float(values.mean()),
                        "median_similarity": float(np.median(values)),
                        "std_similarity": float(values.std()),
                    }
                )

        homologs = pairs[pairs["homology_hit"]]
        near_ligand_homologs = homologs[homologs["ligand_similarity"] >= 0.80]
        exact_seq = pairs[pairs["same_sequence"]]
        cliff = cases[cases["case"] == "near_seq_near_lig_cliff"]
        smooth = cases[cases["case"] == "near_seq_near_lig_smooth"]
        remote_smooth = cases[cases["case"] == "remote_seq_near_lig_smooth"]
        remote_divergent = cases[cases["case"] == "remote_seq_near_lig_divergent"]
        diagnostics[model] = {
            "spearman_sequence_identity_vs_similarity": _safe_spearman(
                homologs["seq_identity"], homologs[sim_col]
            ),
            "spearman_sequence_identity_vs_similarity_near_ligand": _safe_spearman(
                near_ligand_homologs["seq_identity"], near_ligand_homologs[sim_col]
            ),
            "spearman_pocket_sequence_identity_vs_similarity": _safe_spearman(
                pairs["pocket_seq_identity"], pairs[sim_col]
            ),
            "spearman_pocket_sequence_identity_vs_similarity_near_ligand": _safe_spearman(
                sequence_sets["ligand_similarity>=0.8"]["pocket_seq_identity"],
                sequence_sets["ligand_similarity>=0.8"][sim_col],
            ),
            "spearman_pocket_sequence_identity_vs_similarity_homolog_pairs": _safe_spearman(
                homologs["pocket_seq_identity"], homologs[sim_col]
            ),
            "spearman_pocket_sequence_identity_vs_similarity_near_ligand_homolog_pairs": _safe_spearman(
                near_ligand_homologs["pocket_seq_identity"],
                near_ligand_homologs[sim_col],
            ),
            "spearman_exact_sequence_ligand_similarity_vs_representation_similarity": _safe_spearman(
                exact_seq["ligand_similarity"], exact_seq[sim_col]
            ),
            "spearman_near_seq_near_lig_affinity_gap_vs_similarity": _safe_spearman(
                pd.concat([smooth["affinity_gap"], cliff["affinity_gap"]]),
                pd.concat([smooth[sim_col], cliff[sim_col]]),
            ),
            "near_seq_near_lig_smooth_mean_similarity": (
                float(smooth[sim_col].mean()) if len(smooth) else None
            ),
            "near_seq_near_lig_cliff_mean_similarity": (
                float(cliff[sim_col].mean()) if len(cliff) else None
            ),
            "cliff_resolution_delta_smooth_minus_cliff": (
                float(smooth[sim_col].mean() - cliff[sim_col].mean())
                if len(smooth) and len(cliff)
                else None
            ),
            "cliff_vs_smooth_auc_lower_similarity_is_cliff": _lower_similarity_auc(
                smooth[sim_col], cliff[sim_col]
            ),
            "remote_same_ligand_delta_smooth_minus_divergent": (
                float(remote_smooth[sim_col].mean() - remote_divergent[sim_col].mean())
                if len(remote_smooth) and len(remote_divergent)
                else None
            ),
            "remote_same_ligand_auc_lower_similarity_is_divergent": _lower_similarity_auc(
                remote_smooth[sim_col], remote_divergent[sim_col]
            ),
        }
    return (
        pd.DataFrame(seq_rows),
        pd.DataFrame(pocket_seq_rows),
        pd.DataFrame(case_rows),
        diagnostics,
    )


def save_plots(
    sequence_summary: pd.DataFrame,
    pocket_sequence_summary: pd.DataFrame,
    case_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    mpl_config = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/voxbind-matplotlib-cache"))
    mpl_config.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not sequence_summary.empty:
        controls = [
            control for control in LIGAND_CONTROL_ORDER
            if control in set(sequence_summary["ligand_control"])
        ]
        fig, axes = plt.subplots(1, len(controls), figsize=(7.2 * len(controls), 5.2), squeeze=False)
        x = np.arange(len(SEQ_BIN_ORDER))
        for ax, control in zip(axes[0], controls):
            subset = sequence_summary[sequence_summary["ligand_control"] == control]
            for model, frame in subset.groupby("model", sort=False):
                frame = frame.set_index("sequence_bin").reindex(SEQ_BIN_ORDER)
                ax.errorbar(
                    x,
                    frame["mean_similarity"],
                    yerr=frame["sem_similarity"],
                    marker="o",
                    linewidth=1.8,
                    capsize=2,
                    label=model,
                )
            ax.set_xticks(x, SEQ_BIN_ORDER, rotation=15)
            ax.set_xlabel("Protein sequence identity (MMseqs2)")
            ax.set_ylabel("Mean standardized cosine similarity")
            title = "All ligands" if control == "all_ligands" else "Ligand similarity >= 0.8"
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        axes[0][0].legend(frameon=False, fontsize=9)
        fig.suptitle("Does each complex representation preserve protein homology?", fontsize=15)
        fig.tight_layout()
        fig.savefig(output_dir / "sequence_similarity_curve.png", dpi=180)
        plt.close(fig)

    if not pocket_sequence_summary.empty:
        controls = [
            control for control in LIGAND_CONTROL_ORDER
            if control in set(pocket_sequence_summary["ligand_control"])
        ]
        fig, axes = plt.subplots(1, len(controls), figsize=(7.2 * len(controls), 5.2), squeeze=False)
        x = np.arange(len(POCKET_SEQ_BIN_ORDER))
        for ax, control in zip(axes[0], controls):
            subset = pocket_sequence_summary[
                pocket_sequence_summary["ligand_control"] == control
            ]
            for model, frame in subset.groupby("model", sort=False):
                frame = frame.set_index("pocket_sequence_bin").reindex(POCKET_SEQ_BIN_ORDER)
                ax.errorbar(
                    x,
                    frame["mean_similarity"],
                    yerr=frame["sem_similarity"],
                    marker="o",
                    linewidth=1.8,
                    capsize=2,
                    label=model,
                )
            ax.set_xticks(x, POCKET_SEQ_BIN_ORDER, rotation=15)
            ax.set_xlabel("Pocket sequence identity (global alignment)")
            ax.set_ylabel("Mean standardized cosine similarity")
            title = "All ligands" if control == "all_ligands" else "Ligand similarity >= 0.8"
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        axes[0][0].legend(frameon=False, fontsize=9)
        fig.suptitle("Does each complex representation preserve pocket sequence?", fontsize=15)
        fig.tight_layout()
        fig.savefig(output_dir / "pocket_sequence_similarity_curve.png", dpi=180)
        plt.close(fig)

    if not case_summary.empty:
        available_cases = [case for case in CASE_ORDER if case in set(case_summary["case"])]
        models = list(dict.fromkeys(case_summary["model"]))
        width = 0.8 / max(len(models), 1)
        x = np.arange(len(available_cases))
        fig, ax = plt.subplots(figsize=(max(9.0, len(available_cases) * 1.65), 5.4))
        for offset, model in enumerate(models):
            frame = case_summary[case_summary["model"] == model].set_index("case")
            values = [frame.loc[c, "mean_similarity"] if c in frame.index else np.nan for c in available_cases]
            ax.bar(x - 0.4 + width / 2 + offset * width, values, width=width, label=model)
        ax.set_xticks(x, [case.replace("_", "\n") for case in available_cases], fontsize=8)
        ax.set_ylabel("Mean standardized cosine similarity")
        ax.set_title("Representation behavior in binding-relevant pair cases")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / "case_similarity.png", dpi=180)
        plt.close(fig)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="cached pid->vector representation (repeatable); defaults to the current four-cache preset",
    )
    parser.add_argument(
        "--include-affinity-baselines",
        action="store_true",
        help="add the CPU-dumped CheapNet-seed0 and GET-seed0 caches",
    )
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--pocket-records",
        type=Path,
        default=DEFAULT_POCKET_RECORDS,
        help="structure-derived seq/pocket_mask records; use --no-pocket-similarity to skip",
    )
    parser.add_argument("--no-pocket-similarity", action="store_true")
    parser.add_argument("--mmseqs", type=Path, default=DEFAULT_MMSEQS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-seq-id", type=float, default=0.20)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--random-controls", type=int, default=20000)
    parser.add_argument(
        "--same-ligand-controls",
        type=int,
        default=20000,
        help="targeted exact-ligand pairs absent from the sequence-homology result",
    )
    parser.add_argument("--max-homolog-pairs", type=int, default=250000)
    parser.add_argument("--max-pids", type=int, default=0, help="deterministic smoke-test subset; 0 uses all")
    parser.add_argument("--force-sequence-search", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print cohort inventory only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    rng = np.random.default_rng(args.seed)

    specs = parse_representation_specs(args.representation)
    if args.include_affinity_baselines:
        duplicates = set(specs) & set(AFFINITY_BASELINE_REPRESENTATIONS)
        if duplicates:
            raise ValueError(
                "--include-affinity-baselines duplicates explicit representations: "
                + ", ".join(sorted(duplicates))
            )
        specs.update(AFFINITY_BASELINE_REPRESENTATIONS)
    representations = [load_representation(name, path) for name, path in specs.items()]
    meta = cohort_for(representations, load_metadata(args.index_csv, args.split_csv))
    if args.max_pids and len(meta) > args.max_pids:
        chosen = np.sort(rng.choice(len(meta), size=args.max_pids, replace=False))
        meta = meta.iloc[chosen].sort_values("pid").reset_index(drop=True)

    pocket_sequences = {}
    if not args.no_pocket_similarity:
        pocket_sequences = load_pocket_sequences(args.pocket_records.resolve())
    pocket_lengths = [len(pocket_sequences.get(pid, "")) for pid in meta["pid"]]
    n_with_pocket = sum(length > 0 for length in pocket_lengths)

    inventory = {
        "cpu_only": True,
        "index_csv": str(args.index_csv.resolve()),
        "split_csv": str(args.split_csv.resolve()),
        "n_common": len(meta),
        "split_counts": {str(k): int(v) for k, v in meta["split"].value_counts().items()},
        "pocket_sequences": {
            "enabled": not args.no_pocket_similarity,
            "path": str(args.pocket_records.resolve()),
            "n_available_in_cohort": n_with_pocket,
            "min_length": min((length for length in pocket_lengths if length), default=None),
            "median_length": (
                float(np.median([length for length in pocket_lengths if length]))
                if n_with_pocket
                else None
            ),
            "max_length": max(pocket_lengths, default=None),
        },
        "representations": {
            rep.name: {
                "path": str(rep.path.resolve()),
                "feature_key": rep.feature_key,
                "n_cached": len(rep.vectors),
                "dim": rep.dim,
            }
            for rep in representations
        },
    }
    print(json.dumps(inventory, indent=2))
    if args.dry_run:
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta = output_dir / "cohort.fasta"
    hits_path = output_dir / "sequence_hits.tsv"
    search_meta_path = output_dir / "sequence_search.json"
    write_fasta(meta, fasta)
    fasta_sha256 = hashlib.sha256(fasta.read_bytes()).hexdigest()
    search_signature = {
        "fasta_sha256": fasta_sha256,
        "n_sequences": len(meta),
        "min_seq_id": args.min_seq_id,
        "coverage": args.coverage,
        "mmseqs": str(args.mmseqs.resolve()),
    }
    saved_signature = None
    if search_meta_path.exists():
        try:
            saved_signature = json.loads(search_meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    if args.force_sequence_search or not hits_path.exists() or saved_signature != search_signature:
        run_mmseqs(
            args.mmseqs.resolve(),
            fasta,
            hits_path,
            output_dir / "mmseqs_tmp" / fasta_sha256[:16],
            args.threads,
            args.min_seq_id,
            args.coverage,
        )
        write_json(search_meta_path, search_signature)
    else:
        print(f"[sequence] reusing {hits_path}")

    pid_to_index = {pid: i for i, pid in enumerate(meta["pid"])}
    all_homologs = load_homolog_pairs(hits_path, pid_to_index, len(meta))
    n_homologs_raw = len(all_homologs)
    forbidden = set(all_homologs["pair_key"].astype(int))
    homologs = cap_homolog_pairs(all_homologs, args.max_homolog_pairs, rng)
    ligand_controls = sample_same_ligand_non_hits(
        meta, args.same_ligand_controls, len(meta), forbidden, rng
    )
    random_forbidden = forbidden | set(ligand_controls["pair_key"].astype(int))
    controls = sample_non_hits(args.random_controls, len(meta), random_forbidden, rng)
    pairs = pd.concat([homologs, controls, ligand_controls], ignore_index=True)
    pairs = add_pair_metadata(pairs, meta)
    pairs = add_ligand_similarity(pairs, meta)
    if not args.no_pocket_similarity:
        print(f"[pocket] globally aligning {len(pairs)} structure-derived pocket pairs")
        pairs = add_pocket_sequence_similarity(pairs, meta, pocket_sequences)
        pairs = assign_pocket_sequence_bins(pairs)
    else:
        pairs["pocket_seq_identity"] = np.nan
        pairs["same_pocket_sequence"] = False
        pairs["pocket_len_a"] = 0
        pairs["pocket_len_b"] = 0
        pairs["pocket_length_ratio"] = np.nan
        pairs["pocket_sequence_bin"] = pd.Categorical(
            [None] * len(pairs), categories=POCKET_SEQ_BIN_ORDER, ordered=True
        )
    pairs, metric_columns = add_representation_metrics(pairs, meta, representations)
    pairs = assign_sequence_bins(pairs)
    cases = classify_cases(pairs)
    sequence_summary, pocket_sequence_summary, case_summary, diagnostics = build_summaries(
        pairs, cases, metric_columns
    )

    pair_columns = [
        "pid_a",
        "pid_b",
        "split_a",
        "split_b",
        "homology_hit",
        "control_type",
        "same_sequence",
        "seq_identity",
        "qcov",
        "tcov",
        "length_ratio",
        "sequence_bin",
        "same_pocket_sequence",
        "pocket_seq_identity",
        "pocket_len_a",
        "pocket_len_b",
        "pocket_length_ratio",
        "pocket_sequence_bin",
        "same_ligand",
        "ligand_similarity",
        "pK_a",
        "pK_b",
        "affinity_gap",
    ]
    for columns in metric_columns.values():
        pair_columns.extend(columns.values())
    with gzip.open(output_dir / "pair_metrics.csv.gz", "wt", newline="") as handle:
        pairs[pair_columns].to_csv(handle, index=False)
    with gzip.open(output_dir / "case_pairs.csv.gz", "wt", newline="") as handle:
        cases[["case", *pair_columns]].to_csv(handle, index=False)
    sequence_summary.to_csv(output_dir / "sequence_summary.csv", index=False)
    pocket_sequence_summary.to_csv(output_dir / "pocket_sequence_summary.csv", index=False)
    case_summary.to_csv(output_dir / "case_summary.csv", index=False)

    summary = {
        **inventory,
        "parameters": {
            "seed": args.seed,
            "include_affinity_baselines": args.include_affinity_baselines,
            "min_seq_id": args.min_seq_id,
            "coverage": args.coverage,
            "random_controls": args.random_controls,
            "same_ligand_controls": args.same_ligand_controls,
            "max_homolog_pairs": args.max_homolog_pairs,
            "pocket_similarity": not args.no_pocket_similarity,
        },
        "pair_counts": {
            "homologs_before_cap": n_homologs_raw,
            "homologs_analyzed": len(homologs),
            "random_non_hit_controls": len(controls),
            "same_ligand_non_hit_controls": len(ligand_controls),
            "total": len(pairs),
            "cases": {str(k): int(v) for k, v in cases["case"].value_counts().items()},
        },
        "primary_metric": "per-feature standardized cosine similarity",
        "metric_columns": metric_columns,
        "diagnostics": diagnostics,
        "notes": {
            "non_hit": (
                "Random pairs absent from the MMseqs2 result at the configured identity/coverage; "
                "absence is a control label, not an exact global-identity estimate. Same-ligand "
                "targeted controls are excluded from the sequence-bin summary."
            ),
            "positive_cliff_resolution_delta": (
                "Higher means the representation separates near-sequence/near-ligand affinity cliffs "
                "more than matched smooth pairs."
            ),
            "pocket_sequence_identity": (
                "BLOSUM62-guided global alignment identity with gap-open -10 and gap-extension "
                "-0.5. Pocket residues are selected from the structure-derived full sequence by "
                "the cached pocket mask and concatenated in parent-sequence order; they need not "
                "form a contiguous sequence segment."
            ),
        },
    }
    write_json(output_dir / "summary.json", summary)
    if not args.no_plots:
        save_plots(sequence_summary, pocket_sequence_summary, case_summary, output_dir)
    print(
        f"[done] {len(meta)} complexes, {len(pairs)} pairs, {len(cases)} case rows -> {output_dir}"
    )


if __name__ == "__main__":
    main()
