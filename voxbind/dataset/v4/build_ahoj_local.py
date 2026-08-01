"""Build locally cached, experimental AHoJ apo/holo pairs for VoxBind v4.

This stage deliberately emits *unsplit* candidate records.  It intersects the
full AHoJ relation table with the existing ``dataset/data/cif`` and
``dataset/data/ccp4`` caches, transfers the 10 A VoxBind holo pocket into each
apo crystal by sequence-aware local superposition, and validates the apo map at
the transferred site.  The final v4 builder assigns merged leakage components
and train/validation/test partitions later.

The model-facing files follow the established PLINDER contract:

* ``pairs.pt`` contains aligned holo/apo records plus provenance;
* ``apo_tuples.pt`` contains ``(pocket_dict, anchor_ligand_dict)`` tuples;
* ``holo_tuples.pt`` contains ``(pocket_dict, ligand_dict)`` tuples;
* no train/validation manifest is written at this stage.

Only AHoJ queries whose query PDB already belongs to a PLINDER pocket component
are selected.  Unmatched AHoJ proteins are not assigned an ad-hoc split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import gemmi  # import before torch (shared libstdc++ compatibility)
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "data"
DEFAULT_OUTPUT = DATA / "v4" / "ahoj_local_candidates"
RELATIONS = DATA / "v4" / "ahoj_relations.parquet"
ANNOTATIONS = DATA / "plinder" / "index" / "annotation_table.parquet"
CIF_DIR = DATA / "cif"
MAP_DIR = DATA / "ccp4"
CLUSTER_COLUMN = "pocket_fident__50__weak__component"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


APO = load_module(
    ROOT / "dataset" / "plinder" / "06_build_apo_pairs.py",
    "voxbind_plinder_apo_builder",
)
PLINDER_BUILD = APO.PLINDER_BUILD
PDBB = APO.PDBB


SELECTION_COLUMNS = [
    "relation_id",
    "ahoj_query_id",
    "query_pdb_id",
    "query_chain",
    "query_ligand_ccd",
    "query_ligand_residue",
    "target_pdb_id",
    "target_pocket_index",
    "state",
    "uniprot_acc",
    "chains3",
    "resolution_A",
    "experimental_method",
    "mapped_sequence_percent",
    "mapped_observed_percent",
    "pocket_rmsd_A",
    "pocket_length",
    "pocket_distance_A",
    "alignment_matrix",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_pdb_ids() -> set[str]:
    cifs = {path.stem.lower() for path in CIF_DIR.glob("*.cif")}
    maps = {path.stem.lower() for path in MAP_DIR.glob("*.ccp4")}
    return cifs & maps


def backbone_pdb_ids() -> set[str]:
    table = pq.read_table(
        ANNOTATIONS,
        columns=["entry_pdb_id", CLUSTER_COLUMN],
    )
    frame = table.to_pandas()
    frame = frame.dropna(subset=[CLUSTER_COLUMN])
    return set(frame["entry_pdb_id"].astype(str).str.lower())


def _count(mask: pa.Array) -> int:
    return int(pc.sum(mask).as_py() or 0)


def select_candidates(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    """Stream-filter the 58M-row relation table and retain one anchor per site."""
    local = local_pdb_ids()
    backbone = backbone_pdb_ids()
    local_values = pa.array(sorted(local))
    backbone_values = pa.array(sorted(backbone))
    selected_batches: list[pa.Table] = []
    funnel: Counter[str] = Counter()

    parquet = pq.ParquetFile(args.relations)
    for batch in parquet.iter_batches(columns=SELECTION_COLUMNS, batch_size=500_000):
        mask = pc.equal(batch.column("state"), pa.scalar("apo"))
        funnel["apo"] += _count(mask)
        mask = pc.and_(
            mask,
            pc.equal(
                batch.column("experimental_method"),
                pa.scalar("X-RAY_DIFFRACTION"),
            ),
        )
        funnel["xray"] += _count(mask)
        resolution = batch.column("resolution_A")
        mask = pc.and_(
            mask,
            pc.and_(
                pc.is_valid(resolution),
                pc.less_equal(
                    resolution,
                    pa.scalar(args.max_resolution, pa.float32()),
                ),
            ),
        )
        funnel["resolution"] += _count(mask)
        mask = pc.and_(
            mask,
            pc.and_(
                pc.greater_equal(
                    batch.column("mapped_sequence_percent"),
                    pa.scalar(args.min_mapped_sequence, pa.float32()),
                ),
                pc.greater_equal(
                    batch.column("mapped_observed_percent"),
                    pa.scalar(args.min_mapped_observed, pa.float32()),
                ),
            ),
        )
        funnel["mapping"] += _count(mask)
        mask = pc.and_(
            mask,
            pc.is_in(batch.column("query_pdb_id"), value_set=backbone_values),
        )
        funnel["plinder_backbone_query"] += _count(mask)
        mask = pc.and_(
            mask,
            pc.is_in(batch.column("query_pdb_id"), value_set=local_values),
        )
        funnel["local_query"] += _count(mask)
        mask = pc.and_(
            mask,
            pc.is_in(batch.column("target_pdb_id"), value_set=local_values),
        )
        funnel["local_query_target_density"] += _count(mask)
        if _count(mask):
            selected_batches.append(pa.Table.from_batches([batch.filter(mask)]))

    if not selected_batches:
        raise RuntimeError("AHoJ selection produced no local candidates")
    frame = pa.concat_tables(selected_batches).to_pandas()
    for column in ("pocket_distance_A", "pocket_rmsd_A", "resolution_A"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Cheap exact-release prefilter for the established VoxBind 6..50 ligand
    # heavy-atom rule. Unknown PDB+CCD keys are retained and checked directly
    # from mmCIF by query_ligand(); known ineligible ligands are skipped before
    # expensive target/query structure parsing.
    ligand_annotations = pq.read_table(
        ANNOTATIONS,
        columns=["entry_pdb_id", "ligand_ccd_code", "ligand_num_heavy_atoms"],
    ).to_pandas()
    ligand_annotations["query_key"] = (
        ligand_annotations["entry_pdb_id"].astype(str).str.lower()
        + "|"
        + ligand_annotations["ligand_ccd_code"].astype(str).str.upper()
    )
    ligand_annotations["eligible"] = pd.to_numeric(
        ligand_annotations["ligand_num_heavy_atoms"],
        errors="coerce",
    ).between(6, 50)
    eligible_by_key = ligand_annotations.groupby("query_key")["eligible"].max()
    query_keys = (
        frame["query_pdb_id"].astype(str).str.lower()
        + "|"
        + frame["query_ligand_ccd"].astype(str).str.upper()
    )
    annotated_eligible = query_keys.map(eligible_by_key)
    frame = frame[annotated_eligible.ne(False)].copy()
    funnel["query_ligand_6_50_or_unknown"] = len(frame)
    frame = frame.sort_values(
        [
            "target_pdb_id",
            "chains3",
            "target_pocket_index",
            "pocket_distance_A",
            "pocket_rmsd_A",
            "resolution_A",
            "ahoj_query_id",
        ],
        na_position="last",
    )
    # AHoJ's target pocket index is structure-local.  Chain set disambiguates
    # symmetric/interface copies.  Alternative query holo anchors for the same
    # physical target site are redundant and are ranked by geometry above.
    frame = frame.drop_duplicates(
        ["target_pdb_id", "chains3", "target_pocket_index"],
        keep="first",
    )
    if args.limit:
        frame = frame.head(args.limit).copy()
    frame = frame.reset_index(drop=True)
    report = {
        "kind": "voxbind_v4_ahoj_local_selection",
        "relations": str(args.relations),
        "relations_sha256": sha256(args.relations),
        "cluster_backbone": CLUSTER_COLUMN,
        "canonical_pocket_definition": (
            "VoxBind supported protein heavy atoms with nearest holo-ligand "
            "heavy-atom distance <= 10 A; no whole-residue expansion"
        ),
        "filters": {
            "state": "apo",
            "experimental_method": "X-RAY_DIFFRACTION",
            "max_resolution_A": args.max_resolution,
            "min_mapped_sequence_percent": args.min_mapped_sequence,
            "min_mapped_observed_percent": args.min_mapped_observed,
            "requires_local_query_cif_and_map": True,
            "requires_local_target_cif_and_map": True,
            "requires_query_plinder_component": True,
            "query_ligand_heavy_atoms": "6..50 when annotated; exact mmCIF recheck",
        },
        "funnel": dict(funnel),
        "n_selected_physical_keys": int(len(frame)),
        "n_query_pockets": int(frame["ahoj_query_id"].nunique()),
        "n_target_pdb": int(frame["target_pdb_id"].nunique()),
        "physical_key": ["target_pdb_id", "chains3", "target_pocket_index"],
    }
    return frame, report


def prepared_structure(path: Path) -> gemmi.Structure:
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    return structure


def _residue_number(value: object) -> tuple[int | None, str]:
    match = re.match(r"^\s*(-?\d+)([A-Za-z]?)", str(value))
    if not match:
        return None, ""
    return int(match.group(1)), match.group(2)


def query_ligand(
    model: gemmi.Model,
    *,
    chain_id: str,
    ccd: str,
    residue_id: str,
) -> dict[str, object]:
    """Resolve one AHoJ query ligand and encode it as PLINDER per-element+other."""
    number, insertion = _residue_number(residue_id)
    ccd = str(ccd).strip().upper()
    candidates: list[tuple[gemmi.Chain, gemmi.Residue]] = []
    for chain in model:
        for residue in chain:
            if residue.name.strip().upper() != ccd:
                continue
            if number is not None and residue.seqid.num != number:
                continue
            if insertion and residue.seqid.icode.strip() != insertion:
                continue
            candidates.append((chain, residue))
    if not candidates:
        raise ValueError("query_ligand_missing")
    chain, residue = next(
        (
            item
            for item in candidates
            if item[0].name == str(chain_id)
            or item[1].subchain == str(chain_id)
        ),
        candidates[0],
    )

    all_xyz: list[list[float]] = []
    xyz: list[list[float]] = []
    channels: list[int] = []
    radii: list[float] = []
    for atom in residue:
        element = PDBB._normalise_element(atom.element.name)
        if not PDBB._is_heavy_element(element):
            continue
        point = [atom.pos.x, atom.pos.y, atom.pos.z]
        all_xyz.append(point)
        channel = PDBB._channel_of(element, PDBB.N_LIG_CH)
        if channel is None:
            if not PLINDER_BUILD.is_diverse_role_atom(element):
                continue
            channel = PDBB.N_LIG_CH
        xyz.append(point)
        channels.append(channel)
        radii.append(PLINDER_BUILD.vdw_radius(element))
    if not (6 <= len(all_xyz) <= 50):
        raise ValueError("query_ligand_heavy_atom_count")
    if not xyz:
        raise ValueError("query_ligand_no_supported_atoms")
    return {
        "chain": chain.name,
        "residue": f"{chain.name}:{residue.name}:{residue.seqid}",
        "all_xyz": np.asarray(all_xyz, dtype=np.float64),
        "xyz": torch.tensor(xyz, dtype=torch.float32),
        "channels": torch.tensor(channels, dtype=torch.uint8),
        "radii": torch.tensor(radii, dtype=torch.float32),
    }


def chain_ids(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    return [field for field in text.split("-") if field]


def query_alignment_chain_ids(row: dict[str, object]) -> list[str]:
    matrix = str(row.get("alignment_matrix") or "")
    marker = f"_to_{row['query_pdb_id']}"
    if marker in matrix:
        suffix = matrix.split(marker, 1)[1].rsplit(".txt", 1)[0]
        parsed = chain_ids(suffix)
        if parsed:
            return parsed
    return [str(row["query_chain"])]


def _protein_chains(
    model: gemmi.Model,
    preferred: Iterable[str],
) -> list[gemmi.Chain]:
    wanted = set(preferred)
    chains = [chain for chain in model if APO.protein_residues(chain)]
    selected = [chain for chain in chains if chain.name in wanted]
    return selected or chains


def query_pocket_chains(
    query_model: gemmi.Model,
    ligand_xyz: np.ndarray,
    pocket_radius: float,
) -> set[str]:
    tree = cKDTree(ligand_xyz)
    return {
        chain.name
        for chain in query_model
        if any(
            APO.residue_near_ligand(residue, tree, pocket_radius)
            for residue in APO.protein_residues(chain)
        )
    }


def multi_chain_alignment(
    query_model: gemmi.Model,
    target_model: gemmi.Model,
    ligand_xyz: np.ndarray,
    query_preferred: list[str],
    target_preferred: list[str],
    pocket_radius: float,
    min_identity: float,
    min_coverage: float,
) -> dict[str, object]:
    pocket_chains = query_pocket_chains(query_model, ligand_xyz, pocket_radius)
    query_chains = [
        chain
        for chain in _protein_chains(query_model, query_preferred)
        if chain.name in pocket_chains
    ]
    if not query_chains:
        query_chains = [
            chain for chain in query_model if chain.name in pocket_chains
        ]
    target_chains = _protein_chains(target_model, target_preferred)
    scored: list[tuple[tuple[float, ...], gemmi.Chain, gemmi.Chain, dict]] = []
    for query_chain in query_chains:
        for target_chain in target_chains:
            result = APO.alignment_for_chain(
                query_chain,
                target_chain,
                ligand_xyz,
                pocket_radius,
            )
            if result is None:
                continue
            score = (
                result["identity"],
                result["coverage"],
                result["n_local_residue_pairs"],
                -result["rmsd"],
            )
            scored.append((score, query_chain, target_chain, result))
    chosen: list[tuple[gemmi.Chain, gemmi.Chain, dict]] = []
    used_query: set[str] = set()
    used_target: set[str] = set()
    for _score, query_chain, target_chain, result in sorted(
        scored, key=lambda item: item[0], reverse=True
    ):
        if query_chain.name in used_query or target_chain.name in used_target:
            continue
        if result["identity"] < min_identity or result["coverage"] < min_coverage:
            continue
        chosen.append((query_chain, target_chain, result))
        used_query.add(query_chain.name)
        used_target.add(target_chain.name)
    if not chosen:
        raise ValueError("alignment_failed")
    if pocket_chains - used_query:
        raise ValueError("interface_chain_unmapped")

    tree = cKDTree(ligand_xyz)
    query_points: list[list[float]] = []
    target_points: list[list[float]] = []
    residue_pairs = 0
    local_pairs = 0
    for query_chain, target_chain, _result in chosen:
        pairs, _identity, _coverage = APO.aligned_residue_pairs(
            APO.protein_residues(query_chain),
            APO.protein_residues(target_chain),
        )
        residue_pairs += len(pairs)
        local = [
            pair
            for pair in pairs
            if APO.residue_near_ligand(pair[0], tree, pocket_radius + 2.0)
        ]
        local_pairs += len(local)
        fit_pairs = local if len(local) >= 3 else pairs
        for query_residue, target_residue in fit_pairs:
            for atom_name in ("N", "CA", "C", "O"):
                query_atom = APO.residue_atom(query_residue, atom_name)
                target_atom = APO.residue_atom(target_residue, atom_name)
                if query_atom is None or target_atom is None:
                    continue
                query_points.append(
                    [query_atom.pos.x, query_atom.pos.y, query_atom.pos.z]
                )
                target_points.append(
                    [target_atom.pos.x, target_atom.pos.y, target_atom.pos.z]
                )
    if len(query_points) < 12:
        raise ValueError("alignment_too_few_atoms")
    rotation, translation, rmsd = APO.kabsch(
        np.asarray(query_points, dtype=np.float64),
        np.asarray(target_points, dtype=np.float64),
    )
    return {
        "R": rotation,
        "t": translation,
        "rmsd": rmsd,
        "identity": min(item[2]["identity"] for item in chosen),
        "coverage": min(item[2]["coverage"] for item in chosen),
        "n_residue_pairs": residue_pairs,
        "n_local_residue_pairs": local_pairs,
        "n_fit_atoms": len(query_points),
        "query_chains": [item[0].name for item in chosen],
        "target_chains": [item[1].name for item in chosen],
        "is_interface": len(chosen) > 1,
    }


def pocket_record(
    model: gemmi.Model,
    ligand_xyz: np.ndarray,
    *,
    sample_id: str,
    pocket_radius: float,
) -> tuple[dict[str, object], str, bool]:
    """Return supported pocket atoms plus a canonical residue identity string."""
    ligand_tree = cKDTree(ligand_xyz)
    coords: list[list[float]] = []
    channels: list[int] = []
    radii: list[float] = []
    residue_ids: list[str] = []
    pocket_chain_ids: set[str] = set()
    for chain in model:
        for residue in APO.protein_residues(chain):
            residue_kept = False
            for atom in residue:
                element = PDBB._normalise_element(atom.element.name)
                if not PDBB._is_heavy_element(element):
                    continue
                channel = PDBB._channel_of(element, PDBB.N_POC_CH)
                if channel is None:
                    continue
                point = [atom.pos.x, atom.pos.y, atom.pos.z]
                distance, _ = ligand_tree.query(point, k=1)
                if float(distance) > pocket_radius:
                    continue
                coords.append(point)
                channels.append(channel)
                radii.append(PLINDER_BUILD.vdw_radius(element))
                residue_kept = True
            if residue_kept:
                residue_ids.append(
                    f"{chain.name}:{residue.seqid}:{residue.name.strip()}"
                )
                pocket_chain_ids.add(chain.name)
    if not coords:
        raise ValueError("apo_pocket_empty")
    canonical = ";".join(sorted(set(residue_ids)))
    return (
        {
            "id": sample_id,
            "coords": torch.tensor(coords, dtype=torch.float32),
            "atoms_channel": torch.tensor(channels, dtype=torch.uint8),
            "radius": torch.tensor(radii, dtype=torch.float32),
        },
        canonical,
        len(pocket_chain_ids) > 1,
    )


def ligand_record(
    ligand: dict[str, object],
    *,
    sample_id: str,
    coords: np.ndarray,
    ligand_present: bool,
) -> dict[str, object]:
    center = coords.mean(0)
    max_len = round(float(np.ptp(coords, axis=0).max()), 2)
    if max_len > 30.0:
        raise ValueError("query_ligand_too_large")
    record = {
        "id": sample_id,
        "coords": torch.from_numpy(coords.astype(np.float32)),
        "atoms_channel": ligand["channels"].clone(),
        "radius": ligand["radii"].clone(),
        "max_len": max_len,
        "ligand_present": ligand_present,
    }
    if not ligand_present:
        record.update(
            center_coords=torch.from_numpy(center.astype(np.float32)),
            role="holo_alignment_anchor",
        )
    return record


def _cache_get(
    cache: OrderedDict[str, gemmi.Structure],
    pdb_id: str,
    max_size: int = 16,
) -> gemmi.Structure:
    if pdb_id in cache:
        cache.move_to_end(pdb_id)
        return cache[pdb_id]
    structure = prepared_structure(CIF_DIR / f"{pdb_id}.cif")
    if len(cache) >= max_size:
        cache.popitem(last=False)
    cache[pdb_id] = structure
    return structure


def build_relation(
    row: dict[str, object],
    *,
    target_structure: gemmi.Structure,
    target_grid,
    query_cache: OrderedDict[str, gemmi.Structure],
    args: dict[str, object],
) -> tuple[dict[str, object] | None, dict[str, object]]:
    status: dict[str, object] = {
        "relation_id": row["relation_id"],
        "ahoj_query_id": row["ahoj_query_id"],
        "target_pdb_id": row["target_pdb_id"],
        "target_pocket_index": row["target_pocket_index"],
        "status": "error",
        "reason": "",
    }
    try:
        query_pdb = str(row["query_pdb_id"]).lower()
        target_pdb = str(row["target_pdb_id"]).lower()
        query_structure = _cache_get(query_cache, query_pdb)
        query_model = query_structure[0]
        target_model = target_structure[0]
        ligand = query_ligand(
            query_model,
            chain_id=str(row["query_chain"]),
            ccd=str(row["query_ligand_ccd"]),
            residue_id=str(row["query_ligand_residue"]),
        )
        ligand_all = np.asarray(ligand["all_xyz"], dtype=np.float64)
        alignment = multi_chain_alignment(
            query_model,
            target_model,
            ligand_all,
            query_alignment_chain_ids(row),
            chain_ids(row.get("chains3")),
            float(args["pocket_radius"]),
            float(args["min_alignment_identity"]),
            float(args["min_alignment_coverage"]),
        )
        if float(alignment["rmsd"]) > float(args["max_alignment_rmsd"]):
            raise ValueError("high_binding_site_rmsd")
        transformed_all = APO.transform(
            ligand_all,
            alignment["R"],
            alignment["t"],
        )
        transformed_supported = APO.transform(
            ligand["xyz"],
            alignment["R"],
            alignment["t"],
        )
        center = transformed_supported.mean(0)
        apo_id = (
            f"ahoj_apo/{target_pdb}_{row['target_pocket_index']}_"
            f"{str(row.get('chains3') or 'all').replace('-', '+')}"
        )
        holo_id = (
            f"ahoj_holo/{query_pdb}_{row['query_chain']}_"
            f"{row['query_ligand_ccd']}_{row['query_ligand_residue']}"
        )
        apo_pocket, apo_residues, interface = pocket_record(
            target_model,
            transformed_all,
            sample_id=apo_id,
            pocket_radius=float(args["pocket_radius"]),
        )
        holo_pocket, holo_residues, holo_interface = pocket_record(
            query_model,
            ligand_all,
            sample_id=holo_id,
            pocket_radius=float(args["pocket_radius"]),
        )
        hetero_total, hetero_organic, hetero_residues = APO.nearby_hetero_atoms(
            target_model,
            transformed_all,
            float(args["nearby_hetero_radius"]),
        )
        if hetero_organic and not bool(args["allow_nearby_organic"]):
            raise ValueError("nearby_organic_ligand")

        pocket_values = APO.sample_grid(
            target_grid,
            apo_pocket["coords"].numpy(),
        )
        density_array = target_grid[0]
        density_z = float(
            (pocket_values.mean() - density_array.mean()) / density_array.std()
        )
        density_coverage = float(
            np.mean(
                (pocket_values - density_array.mean()) / density_array.std()
                >= float(args["density_atom_z"])
            )
        )
        if density_z < float(args["min_density_z"]):
            raise ValueError("weak_apo_pocket_density")
        if not APO.crop_is_finite(target_grid, center):
            raise ValueError("invalid_density_crop")

        apo_anchor = ligand_record(
            ligand,
            sample_id=apo_id,
            coords=transformed_supported,
            ligand_present=False,
        )
        holo_ligand = ligand_record(
            ligand,
            sample_id=holo_id,
            coords=ligand["xyz"].numpy(),
            ligand_present=True,
        )
        # Pair-evaluation copy of the holo state in the apo/map coordinate frame.
        aligned_holo_pocket = dict(holo_pocket)
        aligned_holo_pocket["coords"] = torch.from_numpy(
            APO.transform(
                holo_pocket["coords"], alignment["R"], alignment["t"]
            ).astype(np.float32)
        )
        aligned_holo_ligand = ligand_record(
            ligand,
            sample_id=holo_id,
            coords=transformed_supported,
            ligand_present=True,
        )
        metadata = {
            "relation_id": row["relation_id"],
            "ahoj_query_id": row["ahoj_query_id"],
            "query_pdb_id": query_pdb,
            "target_pdb_id": target_pdb,
            "target_pocket_index": row["target_pocket_index"],
            "target_chains": row.get("chains3"),
            "uniprot_acc": row.get("uniprot_acc"),
            "resolution_A": float(row["resolution_A"]),
            "pocket_rmsd_A": float(row["pocket_rmsd_A"]),
            "pocket_distance_A": float(row["pocket_distance_A"]),
            "pocket_radius_A": float(args["pocket_radius"]),
            "apo_map_path": str(MAP_DIR / f"{target_pdb}.ccp4"),
            "holo_map_path": str(MAP_DIR / f"{query_pdb}.ccp4"),
            "apo_canonical_pocket_residue_set": apo_residues,
            "holo_canonical_pocket_residue_set": holo_residues,
            "is_interface_pocket": bool(interface or holo_interface),
            "alignment": {
                key: (
                    value.astype(np.float32)
                    if isinstance(value, np.ndarray)
                    else value
                )
                for key, value in alignment.items()
            },
            "nearby_hetero_atoms": hetero_total,
            "nearby_organic_atoms": hetero_organic,
            "nearby_hetero_residues": hetero_residues,
            "pocket_density_mean_z": density_z,
            "pocket_density_coverage": density_coverage,
        }
        pair = {
            "id": str(row["relation_id"]),
            "holo": {
                "pocket": holo_pocket,
                "ligand": holo_ligand,
                "aligned_pocket": aligned_holo_pocket,
                "aligned_ligand": aligned_holo_ligand,
            },
            "apo": {"pocket": apo_pocket, "anchor_ligand": apo_anchor},
            "metadata": metadata,
        }
        status.update(
            status="ok",
            reason="ok",
            query_pdb_id=query_pdb,
            apo_pdb_id=target_pdb,
            apo_sample_id=apo_id,
            holo_sample_id=holo_id,
            alignment_identity=alignment["identity"],
            alignment_coverage=alignment["coverage"],
            binding_site_rmsd=alignment["rmsd"],
            n_alignment_atoms=alignment["n_fit_atoms"],
            n_apo_pocket_atoms=len(apo_pocket["coords"]),
            nearby_hetero_atoms=hetero_total,
            nearby_organic_atoms=hetero_organic,
            pocket_density_mean_z=density_z,
            pocket_density_coverage=density_coverage,
            apo_resolution=float(row["resolution_A"]),
            pocket_rmsd_A=float(row["pocket_rmsd_A"]),
            canonical_pocket_residue_set=apo_residues,
            is_interface_pocket=bool(interface or holo_interface),
        )
        return pair, status
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}:{exc}"
        status["reason"] = reason[:300]
        return None, status


def build_target_shard(payload: tuple[str, list[dict], str, dict]) -> dict[str, object]:
    target_pdb, rows, output_dir_text, build_args = payload
    output_dir = Path(output_dir_text)
    shard_path = output_dir / "shards" / f"{target_pdb}.pt"
    status_path = output_dir / "status" / f"{target_pdb}.json"
    if bool(build_args["resume"]) and shard_path.exists() and status_path.exists():
        statuses = json.loads(status_path.read_text())
        return {
            "target_pdb_id": target_pdb,
            "reused": True,
            "built": sum(row["status"] == "ok" for row in statuses),
            "attempted": len(statuses),
        }
    target_structure = prepared_structure(CIF_DIR / f"{target_pdb}.cif")
    target_grid = APO.load_density_grid(MAP_DIR / f"{target_pdb}.ccp4")
    query_cache: OrderedDict[str, gemmi.Structure] = OrderedDict()
    pairs: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    for row in rows:
        pair, status = build_relation(
            row,
            target_structure=target_structure,
            target_grid=target_grid,
            query_cache=query_cache,
            args=build_args,
        )
        statuses.append(status)
        if pair is not None:
            pairs.append(pair)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pairs, shard_path)
    status_path.write_text(json.dumps(statuses))
    return {
        "target_pdb_id": target_pdb,
        "reused": False,
        "built": len(pairs),
        "attempted": len(rows),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def consolidate(
    output_dir: Path,
    report: dict[str, object],
    target_pdb_ids: set[str],
) -> dict[str, object]:
    pairs: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    for path in sorted((output_dir / "shards").glob("*.pt")):
        if path.stem not in target_pdb_ids:
            continue
        pairs.extend(torch.load(path, map_location="cpu", weights_only=False))
    for path in sorted((output_dir / "status").glob("*.json")):
        if path.stem not in target_pdb_ids:
            continue
        statuses.extend(json.loads(path.read_text()))

    # Canonical cross-query dedup: one physical apo pocket per target structure
    # and transferred residue set.  Keep the best AHoJ pocket RMSD/resolution.
    pairs.sort(
        key=lambda pair: (
            float(pair["metadata"]["pocket_rmsd_A"]),
            float(pair["metadata"]["resolution_A"]),
            pair["id"],
        )
    )
    kept: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    dropped_ids: set[str] = set()
    for pair in pairs:
        metadata = pair["metadata"]
        key = (
            metadata["target_pdb_id"],
            metadata["apo_canonical_pocket_residue_set"],
        )
        if key in seen:
            dropped_ids.add(pair["id"])
            continue
        seen.add(key)
        kept.append(pair)
    for status in statuses:
        if status["relation_id"] in dropped_ids:
            status["status"] = "skipped"
            status["reason"] = "duplicate_target_residue_set"

    ok_rows = [row for row in statuses if row["status"] == "ok"]
    skipped_rows = [row for row in statuses if row["status"] != "ok"]
    ok_ids = {row["relation_id"] for row in ok_rows}
    kept = [pair for pair in kept if pair["id"] in ok_ids]
    torch.save(kept, output_dir / "pairs.pt")
    torch.save(
        [
            (pair["apo"]["pocket"], pair["apo"]["anchor_ligand"])
            for pair in kept
        ],
        output_dir / "apo_tuples.pt",
    )
    # One independent query holo state per stable AHoJ query pocket.
    holo_by_id: dict[str, tuple[dict, dict]] = {}
    for pair in kept:
        holo = pair["holo"]
        holo_by_id.setdefault(
            holo["pocket"]["id"],
            (holo["pocket"], holo["ligand"]),
        )
    torch.save(list(holo_by_id.values()), output_dir / "holo_tuples.pt")
    write_csv(output_dir / "pair_index.csv", ok_rows)
    write_csv(output_dir / "skipped.csv", skipped_rows)

    reasons = Counter(row["reason"] for row in skipped_rows)
    report.update(
        {
            "kind": "voxbind_v4_ahoj_local_build",
            "n_attempted": len(statuses),
            "n_built_pairs": len(kept),
            "n_unique_apo_pdb": len(
                {pair["metadata"]["target_pdb_id"] for pair in kept}
            ),
            "n_unique_holo_queries": len(holo_by_id),
            "n_interface_pairs": sum(
                bool(pair["metadata"]["is_interface_pocket"])
                for pair in kept
            ),
            "n_skipped": len(skipped_rows),
            "skip_reasons": dict(reasons),
            "outputs": {
                "pairs": str(output_dir / "pairs.pt"),
                "apo_tuples": str(output_dir / "apo_tuples.pt"),
                "holo_tuples": str(output_dir / "holo_tuples.pt"),
                "pair_index": str(output_dir / "pair_index.csv"),
                "skipped": str(output_dir / "skipped.csv"),
            },
        }
    )
    (output_dir / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--relations", type=Path, default=RELATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-resolution", type=float, default=2.5)
    parser.add_argument("--min-mapped-sequence", type=float, default=99.0)
    parser.add_argument("--min-mapped-observed", type=float, default=80.0)
    parser.add_argument("--pocket-radius", type=float, default=10.0)
    parser.add_argument("--nearby-hetero-radius", type=float, default=6.0)
    parser.add_argument("--allow-nearby-organic", action="store_true")
    parser.add_argument("--min-alignment-identity", type=float, default=0.90)
    parser.add_argument("--min-alignment-coverage", type=float, default=0.80)
    parser.add_argument("--max-alignment-rmsd", type=float, default=2.5)
    parser.add_argument("--min-density-z", type=float, default=0.5)
    parser.add_argument("--density-atom-z", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected, report = select_candidates(args)
    selected.to_parquet(args.output_dir / "selected_relations.parquet", index=False)
    (args.output_dir / "selection_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        f"[selection] {len(selected):,} physical keys / "
        f"{selected.target_pdb_id.nunique():,} target PDBs"
    )
    if args.selection_only:
        return

    build_args = {
        "resume": args.resume,
        "pocket_radius": args.pocket_radius,
        "nearby_hetero_radius": args.nearby_hetero_radius,
        "allow_nearby_organic": args.allow_nearby_organic,
        "min_alignment_identity": args.min_alignment_identity,
        "min_alignment_coverage": args.min_alignment_coverage,
        "max_alignment_rmsd": args.max_alignment_rmsd,
        "min_density_z": args.min_density_z,
        "density_atom_z": args.density_atom_z,
    }
    payloads = [
        (
            str(target_pdb),
            group.to_dict(orient="records"),
            str(args.output_dir),
            build_args,
        )
        for target_pdb, group in selected.groupby("target_pdb_id", sort=True)
    ]
    if args.jobs <= 1:
        results = [
            build_target_shard(payload)
            for payload in tqdm(payloads, desc="AHoJ target PDB", unit="pdb")
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(build_target_shard, payload) for payload in payloads]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="AHoJ target PDB",
                unit="pdb",
            ):
                results.append(future.result())
    print(
        f"[shards] attempted {sum(item['attempted'] for item in results):,}; "
        f"built before canonical dedup {sum(item['built'] for item in results):,}"
    )
    final = consolidate(
        args.output_dir,
        report,
        {str(item[0]) for item in payloads},
    )
    print(json.dumps({
        key: final[key]
        for key in (
            "n_attempted", "n_built_pairs", "n_unique_apo_pdb",
            "n_unique_holo_queries", "n_interface_pairs", "n_skipped",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
