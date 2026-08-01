"""Freeze a leakage-safe, training-ready VoxBind v4 corpus.

Inputs are the existing PLINDER v3 per-element tuples, validated PLINDER linked
apo pairs, and the locally validated AHoJ apo/holo pairs.  Outputs preserve the
existing VoxBind training layout while adding the explicit v4 provenance tables:

* ``dataset/data/v4/manifest.parquet`` and ``pair_edges.parquet``;
* ``dataset/data/pretrain/data_train_plinder_v4_perelem.pt``;
* ``dataset/data/pretrain/xray_resample_plinder_v4_perelem/``.

The serialized tuple list is deliberately arranged so the legacy
``Random(1234).shuffle -> drop tail 100 -> max_len`` loader produces the exact
position order recorded in ``train_manifest.npz``.  ``subset_n`` selects only
training components; the next 100 rows form the validation loader pool.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import gemmi  # import before torch (shared libstdc++ compatibility)
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "data"
PRETRAIN = DATA / "pretrain"
V4_DATA = DATA / "v4"
SPLITS = ROOT / "splits" / "v4"
CIF_DIR = DATA / "cif"
MAP_DIR = DATA / "ccp4"
MMS_PATH = DATA / "plinder" / "mmp" / "plinder_mmp_series.parquet"
CLUSTER_COLUMN = "pocket_fident__50__weak__component"
GRID_DIM = 64
GRID_SPACING = 0.25
SHUFFLE_SEED = 1234
LEGACY_TAIL = 100
_CANONICAL_RECORDS: dict[str, tuple[dict, dict]] = {}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


APO = load_module(
    ROOT / "dataset" / "plinder" / "06_build_apo_pairs.py",
    "voxbind_plinder_apo_builder_for_v4",
)
PDBB = APO.PDBB


def digest_text(value: str, size: int = 16) -> str:
    return hashlib.sha1(value.encode(), usedforsecurity=False).hexdigest()[:size]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def canonical_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def uniprot_tokens(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {
        token
        for token in re.split(r"[\s,;|]+", str(value).strip())
        if token and token.lower() != "nan"
    }


def tuple_pdb_id(item: tuple[dict, dict]) -> str:
    return item[0]["id"].split("/", 1)[1].split("_", 1)[0].lower()


def tuple_center(item: tuple[dict, dict]) -> np.ndarray:
    ligand = item[1]
    center = ligand.get("center_coords")
    if center is None:
        center = ligand["coords"].to(torch.float32).mean(0)
    if isinstance(center, torch.Tensor):
        center = center.detach().cpu().numpy()
    return np.asarray(center, dtype=np.float32)


def tuple_max_len(item: tuple[dict, dict]) -> float:
    return float(item[1].get("max_len", 0.0))


@dataclass
class Sample:
    sample_id: str
    pdb_id: str
    state: str
    sources: set[str]
    item: tuple[dict, dict]
    canonical_residues: str
    component_ids: set[str]
    uniprots: set[str]
    mms_ids: set[str]
    resolution_A: float | None
    assembly: str = "1"
    is_interface: bool = False
    aliases: set[str] = field(default_factory=set)
    map_R: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float32)
    )
    map_t: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )


@dataclass
class PairEdge:
    sample_id_a: str
    sample_id_b: str
    source: str
    source_relation_id: str
    pocket_rmsd_apo_holo: float | None
    # R/t map sample A's deposited coordinates into sample B's deposited
    # coordinates: x_b = x_a @ R_a_to_b.T + t_a_to_b.
    R_a_to_b: np.ndarray
    t_a_to_b: np.ndarray


def inverse_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    return rotation.T, -translation @ rotation


def compose_transforms(
    first_R: np.ndarray,
    first_t: np.ndarray,
    second_R: np.ndarray,
    second_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose A->B then B->C transforms in the row-vector convention."""
    first_R = np.asarray(first_R, dtype=np.float64)
    first_t = np.asarray(first_t, dtype=np.float64)
    second_R = np.asarray(second_R, dtype=np.float64)
    second_t = np.asarray(second_t, dtype=np.float64)
    return second_R @ first_R, first_t @ second_R.T + second_t


def transform_points(
    points: torch.Tensor,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> torch.Tensor:
    array = points.detach().cpu().numpy().astype(np.float64, copy=False)
    transformed = array @ np.asarray(rotation, dtype=np.float64).T
    transformed += np.asarray(translation, dtype=np.float64).reshape(1, 3)
    return torch.from_numpy(transformed.astype(np.float32))


def transformed_item(
    item: tuple[dict, dict],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[dict, dict]:
    pocket = dict(item[0])
    ligand = dict(item[1])
    pocket["coords"] = transform_points(pocket["coords"], rotation, translation)
    ligand["coords"] = transform_points(ligand["coords"], rotation, translation)
    if "center_coords" in ligand:
        center = ligand["center_coords"].reshape(1, 3)
        ligand["center_coords"] = transform_points(
            center,
            rotation,
            translation,
        ).reshape(3)
    return pocket, ligand


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        self.add(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, first: str, second: str) -> str:
        root_a, root_b = self.find(first), self.find(second)
        if root_a == root_b:
            return root_a
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1
        return root_a

    def union_all(self, items: Iterable[str]) -> str | None:
        values = list(dict.fromkeys(items))
        if not values:
            return None
        root = values[0]
        self.add(root)
        for value in values[1:]:
            root = self.union(root, value)
        return self.find(root)


def prepared_structure(pdb_id: str) -> gemmi.Structure:
    structure = gemmi.read_structure(str(CIF_DIR / f"{pdb_id}.cif"))
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    return structure


def structure_atom_index(
    pdb_id: str,
) -> tuple[dict[tuple[float, float, float], str], np.ndarray, list[str]]:
    structure = prepared_structure(pdb_id)
    lookup: dict[tuple[float, float, float], str] = {}
    coords: list[list[float]] = []
    residues: list[str] = []
    for chain in structure[0]:
        for residue in APO.protein_residues(chain):
            residue_id = f"{chain.name}:{residue.seqid}:{residue.name.strip()}"
            for atom in residue:
                element = PDBB._normalise_element(atom.element.name)
                if not PDBB._is_heavy_element(element):
                    continue
                if PDBB._channel_of(element, PDBB.N_POC_CH) is None:
                    continue
                point = [atom.pos.x, atom.pos.y, atom.pos.z]
                key = tuple(round(value, 3) for value in point)
                lookup[key] = residue_id
                coords.append(point)
                residues.append(residue_id)
    return lookup, np.asarray(coords, dtype=np.float64), residues


def compute_plinder_residue_cache(
    records: dict[str, tuple[dict, dict]],
    cache_path: Path,
    *,
    rebuild: bool,
    jobs: int = 1,
) -> pd.DataFrame:
    if cache_path.exists() and not rebuild:
        cached = pd.read_parquet(cache_path)
        if set(records).issubset(set(cached["sample_id"])):
            return cached[cached["sample_id"].isin(records)].copy()
    by_pdb: dict[str, list[str]] = defaultdict(list)
    for sample_id, item in records.items():
        by_pdb[tuple_pdb_id(item)].append(sample_id)
    pdb_groups = sorted(by_pdb.items())
    if jobs <= 1:
        payloads = [
            (pdb_id, [(sample_id, records[sample_id]) for sample_id in sample_ids])
            for pdb_id, sample_ids in pdb_groups
        ]
        iterator = map(canonicalize_pdb_records, payloads)
        pool = None
    else:
        # Linux/fork keeps the large tuple dictionary copy-on-write instead of
        # serializing it once per task. Workers receive only PDB + sample IDs.
        global _CANONICAL_RECORDS
        _CANONICAL_RECORDS = records
        pool = mp.get_context("fork").Pool(processes=jobs)
        iterator = pool.imap(canonicalize_global_pdb_group, pdb_groups, chunksize=8)
    rows = []
    for result in tqdm(
        iterator,
        total=len(pdb_groups),
        desc="canonical PLINDER pockets",
        unit="pdb",
    ):
        rows.extend(result)
    if pool is not None:
        pool.close()
        pool.join()
        _CANONICAL_RECORDS = {}
    frame = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    return frame


def canonicalize_global_pdb_group(
    payload: tuple[str, list[str]],
) -> list[dict[str, object]]:
    pdb_id, sample_ids = payload
    return canonicalize_pdb_records(
        (
            pdb_id,
            [(sample_id, _CANONICAL_RECORDS[sample_id]) for sample_id in sample_ids],
        )
    )


def canonicalize_pdb_records(
    payload: tuple[str, list[tuple[str, tuple[dict, dict]]]],
) -> list[dict[str, object]]:
    pdb_id, sample_records = payload
    rows: list[dict[str, object]] = []
    sample_ids = [sample_id for sample_id, _item in sample_records]
    try:
        lookup, structure_coords, structure_residues = structure_atom_index(pdb_id)
        tree = cKDTree(structure_coords) if len(structure_coords) else None
    except Exception as exc:
        return [
            {
                "sample_id": sample_id,
                "pdb_id": pdb_id,
                "canonical_pocket_residue_set": "",
                "chains": "",
                "is_interface_pocket": False,
                "error": f"{type(exc).__name__}:{exc}"[:240],
            }
            for sample_id in sample_ids
        ]
    for sample_id, item in sample_records:
        pocket = item[0]
        pocket_coords = pocket["coords"].detach().cpu().numpy()
        residue_ids: set[str] = set()
        missing: list[np.ndarray] = []
        for point in pocket_coords:
            residue_id = lookup.get(tuple(round(float(value), 3) for value in point))
            if residue_id is None:
                missing.append(point)
            else:
                residue_ids.add(residue_id)
        if missing and tree is not None:
            distances, indices = tree.query(np.asarray(missing), k=1)
            for distance, index in zip(distances, indices):
                if float(distance) <= 0.08:
                    residue_ids.add(structure_residues[int(index)])
        canonical = ";".join(sorted(residue_ids))
        chains = sorted({value.split(":", 1)[0] for value in residue_ids})
        rows.append(
            {
                "sample_id": sample_id,
                "pdb_id": pdb_id,
                "canonical_pocket_residue_set": canonical,
                "chains": ";".join(chains),
                "is_interface_pocket": len(chains) > 1,
                "error": "" if canonical else "residue_mapping_empty",
            }
        )
    return rows


def metadata_tables():
    annotations = pq.read_table(
        DATA / "plinder" / "index" / "annotation_table.parquet",
        columns=["system_id", "entry_pdb_id", "entry_resolution", CLUSTER_COLUMN],
    ).to_pandas()
    annotations["pdb_id"] = annotations["entry_pdb_id"].astype(str).str.lower()
    annotations["component_id"] = annotations[CLUSTER_COLUMN].map(component_text)
    annotations = annotations.drop_duplicates("system_id")
    system_to_component = dict(
        zip(annotations["system_id"].astype(str), annotations["component_id"])
    )
    pdb_to_components: dict[str, set[str]] = defaultdict(set)
    for row in annotations.itertuples(index=False):
        if row.component_id:
            pdb_to_components[row.pdb_id].add(row.component_id)
    resolution_rows = annotations.assign(
        _resolution=pd.to_numeric(annotations["entry_resolution"], errors="coerce")
    ).dropna(subset=["_resolution"])
    pdb_to_resolution = (
        resolution_rows.groupby("pdb_id")["_resolution"].min().to_dict()
    )

    if not MMS_PATH.exists():
        raise FileNotFoundError(
            f"missing PLINDER matched-series leakage artifact: {MMS_PATH}"
        )
    mms = pq.read_table(
        MMS_PATH,
        columns=[
            "system_id",
            "congeneric_id",
            "prot_pocket_set_shared",
            "mms_unique_count",
        ],
    ).to_pandas()
    mms = mms[pd.to_numeric(mms["mms_unique_count"], errors="coerce").ge(3)].copy()
    mms["mms_id"] = (
        "mms:"
        + mms["prot_pocket_set_shared"].astype(str)
        + ":"
        + mms["congeneric_id"].astype(str)
    )
    system_to_mms: dict[str, set[str]] = defaultdict(set)
    mms_to_components: dict[str, set[str]] = defaultdict(set)
    for row in mms.itertuples(index=False):
        system_id = str(row.system_id)
        mms_id = str(row.mms_id)
        system_to_mms[system_id].add(mms_id)
        component = system_to_component.get(system_id)
        if component:
            mms_to_components[mms_id].add(component)
    pdb_to_mms: dict[str, set[str]] = defaultdict(set)
    for row in annotations.itertuples(index=False):
        pdb_to_mms[row.pdb_id].update(system_to_mms.get(str(row.system_id), set()))

    selection = pd.read_csv(ROOT / "splits" / "plinder" / "v3" / "plinder_selected.csv")
    selection["pdb_id"] = selection["entry_pdb_id"].astype(str).str.lower()
    selection["sample_id"] = (
        "plinder/"
        + selection["pdb_id"]
        + "_"
        + selection["ligand_asym_id"].astype(str)
    )
    selected_by_sample = {
        sample_id: group.copy()
        for sample_id, group in selection.groupby("sample_id", sort=False)
    }
    selected_by_system = {
        system_id: group.copy()
        for system_id, group in selection.groupby("system_id", sort=False)
    }
    return (
        annotations,
        system_to_component,
        pdb_to_components,
        selected_by_sample,
        selected_by_system,
        system_to_mms,
        mms_to_components,
        pdb_to_mms,
        pdb_to_resolution,
    )


def add_or_merge(samples: dict[str, Sample], sample: Sample) -> None:
    existing = samples.get(sample.sample_id)
    if existing is None:
        samples[sample.sample_id] = sample
        return
    existing.sources.update(sample.sources)
    existing.component_ids.update(sample.component_ids)
    existing.uniprots.update(sample.uniprots)
    existing.mms_ids.update(sample.mms_ids)
    existing.aliases.update(sample.aliases)


def load_source_samples(args: argparse.Namespace):
    (
        _annotations,
        system_to_component,
        pdb_to_components,
        selected_by_sample,
        selected_by_system,
        system_to_mms,
        mms_to_components,
        pdb_to_mms,
        pdb_to_resolution,
    ) = metadata_tables()
    v3_tuples = torch.load(args.v3_tuples, map_location="cpu", weights_only=False)
    plinder_pairs = torch.load(
        args.plinder_apo_pairs, map_location="cpu", weights_only=False
    )
    ahoj_pairs = torch.load(args.ahoj_pairs, map_location="cpu", weights_only=False)

    coordinate_records: dict[str, tuple[dict, dict]] = {
        item[0]["id"]: item for item in v3_tuples
    }
    for pair in plinder_pairs:
        item = (pair["apo"]["pocket"], pair["apo"]["anchor_ligand"])
        coordinate_records[item[0]["id"]] = item
    residue_cache = compute_plinder_residue_cache(
        coordinate_records,
        args.canonical_cache,
        rebuild=args.rebuild_canonical_cache,
        jobs=args.canonical_jobs,
    ).set_index("sample_id")

    samples: dict[str, Sample] = {}
    edges: list[PairEdge] = []
    source_report: Counter[str] = Counter()
    v3_ids: set[str] = set()
    for item in v3_tuples:
        sample_id = item[0]["id"]
        if tuple_max_len(item) > args.max_len or sample_id not in residue_cache.index:
            source_report["plinder_holo_size_or_cache_skip"] += 1
            continue
        canonical = canonical_text(
            residue_cache.at[sample_id, "canonical_pocket_residue_set"]
        )
        if not canonical:
            source_report["plinder_holo_residue_skip"] += 1
            continue
        selected = selected_by_sample.get(sample_id)
        if selected is None:
            source_report["plinder_holo_selection_skip"] += 1
            continue
        components = {
            value
            for value in (
                system_to_component.get(str(system_id))
                for system_id in selected["system_id"]
            )
            if value
        }
        if not components:
            source_report["plinder_holo_component_skip"] += 1
            continue
        uniprots = set().union(
            *(uniprot_tokens(value) for value in selected["system_pocket_UniProt"])
        )
        mms_ids = set().union(
            *(system_to_mms.get(str(value), set()) for value in selected["system_id"])
        )
        resolution = pd.to_numeric(selected["entry_resolution"], errors="coerce").min()
        assembly = str(selected["system_biounit_id"].iloc[0])
        add_or_merge(
            samples,
            Sample(
                sample_id=sample_id,
                pdb_id=tuple_pdb_id(item),
                state="holo",
                sources={"plinder"},
                item=item,
                canonical_residues=canonical,
                component_ids=components,
                uniprots=uniprots,
                mms_ids=mms_ids,
                resolution_A=float(resolution) if pd.notna(resolution) else None,
                assembly=assembly,
                is_interface=bool(residue_cache.at[sample_id, "is_interface_pocket"]),
            ),
        )
        v3_ids.add(sample_id)
        source_report["plinder_holo"] += 1

    for pair in plinder_pairs:
        metadata = pair["metadata"]
        item = (pair["apo"]["pocket"], pair["apo"]["anchor_ligand"])
        sample_id = item[0]["id"]
        if tuple_max_len(item) > args.max_len or sample_id not in residue_cache.index:
            source_report["plinder_apo_size_or_cache_skip"] += 1
            continue
        canonical = canonical_text(
            residue_cache.at[sample_id, "canonical_pocket_residue_set"]
        )
        component = system_to_component.get(str(metadata["reference_system_id"]))
        if not canonical or not component:
            source_report["plinder_apo_residue_or_component_skip"] += 1
            continue
        selected = selected_by_system.get(str(metadata["reference_system_id"]))
        uniprots = (
            set().union(
                *(uniprot_tokens(value) for value in selected["system_pocket_UniProt"])
            )
            if selected is not None
            else set()
        )
        components = {component} | pdb_to_components.get(
            str(metadata["apo_pdb_id"]).lower(), set()
        )
        mms_ids = system_to_mms.get(str(metadata["reference_system_id"]), set())
        add_or_merge(
            samples,
            Sample(
                sample_id=sample_id,
                pdb_id=str(metadata["apo_pdb_id"]).lower(),
                state="apo",
                sources={"plinder"},
                item=item,
                canonical_residues=canonical,
                component_ids=set(components),
                uniprots=uniprots,
                mms_ids=set(mms_ids),
                resolution_A=float(metadata["apo_resolution"]),
                is_interface=bool(residue_cache.at[sample_id, "is_interface_pocket"]),
            ),
        )
        source_report["plinder_apo"] += 1
        match = re.search(r"__lig_([^_]+)_", str(pair["id"]))
        holo_id = (
            f"plinder/{str(metadata['holo_pdb_id']).lower()}_{match.group(1)}"
            if match
            else ""
        )
        if holo_id in v3_ids:
            # PLINDER stores the Kabsch transform from the holo crystal into
            # the apo crystal.  PairEdge is standardized as apo -> holo.
            apo_to_holo_R, apo_to_holo_t = inverse_transform(
                metadata["alignment"]["R"],
                metadata["alignment"]["t"],
            )
            edges.append(
                PairEdge(
                    sample_id_a=sample_id,
                    sample_id_b=holo_id,
                    source="plinder",
                    source_relation_id=str(pair["id"]),
                    pocket_rmsd_apo_holo=float(metadata["alignment"]["rmsd"]),
                    R_a_to_b=apo_to_holo_R,
                    t_a_to_b=apo_to_holo_t,
                )
            )

    for pair in ahoj_pairs:
        metadata = pair["metadata"]
        query_pdb = str(metadata["query_pdb_id"]).lower()
        target_pdb = str(metadata["target_pdb_id"]).lower()
        components = set(pdb_to_components.get(query_pdb, set())) | set(
            pdb_to_components.get(target_pdb, set())
        )
        if not components:
            source_report["ahoj_no_backbone_component"] += 1
            continue
        uniprots = uniprot_tokens(metadata.get("uniprot_acc"))
        mms_ids = set(pdb_to_mms.get(query_pdb, set()))
        apo_item = (pair["apo"]["pocket"], pair["apo"]["anchor_ligand"])
        holo_item = (pair["holo"]["pocket"], pair["holo"]["ligand"])
        if tuple_max_len(apo_item) > args.max_len:
            source_report["ahoj_size_skip"] += 1
            continue
        apo_id, holo_id = apo_item[0]["id"], holo_item[0]["id"]
        add_or_merge(
            samples,
            Sample(
                sample_id=apo_id,
                pdb_id=target_pdb,
                state="apo",
                sources={"ahoj"},
                item=apo_item,
                canonical_residues=str(
                    metadata["apo_canonical_pocket_residue_set"]
                ),
                component_ids=set(components),
                uniprots=set(uniprots),
                mms_ids=set(mms_ids),
                resolution_A=float(metadata["resolution_A"]),
                is_interface=bool(metadata["is_interface_pocket"]),
            ),
        )
        add_or_merge(
            samples,
            Sample(
                sample_id=holo_id,
                pdb_id=query_pdb,
                state="holo",
                sources={"ahoj"},
                item=holo_item,
                canonical_residues=str(
                    metadata["holo_canonical_pocket_residue_set"]
                ),
                component_ids=set(components),
                uniprots=set(uniprots),
                mms_ids=set(mms_ids),
                resolution_A=(
                    float(pdb_to_resolution[query_pdb])
                    if query_pdb in pdb_to_resolution
                    else None
                ),
                is_interface=bool(metadata["is_interface_pocket"]),
            ),
        )
        apo_to_holo_R, apo_to_holo_t = inverse_transform(
            metadata["alignment"]["R"],
            metadata["alignment"]["t"],
        )
        edges.append(
            PairEdge(
                sample_id_a=apo_id,
                sample_id_b=holo_id,
                source="ahoj",
                source_relation_id=str(metadata["relation_id"]),
                pocket_rmsd_apo_holo=float(metadata["pocket_rmsd_A"]),
                R_a_to_b=apo_to_holo_R,
                t_a_to_b=apo_to_holo_t,
            )
        )
        source_report["ahoj_apo"] += 1
        source_report["ahoj_holo_relation"] += 1
    return (
        samples,
        edges,
        source_report,
        system_to_component,
        pdb_to_components,
        mms_to_components,
    )


def deduplicate_samples(
    samples: dict[str, Sample],
    edges: list[PairEdge],
) -> tuple[dict[str, Sample], list[PairEdge], dict[str, str]]:
    by_physical: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for sample in samples.values():
        by_physical[(sample.pdb_id, sample.canonical_residues)].append(sample)
    priority = {
        ("plinder", "holo"): 0,
        ("plinder", "apo"): 1,
        ("ahoj", "holo"): 2,
        ("ahoj", "apo"): 3,
    }
    alias: dict[str, str] = {}
    deduplicated: dict[str, Sample] = {}
    for group in by_physical.values():
        group.sort(
            key=lambda sample: (
                min(
                    priority.get((source, sample.state), 9)
                    for source in sample.sources
                ),
                sample.sample_id,
            )
        )
        kept = group[0]
        for duplicate in group[1:]:
            kept.sources.update(duplicate.sources)
            kept.component_ids.update(duplicate.component_ids)
            kept.uniprots.update(duplicate.uniprots)
            kept.mms_ids.update(duplicate.mms_ids)
            kept.aliases.add(duplicate.sample_id)
            kept.aliases.update(duplicate.aliases)
            if duplicate.state == "holo" and kept.state == "apo":
                kept.state = "holo"
                kept.item = duplicate.item
            alias[duplicate.sample_id] = kept.sample_id
        deduplicated[kept.sample_id] = kept
        alias[kept.sample_id] = kept.sample_id

    rewritten: dict[tuple[str, str, str], PairEdge] = {}
    for edge in edges:
        first = alias.get(edge.sample_id_a, edge.sample_id_a)
        second = alias.get(edge.sample_id_b, edge.sample_id_b)
        if first == second or first not in deduplicated or second not in deduplicated:
            continue
        key = tuple(sorted((first, second))) + (edge.source,)
        current = rewritten.get(key)
        if current is None or (
            edge.pocket_rmsd_apo_holo is not None
            and (
                current.pocket_rmsd_apo_holo is None
                or edge.pocket_rmsd_apo_holo
                > current.pocket_rmsd_apo_holo
            )
        ):
            rewritten[key] = PairEdge(
                sample_id_a=first,
                sample_id_b=second,
                source=edge.source,
                source_relation_id=edge.source_relation_id,
                pocket_rmsd_apo_holo=edge.pocket_rmsd_apo_holo,
                R_a_to_b=edge.R_a_to_b,
                t_a_to_b=edge.t_a_to_b,
            )
    return deduplicated, list(rewritten.values()), alias


def deduplicate_same_pdb_pocket(
    samples: dict[str, Sample],
    edges: list[PairEdge],
) -> tuple[dict[str, Sample], list[PairEdge], int]:
    """Collapse same-deposition symmetry/site copies of one logical pocket.

    The first canonicalization key intentionally preserves distinct residue
    signatures. Pair relationships can subsequently show that two such copies
    represent one pocket identity. In that case A3 permits one row per PDB and
    pocket. Edges belonging only to the discarded spatial copy are dropped:
    reusing them would falsely treat two chain-copy transforms as identical.
    """
    logical_pockets = pocket_assignments(samples, edges)
    grouped: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    degree = Counter(
        sample_id
        for edge in edges
        for sample_id in (edge.sample_id_a, edge.sample_id_b)
    )
    for sample_id, sample in samples.items():
        grouped[(sample.pdb_id, logical_pockets[sample_id])].append(sample)

    removed: set[str] = set()
    for group in grouped.values():
        if len(group) <= 1:
            continue
        group.sort(
            key=lambda sample: (
                sample.state != "holo",
                "plinder" not in sample.sources,
                -degree[sample.sample_id],
                sample.sample_id,
            )
        )
        kept = group[0]
        for duplicate in group[1:]:
            kept.sources.update(duplicate.sources)
            kept.component_ids.update(duplicate.component_ids)
            kept.uniprots.update(duplicate.uniprots)
            kept.mms_ids.update(duplicate.mms_ids)
            kept.aliases.add(duplicate.sample_id)
            kept.aliases.update(duplicate.aliases)
            removed.add(duplicate.sample_id)
    if not removed:
        return samples, edges, 0
    deduplicated = {
        sample_id: sample
        for sample_id, sample in samples.items()
        if sample_id not in removed
    }
    retained_edges = [
        edge
        for edge in edges
        if edge.sample_id_a not in removed and edge.sample_id_b not in removed
    ]
    return deduplicated, retained_edges, len(removed)


def leakage_assignments(
    samples: dict[str, Sample],
    edges: list[PairEdge],
    system_to_component: dict[str, str | None],
    pdb_to_components: dict[str, set[str]],
    mms_to_components: dict[str, set[str]],
) -> tuple[dict[str, str], dict[str, str], UnionFind]:
    union = UnionFind()
    for sample in samples.values():
        union.union_all(sorted(sample.component_ids))
    # Different pockets in one deposited structure and exact UniProt groups are
    # conservatively co-assigned.  This prevents cross-PDB/cross-source twins.
    pdb_components: dict[str, set[str]] = defaultdict(set)
    uniprot_components: dict[str, set[str]] = defaultdict(set)
    for sample in samples.values():
        pdb_components[sample.pdb_id].update(sample.component_ids)
        for accession in sample.uniprots:
            uniprot_components[accession].update(sample.component_ids)
    for values in pdb_components.values():
        union.union_all(sorted(values))
    for values in uniprot_components.values():
        union.union_all(sorted(values))
    # PLINDER defines an MMS as a congeneric core within a shared protein-pocket
    # set. Its official split uses series with >=3 unique members; unioning those
    # same component memberships makes L4 explicit after apo/AHoJ extension.
    for values in mms_to_components.values():
        union.union_all(sorted(values))
    for edge in edges:
        first = samples[edge.sample_id_a]
        second = samples[edge.sample_id_b]
        union.union_all(sorted(first.component_ids | second.component_ids))

    requests: dict[str, set[str]] = defaultdict(set)
    split_table = pq.read_table(
        DATA / "plinder" / "splits" / "split.parquet",
        columns=["system_id", "split"],
    ).to_pandas()
    for row in split_table.itertuples(index=False):
        component = system_to_component.get(str(row.system_id))
        if component and str(row.split) in {"train", "val", "test"}:
            requests[union.find(component)].add(str(row.split))
    protected = pd.read_csv(SPLITS / "protected_pdb_ids.csv")
    for row in protected.itertuples(index=False):
        for component in pdb_to_components.get(str(row.pdb_id).lower(), set()):
            requests[union.find(component)].add(str(row.required_split))

    precedence = {"train": 0, "val": 1, "test": 2}
    root_split: dict[str, str] = {}
    for component in union.parent:
        root = union.find(component)
        labels = requests.get(root, {"train"})
        root_split[root] = max(labels, key=lambda value: precedence[value])
    sample_cluster: dict[str, str] = {}
    sample_split: dict[str, str] = {}
    root_members: dict[str, list[str]] = defaultdict(list)
    for component in union.parent:
        root_members[union.find(component)].append(component)
    root_cluster = {
        root: f"v4c_{digest_text('|'.join(sorted(members)), 20)}"
        for root, members in root_members.items()
    }
    for sample_id, sample in samples.items():
        roots = {union.find(component) for component in sample.component_ids}
        if len(roots) != 1:
            raise AssertionError(f"sample {sample_id} has unmerged roots: {roots}")
        root = next(iter(roots))
        sample_cluster[sample_id] = root_cluster[root]
        sample_split[sample_id] = root_split[root]
    return sample_cluster, sample_split, union


def pocket_assignments(samples: dict[str, Sample], edges: list[PairEdge]) -> dict[str, str]:
    union = UnionFind()
    for sample_id in samples:
        union.add(sample_id)
    for edge in edges:
        union.union(edge.sample_id_a, edge.sample_id_b)
    members: dict[str, list[str]] = defaultdict(list)
    for sample_id in samples:
        members[union.find(sample_id)].append(sample_id)
    pocket_by_root = {
        root: f"v4p_{digest_text('|'.join(sorted(values)), 20)}"
        for root, values in members.items()
    }
    return {
        sample_id: pocket_by_root[union.find(sample_id)] for sample_id in samples
    }


def apply_pair_coordinate_frames(
    samples: dict[str, Sample],
    edges: list[PairEdge],
) -> dict[str, object]:
    """Put each connected apo/holo pocket into one real local coordinate frame.

    Tuple coordinates are rewritten from their deposited crystal frame into a
    deterministic pocket-root frame.  ``Sample.map_R/map_t`` retain the inverse
    root->deposited transform used by the OTF density loader, so every sample
    still reads density from its own experimental map.
    """
    adjacency: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.sample_id_a].append(
            (edge.sample_id_b, edge.R_a_to_b, edge.t_a_to_b)
        )
        reverse_R, reverse_t = inverse_transform(edge.R_a_to_b, edge.t_a_to_b)
        adjacency[edge.sample_id_b].append(
            (edge.sample_id_a, reverse_R, reverse_t)
        )

    unseen = set(adjacency)
    n_components = 0
    max_cycle_rotation_error = 0.0
    max_cycle_translation_error = 0.0
    while unseen:
        component_nodes: set[str] = set()
        stack = [next(iter(unseen))]
        while stack:
            node = stack.pop()
            if node in component_nodes:
                continue
            component_nodes.add(node)
            stack.extend(neighbor for neighbor, _rotation, _translation in adjacency[node])
        unseen.difference_update(component_nodes)
        n_components += 1
        root = min(
            component_nodes,
            key=lambda sample_id: (
                samples[sample_id].state != "holo",
                "plinder" not in samples[sample_id].sources,
                sample_id,
            ),
        )
        to_root: dict[str, tuple[np.ndarray, np.ndarray]] = {
            root: (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
        }
        queue = deque([root])
        while queue:
            current = queue.popleft()
            current_R, current_t = to_root[current]
            for neighbor, current_to_neighbor_R, current_to_neighbor_t in adjacency[current]:
                neighbor_to_current_R, neighbor_to_current_t = inverse_transform(
                    current_to_neighbor_R,
                    current_to_neighbor_t,
                )
                candidate_R, candidate_t = compose_transforms(
                    neighbor_to_current_R,
                    neighbor_to_current_t,
                    current_R,
                    current_t,
                )
                if neighbor in to_root:
                    known_R, known_t = to_root[neighbor]
                    max_cycle_rotation_error = max(
                        max_cycle_rotation_error,
                        float(np.linalg.norm(candidate_R - known_R)),
                    )
                    max_cycle_translation_error = max(
                        max_cycle_translation_error,
                        float(np.linalg.norm(candidate_t - known_t)),
                    )
                    continue
                to_root[neighbor] = (candidate_R, candidate_t)
                queue.append(neighbor)
        for sample_id, (to_root_R, to_root_t) in to_root.items():
            sample = samples[sample_id]
            sample.item = transformed_item(sample.item, to_root_R, to_root_t)
            map_R, map_t = inverse_transform(to_root_R, to_root_t)
            sample.map_R = map_R.astype(np.float32)
            sample.map_t = map_t.astype(np.float32)

    return {
        "n_paired_coordinate_components": n_components,
        "n_samples_transformed_to_common_frame": len(adjacency),
        "max_cycle_rotation_matrix_error": max_cycle_rotation_error,
        "max_cycle_translation_error_A": max_cycle_translation_error,
    }


def manifest_and_pairs(
    samples: dict[str, Sample],
    edges: list[PairEdge],
    sample_cluster: dict[str, str],
    sample_split: dict[str, str],
    pocket_id: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    partners: dict[str, list[tuple[str, float | None]]] = defaultdict(list)
    pair_rows: list[dict[str, object]] = []
    for edge in edges:
        if sample_split[edge.sample_id_a] != sample_split[edge.sample_id_b]:
            raise AssertionError(f"pair split mismatch: {edge}")
        partners[edge.sample_id_a].append(
            (edge.sample_id_b, edge.pocket_rmsd_apo_holo)
        )
        partners[edge.sample_id_b].append(
            (edge.sample_id_a, edge.pocket_rmsd_apo_holo)
        )
        sample_a = samples[edge.sample_id_a]
        sample_b = samples[edge.sample_id_b]
        a_to_root_R, a_to_root_t = inverse_transform(
            sample_a.map_R,
            sample_a.map_t,
        )
        common_R_a_to_b, common_t_a_to_b = compose_transforms(
            a_to_root_R,
            a_to_root_t,
            sample_b.map_R,
            sample_b.map_t,
        )
        pair_rows.append(
            {
                "sample_id_a": edge.sample_id_a,
                "sample_id_b": edge.sample_id_b,
                "relation": (
                    "apo_holo"
                    if {samples[edge.sample_id_a].state, samples[edge.sample_id_b].state}
                    == {"apo", "holo"}
                    else "holo_holo"
                ),
                "source": edge.source,
                "source_relation_id": edge.source_relation_id,
                "pocket_rmsd_apo_holo": edge.pocket_rmsd_apo_holo,
                "R_a_to_b": np.asarray(common_R_a_to_b, dtype=np.float32)
                .reshape(-1)
                .tolist(),
                "t_a_to_b": np.asarray(
                    common_t_a_to_b,
                    dtype=np.float32,
                ).tolist(),
                "source_alignment_R_a_to_b": np.asarray(
                    edge.R_a_to_b,
                    dtype=np.float32,
                )
                .reshape(-1)
                .tolist(),
                "source_alignment_t_a_to_b": np.asarray(
                    edge.t_a_to_b,
                    dtype=np.float32,
                ).tolist(),
                "split": sample_split[edge.sample_id_a],
                "pocket_id": pocket_id[edge.sample_id_a],
            }
        )

    cluster_counts = Counter(sample_cluster.values())
    rows: list[dict[str, object]] = []
    for sample_id, sample in sorted(samples.items()):
        related = sorted(partners.get(sample_id, []), key=lambda value: value[0])
        paired_holo = next(
            (
                other
                for other, _rmsd in related
                if samples[other].state in {"holo", "holo_alt"}
            ),
            None,
        )
        paired_apo = next(
            (other for other, _rmsd in related if samples[other].state == "apo"),
            None,
        )
        rmsd = next((value for _other, value in related if value is not None), None)
        chains = sorted(
            {
                residue.split(":", 1)[0]
                for residue in sample.canonical_residues.split(";")
                if residue
            }
        )
        cluster_id = sample_cluster[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "pdb_id": sample.pdb_id,
                "assembly": sample.assembly,
                "chains": ";".join(chains),
                "uniprot_acc": ";".join(sorted(sample.uniprots)),
                "pocket_id": pocket_id[sample_id],
                "canonical_pocket_residue_set": sample.canonical_residues,
                "cluster_id": cluster_id,
                "split": sample_split[sample_id],
                "state": sample.state,
                "source": "both" if len(sample.sources) > 1 else next(iter(sample.sources)),
                "structure_origin": "experimental",
                "mms_id": ";".join(sorted(sample.mms_ids)),
                "sample_weight": min(1.0, 32.0 / cluster_counts[cluster_id]),
                "resolution_A": sample.resolution_A,
                "has_structure_factors": True,
                "density_path": f"ccp4/{sample.pdb_id}.ccp4",
                "density_source_pdb_id": sample.pdb_id,
                "grid_frame_id": f"grid:{pocket_id[sample_id]}",
                "density_map_R": sample.map_R.reshape(-1).tolist(),
                "density_map_t": sample.map_t.tolist(),
                "grid_spacing_A": GRID_SPACING,
                "grid_dim": GRID_DIM,
                "density_registration_ok": True,
                "paired_holo_id": paired_holo,
                "paired_apo_id": paired_apo,
                "pocket_rmsd_apo_holo": rmsd,
                "is_interface_pocket": sample.is_interface,
                "aliases": ";".join(sorted(sample.aliases)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def stable_order(sample_ids: Iterable[str]) -> list[str]:
    return sorted(sample_ids, key=lambda value: (digest_text(value, 40), value))


def balanced_cluster_cap(
    manifest: pd.DataFrame,
    *,
    split: str,
    cap: int,
) -> list[str]:
    selected: list[str] = []
    subset = manifest[manifest["split"].eq(split)]
    for _cluster_id, group in subset.groupby("cluster_id", sort=True):
        buckets = {
            state: stable_order(group.loc[group["state"].eq(state), "sample_id"])
            for state in ("apo", "holo", "holo_alt")
        }
        cluster_rows: list[str] = []
        while len(cluster_rows) < cap and any(buckets.values()):
            for state in ("holo", "apo", "holo_alt"):
                if buckets[state] and len(cluster_rows) < cap:
                    cluster_rows.append(buckets[state].pop(0))
        selected.extend(cluster_rows)
    return selected


def diverse_validation_ids(manifest: pd.DataFrame, count: int) -> list[str]:
    subset = manifest[manifest["split"].eq("val")].copy()
    per_cluster = {
        cluster: stable_order(group["sample_id"])
        for cluster, group in subset.groupby("cluster_id", sort=True)
    }
    selected: list[str] = []
    while len(selected) < count and any(per_cluster.values()):
        for cluster in sorted(per_cluster):
            if per_cluster[cluster] and len(selected) < count:
                selected.append(per_cluster[cluster].pop(0))
    if len(selected) < count:
        raise RuntimeError(f"need {count} validation samples; found {len(selected)}")
    return selected


def inverse_legacy_shuffle(desired: list[Sample]) -> list[Sample]:
    permutation = list(range(len(desired)))
    random.Random(SHUFFLE_SEED).shuffle(permutation)
    raw: list[Sample | None] = [None] * len(desired)
    for shuffled_index, original_index in enumerate(permutation):
        raw[original_index] = desired[shuffled_index]
    return [sample for sample in raw if sample is not None]


def write_training_artifacts(
    args: argparse.Namespace,
    samples: dict[str, Sample],
    manifest: pd.DataFrame,
) -> dict[str, object]:
    train_ids = balanced_cluster_cap(
        manifest,
        split="train",
        cap=args.max_train_per_cluster,
    )
    validation_ids = diverse_validation_ids(manifest, args.loader_val_n + LEGACY_TAIL)
    active_val = validation_ids[: args.loader_val_n]
    tail_val = validation_ids[args.loader_val_n :]
    desired_ids = train_ids + active_val + tail_val
    desired = [samples[sample_id] for sample_id in desired_ids]
    raw = inverse_legacy_shuffle(desired)
    args.output_tuples.parent.mkdir(parents=True, exist_ok=True)
    torch.save([sample.item for sample in raw], args.output_tuples)

    loader_samples = desired[: len(desired) - LEGACY_TAIL]
    pdb_ids = np.asarray([sample.pdb_id for sample in loader_samples])
    centroids = np.stack([tuple_center(sample.item) for sample in loader_samples])
    count = len(loader_samples)
    args.output_resample.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_resample / "train_manifest.npz",
        pdb_id=pdb_ids,
        centroid=centroids.astype(np.float32),
        R=np.stack([sample.map_R for sample in loader_samples]).astype(np.float32),
        t=np.stack([sample.map_t for sample in loader_samples]).astype(np.float32),
        ok=np.ones(count, dtype=bool),
        sample_id=np.asarray([sample.sample_id for sample in loader_samples]),
        split=np.asarray(["train"] * len(train_ids) + ["val"] * len(active_val)),
    )
    np.save(
        args.output_resample / "train_available.npy",
        np.ones(count, dtype=bool),
    )
    source_recipe = json.loads(
        (PRETRAIN / "xray_resample_plinder_v3_perelem" / "resample.json").read_text()
    )
    recipe = {
        "kind": "resample_manifest",
        "grid_dim": GRID_DIM,
        "resolution": GRID_SPACING,
        "ccp4_dir": str(MAP_DIR.resolve()),
        "ccp4_ext": ".ccp4",
        "frame": (
            "one deterministic local frame per paired pocket component; R/t map "
            "that frame back into each structure's own deposited experimental map"
        ),
        "norm_version": source_recipe.get("norm_version", "v6"),
        "normalization": source_recipe["normalization"],
        "train_time_note": (
            "VoxBind v4 PLINDER holo + PLINDER/AHoJ apo pocket states. "
            "Apo tuples expose zero ligand atoms; the aligned holo ligand is only "
            "the density crop/recentering anchor."
        ),
        "data_file": str(args.output_tuples.relative_to(DATA)),
        "manifest": str(args.output_manifest.relative_to(DATA)),
        "pair_edges": str(args.output_pairs.relative_to(DATA)),
        "shuffle_seed": SHUFFLE_SEED,
        "legacy_tail_drop": LEGACY_TAIL,
        "subset_n": len(train_ids),
        "subset_val_n": len(active_val),
        "n_lig_ch": 8,
        "n_poc_ch": 4,
        "max_train_per_cluster": args.max_train_per_cluster,
    }
    (args.output_resample / "resample.json").write_text(
        json.dumps(recipe, indent=2) + "\n"
    )
    contract = {
        "kind": "voxbind_v4_training_loader_contract",
        "data_file": str(args.output_tuples),
        "data_file_sha256": file_sha256(args.output_tuples),
        "resample_dir": str(args.output_resample),
        "post_shuffle_train_samples": len(train_ids),
        "post_shuffle_validation_samples": len(active_val),
        "reserved_legacy_tail_samples": len(tail_val),
        "raw_tuple_count": len(raw),
        "manifest_position_count": count,
        "subset_n": len(train_ids),
        "subset_val_n": len(active_val),
        "shuffle_seed": SHUFFLE_SEED,
        "tail_sample_ids": tail_val,
    }
    (args.output_resample / "loader_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    manifest["in_training_corpus"] = manifest["sample_id"].isin(train_ids)
    manifest["in_loader_validation"] = manifest["sample_id"].isin(active_val)
    manifest["reserved_loader_tail"] = manifest["sample_id"].isin(tail_val)
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--v3-tuples",
        type=Path,
        default=PRETRAIN / "data_train_plinder_v3_perelem.pt",
    )
    parser.add_argument(
        "--plinder-apo-pairs",
        type=Path,
        default=V4_DATA / "plinder_v3_apo_candidates" / "pairs.pt",
    )
    parser.add_argument(
        "--ahoj-pairs",
        type=Path,
        default=V4_DATA / "ahoj_local_candidates" / "pairs.pt",
    )
    parser.add_argument(
        "--canonical-cache",
        type=Path,
        default=V4_DATA / "canonical_plinder_samples.parquet",
    )
    parser.add_argument("--rebuild-canonical-cache", action="store_true")
    parser.add_argument(
        "--canonical-jobs",
        type=int,
        default=8,
        help="threads used only when the reusable canonical cache is absent",
    )
    parser.add_argument(
        "--prepare-canonical-cache",
        action="store_true",
        help="build only the reusable PLINDER tuple-to-residue cache, then exit",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=V4_DATA / "manifest.parquet",
    )
    parser.add_argument(
        "--output-pairs",
        type=Path,
        default=V4_DATA / "pair_edges.parquet",
    )
    parser.add_argument(
        "--output-tuples",
        type=Path,
        default=PRETRAIN / "data_train_plinder_v4_perelem.pt",
    )
    parser.add_argument(
        "--output-resample",
        type=Path,
        default=PRETRAIN / "xray_resample_plinder_v4_perelem",
    )
    parser.add_argument(
        "--output-induced",
        type=Path,
        default=V4_DATA / "induced_fit_eval.parquet",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=V4_DATA / "build_report.json",
    )
    parser.add_argument("--max-len", type=float, default=30.0)
    parser.add_argument("--max-train-per-cluster", type=int, default=128)
    parser.add_argument("--loader-val-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_canonical_cache:
        for path in (args.v3_tuples, args.plinder_apo_pairs):
            if not path.exists():
                raise SystemExit(f"[fatal] missing canonical-cache input: {path}")
        v3_tuples = torch.load(
            args.v3_tuples,
            map_location="cpu",
            weights_only=False,
        )
        plinder_pairs = torch.load(
            args.plinder_apo_pairs,
            map_location="cpu",
            weights_only=False,
        )
        records = {item[0]["id"]: item for item in v3_tuples}
        for pair in plinder_pairs:
            item = (pair["apo"]["pocket"], pair["apo"]["anchor_ligand"])
            records[item[0]["id"]] = item
        cache = compute_plinder_residue_cache(
            records,
            args.canonical_cache,
            rebuild=args.rebuild_canonical_cache,
            jobs=args.canonical_jobs,
        )
        print(
            json.dumps(
                {
                    "canonical_cache": str(args.canonical_cache),
                    "n_rows": len(cache),
                    "n_valid": int(
                        cache["canonical_pocket_residue_set"].astype(bool).sum()
                    ),
                },
                indent=2,
            )
        )
        return
    for path in (args.v3_tuples, args.plinder_apo_pairs, args.ahoj_pairs):
        if not path.exists():
            raise SystemExit(f"[fatal] missing build input: {path}")
    (
        samples,
        edges,
        source_report,
        system_to_component,
        pdb_to_components,
        mms_to_components,
    ) = load_source_samples(args)
    before_dedup = len(samples)
    samples, edges, _aliases = deduplicate_samples(samples, edges)
    after_physical_dedup = len(samples)
    samples, edges, same_pdb_pocket_removed = deduplicate_same_pdb_pocket(
        samples,
        edges,
    )
    coordinate_frame_report = apply_pair_coordinate_frames(samples, edges)
    sample_cluster, sample_split, _component_union = leakage_assignments(
        samples,
        edges,
        system_to_component,
        pdb_to_components,
        mms_to_components,
    )
    pocket_id = pocket_assignments(samples, edges)
    manifest, pair_frame = manifest_and_pairs(
        samples,
        edges,
        sample_cluster,
        sample_split,
        pocket_id,
    )
    contract = write_training_artifacts(args, samples, manifest)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.output_manifest, index=False)
    pair_frame.to_parquet(args.output_pairs, index=False)

    test_pairs = pair_frame[
        pair_frame["split"].eq("test")
        & pair_frame["pocket_rmsd_apo_holo"].notna()
    ].copy()
    if len(test_pairs):
        threshold = max(
            1.5,
            float(test_pairs["pocket_rmsd_apo_holo"].quantile(0.90)),
        )
        induced = test_pairs[
            test_pairs["pocket_rmsd_apo_holo"].ge(threshold)
        ].copy()
    else:
        threshold = None
        induced = test_pairs
    args.output_induced.parent.mkdir(parents=True, exist_ok=True)
    induced.to_parquet(args.output_induced, index=False)

    report = {
        "kind": "voxbind_v4_corpus_build",
        "method": {
            "canonical_pocket": (
                "all supported protein heavy atoms whose nearest holo-ligand "
                "heavy atom is within 10 A"
            ),
            "grid_dim": GRID_DIM,
            "grid_spacing_A": GRID_SPACING,
            "leakage_backbone": CLUSTER_COLUMN,
            "split_precedence": ["test", "val", "train"],
            "max_train_samples_per_cluster": args.max_train_per_cluster,
        },
        "inputs": {
            "v3_tuples": {
                "path": str(args.v3_tuples),
                "sha256": file_sha256(args.v3_tuples),
            },
            "plinder_apo_pairs": {
                "path": str(args.plinder_apo_pairs),
                "sha256": file_sha256(args.plinder_apo_pairs),
            },
            "ahoj_pairs": {
                "path": str(args.ahoj_pairs),
                "sha256": file_sha256(args.ahoj_pairs),
            },
            "canonical_cache": {
                "path": str(args.canonical_cache),
                "sha256": file_sha256(args.canonical_cache),
            },
        },
        "source_counts": dict(source_report),
        "samples_before_cross_source_dedup": before_dedup,
        "samples_after_cross_source_dedup": len(manifest),
        "cross_source_duplicates_removed": before_dedup - after_physical_dedup,
        "same_pdb_logical_pocket_copies_removed": same_pdb_pocket_removed,
        "coordinate_frames": coordinate_frame_report,
        "mms_leakage": {
            "artifact": str(MMS_PATH),
            "artifact_sha256": file_sha256(MMS_PATH),
            "minimum_unique_members": 3,
            "n_series_union_groups": len(mms_to_components),
            "n_samples_with_mms": int(manifest["mms_id"].astype(bool).sum()),
        },
        "state_counts": manifest["state"].value_counts().to_dict(),
        "source_counts_final": manifest["source"].value_counts().to_dict(),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "split_cluster_counts": (
            manifest.groupby("split")["cluster_id"].nunique().to_dict()
        ),
        "n_pairs": len(pair_frame),
        "n_induced_fit_eval_pairs": len(induced),
        "induced_fit_rmsd_threshold_A": threshold,
        "training_loader": contract,
        "outputs": {
            "manifest": str(args.output_manifest),
            "pair_edges": str(args.output_pairs),
            "induced_fit_eval": str(args.output_induced),
            "tuples": str(args.output_tuples),
            "resample": str(args.output_resample),
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
