"""Build density-backed, pocket-specified PLINDER apo/holo pairs.

The existing PLINDER corpora are ligand-centred holo complexes.  PLINDER v2 also
publishes experimental apo links.  This script intersects a frozen VoxBind
PLINDER selection with those links and keeps pairs for which all of the
following are available locally:

* the reference holo mmCIF used by the existing PLINDER build;
* the deposited apo mmCIF;
* the apo 2Fo-Fc CCP4 map;
* the linked apo chain in PLINDER's ``linked_structures/apo.zip`` archive.

For every retained link, the holo receptor chain is sequence-aligned to the apo
chain and a binding-site Kabsch transform maps the holo ligand into the apo
crystal/map frame.  The apo pocket is then the apo-chain protein atoms within
``--pocket-radius`` of that transformed ligand.  The ligand is retained only as
an alignment/crop anchor and is marked ``ligand_present=False`` in the
model-ready tuple, so compatible dataset loaders return zero ligand atoms.

Outputs (for ``--version v2``):

* ``data/pretrain/plinder_v2_apo/selected_links.csv`` -- metadata/provenance;
* ``data/pretrain/plinder_v2_apo/pair_index.csv`` -- successfully built pairs;
* ``data/pretrain/plinder_v2_apo_pairs.pt`` -- paired holo/apo records;
* ``data/pretrain/data_train_plinder_v2_apo.pt`` -- apo model tuples;
* ``data/pretrain/xray_resample_plinder_v2_apo/`` -- OTF density manifest.

The default is deliberately local-only: "density exists" means a non-empty
CCP4 map is already present in ``data/ccp4`` or ``data/plinder/ccp4``.  Run the
selection report first, acquire additional target CIF/maps if desired, then
rerun; no network access occurs in this script.

Examples
--------
    cd voxbind
    python dataset/plinder/06_build_apo_pairs.py --version v2 --selection-only
    python dataset/plinder/06_build_apo_pairs.py --version v2 --limit 20
    python dataset/plinder/06_build_apo_pairs.py --version v2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import gemmi  # import before torch (shared libstdc++ compatibility)
import numpy as np
import pandas as pd
import scipy.ndimage
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "2")
torch.set_num_threads(2)

VOX_ROOT = Path(__file__).resolve().parents[2]
DATA = VOX_ROOT / "dataset" / "data"
PRETRAIN = DATA / "pretrain"
HOLO_CIF_DIR = DATA / "cif"
LINKS = DATA / "plinder" / "links" / "kind=apo" / "links.parquet"
APO_ZIP = DATA / "plinder" / "linked_structures" / "apo.zip"
MAP_DIRS = (DATA / "ccp4", DATA / "plinder" / "ccp4")
VAL_SIZE = 100
MAX_LEN = 30.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PLINDER_BUILD = load_module(
    VOX_ROOT / "dataset" / "legacy" / "03c_plinder_preprocess.py",
    "voxbind_plinder_legacy_03c",
)
PDBB = PLINDER_BUILD.PDBB


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def output_paths(version: str, *, v4_candidates: bool = False) -> dict[str, Path]:
    base = (
        DATA / "v4" / f"plinder_{version}_apo_candidates"
        if v4_candidates
        else PRETRAIN / f"plinder_{version}_apo"
    )
    return {
        "base": base,
        "selected": base / "selected_links.csv",
        "selection_report": base / "selection_report.json",
        "pair_index": base / "pair_index.csv",
        "skipped": base / "skipped.csv",
        "pairs": (
            base / "pairs.pt"
            if v4_candidates
            else PRETRAIN / f"plinder_{version}_apo_pairs.pt"
        ),
        "tuples": (
            base / "apo_tuples.pt"
            if v4_candidates
            else PRETRAIN / f"data_train_plinder_{version}_apo.pt"
        ),
        "resample": (
            base / "xray_resample"
            if v4_candidates
            else PRETRAIN / f"xray_resample_plinder_{version}_apo"
        ),
    }


def frozen_selection(version: str) -> tuple[pd.DataFrame, Path]:
    freeze = VOX_ROOT / "splits" / "plinder" / version
    csv_path = freeze / "plinder_selected.csv"
    inputs_path = freeze / "plinder_inputs.json"
    if not csv_path.exists():
        raise SystemExit(f"[fatal] missing frozen PLINDER selection: {csv_path}")
    if inputs_path.exists():
        want = json.loads(inputs_path.read_text()).get("selection_sha256")
        got = sha256(csv_path)
        if want and got != want:
            raise SystemExit(
                "[fatal] frozen PLINDER selection hash mismatch:\n"
                f"  expected {want}\n  observed {got}"
            )
    return pd.read_csv(csv_path), csv_path


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def local_map_path(pdb_id: str) -> Path | None:
    return first_existing(d / f"{pdb_id.lower()}.ccp4" for d in MAP_DIRS)


def system_receptor_chains(system_id: str) -> list[str]:
    """Parse receptor chain IDs from ``pdb__biounit__1.A_1.B__1.C``."""
    parts = str(system_id).split("__")
    if len(parts) < 3:
        return []
    chains = []
    for token in parts[2].split("_"):
        chain = token.split(".", 1)[-1].strip()
        if chain:
            chains.append(chain)
    return chains


def apo_chain_id(link_id: str) -> str:
    """``2vb1_A`` -> ``A``; PLINDER permits multi-character chain IDs."""
    return str(link_id).split("_", 1)[1]


def archived_apo_chain_ids(link_ids: Iterable[str]) -> dict[str, str]:
    """Resolve PLINDER link IDs to chain names in the deposited-coordinate file.

    The suffix in a link ID is not always the chain name retained in the
    linked-structure mmCIF (for example, ``1nvv_C.cif`` contains chain ``S``).
    PLINDER's archived file is authoritative because its coordinates are in the
    same frame as the raw target mmCIF and CCP4 map.
    """
    resolved: dict[str, str] = {}
    with zipfile.ZipFile(APO_ZIP) as archive:
        names = set(archive.namelist())
        for link_id in sorted(set(map(str, link_ids))):
            member = f"{link_id}.cif"
            if member not in names:
                continue
            document = gemmi.cif.read_string(archive.read(member).decode("utf-8"))
            structure = gemmi.make_structure_from_block(document.sole_block())
            protein_chains = [
                chain for chain in structure[0] if protein_residues(chain)
            ]
            if protein_chains:
                resolved[link_id] = max(
                    protein_chains, key=lambda chain: len(protein_residues(chain))
                ).name
    return resolved


def is_protein_residue(residue: gemmi.Residue) -> bool:
    info = gemmi.find_tabulated_residue(residue.name)
    return bool(info and info.is_amino_acid())


def protein_residues(chain: gemmi.Chain) -> list[gemmi.Residue]:
    return [res for res in chain if is_protein_residue(res)]


def residue_atom(residue: gemmi.Residue, name: str):
    for atom in residue:
        if atom.name.strip() == name:
            return atom
    return None


_CIGAR = re.compile(r"(\d+)([MID])")


def aligned_residue_pairs(
    query: list[gemmi.Residue],
    target: list[gemmi.Residue],
) -> tuple[list[tuple[gemmi.Residue, gemmi.Residue]], float, float]:
    """Global sequence alignment; return residue pairs, identity, and coverage."""
    if not query or not target:
        return [], 0.0, 0.0
    q_names = [r.name.strip().upper() for r in query]
    t_names = [r.name.strip().upper() for r in target]
    result = gemmi.align_string_sequences(q_names, t_names, [0] * len(t_names))
    qi = ti = exact = 0
    pairs: list[tuple[gemmi.Residue, gemmi.Residue]] = []
    for count_s, op in _CIGAR.findall(result.cigar_str()):
        count = int(count_s)
        for _ in range(count):
            if op == "M":
                if qi < len(query) and ti < len(target):
                    pairs.append((query[qi], target[ti]))
                    exact += int(q_names[qi] == t_names[ti])
                qi += 1
                ti += 1
            elif op == "I":
                qi += 1
            else:  # D
                ti += 1
    identity = exact / max(len(pairs), 1)
    coverage = len(pairs) / max(min(len(query), len(target)), 1)
    return pairs, identity, coverage


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigid transform mapping ``P @ R.T + t`` onto ``Q``."""
    p0, q0 = P.mean(0), Q.mean(0)
    U, _S, Vt = np.linalg.svd((P - p0).T @ (Q - q0))
    det = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, det]) @ U.T
    t = q0 - p0 @ R.T
    rmsd = float(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, axis=1))))
    return R, t, rmsd


def residue_near_ligand(residue: gemmi.Residue, ligand_tree: cKDTree, radius: float) -> bool:
    points = [
        [atom.pos.x, atom.pos.y, atom.pos.z]
        for atom in residue
        if PDBB._is_heavy_element(PDBB._normalise_element(atom.element.name))
    ]
    if not points:
        return False
    distances, _ = ligand_tree.query(np.asarray(points), k=1)
    return bool(np.min(distances) <= radius)


def alignment_for_chain(
    holo_chain: gemmi.Chain,
    apo_chain: gemmi.Chain,
    ligand_xyz: np.ndarray,
    pocket_radius: float,
) -> dict | None:
    h_res = protein_residues(holo_chain)
    a_res = protein_residues(apo_chain)
    pairs, identity, coverage = aligned_residue_pairs(h_res, a_res)
    if not pairs:
        return None

    ligand_tree = cKDTree(ligand_xyz)
    local_pairs = [
        (h, a) for h, a in pairs
        if residue_near_ligand(h, ligand_tree, pocket_radius + 2.0)
    ]
    fit_pairs = local_pairs if len(local_pairs) >= 3 else pairs

    P: list[list[float]] = []
    Q: list[list[float]] = []
    for h_residue, a_residue in fit_pairs:
        for atom_name in ("N", "CA", "C", "O"):
            h_atom = residue_atom(h_residue, atom_name)
            a_atom = residue_atom(a_residue, atom_name)
            if h_atom is None or a_atom is None:
                continue
            P.append([h_atom.pos.x, h_atom.pos.y, h_atom.pos.z])
            Q.append([a_atom.pos.x, a_atom.pos.y, a_atom.pos.z])
    if len(P) < 12:
        return None
    R, t, rmsd = kabsch(np.asarray(P, dtype=np.float64), np.asarray(Q, dtype=np.float64))
    return {
        "R": R,
        "t": t,
        "rmsd": rmsd,
        "identity": identity,
        "coverage": coverage,
        "n_residue_pairs": len(pairs),
        "n_local_residue_pairs": len(local_pairs),
        "n_fit_atoms": len(P),
        "holo_chain": holo_chain.name,
        "apo_chain": apo_chain.name,
    }


def find_chain(model: gemmi.Model, chain_id: str) -> gemmi.Chain | None:
    for chain in model:
        if chain.name == chain_id:
            return chain
    return None


def best_holo_to_apo_alignment(
    holo_model: gemmi.Model,
    apo_chain: gemmi.Chain,
    ligand_xyz: np.ndarray,
    reference_system_id: str,
    pocket_radius: float,
) -> dict | None:
    preferred = system_receptor_chains(reference_system_id)
    candidates = [
        chain for chain in holo_model
        if protein_residues(chain) and (not preferred or chain.name in preferred)
    ]
    if not candidates:
        candidates = [chain for chain in holo_model if protein_residues(chain)]
    scored = []
    for chain in candidates:
        result = alignment_for_chain(chain, apo_chain, ligand_xyz, pocket_radius)
        if result is not None:
            scored.append(result)
    if not scored:
        return None
    return max(
        scored,
        key=lambda r: (
            r["identity"],
            r["coverage"],
            r["n_local_residue_pairs"],
            -r["rmsd"],
        ),
    )


def transform(xyz: torch.Tensor | np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    arr = xyz.detach().cpu().numpy() if isinstance(xyz, torch.Tensor) else np.asarray(xyz)
    return arr.astype(np.float64) @ R.T + t


def apo_pocket(
    chain: gemmi.Chain,
    transformed_ligand: np.ndarray,
    pocket_radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows: list[tuple[int, float, float, float, float]] = []
    for residue in protein_residues(chain):
        for atom in residue:
            element = PDBB._normalise_element(atom.element.name)
            if not PDBB._is_heavy_element(element):
                continue
            channel = PDBB._channel_of(element, PDBB.N_POC_CH)
            if channel is None:
                continue
            rows.append(
                (channel, atom.pos.x, atom.pos.y, atom.pos.z, PLINDER_BUILD.vdw_radius(element))
            )
    if not rows:
        raise ValueError("apo_chain_has_no_supported_atoms")
    xyz = np.asarray([[x, y, z] for _c, x, y, z, _r in rows], dtype=np.float64)
    distance, _ = cKDTree(transformed_ligand).query(xyz, k=1)
    keep = distance <= pocket_radius
    if not np.any(keep):
        raise ValueError("apo_pocket_empty")
    channels = np.asarray([row[0] for row in rows], dtype=np.uint8)[keep]
    radii = np.asarray([row[4] for row in rows], dtype=np.float32)[keep]
    return (
        torch.from_numpy(xyz[keep].astype(np.float32)),
        torch.from_numpy(channels),
        torch.from_numpy(radii),
    )


def nearby_hetero_atoms(
    model: gemmi.Model,
    transformed_ligand: np.ndarray,
    radius: float,
) -> tuple[int, int, list[str]]:
    """Count non-water, non-protein atoms near the nominal apo binding site."""
    tree = cKDTree(transformed_ligand)
    total = organic = 0
    residues: set[str] = set()
    for chain in model:
        for residue in chain:
            if is_protein_residue(residue) or residue.is_water():
                continue
            heavy = []
            has_carbon = False
            for atom in residue:
                element = PDBB._normalise_element(atom.element.name)
                if not PDBB._is_heavy_element(element):
                    continue
                heavy.append([atom.pos.x, atom.pos.y, atom.pos.z])
                has_carbon = has_carbon or element == "C"
            if not heavy:
                continue
            distances, _ = tree.query(np.asarray(heavy), k=1)
            count = int(np.sum(distances <= radius))
            if count:
                total += count
                organic += count if has_carbon else 0
                residues.add(f"{chain.name}:{residue.name}:{residue.seqid}")
    return total, organic, sorted(residues)


def load_density_grid(path: Path):
    density_map = gemmi.read_ccp4_map(str(path))
    density_map.setup(float("nan"))
    grid = density_map.grid
    arr = np.array(grid, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("density_nonfinite")
    sigma = float(arr.std())
    if not np.isfinite(sigma) or sigma < 1e-8:
        raise ValueError("density_flat")
    orthogonal = np.array(grid.unit_cell.orth.mat.tolist(), dtype=np.float64)
    frac_T = np.linalg.inv(orthogonal).T
    return arr, frac_T, *arr.shape


def sample_grid(grid, xyz: np.ndarray) -> np.ndarray:
    arr, frac_T, nu, nv, nw = grid
    fractional = np.asarray(xyz, dtype=np.float64) @ frac_T
    indices = [
        (fractional[:, 0] * nu) % nu,
        (fractional[:, 1] * nv) % nv,
        (fractional[:, 2] * nw) % nw,
    ]
    return scipy.ndimage.map_coordinates(
        arr, indices, order=1, mode="wrap", prefilter=False
    )


def crop_is_finite(grid, center: np.ndarray, grid_dim: int = 64, resolution: float = 0.25) -> bool:
    arr, frac_T, nu, nv, nw = grid
    crop = PLINDER_BUILD._crop_density(
        arr, frac_T, nu, nv, nw, center,
        G=grid_dim, res=resolution, transform=None,
    )
    return bool(np.isfinite(crop).all() and float(crop.std()) > 1e-8)


def pair_id_for_row(row, args) -> str:
    """Return a stable candidate key without collapsing distinct holo pockets."""
    base = f"{row.reference_system_id}__apo_{row.id}"
    if not args.v4_candidates:
        return base
    ligand_asym = re.sub(r"[^A-Za-z0-9.-]+", "-", str(row.ligand_asym_id))
    ligand_ccd = re.sub(r"[^A-Za-z0-9.-]+", "-", str(row.ligand_ccd_code))
    return f"{base}__lig_{ligand_asym}_{ligand_ccd}"


def select_links(args, paths: dict[str, Path]) -> pd.DataFrame:
    selection, selection_path = frozen_selection(args.version)
    if not LINKS.exists():
        raise SystemExit(
            f"[fatal] missing {LINKS}\n"
            "Download PLINDER 2024-06/v2/links/kind=apo/links.parquet first."
        )
    if not APO_ZIP.exists():
        raise SystemExit(
            f"[fatal] missing {APO_ZIP}\n"
            "Download PLINDER 2024-06/v2/linked_structures/apo.zip first."
        )

    wanted = set(selection["system_id"].astype(str))
    columns = [
        "reference_system_id", "id", "target_id", "pocket_fident", "pocket_lddt",
        "protein_lddt_weighted_sum", "sort_score", "fraction_reference_proteins_mapped",
        "fraction_model_proteins_mapped", "num_reference_proteins", "num_model_proteins",
        "receptor_file",
    ]
    links = pd.read_parquet(LINKS, columns=columns)
    links = links[links["reference_system_id"].isin(wanted)].copy()
    funnel = [{"stage": "linked", "links": len(links),
               "systems": int(links.reference_system_id.nunique())}]

    filters = [
        ("pocket_fident", links.pocket_fident >= args.min_pocket_fident),
        ("pocket_lddt", links.pocket_lddt >= args.min_pocket_lddt),
        ("apo_resolution", links.sort_score <= args.max_apo_resolution),
        ("single_chain", (links.num_reference_proteins == 1) & (links.num_model_proteins == 1)),
        ("fully_mapped", (links.fraction_reference_proteins_mapped == 1.0)
         & (links.fraction_model_proteins_mapped == 1.0)),
    ]
    mask = pd.Series(True, index=links.index)
    for name, condition in filters:
        mask &= condition.fillna(False)
        current = links[mask]
        funnel.append({"stage": name, "links": len(current),
                       "systems": int(current.reference_system_id.nunique())})
    links = links[mask].copy()

    with zipfile.ZipFile(APO_ZIP) as archive:
        apo_ids = {Path(name).stem for name in archive.namelist() if name.endswith(".cif")}
    links["has_linked_apo"] = links["id"].isin(apo_ids)
    links["apo_cif_path"] = links["target_id"].map(
        lambda pid: str(HOLO_CIF_DIR / f"{str(pid).lower()}.cif")
    )
    links["has_apo_cif"] = links["apo_cif_path"].map(
        lambda path: Path(path).exists() and Path(path).stat().st_size > 0
    )
    links["apo_map_path"] = links["target_id"].map(
        lambda pid: str(local_map_path(str(pid)) or "")
    )
    links["has_apo_density"] = links["apo_map_path"].str.len() > 0
    links["holo_pdb_id"] = links.reference_system_id.str.slice(0, 4).str.lower()
    links["holo_cif_path"] = links.holo_pdb_id.map(
        lambda pid: str(HOLO_CIF_DIR / f"{pid}.cif")
    )
    links["has_holo_cif"] = links.holo_cif_path.map(
        lambda path: Path(path).exists() and Path(path).stat().st_size > 0
    )
    links["holo_map_path"] = links.holo_pdb_id.map(
        lambda pid: str(local_map_path(str(pid)) or "")
    )
    links["has_holo_density"] = links["holo_map_path"].str.len() > 0
    ready = (
        links.has_linked_apo & links.has_apo_cif
        & links.has_apo_density & links.has_holo_cif & links.has_holo_density
    )
    links = links[ready].copy()
    funnel.append({"stage": "local_structure_and_density", "links": len(links),
                   "systems": int(links.reference_system_id.nunique())})

    n_density_backed_links = len(links)
    n_density_backed_systems = int(links.reference_system_id.nunique())
    n_density_backed_apo_pdb = int(links.target_id.nunique())

    # Legacy corpora keep one best apo link per reference system. V4 retains
    # all experimentally density-backed state links and canonicalizes physical
    # pockets only after cross-source relation construction.
    links = links.sort_values(
        [
            "reference_system_id", "pocket_fident", "pocket_lddt",
            "protein_lddt_weighted_sum", "sort_score", "id",
        ],
        ascending=[True, False, False, False, True, True],
    )
    if not args.all_links:
        links = links.drop_duplicates("reference_system_id", keep="first")

    # Repeated frozen rows with the same ligand asym/CCD denote the same
    # physical ligand site. Distinct ligand asyms remain separate pockets.
    selection_for_merge = selection.drop_duplicates(
        ["system_id", "ligand_asym_id", "ligand_ccd_code"],
        keep="first",
    )
    links = links.merge(
        selection_for_merge,
        left_on="reference_system_id",
        right_on="system_id",
        how="left",
        # V4 preserves several apo links and several physical ligand pockets
        # for one system; legacy best-link mode has only one left-side row.
        validate="many_to_many" if args.all_links else "one_to_many",
        suffixes=("", "_selection"),
    )
    links = links.drop_duplicates(
        [
            "reference_system_id",
            "id",
            "target_id",
            "ligand_asym_id",
            "ligand_ccd_code",
        ],
        keep="first",
    )
    if args.limit:
        links = links.head(args.limit).copy()

    paths["base"].mkdir(parents=True, exist_ok=True)
    links.to_csv(paths["selected"], index=False)
    report = {
        "kind": "plinder_apo_selection",
        "version": args.version,
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "apo_links": str(LINKS),
        "apo_links_sha256": sha256(LINKS),
        "apo_archive": str(APO_ZIP),
        "filters": {
            "min_pocket_fident_percent": args.min_pocket_fident,
            "min_pocket_lddt_percent": args.min_pocket_lddt,
            "max_apo_resolution_angstrom": args.max_apo_resolution,
            "requires_single_chain": True,
            "requires_fully_mapped": True,
            "requires_local_apo_cif": True,
            "requires_local_apo_density": True,
            "requires_local_holo_density": True,
        },
        "link_policy": "all_density_backed_links" if args.all_links else "best_per_system",
        "funnel": funnel,
        "n_density_backed_links_before_policy": n_density_backed_links,
        "n_density_backed_systems_before_policy": n_density_backed_systems,
        "n_density_backed_apo_pdb_before_policy": n_density_backed_apo_pdb,
        "n_selected_rows": len(links),
        "n_selected_holo_systems": int(links.reference_system_id.nunique()),
        "n_selected_apo_pdb": int(links.target_id.nunique()),
    }
    paths["selection_report"].write_text(json.dumps(report, indent=2))
    print(
        f"[selection] {len(links):,} candidate rows / "
        f"{links.reference_system_id.nunique():,} holo systems / "
        f"{links.target_id.nunique():,} apo PDBs"
    )
    print(f"            -> {paths['selected']}")
    return links


def build_pair(
    row,
    args,
    density_cache: dict[str, object],
    archive_chain_ids: dict[str, str],
) -> tuple[dict | None, dict]:
    pair_id = pair_id_for_row(row, args)
    status = {
        "pair_id": pair_id,
        "reference_system_id": row.reference_system_id,
        "apo_id": row.id,
        "apo_pdb_id": row.target_id,
        "status": "error",
        "reason": "",
    }
    try:
        holo_path = Path(row.holo_cif_path)
        apo_path = Path(row.apo_cif_path)
        map_path = Path(row.apo_map_path)

        parsed = PLINDER_BUILD.parse_cif_system(
            holo_path,
            row.ligand_asym_id,
            row.ligand_ccd_code,
            args.pocket_radius,
            diverse=False,
            other_channel=True,
        )
        if parsed is None:
            raise ValueError("holo_parse_failed")
        (
            lig_xyz, lig_ch, holo_poc_xyz, holo_poc_ch,
            _lig_unsupported, _poc_unsupported, lig_rad, holo_poc_rad,
        ) = parsed

        holo_structure = gemmi.read_structure(str(holo_path))
        apo_structure = gemmi.read_structure(str(apo_path))
        holo_structure.remove_alternative_conformations()
        holo_structure.remove_hydrogens()
        apo_structure.remove_alternative_conformations()
        apo_structure.remove_hydrogens()
        holo_model = holo_structure[0]
        apo_model = apo_structure[0]
        chain_id = archive_chain_ids.get(str(row.id), apo_chain_id(row.id))
        apo_chain = find_chain(apo_model, chain_id)
        if apo_chain is None:
            raise ValueError(f"apo_chain_missing:{chain_id}")

        alignment = best_holo_to_apo_alignment(
            holo_model,
            apo_chain,
            lig_xyz.numpy(),
            row.reference_system_id,
            args.pocket_radius,
        )
        if alignment is None:
            raise ValueError("alignment_failed")
        if alignment["identity"] < args.min_alignment_identity:
            raise ValueError("low_alignment_identity")
        if alignment["coverage"] < args.min_alignment_coverage:
            raise ValueError("low_alignment_coverage")
        if alignment["rmsd"] > args.max_alignment_rmsd:
            raise ValueError("high_binding_site_rmsd")

        R, t = alignment["R"], alignment["t"]
        lig_apo = transform(lig_xyz, R, t)
        holo_poc_apo = transform(holo_poc_xyz, R, t)
        center = lig_apo.mean(0)
        apo_xyz, apo_ch, apo_rad = apo_pocket(
            apo_chain, lig_apo, args.pocket_radius
        )

        hetero_total, hetero_organic, hetero_residues = nearby_hetero_atoms(
            apo_model, lig_apo, args.nearby_hetero_radius
        )
        if hetero_organic and not args.allow_nearby_organic:
            raise ValueError("nearby_organic_ligand")

        target_id = str(row.target_id)
        if target_id not in density_cache:
            density_cache[target_id] = load_density_grid(map_path)
        grid = density_cache[target_id]
        pocket_values = sample_grid(grid, apo_xyz.numpy())
        arr = grid[0]
        pocket_density_z = float((pocket_values.mean() - arr.mean()) / arr.std())
        pocket_density_coverage = float(np.mean(
            (pocket_values - arr.mean()) / arr.std() >= args.density_atom_z
        ))
        if pocket_density_z < args.min_density_z:
            raise ValueError("weak_apo_pocket_density")
        if not crop_is_finite(grid, center):
            raise ValueError("invalid_density_crop")

        max_len = round(float(np.ptp(lig_apo, axis=0).max()), 2)
        sample_id = f"plinder_apo/{target_id}_{chain_id}__{row.reference_system_id}"
        apo_pocket_record = {
            "id": sample_id,
            "coords": apo_xyz.to(torch.float32),
            "atoms_channel": apo_ch.to(torch.uint8),
            "radius": apo_rad.to(torch.float32),
        }
        anchor_ligand = {
            "id": sample_id,
            "coords": torch.from_numpy(lig_apo.astype(np.float32)),
            "center_coords": torch.from_numpy(center.astype(np.float32)),
            "atoms_channel": lig_ch.to(torch.uint8),
            "radius": lig_rad.to(torch.float32),
            "max_len": max_len,
            "ligand_present": False,
            "role": "holo_alignment_anchor",
        }
        holo_pocket_record = {
            "id": f"plinder_holo/{row.reference_system_id}",
            "coords": torch.from_numpy(holo_poc_apo.astype(np.float32)),
            "atoms_channel": holo_poc_ch.to(torch.uint8),
            "radius": holo_poc_rad.to(torch.float32),
        }
        holo_ligand_record = {
            "id": f"plinder_holo/{row.reference_system_id}",
            "coords": torch.from_numpy(lig_apo.astype(np.float32)),
            "atoms_channel": lig_ch.to(torch.uint8),
            "radius": lig_rad.to(torch.float32),
            "max_len": max_len,
            "ligand_present": True,
        }
        metadata = {
            "pair_id": pair_id,
            "reference_system_id": row.reference_system_id,
            "holo_pdb_id": row.holo_pdb_id,
            "holo_map_path": str(row.holo_map_path),
            "holo_coordinate_frame": "rigidly aligned into apo deposited/map frame",
            "apo_id": row.id,
            "apo_pdb_id": target_id,
            "apo_chain": chain_id,
            "apo_link_id_chain": apo_chain_id(row.id),
            "apo_map_path": str(map_path),
            "pocket_radius": args.pocket_radius,
            "pocket_fident": float(row.pocket_fident),
            "pocket_lddt": float(row.pocket_lddt),
            "apo_resolution": float(row.sort_score),
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
            "pocket_density_mean_z": pocket_density_z,
            "pocket_density_coverage": pocket_density_coverage,
        }
        pair = {
            "id": pair_id,
            "holo": {"pocket": holo_pocket_record, "ligand": holo_ligand_record},
            "apo": {"pocket": apo_pocket_record, "anchor_ligand": anchor_ligand},
            "metadata": metadata,
        }
        status.update(
            status="ok",
            reason="ok",
            holo_chain=alignment["holo_chain"],
            apo_chain=alignment["apo_chain"],
            alignment_identity=alignment["identity"],
            alignment_coverage=alignment["coverage"],
            binding_site_rmsd=alignment["rmsd"],
            n_alignment_atoms=alignment["n_fit_atoms"],
            n_apo_pocket_atoms=len(apo_xyz),
            nearby_hetero_atoms=hetero_total,
            nearby_organic_atoms=hetero_organic,
            pocket_density_mean_z=pocket_density_z,
            pocket_density_coverage=pocket_density_coverage,
            apo_resolution=float(row.sort_score),
            pocket_fident=float(row.pocket_fident),
            pocket_lddt=float(row.pocket_lddt),
        )
        return pair, status
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}:{exc}"
        status["reason"] = reason[:240]
        return None, status


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_status_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict(orient="records")


def write_model_outputs(
    pairs: list[dict],
    args,
    paths: dict[str, Path],
) -> None:
    PRETRAIN.mkdir(parents=True, exist_ok=True)
    torch.save(pairs, paths["pairs"])
    tuples = [
        (pair["apo"]["pocket"], pair["apo"]["anchor_ligand"])
        for pair in pairs
    ]
    torch.save(tuples, paths["tuples"])

    if args.v4_candidates:
        # Never create tuple-level train/validation splits here. The v4
        # manifest builder assigns complete merged leakage components later.
        return

    # Match DatasetCrossDockedDensity's deterministic train/val ordering.
    order = list(tuples)
    random.Random(1234).shuffle(order)
    train_order = order[: max(len(order) - VAL_SIZE, 0)]
    val_order = order[max(len(order) - VAL_SIZE, 0):]
    train_order = [item for item in train_order if item[1]["max_len"] <= MAX_LEN]
    val_order = [item for item in val_order if item[1]["max_len"] <= MAX_LEN]

    resample = paths["resample"]
    map_view = resample / "maps"
    map_view.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        source = Path(pair["metadata"]["apo_map_path"]).resolve()
        destination = map_view / f"{pair['metadata']['apo_pdb_id']}.ccp4"
        if not destination.exists():
            destination.symlink_to(source)

    def write_manifest(split: str, items: list[tuple[dict, dict]]) -> None:
        n_items = len(items)
        pdb_ids = np.asarray([
            pocket["id"].split("/")[1].split("_")[0].lower()
            for pocket, _ligand in items
        ])
        centroids = (
            np.stack([
                ligand["center_coords"].numpy() for _pocket, ligand in items
            ]).astype(np.float32)
            if n_items else np.zeros((0, 3), dtype=np.float32)
        )
        np.savez(
            resample / f"{split}_manifest.npz",
            pdb_id=pdb_ids,
            centroid=centroids,
            R=np.broadcast_to(
                np.eye(3, dtype=np.float32), (n_items, 3, 3)
            ).copy(),
            t=np.zeros((n_items, 3), dtype=np.float32),
            ok=np.ones(n_items, dtype=bool),
        )
        np.save(
            resample / f"{split}_available.npy",
            np.ones(n_items, dtype=bool),
        )

    write_manifest("train", train_order)
    write_manifest("val", val_order)

    norm_source = (
        PRETRAIN / f"xray_resample_plinder_{args.version}_perelem" / "resample.json"
    )
    if not norm_source.exists():
        norm_source = DATA / "plinder" / "xray_resample_plinder" / "resample.json"
    normalization = json.loads(norm_source.read_text())["normalization"]
    recipe = {
        "kind": "resample_manifest",
        "grid_dim": 64,
        "resolution": 0.25,
        "ccp4_dir": str(map_view.resolve()),
        "ccp4_ext": ".ccp4",
        "frame": (
            "apo deposited/map frame (transform=None): holo ligand is an alignment "
            "anchor only; apo tuple has ligand_present=false"
        ),
        "norm_version": "v6",
        "normalization": normalization,
        "train_time_note": (
            f"PLINDER {args.version} linked experimental apo pockets. Protein coordinates "
            "and 2Fo-Fc density share the deposited apo PDB frame. Ligand channels are zero; "
            "the aligned holo ligand supplies only the crop/recentering centre."
        ),
        "n_pairs": len(pairs),
        "n_train_after_tail_and_size_filter": len(train_order),
        "n_val_after_tail_and_size_filter": len(val_order),
        "data_file": str(paths["tuples"]),
        "paired_records": str(paths["pairs"]),
    }
    (resample / "resample.json").write_text(json.dumps(recipe, indent=2))


def build(args, paths: dict[str, Path], selected: pd.DataFrame) -> None:
    pair_by_id: dict[str, dict] = {}
    status_by_id: dict[str, dict] = {}
    density_cache: dict[str, object] = {}
    # Sorting by target reuses each potentially large CCP4 array before moving on.
    selected = selected.sort_values(["target_id", "reference_system_id"])
    selected_rows = list(selected.itertuples(index=False))

    if args.resume and paths["pairs"].exists():
        previous_pairs = torch.load(paths["pairs"], map_location="cpu", weights_only=False)
        # Backfill provenance when resuming records made before holo-density
        # visualization support was added.
        for pair in previous_pairs:
            metadata = pair["metadata"]
            if not metadata.get("holo_map_path"):
                map_path = local_map_path(str(metadata["holo_pdb_id"]))
                if map_path is not None:
                    metadata["holo_map_path"] = str(map_path)
            metadata.setdefault(
                "holo_coordinate_frame",
                "rigidly aligned into apo deposited/map frame",
            )
        pair_by_id = {pair["id"]: pair for pair in previous_pairs}
        previous_statuses = (
            read_status_rows(paths["pair_index"]) + read_status_rows(paths["skipped"])
        )
        status_by_id = {
            str(status["pair_id"]): status for status in previous_statuses
        }

    pending = []
    n_reused_ok = 0
    for row in selected_rows:
        pair_id = pair_id_for_row(row, args)
        previous = status_by_id.get(pair_id)
        if previous is None:
            pending.append(row)
        elif str(previous.get("status")) == "ok" and pair_id in pair_by_id:
            n_reused_ok += 1
        elif (
            args.retry_reason_prefix
            and not str(previous.get("reason", "")).startswith(args.retry_reason_prefix)
        ):
            continue
        else:
            pending.append(row)

    archive_chain_ids = archived_apo_chain_ids(row.id for row in pending)
    if args.resume:
        print(
            f"[resume] reuse {n_reused_ok:,} valid pairs; "
            f"re-evaluate {len(pending):,}/{len(selected_rows):,} rows"
        )
    for row in tqdm(
        pending,
        desc=f"apo-pairs-{args.version}",
        unit="pair",
    ):
        pair, status = build_pair(row, args, density_cache, archive_chain_ids)
        status_by_id[status["pair_id"]] = status
        if pair is not None:
            pair_by_id[pair["id"]] = pair
        else:
            pair_by_id.pop(status["pair_id"], None)
        # Keep only one raw map resident; links are sorted by target PDB.
        if len(density_cache) > 1:
            newest = str(row.target_id)
            density_cache = {newest: density_cache[newest]}

    ordered_ids = [
        pair_id_for_row(row, args) for row in selected_rows
    ]
    statuses = [status_by_id[pair_id] for pair_id in ordered_ids if pair_id in status_by_id]
    pairs = [pair_by_id[pair_id] for pair_id in ordered_ids if pair_id in pair_by_id]
    ok_rows = [row for row in statuses if row["status"] == "ok"]
    skip_rows = [row for row in statuses if row["status"] != "ok"]
    write_csv(paths["pair_index"], ok_rows)
    write_csv(paths["skipped"], skip_rows)
    if not pairs:
        raise SystemExit("[fatal] no apo pairs passed build validation")
    write_model_outputs(pairs, args, paths)

    reasons = Counter(row["reason"] for row in skip_rows)
    build_report = json.loads(paths["selection_report"].read_text())
    build_report.update({
        "kind": "plinder_apo_build",
        "n_attempted": len(statuses),
        "n_built": len(pairs),
        "n_skipped": len(skip_rows),
        "n_reused_ok": n_reused_ok,
        "skip_reasons": dict(reasons.most_common()),
        "n_unique_apo_pdb_built": len(
            {pair["metadata"]["apo_pdb_id"] for pair in pairs}
        ),
        "outputs": {key: str(path) for key, path in paths.items()},
        "tuple_contract": {
            "pocket": "true apo protein atoms within pocket_radius of aligned holo ligand",
            "ligand": "holo alignment anchor; ligand_present=false; loader emits zero ligand atoms",
            "density": "experimental apo-PDB 2Fo-Fc map, cropped at aligned holo ligand centroid",
        },
    })
    paths["selection_report"].write_text(json.dumps(build_report, indent=2))

    print(f"\n[built] {len(pairs):,}/{len(statuses):,} validated apo/holo pairs")
    print(f"        unique apo PDBs: {build_report['n_unique_apo_pdb_built']:,}")
    for reason, count in reasons.most_common():
        print(f"        skip {reason:30s} {count:,}")
    print(f"        tuples   -> {paths['tuples']}")
    print(f"        pairs    -> {paths['pairs']}")
    print(f"        manifest -> {paths['resample']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", default="v2", choices=["v2", "v2p1", "v3"])
    parser.add_argument("--limit", type=int, default=0, help="first N best links; 0 = all")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument(
        "--v4-candidates",
        action="store_true",
        help="write unsplit v4 candidate records under dataset/data/v4",
    )
    parser.add_argument(
        "--all-links",
        action="store_true",
        help="retain every density-backed apo link instead of one per holo system",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse successful pairs and prior statuses from the current output",
    )
    parser.add_argument(
        "--retry-reason-prefix", default="",
        help="with --resume, retry only prior failures whose reason starts with this text",
    )
    parser.add_argument("--pocket-radius", type=float, default=10.0)
    parser.add_argument("--nearby-hetero-radius", type=float, default=6.0)
    parser.add_argument("--allow-nearby-organic", action="store_true")
    parser.add_argument("--min-pocket-fident", type=float, default=99.0,
                        help="PLINDER percentage, not a 0-1 fraction")
    parser.add_argument("--min-pocket-lddt", type=float, default=80.0,
                        help="PLINDER percentage, not a 0-1 fraction")
    parser.add_argument("--max-apo-resolution", type=float, default=2.5)
    parser.add_argument("--min-alignment-identity", type=float, default=0.90)
    parser.add_argument("--min-alignment-coverage", type=float, default=0.80)
    parser.add_argument("--max-alignment-rmsd", type=float, default=2.5)
    parser.add_argument("--min-density-z", type=float, default=0.5,
                        help="minimum mean global-map z-score at apo pocket atoms")
    parser.add_argument("--density-atom-z", type=float, default=0.5,
                        help="threshold reported as pocket atom density coverage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_links and not args.v4_candidates:
        raise SystemExit("[fatal] --all-links requires --v4-candidates")
    paths = output_paths(args.version, v4_candidates=args.v4_candidates)
    selected = select_links(args, paths)
    if args.selection_only:
        return
    build(args, paths, selected)


if __name__ == "__main__":
    main()
