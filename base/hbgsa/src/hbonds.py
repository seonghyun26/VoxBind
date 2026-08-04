"""hbonds.py — extract the protein–ligand hydrogen-bond graph with PyMOL.

Faithful to HBGSA (arXiv 2604.23115): H-bonds are the graph nodes, each carrying
a 9-dim coordinate vector [protein-end xyz, ligand-end xyz, midpoint xyz]. The
paper detects polar contacts with PyMOL; we use PyMOL for structure I/O + H-add,
then apply the paper's stated geometric criteria directly:

    donor–acceptor distance d ≤ 3.5 Å,
    D–H···A angle (vertex H) ≥ 120°,
    polar atoms restricted to N / O / S.

An isolated `pymol2.PyMOL()` instance is used so the extractor is process-safe
(no global singleton) and shardable for parallel cache building.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (CACHE_DIR, HB_ANGLE_MIN, HB_DIST_MAX, HB_KNN_K, HB_MAX_NODES,
                    HB_POLAR_ELEMS)
from manifest import build_manifest

HB_CACHE = CACHE_DIR / "hbonds"
HB_CACHE.mkdir(parents=True, exist_ok=True)
_POLAR = set(HB_POLAR_ELEMS)


def _polar_atoms_with_h(model) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """For one PyMOL model: list of (heavy-polar-atom xyz, [bonded H xyz...])."""
    atoms = model.atom
    coords = [np.asarray(a.coord, dtype=np.float64) for a in atoms]
    bonded_h: dict[int, list[int]] = {}
    for b in model.bond:                          # bond.index is 0-based into atoms[]
        i, j = b.index
        if atoms[i].symbol == "H":
            bonded_h.setdefault(j, []).append(i)
        if atoms[j].symbol == "H":
            bonded_h.setdefault(i, []).append(j)
    out = []
    for i, a in enumerate(atoms):
        if a.symbol in _POLAR:
            out.append((coords[i], [coords[k] for k in bonded_h.get(i, [])]))
    return out


def _angle_at_h(D: np.ndarray, H: np.ndarray, A: np.ndarray) -> float:
    """D–H···A angle in degrees, vertex at H."""
    v1, v2 = D - H, A - H
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def extract_hbond_graph(
    protein_pdb: str,
    ligand_sdf: str,
    *,
    dist_max: float = HB_DIST_MAX,
    angle_min: float = HB_ANGLE_MIN,
    max_nodes: int = HB_MAX_NODES,
    site_radius: float = 6.0,
) -> dict:
    """Return {'node_feats': (N,9) float32, 'mid': (N,3), 'n': N} for one complex."""
    import pymol2

    with pymol2.PyMOL() as p:
        cmd = p.cmd
        cmd.set("retain_order", 1)
        cmd.load(protein_pdb, "prot")
        cmd.load(ligand_sdf, "lig")
        cmd.remove("solvent")
        cmd.remove("prot and not polymer")          # strip ions/hetero from protein obj
        cmd.h_add("prot")                            # protein PDBs ship without H
        if cmd.count_atoms("lig and hydro") == 0:
            cmd.h_add("lig")
        # restrict protein to the binding-site shell to keep the pair loop cheap
        cmd.select("site", f"byres (prot within {site_radius} of lig)")
        prot_model = cmd.get_model("site")
        lig_model = cmd.get_model("lig")

    lig_polar = _polar_atoms_with_h(lig_model)
    prot_polar = _polar_atoms_with_h(prot_model)

    hbonds = []   # (prot_xyz, lig_xyz, dist)
    for lc, lH in lig_polar:
        for pc, pH in prot_polar:
            d = float(np.linalg.norm(lc - pc))
            if d > dist_max:
                continue
            best = 0.0
            for h in lH:                              # ligand donor → protein acceptor
                best = max(best, _angle_at_h(lc, h, pc))
            for h in pH:                              # protein donor → ligand acceptor
                best = max(best, _angle_at_h(pc, h, lc))
            if best >= angle_min:
                hbonds.append((pc, lc, d))

    if not hbonds:
        return {"node_feats": np.zeros((0, 9), np.float32),
                "mid": np.zeros((0, 3), np.float32), "n": 0}

    hbonds.sort(key=lambda t: t[2])                   # keep the closest if > max_nodes
    hbonds = hbonds[:max_nodes]
    prot_xyz = np.stack([h[0] for h in hbonds])
    lig_xyz = np.stack([h[1] for h in hbonds])
    mid = 0.5 * (prot_xyz + lig_xyz)
    node_feats = np.concatenate([prot_xyz, lig_xyz, mid], axis=1).astype(np.float32)
    return {"node_feats": node_feats, "mid": mid.astype(np.float32), "n": len(hbonds)}


def cache_path(pdb_id: str) -> Path:
    return HB_CACHE / f"{pdb_id}.npz"


def build_cache(manifest, *, shard: int = 0, nshards: int = 1, overwrite: bool = False) -> None:
    rows = manifest.iloc[shard::nshards]
    n_ok = n_empty = n_err = n_skip = 0
    for k, (_, r) in enumerate(rows.iterrows()):
        out = cache_path(r.pdb_id)
        if out.exists() and not overwrite:
            n_skip += 1
            continue
        try:
            g = extract_hbond_graph(r.protein_pdb, r.ligand_sdf)
            np.savez_compressed(out, node_feats=g["node_feats"], n=g["n"])
            n_ok += 1
            n_empty += int(g["n"] == 0)
        except Exception as e:                        # noqa: BLE001
            n_err += 1
            print(f"  ERR {r.pdb_id}: {type(e).__name__}: {e}", flush=True)
        if (k + 1) % 100 == 0:
            print(f"  shard {shard}/{nshards}  {k + 1}/{len(rows)}  "
                  f"ok={n_ok} empty={n_empty} err={n_err} skip={n_skip}", flush=True)
    print(f"[shard {shard}] DONE  ok={n_ok} empty={n_empty} err={n_err} skip={n_skip}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build per-complex H-bond graph cache (PyMOL)")
    from config import DEFAULT_SPLIT_CSV
    ap.add_argument("--split_csv", default=str(DEFAULT_SPLIT_CSV))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--cl1_only", action="store_true")
    ap.add_argument("--keep_covalent", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="only process first N complexes (test)")
    args = ap.parse_args()

    # require_smiles=False: the H-bond cache only needs structures, so cover every
    # split pid regardless of SMILES resolvability (keeps this env RDKit-free).
    m = build_manifest(split_csv=Path(args.split_csv), cl1_only=args.cl1_only,
                       drop_covalent=not args.keep_covalent, require_smiles=False,
                       verbose=(args.shard == 0))
    if args.smoke:
        m = m.head(args.smoke)
    build_cache(m, shard=args.shard, nshards=args.nshards, overwrite=args.overwrite)
