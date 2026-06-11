"""extract_bfactors.py — per-complex pocket B-factor targets for the PDBbind probe.

B-factors are an *atomic flexibility* signal: high B ↔ diffuse electron density ↔
low ‖∇ρ‖. They are not a separate dataset — they already live in the PDBbind
structure files (`structures/pbpp-2020/<id>/<id>_{pocket,protein}.pdb`, the
temperature-factor column). This script parses them into a small JSON map that
`01c_pdbbind_probe.py --target bfactor[_rel]` regresses on the *identical*
LP_PDBBind splits used for affinity, exactly like the H-bond-count target.

Per complex we record (heavy atoms only — added H have B=0; waters dropped):
    mean_pocket_b   mean B over `<id>_pocket.pdb`   (the binding-site residues)
    mean_protein_b  mean B over `<id>_protein.pdb`  (whole chain)
    rel_pocket_b    mean_pocket_b / mean_protein_b

Why two targets:
  • `mean_pocket_b` (raw)  — absolute flexibility. CAVEAT: raw B is not comparable
    across structures; it scales with resolution and the refinement's overall
    B-scaling, so a probe on raw pocket-mean B partly predicts *resolution*. The
    density channel encodes sharpness/resolution directly, so density is *expected*
    to help here — a true but partly-confounded positive.
  • `rel_pocket_b` (ratio) — pocket flexibility *relative to its own protein*. The
    per-structure scale cancels, so it is resolution-robust and reads as "is this
    site more rigid (<1) or more mobile (>1) than the protein average" — the cleaner
    target for the density-vs-coords ablation.

Granularity note: this is a per-complex SCALAR (drops into the existing frozen
feature → MLP probe with no architecture change). It is a *weak* test — mean-pooling
washes out the local flexibility that ‖∇ρ‖ should capture; a per-atom readout head
is the stronger follow-up if the scalar shows signal.

Alignment note: `pocket.pdb` is PDBbind's ~10 Å binding-site shell, a close but not
exact match to the encoder's cube crop around the ligand centroid. Fine as a v1
target; tighten to the crop atoms only if the scalar warrants the per-atom version.

Pure gemmi + stdlib — no torch import (sidesteps the gemmi/torch import-order trap).

    cd voxbind
    python dataset/extract_bfactors.py              # all complexes → data/pdbbind/bfactors.json
    python dataset/extract_bfactors.py --limit 20   # smoke test
"""

import argparse
import json
from pathlib import Path

import gemmi
from tqdm import tqdm

PDBBIND_DIR = Path(__file__).parent / "data" / "pdbbind"
STRUCT_DIR  = PDBBIND_DIR / "structures" / "pbpp-2020"
OUT_PATH    = PDBBIND_DIR / "bfactors.json"

WATER_NAMES = {"HOH", "WAT", "DOD", "H2O"}


def mean_heavy_bfactor(pdb_path: Path) -> tuple[float, int]:
    """Mean isotropic B over heavy, non-water, primary-conformer atoms.

    Returns (mean_b, n_atoms). n_atoms == 0 (mean NaN) when the file is missing,
    unreadable, or has no qualifying atoms — the caller drops those complexes.
    """
    if not pdb_path.exists():
        return float("nan"), 0
    try:
        st = gemmi.read_structure(str(pdb_path), format=gemmi.CoorFormat.Pdb)
    except Exception:
        return float("nan"), 0
    if len(st) == 0:
        return float("nan"), 0

    total, n = 0.0, 0
    for chain in st[0]:
        for residue in chain:
            if residue.name in WATER_NAMES:
                continue
            for atom in residue:
                if atom.element.atomic_number == 1:      # skip H / D (B=0 placeholder)
                    continue
                if atom.occ <= 0.0:                      # skip zero-occupancy atoms
                    continue
                # gemmi encodes "no altloc" as '\x00' (not ''); keep that + conformer A only.
                if atom.altloc not in ("", "A", "\x00"):
                    continue
                total += atom.b_iso
                n += 1
    if n == 0:
        return float("nan"), 0
    return total / n, n


def isfinite_pos(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf")) and x > 0.0


def extract_one(pdb_id: str, struct_dir: Path) -> dict:
    cdir = struct_dir / pdb_id
    pocket_b, n_pocket = mean_heavy_bfactor(cdir / f"{pdb_id}_pocket.pdb")
    protein_b, n_protein = mean_heavy_bfactor(cdir / f"{pdb_id}_protein.pdb")
    rel = pocket_b / protein_b if (isfinite_pos(pocket_b) and isfinite_pos(protein_b)) else float("nan")
    return {
        "mean_pocket_b":  pocket_b,
        "mean_protein_b": protein_b,
        "rel_pocket_b":   rel,
        "n_pocket_heavy":  n_pocket,
        "n_protein_heavy": n_protein,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--struct_dir", type=Path, default=STRUCT_DIR,
                    help="dir of per-complex <id>/<id>_{pocket,protein}.pdb")
    ap.add_argument("--out", type=Path, default=OUT_PATH,
                    help="output JSON map (pdb_id -> bfactor record)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N complexes (smoke test)")
    args = ap.parse_args()

    pdb_ids = sorted(p.name for p in args.struct_dir.iterdir() if p.is_dir())
    if args.limit:
        pdb_ids = pdb_ids[: args.limit]
    print(f"=== PDBbind pocket B-factor extraction ===")
    print(f"  struct_dir : {args.struct_dir}")
    print(f"  complexes  : {len(pdb_ids):,}")
    print(f"  out        : {args.out}")

    out: dict[str, dict] = {}
    n_pocket_ok = n_rel_ok = 0
    for pid in tqdm(pdb_ids, unit="cplx", desc="bfactors"):
        rec = extract_one(pid, args.struct_dir)
        out[pid.lower()] = rec
        if rec["n_pocket_heavy"] > 0:
            n_pocket_ok += 1
        if isfinite_pos(rec["rel_pocket_b"]):
            n_rel_ok += 1

    # NaN is not valid JSON per spec but json.dump emits it and the stdlib loader
    # round-trips it; the probe loader filters non-finite values, so missing
    # complexes are simply absent from the regression pool.
    args.out.write_text(json.dumps(out, indent=0))
    print(f"\n  wrote {len(out):,} records → {args.out}  ({args.out.stat().st_size/1e3:.0f} KB)")
    print(f"  usable mean_pocket_b : {n_pocket_ok:,}")
    print(f"  usable rel_pocket_b  : {n_rel_ok:,}")

    # Quick distribution sanity print over usable values.
    pk = [r["mean_pocket_b"] for r in out.values() if r["n_pocket_heavy"] > 0]
    rl = [r["rel_pocket_b"] for r in out.values() if isfinite_pos(r["rel_pocket_b"])]
    if pk:
        pk_sorted = sorted(pk)
        med = pk_sorted[len(pk_sorted) // 2]
        print(f"  mean_pocket_b : min={min(pk):.1f} median={med:.1f} max={max(pk):.1f}")
    if rl:
        rl_sorted = sorted(rl)
        med = rl_sorted[len(rl_sorted) // 2]
        print(f"  rel_pocket_b  : min={min(rl):.3f} median={med:.3f} max={max(rl):.3f}")


if __name__ == "__main__":
    main()
