"""pose_eval.py — PoseCheck + PoseBusters structural-quality eval for generated ligands.

Runs in the dedicated ``moleval`` conda env (posecheck + posebusters + prolif +
hydride + reduce). It is invoked as a subprocess worker by
``notebook/webapp/metrics.py`` — that pipeline lives in the ``voxbind`` env, which
cannot host prolif/MDAnalysis without risking the training deps, so the two are
bridged process-to-process (same dual-env pattern as MISATO's H5PY worker).

Given a protein pocket PDB and an SDF of ligands, it computes per ligand:

  PoseCheck  (Harris et al., 2023 — the tool VoxBind's paper uses):
      * clashes : # steric clashes between the ligand and the (hydrogenated) pocket
      * strain  : UFF internal-energy strain, as defined by posecheck >= 1.3.1's
                  `calculate_strain_energy`:

                      E(pose after position-constrained UFF relaxation, maxDispl 0.1 Å)
                    − min E over {50 re-embedded + UFF-minimised conformers}
                                 ∪ {unconstrained UFF minimisation from the given pose}

                  i.e. locally-relaxed pose vs. an estimate of the global minimum —
                  NOT the raw pose, and NOT the nearest conformer.  Upstream changed
                  this in commit e021f3a (2024-01-18), *after* both the PoseCheck and
                  VoxBind papers were written: those report the older
                  E(raw pose) − E(one re-embedded relaxed conformer), which is ~an
                  order of magnitude larger (VoxBind Fig. 6 reference median 102.5).
                  Our numbers are internally consistent and use the current official
                  API, but absolute values must NOT be placed next to the paper's.
      * interactions : prolif protein–ligand interaction fingerprint, summarised to
                       per-type counts (HBAcceptor / HBDonor / Hydrophobic / ...)

  PoseBusters (Buttenschoen et al., 2023 — net-new validity layer):
      * dock-mode checks : bond lengths/angles, ring & double-bond flatness,
                           internal steric clash, internal energy, protein-ligand
                           minimum distance and volume overlap, ...
      * valid : True iff every check passes (the headline "PB-valid" flag)

Output is a JSON list, one entry per input-SDF record, in file order, written to
``--out`` (stdout is left for library/tqdm noise so it never corrupts the JSON):

    [{"posecheck": {...}, "posebusters": {...}}, ...]

Each of the two blocks is computed independently; a failure in one is recorded as
``{"error": "..."}`` in that block only, so the other still lands.

CLI:
    python pose_eval.py --protein pocket.pdb --ligands samples.sdf --out out.json \
        [--pose all|posecheck|posebusters]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# Per-molecule wall-clock cap for the strain-energy calc — UFF relaxation can
# spin for minutes on a pathological generated pose. Bounds each ligand so one
# bad mol records `strain: null` instead of stalling the whole target. 0 = off.
_STRAIN_TIMEOUT_S = float(os.environ.get("POSE_STRAIN_TIMEOUT_S", "60"))


class _Deadline:
    """SIGALRM-based wall-clock cap (main-thread only — the worker is single-threaded)."""

    def __init__(self, seconds: float):
        self.seconds = seconds

    def __enter__(self):
        if self.seconds and self.seconds > 0:
            signal.signal(signal.SIGALRM, self._fire)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        return False

    def _fire(self, *_):
        raise TimeoutError(f"exceeded {self.seconds:g}s")


# ── PoseCheck ─────────────────────────────────────────────────────────────────
def _summarise_interactions(df) -> dict:
    """Collapse a prolif fingerprint DataFrame (1 ligand row) to per-type counts.

    prolif returns a MultiIndex on the columns: (ligand, protein_residue,
    interaction_type). Each cell is a bool "this interaction is present". We sum
    the True cells grouped by the interaction-type level.
    """
    counts: dict[str, int] = {}
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        return {"n_interactions": 0, "interactions": {}}
    row = df.iloc[0]
    for col in df.columns:
        itype = col[-1] if isinstance(col, tuple) else str(col)
        try:
            present = bool(row[col])
        except Exception:  # noqa: BLE001
            present = False
        if present:
            counts[itype] = counts.get(itype, 0) + 1
    return {"n_interactions": int(sum(counts.values())), "interactions": counts}


def _seed_posecheck_strain(seed: int = 0) -> None:
    """Make PoseCheck's strain energy reproducible.

    Strain is (constrained-relaxed energy) minus (best global-relaxed conformer),
    and posecheck builds that global conformer with an UNSEEDED embedding —
    `posecheck/utils/strain.py` ships the seeded call commented out:

        #AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
        AllChem.EmbedMolecule(mol)

    So the same molecule measured twice gives different strain. Measured on our
    78-pocket reference ligands the median moved 33.8 -> 34.6 between runs, which
    is small but means no strain figure can be reproduced exactly.

    Rather than edit the installed package (which a rebuild of the env would undo),
    wrap EmbedMolecule in posecheck's own module namespace so every call it makes
    carries randomSeed=<seed>. Idempotent: re-wrapping is a no-op.
    """
    from posecheck.utils import strain as _pcstrain

    chem = _pcstrain.AllChem
    if getattr(chem.EmbedMolecule, "_voxbind_seeded", False):
        return

    _orig = chem.EmbedMolecule

    def _seeded(mol, *args, **kwargs):
        kwargs.setdefault("randomSeed", seed)
        return _orig(mol, *args, **kwargs)

    _seeded._voxbind_seeded = True          # noqa: SLF001
    chem.EmbedMolecule = _seeded


def _strain_one(pc, m) -> tuple[float | None, str | None]:
    """Strain energy for one ligand, bounded by a per-mol wall-clock deadline.

    Returns (strain, error). UFF relaxation can spin for minutes on a bad pose,
    so the SIGALRM deadline caps each ligand — a timed-out or failed one gets a
    null strain + reason instead of stalling the whole target.
    """
    try:
        with _Deadline(_STRAIN_TIMEOUT_S):
            pc.load_ligands_from_mols([m])
            s = pc.calculate_strain_energy()[0]
        # nan/inf (posecheck's own failure sentinel) -> None so it can't poison
        # the strain_median aggregate downstream.
        ok = isinstance(s, (int, float)) and math.isfinite(s)
        return (float(s) if ok else None, None)
    except Exception as e:  # noqa: BLE001 — TimeoutError or any UFF/RDKit failure
        return (None, f"{type(e).__name__}: {e}")


def _sanitise_receptor(pdb_path: str) -> str | None:
    """Rewrite a receptor PDB into a form prolif can load. Returns a temp path.

    Two features of stock CrossDocked receptors abort the load outright, and
    because the protein is loaded once per target, each one silently costs the
    WHOLE target (every ligand records the same protein-level error):

      * HETATM metals / cofactors — prolif has no van der Waals radius for
        element types like CO (cobalt) and raises
        `ValueError: vdw radii for types: CO`. PoseCheck's clash test is defined
        against the protein, so dropping HETATM is the conventional scope rather
        than a fudge; it does mean a clash with a bound metal is not counted.
      * Negative residue numbers — expression-tag residues numbered -4..-1 are
        cast to uint32 by prolif (residue.py), raising
        `OverflowError: Python integer -2 out of bounds for uint32`. Shifting
        every residue number by one constant preserves geometry and ordering;
        only the labels move.

    Returns None when nothing needed changing (so the caller does not retry a
    load that would fail identically), or when the shift would overflow the
    fixed-width resSeq column.
    """
    from pathlib import Path as _Path

    lines = _Path(pdb_path).read_text().splitlines()
    kept = [ln for ln in lines if not ln.startswith("HETATM")]
    dropped_het = len(kept) != len(lines)

    _RES = ("ATOM", "ANISOU", "TER")
    nums = []
    for ln in kept:
        if ln.startswith(_RES):
            try:
                nums.append(int(ln[22:26]))
            except ValueError:
                pass
    shift = (1 - min(nums)) if nums and min(nums) < 1 else 0

    if not dropped_het and shift == 0:
        return None
    if nums and shift and max(nums) + shift > 9999:
        return None

    out = []
    for ln in kept:
        if shift and ln.startswith(_RES):
            try:
                n = int(ln[22:26]) + shift
            except ValueError:
                out.append(ln)
                continue
            ln = f"{ln[:22]}{n:4d}{ln[26:]}"
        out.append(ln)

    fd, tmp = tempfile.mkstemp(suffix=".pdb")
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return tmp


def run_posecheck(protein_pdb: str, mols: list) -> list[dict]:
    """PoseCheck (clashes, strain, interaction summary) for every ligand.

    The protein is protonated + loaded once (posecheck lru-caches it). Clashes and
    the prolif interaction fingerprint are computed as one batch (the fingerprint
    build is the expensive setup, so doing it once beats per-ligand). Strain energy
    is done per-ligand under a wall-clock deadline so a single pathological pose
    can't stall the target. Batch results are identical to the per-ligand path.
    """
    from posecheck import PoseCheck

    _seed_posecheck_strain()
    pc = PoseCheck()
    # Protonate + load the pocket once. If this fails (hydride/reduce/pdb issue),
    # every ligand's posecheck block carries the same protein-level error.
    #
    # Two loader limitations kill whole targets on stock CrossDocked receptors, so
    # retry once on a sanitised copy (see _sanitise_receptor) before giving up.
    try:
        pc.load_protein_from_pdb(protein_pdb)
    except Exception as e:  # noqa: BLE001
        first = f"{type(e).__name__}: {e}"
        clean = _sanitise_receptor(protein_pdb)
        if clean is None:
            return [{"error": f"protein load failed: {first}"} for _ in mols]
        try:
            pc = PoseCheck()
            pc.load_protein_from_pdb(clean)
        except Exception as e2:  # noqa: BLE001
            return [{"error": f"protein load failed: {first}; "
                              f"after sanitise: {type(e2).__name__}: {e2}"}
                    for _ in mols]
        finally:
            try:
                os.remove(clean)
            except OSError:
                pass

    valid = [(i, m) for i, m in enumerate(mols) if m is not None]
    out: list[dict] = [{"error": "unparsable mol"} for _ in mols]
    if not valid:
        return out

    # ── batch: clashes + interactions (both read self.ligands) ───────────────
    pc.load_ligands_from_mols([m for _, m in valid])
    try:
        clashes = pc.calculate_clashes()
    except Exception:  # noqa: BLE001
        clashes = [None] * len(valid)
    try:
        ix = pc.calculate_interactions()  # one row per ligand, in load order
    except Exception:  # noqa: BLE001
        ix = None

    # ── per-ligand strain (deadline-bounded) + assemble records ──────────────
    for k, (i, m) in enumerate(valid):
        rec: dict = {}
        c = clashes[k] if k < len(clashes) else None
        rec["clashes"] = int(c) if isinstance(c, (int, float)) else None
        s, serr = _strain_one(pc, m)
        rec["strain"] = s
        if serr is not None:
            rec.setdefault("errors", {})["strain"] = serr
        try:
            rec.update(_summarise_interactions(
                ix.iloc[[k]] if ix is not None and k < ix.shape[0] else None))
        except Exception as e:  # noqa: BLE001
            rec.setdefault("errors", {})["interactions"] = f"{type(e).__name__}: {e}"
        out[i] = rec
    return out


# ── PoseBusters ───────────────────────────────────────────────────────────────
# Informational columns that report "loaded ok" rather than a geometric check —
# excluded from the pass/fail count so `valid` reflects real validity checks.
_PB_INFO_COLS = frozenset({"mol_pred_loaded", "mol_cond_loaded"})


def _pb_row_to_rec(row) -> dict:
    checks = {k: (None if v is None else bool(v)) for k, v in row.items()}
    graded = {k: v for k, v in checks.items()
              if k not in _PB_INFO_COLS and v is not None}
    n_checks = len(graded)
    n_pass = sum(1 for v in graded.values() if v)
    return {
        "valid": bool(n_checks > 0 and n_pass == n_checks),
        "n_pass": n_pass,
        "n_checks": n_checks,
        "checks": checks,
    }


def run_posebusters(protein_pdb: str, mols: list) -> list[dict]:
    """PoseBusters dock-mode validity for every ligand (one batch, per-mol fallback)."""
    from posebusters import PoseBusters

    buster = PoseBusters(config="dock")
    try:
        df = buster.bust(mols, None, protein_pdb, full_report=False)
        # bust() returns one row per input mol, in order.
        return [_pb_row_to_rec(df.iloc[i]) for i in range(len(mols))]
    except Exception:  # noqa: BLE001 — fall back to isolating each ligand
        out: list[dict] = []
        for m in mols:
            if m is None:
                out.append({"error": "unparsable mol"})
                continue
            try:
                df = PoseBusters(config="dock").bust([m], None, protein_pdb,
                                                      full_report=False)
                out.append(_pb_row_to_rec(df.iloc[0]))
            except Exception as e:  # noqa: BLE001
                out.append({"error": f"{type(e).__name__}: {e}"})
        return out


# ── driver ────────────────────────────────────────────────────────────────────
def evaluate(protein_pdb: str, ligands_sdf: str, pose: str = "all") -> list[dict]:
    """Return a list aligned to the SDF record order: one {posecheck, posebusters}."""
    # SDMolSupplier yields one entry per record — None for an unparsable one —
    # so the output stays index-aligned with the caller's row list; the per-mol
    # helpers emit an error record for any None rather than dropping it.
    mols = list(Chem.SDMolSupplier(ligands_sdf, sanitize=True))
    n = len(mols)

    pc_res = run_posecheck(protein_pdb, mols) if pose in ("all", "posecheck") \
        else [None] * n
    pb_res = run_posebusters(protein_pdb, mols) if pose in ("all", "posebusters") \
        else [None] * n

    out: list[dict] = []
    for i in range(n):
        rec: dict = {}
        if pc_res[i] is not None:
            rec["posecheck"] = pc_res[i]
        if pb_res[i] is not None:
            rec["posebusters"] = pb_res[i]
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protein", required=True, help="pocket PDB (conditioning)")
    ap.add_argument("--ligands", required=True, help="SDF of ligands to score")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--pose", choices=("all", "posecheck", "posebusters"),
                    default="all")
    args = ap.parse_args()

    results = evaluate(args.protein, args.ligands, pose=args.pose)
    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"pose_eval: wrote {len(results)} records -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
