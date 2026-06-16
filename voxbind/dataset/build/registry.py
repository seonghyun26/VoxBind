"""registry.py — per-dataset config + legacy-module dispatch for the unified build pipeline.

The single place the 5 stage files (s1..s5) import. Holds:
  - the VoxBind atom dictionary + vdW radii (from voxbind/constants.py),
  - a DatasetSpec per dataset (frame/align, density match gate, split source,
    stage-4 voxelize mode, target kind, atom-blob radii),
  - run_stage(): dispatch a stage for a dataset to its ordered build steps.

PHASE-1 strategy (faithful + low-risk): branches DELEGATE to the validated
dataset/NN_*.py scripts (subprocess, same code path ⇒ automatic parity). CrossDocked
stays delegated permanently (user: leave it). pdbbind/misato/plinder logic will be
inlined into the branches once parity is locked, then the NN_ scripts retire to legacy/.
See dataset/build/README.md.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATASET_DIR = Path(__file__).resolve().parent.parent        # .../voxbind/dataset
VOX_ROOT = DATASET_DIR.parent                               # .../voxbind

# ── VoxBind atom dictionary (single source of truth) ───────────────────────────
if str(VOX_ROOT) not in sys.path:
    sys.path.insert(0, str(VOX_ROOT))
from constants import ELEMENTS_HASH_CROSSDOCKED, RADIUS_PER_ATOM  # noqa: E402

LIGAND_ELEMENTS = ["C", "O", "N", "S", "F", "Cl", "P"]      # atom-voxel channels 0-6
POCKET_ELEMENTS = ["C", "O", "N", "S"]                      # atom-voxel channels 7-10
VDW_RADII = RADIUS_PER_ATOM["MOL"]                          # element -> vdW radius (Å)

GRID_DIM, RESOLUTION = 64, 0.25

# Atom-blob radius modes (matches crossdocked_xray.py / 01b convention):
#   a positive scalar = ONE uniform blob radius (Å) for every atom of that group;
#   VDW (-1)          = per-element van der Waals radius via VDW_RADII LUT.
VDW = -1.0
UNIFORM = 0.5

DATASETS = ("crossdocked", "pdbbind", "misato", "plinder")

# MISATO's 02b/02c read HDF5 and need an h5py interpreter (NOT the voxbind env).
# Override with MISATO_H5PY_PY=/path/to/python.
H5PY_PY = os.environ.get("MISATO_H5PY_PY", "/home/shpark/.conda/envs/cellmaes/bin/python")


def _script_path(script: str) -> Path:
    """Resolve a delegated NN_ script: prefer dataset/legacy/ (retired pdbbind/misato/plinder
    originals), else dataset/ (CrossDocked's 00*/preprocess stay put)."""
    legacy = DATASET_DIR / "legacy" / script
    return legacy if legacy.exists() else (DATASET_DIR / script)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    frame: str                  # "kabsch" (CrossDocked) | "deposited" (no align)
    match_gate: str             # density↔ligand match criterion (human-readable)
    split_source: str
    voxelize_mode: str          # ATOMS: "on_the_fly" (splat on-GPU at train) | "precompute"
    target: Optional[str]       # "pK" | "qm_md" | None
    ligand_radius: float        # >0 uniform Å, VDW(-1) = per-element vdW LUT
    pocket_radius: float        # >0 uniform Å, VDW(-1) = per-element vdW LUT
    # DENSITY (orthogonal to voxelize_mode, which is about atoms):
    #   "precomputed_crop"  s3 writes frozen 64³ crops; train-time aug shifts them (zero-fill).
    #   "resample_map"      s3 writes only a manifest + recipe and KEEPS the maps; density is
    #                       cropped from the full map at the augmented pose at train time.
    # Only datasets whose s3 delegates to a resample-capable script (00b) can use resample_map.
    density_mode: str = "precomputed_crop"
    # stage -> ordered build steps. Each step is a tuple:
    #   ("run",  script, subcmd|None)   subprocess: python dataset/<script> [subcmd] <passthrough>
    #   ("note", message)               informational, no-op
    #   ("todo", message)               not wired yet -> NotImplementedError (printed under --dry_run)
    stages: dict = field(default_factory=dict)


SPEC = {
    "crossdocked": DatasetSpec(
        name="crossdocked", frame="kabsch",
        match_gate="native_filter tt_min (ligand PDB == receptor PDB) + density@atoms threshold",
        split_source="val_from_train_tail", voxelize_mode="on_the_fly", target=None,
        ligand_radius=UNIFORM, pocket_radius=VDW,
        density_mode="resample_map",   # 00b supports --mode resample (manifest + keep maps)
        stages={
            "s1": [("run", "preprocess_crossdocked.py", None)],
            "s2": [("run", "00a_density_download.py", None)],
            "s3": [("run", "00b_density_preprocess.py", None)],  # align(Kabsch)+crop+normalize; --version vN passthrough
                                                                 # resample: s3_crop --resample → 00b --mode resample
            "s4": [("note", "CrossDocked atoms voxelize on-GPU at train time — no precompute.")],
            "s5": [("note", "CrossDocked is self-supervised — no targets.")],
        },
    ),
    "pdbbind": DatasetSpec(
        name="pdbbind", frame="deposited",
        match_gate="inherent holo crystal (PDBe-EDS coverage)",
        split_source="pdbbind new_split — 2172/480/839 (cl1_only=False, !covalent, has_pK, has_density)",
        voxelize_mode="precompute", target="pK",
        ligand_radius=VDW, pocket_radius=VDW,   # canonical probe set = ligvdw (pass --ligand_vdw to s4)
        stages={
            "s1": [("run", "01a_pdbbind_acquire.py", "structures"),
                   ("run", "01a_pdbbind_acquire.py", "index")],
            "s2": [("run", "01a_pdbbind_acquire.py", "density")],
            "s3": [("note", "PDBbind is deposited-frame (no Kabsch); crop is FUSED into the precompute — see s4.")],
            # Reproduce the canonical PROBE voxel sets (element_filter=all throughout; probe is aug-OFF
            # so precompute is correct here). 3 steps; pass only flags common to all (e.g. --max_complexes).
            "s4": [
                # 1. uniform-ligand atoms (0.5 Å) + v1 density crop  → voxels/
                ("run", "01b_pdbbind_preprocess.py", "voxelize", {"args": ["--element_filter", "all"]}),
                # 2. vdW-ligand atoms (the canonical probe atoms, read via --atom_source ligvdw)  → voxels_ligvdw/
                ("run", "01b_pdbbind_preprocess.py", "voxelize",
                 {"args": ["--ligand_vdw", "--element_filter", "all", "--no_density",
                           "--out_dir", "dataset/data/pdbbind/voxels_ligvdw"]}),
                # 3. pocket-pool v2..v5 density; v5 = canonical scale fit to the CrossDocked v5 ref stats  → voxels_v5/
                ("run", "01b_pdbbind_preprocess.py", "poolnorm",
                 {"args": ["--element_filter", "all", "--v5_stats_source", "reference"]}),
            ],
            "s5": [("note", "pK target lives in pdbbind/index.csv (read directly by 01c probe). "
                            "Run dataset/extract_bfactors.py for the B-factor target.")],
        },
    ),
    "misato": DatasetSpec(
        name="misato", frame="deposited",
        match_gate="QM/MD coverage + built structures",
        split_source="misato 8:1:1 (--split misato)", voxelize_mode="precompute", target="qm_md",
        ligand_radius=VDW, pocket_radius=VDW,
        stages={
            # MISATO spans two envs: 02a/02d run under voxbind; 02b reads HDF5 → needs H5PY_PY.
            "s1": [("run", "02a_misato_acquire.py", "hdf5", {"args": ["--which", "both"]})],  # QM.hdf5 + MD.hdf5
            "s2": [("run", "02a_misato_acquire.py", "pdbe")],   # pass --ids <list> --what both (PDB + EDS maps)
            "s3": [("note", "deposited-frame (no Kabsch); crop fused into s4. PDBbind-missing complexes need "
                            "structures built first: 02c_misato_structures.py build --ids <list>  (run under H5PY_PY).")],
            "s4": [("run", "02d_misato_voxelize.py", None)],    # chunked 01b voxelize+poolnorm + feature extract (voxbind)
            "s5": [("run", "02b_misato_targets.py", "qm", {"py": H5PY_PY}),
                   ("run", "02b_misato_targets.py", "md", {"py": H5PY_PY})],
        },
    ),
    "plinder": DatasetSpec(
        name="plinder", frame="deposited",
        match_gate="RSCC >= 0.8 + 0 unresolved heavy atoms",
        split_source="val_from_train_tail", voxelize_mode="on_the_fly", target=None,
        ligand_radius=VDW, pocket_radius=VDW,
        density_mode="resample_map",   # 03c --mode resample (deposited frame, R=I); maps re-acquired via s2
        stages={
            "s1": [("run", "03a_plinder_select.py", None)],                                   # funnel + RSCC gate + dedup + leakage
            "s2": [("run", "03b_plinder_acquire.py", None, {"args": ["--what", "both"]})],     # cif + ccp4
            "s3": [("run", "03c_plinder_preprocess.py", None)],                                # parse label_asym_id, crop, normalize, tuples
            "s4": [("note", "PLINDER atoms voxelize on-GPU at train time — no precompute.")],
            "s5": [("note", "PLINDER is self-supervised — no targets.")],
        },
    ),
}


def load_legacy(filename: str):
    """Import a numbered dataset/NN_*.py module by path (names start with a digit → not importable normally).

    Mirrors the importlib trick already used by 03b/03c. Returns the module so callers
    can invoke its functions (run_structures, run_voxelize, ...) directly if needed.
    """
    path = _script_path(filename)
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_").lstrip("0123456789_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_stage(stage: str, dataset: str, passthrough: list, dry_run: bool = False) -> None:
    """Dispatch one pipeline stage for one dataset to its ordered build steps.

    `passthrough` (unrecognized CLI args) is forwarded to EVERY 'run' step of the stage, so for a
    multi-step stage use only flags common to all steps (e.g. --max_complexes). Step-specific flags
    (e.g. --ligand_vdw) are injected from the spec during inlining, not passed through here.
    """
    if dataset not in SPEC:
        raise SystemExit(f"unknown --dataset {dataset!r}; choose from {DATASETS}")
    spec = SPEC[dataset]
    print(f"[{stage}] dataset={dataset}  frame={spec.frame}  voxelize_mode={spec.voxelize_mode}"
          f"  density_mode={spec.density_mode}  radii(lig={spec.ligand_radius},poc={spec.pocket_radius})")
    steps = spec.stages.get(stage, [])
    if not steps:
        print(f"  (no steps defined for {dataset} at {stage})")
        return
    for kind, *rest in steps:
        if kind == "note":
            print(f"  • {rest[0]}")
        elif kind == "todo":
            msg = f"{stage}/{dataset} not wired yet — {rest[0]}."
            if dry_run:
                print(f"  [todo] {msg}")
            else:
                raise NotImplementedError(
                    msg + "  (Phase 'extend branches'; run the legacy script directly meanwhile.)")
        elif kind == "run":
            script, subcmd = rest[0], rest[1]
            opts = rest[2] if len(rest) > 2 else {}
            fixed = opts.get("args", [])
            py = opts.get("py", sys.executable)
            sub = [subcmd] if subcmd else []
            argv_tail = [*sub, *fixed, *passthrough]
            pyshow = "python" if py == sys.executable else py
            spath = _script_path(script)                       # dataset/legacy/ (retired) or dataset/ (crossdocked)
            shown = " ".join([pyshow, str(spath.relative_to(VOX_ROOT)), *argv_tail])
            # gemmi users (00b/01b/03c/02d) need the interpreter's own lib on LD_LIBRARY_PATH
            # (CXXABI), per memory; set it per-interpreter so it's correct for H5PY_PY too.
            libdir = str(Path(py).resolve().parent.parent / "lib")
            env = {**os.environ, "LD_LIBRARY_PATH": libdir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")}
            if dry_run:
                print(f"  $ {shown}   (dry-run)")
            else:
                print(f"  $ {shown}")
                subprocess.run([py, str(spath), *argv_tail],
                               cwd=str(VOX_ROOT), check=True, env=env)
        else:
            raise ValueError(f"bad step kind {kind!r}")
