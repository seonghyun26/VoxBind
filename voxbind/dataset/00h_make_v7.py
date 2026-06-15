"""00h_make_v7.py — Build the v7 combined pretraining dataset.

v7 = CrossDocked v6 (ligand-matched) crops  UNION  PDBbind-2020 train-split matched
complexes. In BOTH halves the experimental 2Fo-Fc density corresponds to the pocket
AND the modeled ligand (the receptor's own crystal ligand). For self-supervised
pretraining of the C+D+G encoder on a larger ligand-MATCHED density corpus.

Two halves, one unified CrossDocked-style dataset:

  CrossDocked v6 (5,270 train): the tt_min ligand-matched crops already in
      xray_crops_aligned_v6/ — reused verbatim (same v6 arcsinh+z norm).

  PDBbind-2020 train (new_split==train & EDS density, ~2,704): real holo crystals,
      inherently matched. Parsed into CrossDocked-schema (pocket, ligand) tuples via
      01b's parse_pocket_pdb / parse_ligand_sdf (identical element vocab + radii), and
      density RE-CROPPED from the PDBbind 2Fo-Fc map at the SUPPORTED-atom ligand
      centroid with the SAME v6 arcsinh+z normalization — so atoms↔density alignment
      and the value scale match CrossDocked exactly. Unsupported atoms (catalytic
      metals in pocket, Br/I in ligand) are masked (kept complex), exactly as the
      CrossDocked path masks them — no whole-complex drop.

Only the TRAIN split is combined (the pretraining corpus). The pretraining val is
carved from this pool at train time (dset.subset_val_n). PDBbind val/test are NEVER
included → no leakage into the affinity probe (which tests on PDBbind test).

Outputs (under --out_root, default dataset/data):
  data_train_v7.pt                 combined (pocket, ligand) tuples [pre-shuffle]
  xray_crops_aligned_v7/
    train/{idx:06d}.npy            float16 (64,64,64) density, v6 arcsinh+z, ligand-centroid
    train_available.npy            bool (N,)  ALL True (every v7 sample is matched)
    test/ , test_available.npy     copied verbatim from xray_crops_aligned_v6 (CrossDocked test)
    stats.json

Crop positions mirror DatasetCrossDockedXray's load order EXACTLY
(shuffle seed 1234 → drop last 100 → filter_by_size(max_len<=30)), so the dataset
reads data_train_v7.pt (via dset.data_file) + these crops with no index drift.

Usage
-----
    cd voxbind
    python dataset/00h_make_v7.py                 # build train (+copy v6 test)
    python dataset/00h_make_v7.py --limit_pdb 50  # debug: only 50 PDBbind complexes
"""

import gemmi  # noqa: F401  — before torch (shared-lib load order; see memory)

import argparse
import csv
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

VOX_ROOT = Path(__file__).resolve().parents[1]          # .../voxbind
sys.path.insert(0, str(VOX_ROOT.parent))                # repo root (has voxbind/ package)
from voxbind.dataset.crossdocked_xray import _crop_density  # noqa: E402

GRID_DIM = 64
RESOLUTION = 0.25
VAL_SZ = 100          # DatasetCrossDockedXray hardcoded train/val tail split
MAX_LEN = 30          # DatasetCrossDockedXray._filter_by_size default


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 01b provides the PDBbind parsers + raw-grid loader (identical element vocab/radii).
PDBB = load_module(Path(__file__).parent / "01b_pdbbind_preprocess.py", "pdbb01b")


def replicate_crossdocked_train_order(data_dir: Path):
    """The exact (pocket, ligand) list DatasetCrossDockedXray uses for the train
    split BEFORE subset selection: shuffle(1234) → drop last 100 → filter max_len.
    Mirrors 00b_density_preprocess.load_split, so positions align with v6 crops."""
    data = torch.load(str(data_dir / "data_train.pt"), weights_only=False)
    random.Random(1234).shuffle(data)
    train = data[: len(data) - VAL_SZ]
    return [(p, l) for p, l in train if l["max_len"] <= MAX_LEN]


def crossdocked_v6_pairs(data_dir: Path, v6_dir: Path):
    """[(tuple, ('cd', j)), ...] for every v6-available CrossDocked train sample,
    where j indexes xray_crops_aligned_v6/train/{j:06d}.npy."""
    order = replicate_crossdocked_train_order(data_dir)
    avail = np.load(str(v6_dir / "train_available.npy"))
    if len(avail) != len(order):
        raise RuntimeError(
            f"v6 train_available length {len(avail)} != replicated CrossDocked order "
            f"{len(order)} — preprocessing drift. Re-check seed/filter."
        )
    return [(order[j], ("cd", j)) for j in range(len(order)) if bool(avail[j])]


def pdbbind_train_eds_pids(data_dir: Path):
    """Sorted lowercase pids: new_split==train AND EDS density available."""
    pdb_dir = data_dir / "pdbbind"
    eds = json.loads((pdb_dir / "eds_cache.json").read_text())
    eds_ok = {k.lower() for k, v in eds.items() if v.get("has_density")}
    pids = []
    with (pdb_dir / "index.csv").open() as f:
        for row in csv.DictReader(f):
            pid = row["pdb_id"].lower()
            if row["new_split"] == "train" and pid in eds_ok:
                pids.append(pid)
    return sorted(set(pids))


def build_pdbbind_pairs(data_dir: Path, stage_dir: Path, norm, limit: int = 0):
    """Parse each PDBbind train+EDS complex → CrossDocked-schema tuple, re-crop its
    density to stage_dir/{pid}.npy. Returns ([(tuple, ('pdb', pid)), ...], skipped).

    norm(raw_crop)->float16 applies the v6 arcsinh+z normalization.
    """
    pdb_dir = data_dir / "pdbbind"
    struct_dir = PDBB.STRUCT_DIR
    ccp4_dir = pdb_dir / "ccp4"
    stage_dir.mkdir(parents=True, exist_ok=True)

    pids = pdbbind_train_eds_pids(data_dir)
    if limit:
        pids = pids[:limit]

    pairs, skipped = [], {}
    for pid in tqdm(pids, desc="pdbbind", unit="cplx"):
        try:
            sdf = struct_dir / pid / f"{pid}_ligand.sdf"
            pdb = struct_dir / pid / f"{pid}_pocket.pdb"
            ccp4 = ccp4_dir / f"{pid}.ccp4"
            if not (sdf.exists() and pdb.exists() and ccp4.exists()):
                skipped[pid] = "missing_files"; continue

            lig_xyz, lig_ch, _all_heavy_centroid, _lig_unsup = PDBB.parse_ligand_sdf(sdf)
            poc_xyz, poc_ch, _poc_unsup = PDBB.parse_pocket_pdb(pdb)
            if lig_xyz.numel() == 0 or poc_xyz.numel() == 0:
                skipped[pid] = "empty_supported_atoms"; continue

            # Density crop center = SUPPORTED-atom ligand centroid (matches the
            # dataset's __getitem__ recentering, which masks atoms_channel<7).
            center = lig_xyz.mean(dim=0).numpy().astype(np.float64)

            grid = PDBB.load_raw_grid(ccp4)
            if grid is None:
                skipped[pid] = "bad_map"; continue
            arr, frac_T, nu, nv, nw = grid
            crop = _crop_density(arr, frac_T, nu, nv, nw, center,
                                 G=GRID_DIM, res=RESOLUTION, transform=None)
            if crop is None:
                skipped[pid] = "crop_failed"; continue
            np.save(str(stage_dir / f"{pid}.npy"), norm(crop))

            c = lig_xyz
            max_len = round(float((c.max(0).values - c.min(0).values).max()), 2)
            pocket_ = {"id": f"pdbbind/{pid}",
                       "coords": poc_xyz.to(torch.float32),
                       "atoms_channel": poc_ch.to(torch.uint8)}
            ligand_ = {"id": f"pdbbind/{pid}",
                       "coords": lig_xyz.to(torch.float32),
                       "atoms_channel": lig_ch.to(torch.uint8),
                       "max_len": max_len}
            pairs.append(((pocket_, ligand_), ("pdb", pid)))
        except Exception as e:  # never let one complex kill the build
            skipped[pid] = f"error:{type(e).__name__}"
    return pairs, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data_dir", default=str(VOX_ROOT / "dataset" / "data"))
    ap.add_argument("--out_root", default=str(VOX_ROOT / "dataset" / "data"))
    ap.add_argument("--limit_pdb", type=int, default=0, help="debug: cap PDBbind complexes")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    v6_dir = out_root / "xray_crops_aligned_v6"
    v7_dir = out_root / "xray_crops_aligned_v7"
    stage_dir = v7_dir / ".pdb_stage"

    # v6 normalization (reused for the PDBbind half so both are one value scale).
    v6_stats = json.loads((v6_dir / "stats.json").read_text())
    s5 = float(v6_stats["arcsinh_scale"]); mu_a = float(v6_stats["mu_a"]); sigma_a = float(v6_stats["sigma_a"])

    def norm(raw):
        return ((np.arcsinh(raw.astype(np.float64) / s5) - mu_a) / sigma_a).astype(np.float16)

    print("=== build v7 (CrossDocked v6 ∪ PDBbind-train, ligand-matched) ===")
    print(f"  data_dir : {data_dir}")
    print(f"  out      : {v7_dir}")
    print(f"  v6 norm  : arcsinh(x/{s5}) z-score  μ_a {mu_a:+.6f}  σ_a {sigma_a:.6f}")

    # ── 1. CrossDocked v6 pairs ──────────────────────────────────────────────────
    cd_pairs = crossdocked_v6_pairs(data_dir, v6_dir)
    print(f"  CrossDocked v6 train pairs : {len(cd_pairs):,}")

    # ── 2. PDBbind train+EDS pairs (re-cropped density → stage) ──────────────────
    pdb_pairs, skipped = build_pdbbind_pairs(data_dir, stage_dir, norm, limit=args.limit_pdb)
    print(f"  PDBbind train pairs        : {len(pdb_pairs):,}  (skipped {len(skipped)})")
    if skipped:
        from collections import Counter
        for reason, n in Counter(skipped.values()).most_common():
            print(f"     skip {reason:24s} {n}")

    # ── 3. Combined (pre-shuffle) data file ──────────────────────────────────────
    combined = cd_pairs + pdb_pairs
    tuples = [t for t, _ in combined]
    torch.save(tuples, str(data_dir / "data_train_v7.pt"))
    print(f"  data_train_v7.pt           : {len(tuples):,} tuples")

    # ── 4. Replicate the dataset's train order, lay crops at those positions ─────
    # shuffle(1234) → drop last 100 → filter max_len  (mirrors DatasetCrossDockedXray)
    order = list(combined)
    random.Random(1234).shuffle(order)
    order = order[: len(order) - VAL_SZ]
    order = [(t, src) for t, src in order if t[1]["max_len"] <= MAX_LEN]
    n = len(order)
    print(f"  v7 train positions (post shuffle/split/filter): {n:,}")

    train_out = v7_dir / "train"
    if train_out.exists():
        shutil.rmtree(train_out)
    train_out.mkdir(parents=True, exist_ok=True)

    for i, (_t, src) in enumerate(tqdm(order, desc="write crops", unit="file")):
        dst = train_out / f"{i:06d}.npy"
        if src[0] == "cd":
            shutil.copyfile(v6_dir / "train" / f"{src[1]:06d}.npy", dst)
        else:  # ('pdb', pid)
            shutil.copyfile(stage_dir / f"{src[1]}.npy", dst)
    np.save(str(v7_dir / "train_available.npy"), np.ones(n, dtype=bool))

    # ── 5. Test split = CrossDocked v6 test, copied verbatim ─────────────────────
    for sub in ("test",):
        src_dir = v6_dir / sub
        if src_dir.exists():
            dst_dir = v7_dir / sub
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            shutil.copyfile(v6_dir / f"{sub}_available.npy", v7_dir / f"{sub}_available.npy")

    # ── 6. stats.json + cleanup ──────────────────────────────────────────────────
    (v7_dir / "stats.json").write_text(json.dumps({
        "scheme": "v7 combined: CrossDocked v6 ∪ PDBbind-2020-train, ligand-matched (arcsinh+z, v6 stats)",
        "formula": "x' = (arcsinh(x / s) - mu_a) / sigma_a",
        "arcsinh_scale": s5, "mu_a": mu_a, "sigma_a": sigma_a,
        "n_train": n, "n_crossdocked_v6": len(cd_pairs), "n_pdbbind_train": len(pdb_pairs),
        "pdbbind_skipped": len(skipped),
        "note": "Density corresponds to BOTH pocket and ligand. PDBbind density re-cropped from "
                "2Fo-Fc maps at the supported-atom ligand centroid; unsupported atoms masked "
                "(complex kept), matching the CrossDocked path. Train-split PDBbind only (no probe leakage). "
                "Train data file: data_train_v7.pt (load via dset.data_file).",
    }, indent=2))
    shutil.rmtree(stage_dir, ignore_errors=True)

    print(f"\nDone. v7 train crops: {n:,}  →  {v7_dir}")
    print(f"  pretrain subset sizing: dset.subset_n={n - VAL_SZ} dset.subset_val_n={VAL_SZ}")


if __name__ == "__main__":
    main()
