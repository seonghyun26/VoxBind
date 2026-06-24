"""03c_plinder_preprocess.py — Build the standalone PLINDER density pretraining set.

Turns the 03a selection + 03b downloads into a CrossDocked-schema pretraining
corpus, reusing the exact PDBbind/v7 machinery so atoms↔density alignment and the
value scale match v6/v7 (one canonical density scale across all pretraining sets):

    for each selected (entry_pdb_id, ligand_asym_id):
      1. parse the deposited mmCIF (03b) — locate the ligand by label_asym_id
         (+CCD), take its supported heavy atoms (C/O/N/S/F/Cl/P) and the protein
         pocket heavy atoms within --pocket_radius A. Unsupported atoms (metals,
         Br/I) are masked, complex kept — exactly as the CrossDocked/PDBbind paths.
      2. crop the 2Fo-Fc map (03b) at the SUPPORTED-atom ligand centroid with
         transform=None (Strategy A: mmCIF + map share the deposited crystal frame,
         so no Kabsch — cf. 00h's PDBbind half), via crossdocked_xray._crop_density.
      3. normalize with the v6 arcsinh+z stats (reused from xray_crops_aligned_v6).

Crop laydown mirrors DatasetCrossDockedXray EXACTLY (shuffle(1234) → drop last 100
→ filter max_len<=30), identical to 00h, so the dataset reads data_train_plinder.pt
(via dset.data_file) + these crops with no index drift.

Outputs (under --out_root, default dataset/data):
    data_train_plinder.pt                  combined (pocket, ligand) tuples [pre-shuffle]
    xray_crops_aligned_plinder/
      train/{idx:06d}.npy                  float16 (64,64,64) density, v6 arcsinh+z, ligand-centroid
      train_available.npy                  bool (N,)  ALL True (every PLINDER sample is matched)
      stats.json
(No test split: this is a pretraining corpus; val is carved from train at train
time via dset.subset_val_n. The dataset's test path always uses data_test.pt.)

Usage
-----
    cd voxbind
    python dataset/03c_plinder_preprocess.py --limit 1500   # smoke pilot
    python dataset/03c_plinder_preprocess.py                # full selection (~17.7k)
    python dataset/03c_plinder_preprocess.py --pocket_radius 8
"""

import gemmi  # noqa: F401  — before torch (shared-lib load order; see memory)

import argparse
import importlib.util
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

VOX_ROOT = Path(__file__).resolve().parents[2]              # .../voxbind
VOXDATA = VOX_ROOT / "dataset" / "data"
PLINDER = VOXDATA / "plinder"
sys.path.insert(0, str(VOX_ROOT.parent))                   # repo root (has voxbind/ package)
from voxbind.dataset.crossdocked_xray import _crop_density  # noqa: E402

GRID_DIM = 64
RESOLUTION = 0.25
VAL_SZ = 100          # DatasetCrossDockedXray hardcoded train/val tail split
MAX_LEN = 30          # DatasetCrossDockedXray._filter_by_size default

# Standard protein residues (fallback when gemmi's residue table lacks an entry).
_AA3 = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR "
           "TRP TYR VAL MSE SEC PYL".split())


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 01b provides the element vocab/channel helpers + the raw-grid loader (identical
# to the PDBbind path). We reuse them so PLINDER atoms hash to the same channels.
PDBB = load_module(Path(__file__).parent / "01b_pdbbind_preprocess.py", "pdbb01b")

# Phase-2 diverse (role-split) build: per-atom vdW radius for ANY heavy element +
# the noble-gas/dummy artifact exclusion, so the single role channel can ingest
# diverse atoms (identity encoded by blob SIZE, not a one-hot element vocab).
from voxbind.constants import is_diverse_role_atom, vdw_radius

# 00a provides the resumable thread-pool downloader (reused by --stream mode so we
# never hold all raw maps at once: download a batch → crop → delete the .ccp4).
DL = load_module(Path(__file__).parent.parent / "00a_density_download.py", "dl00a_for03c")
MAP_URL = "https://www.ebi.ac.uk/pdbe/entry-files/{pid}.ccp4"
CIF_URL = "https://www.ebi.ac.uk/pdbe/entry-files/download/{pid}_updated.cif"


def _free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists():            # path may not be created yet
        p = p.parent
    return shutil.disk_usage(str(p)).free / 1e9


def _is_protein_res(res) -> bool:
    info = gemmi.find_tabulated_residue(res.name)
    if info is not None:
        return info.is_amino_acid()
    return res.name.strip().upper() in _AA3


def parse_cif_system(cif_path: Path, ligand_asym_id: str, ccd_codes: str,
                     pocket_radius: float = 10.0, diverse: bool = False):
    """Locate PLINDER's ligand instance in the deposited mmCIF + its pocket.

    Returns (lig_xyz, lig_ch, poc_xyz, poc_ch, lig_unsup, poc_unsup, lig_rad,
    poc_rad) in the deposited frame, or None if the ligand/pocket can't be
    resolved.

    diverse=False (default, per-element build): element→channel via 01b (ligand
    C/O/N/S/F/Cl/P → 0..6, pocket C/O/N/S → 0..3); H + out-of-vocab atoms dropped
    (masked), complex kept; lig_rad/poc_rad are None.

    diverse=True (Phase-2 role build): keep EVERY heavy atom that is_diverse_role_atom
    (drops H + noble-gas/dummy artifacts), assign channel 0 (single role channel)
    and a per-atom vdW radius (lig_rad/poc_rad). The role channel then encodes any
    element by blob SIZE, with no fixed vocabulary.
    """
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()                      # populate subchains (label_asym_id)
    st.remove_alternative_conformations()    # keep a single altloc
    st.remove_hydrogens()
    model = st[0]
    asym = str(ligand_asym_id)
    ccd_set = {c.strip().upper() for c in str(ccd_codes).split("-") if c.strip()} \
        if ccd_codes and str(ccd_codes) != "nan" else set()

    # ── locate ligand residues: by label_asym_id (subchain) / chain, +CCD verify ──
    lig_res_ids = set()
    lig_residues = []
    for chain in model:
        for res in chain:
            if (res.subchain == asym) or (chain.name == asym):
                if (not ccd_set) or (res.name.strip().upper() in ccd_set):
                    lig_residues.append(res); lig_res_ids.add(id(res))
    if not lig_residues and ccd_set:         # fallback: match the CCD anywhere
        for chain in model:
            for res in chain:
                if res.name.strip().upper() in ccd_set:
                    lig_residues.append(res); lig_res_ids.add(id(res))
    if not lig_residues:
        return None

    lig_xyz, lig_ch, lig_rad, lig_heavy, lig_all_heavy = [], [], [], set(), []
    for res in lig_residues:
        for a in res:
            el = PDBB._normalise_element(a.element.name)
            if not PDBB._is_heavy_element(el):
                continue
            lig_heavy.add(el)
            lig_all_heavy.append([a.pos.x, a.pos.y, a.pos.z])  # full extent (incl. unsupported)
            if diverse:
                if not is_diverse_role_atom(el):     # drop noble-gas / dummy artifacts
                    continue
                lig_xyz.append([a.pos.x, a.pos.y, a.pos.z])
                lig_ch.append(0); lig_rad.append(vdw_radius(el))   # single role channel
            else:
                ch = PDBB._channel_of(el, PDBB.N_LIG_CH)
                if ch is None:
                    continue
                lig_xyz.append([a.pos.x, a.pos.y, a.pos.z]); lig_ch.append(ch)
    if not lig_xyz or not lig_all_heavy:     # no usable ligand atoms
        return None

    # ── pocket: protein heavy atoms within pocket_radius of any ligand heavy atom ──
    prot = []  # (ch, x, y, z, rad)   rad used in diverse mode only (0.0 otherwise)
    poc_heavy = set()
    for chain in model:
        for res in chain:
            if id(res) in lig_res_ids or not _is_protein_res(res):
                continue
            for a in res:
                el = PDBB._normalise_element(a.element.name)
                if not PDBB._is_heavy_element(el):
                    continue
                if diverse:
                    if not is_diverse_role_atom(el):
                        continue
                    ch, rad = 0, vdw_radius(el)
                else:
                    ch = PDBB._channel_of(el, PDBB.N_POC_CH)
                    if ch is None:
                        continue
                    rad = 0.0
                prot.append((ch, a.pos.x, a.pos.y, a.pos.z, rad)); poc_heavy.add(el)
    if not prot:
        return None
    P = np.array([[x, y, z] for _, x, y, z, _ in prot], dtype=np.float64)
    d, _ = cKDTree(np.array(lig_all_heavy, dtype=np.float64)).query(P, k=1)
    keep = [row for row, dist in zip(prot, d) if dist <= pocket_radius]
    poc_xyz = [[x, y, z] for _, x, y, z, _ in keep]
    poc_ch = [ch for ch, *_ in keep]
    poc_rad = [r for *_, r in keep]
    if not poc_xyz:
        return None

    lig_rad_t = torch.tensor(lig_rad, dtype=torch.float32) if diverse else None
    poc_rad_t = torch.tensor(poc_rad, dtype=torch.float32) if diverse else None
    return (torch.tensor(lig_xyz, dtype=torch.float32),
            torch.tensor(lig_ch, dtype=torch.long),
            torch.tensor(poc_xyz, dtype=torch.float32),
            torch.tensor(poc_ch, dtype=torch.long),
            PDBB._unsupported_heavy(lig_heavy, PDBB.LIGAND_SUPPORTED_HEAVY),
            PDBB._unsupported_heavy(poc_heavy, PDBB.POCKET_SUPPORTED_HEAVY),
            lig_rad_t, poc_rad_t)


def _process_case(r, cif: Path, ccp4: Path, stage_dir: Path, norm, pocket_radius: float):
    """Parse one case → ((pocket_, ligand_), key) or raises/returns (None, reason)."""
    pid = str(r.entry_pdb_id).lower()
    key = f"{pid}_{r.ligand_asym_id}"
    if not (cif.exists() and ccp4.exists()):
        return None, "missing_files"
    parsed = parse_cif_system(cif, r.ligand_asym_id, r.ligand_ccd_code, pocket_radius)
    if parsed is None:
        return None, "parse_failed"
    lig_xyz, lig_ch, poc_xyz, poc_ch, _lu, _pu, _lr, _pr = parsed   # _lr/_pr None (non-diverse)

    # Crop center = SUPPORTED-atom ligand centroid (matches the dataset's
    # __getitem__ recentering on ligand["coords"] after the <7 mask; 00h-identical).
    center = lig_xyz.mean(dim=0).numpy().astype(np.float64)
    grid = PDBB.load_raw_grid(ccp4)
    if grid is None:
        return None, "bad_map"
    arr, frac_T, nu, nv, nw = grid
    crop = _crop_density(arr, frac_T, nu, nv, nw, center,
                         G=GRID_DIM, res=RESOLUTION, transform=None)
    if crop is None:
        return None, "crop_failed"
    np.save(str(stage_dir / f"{key}.npy"), norm(crop))

    c = lig_xyz
    max_len = round(float((c.max(0).values - c.min(0).values).max()), 2)
    pocket_ = {"id": f"plinder/{key}", "coords": poc_xyz.to(torch.float32),
               "atoms_channel": poc_ch.to(torch.uint8)}
    ligand_ = {"id": f"plinder/{key}", "coords": lig_xyz.to(torch.float32),
               "atoms_channel": lig_ch.to(torch.uint8), "max_len": max_len}
    return ((pocket_, ligand_), key), None


def build_pairs(sel: pd.DataFrame, cif_dir: Path, ccp4_dir: Path, stage_dir: Path,
                norm, pocket_radius: float, stream: bool = False, batch: int = 400,
                workers: int = 16, keep_maps: bool = False, min_free_gb: float = 100.0):
    """Parse + re-crop each selected case → (pairs, skipped), processed in batches.

    stream=True: per batch, download the maps+cifs (00a thread pool), crop, then
    DELETE that batch's raw .ccp4 files (keep the ~0.5 MB crops + small cifs) so
    peak disk stays ~batch*map_size regardless of selection size. Aborts cleanly
    if free disk drops below min_free_gb.

    pairs : [((pocket_, ligand_), stage_key), ...]   stage_key = "{pid}_{asym}".
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    cif_dir.mkdir(parents=True, exist_ok=True)
    ccp4_dir.mkdir(parents=True, exist_ok=True)
    rows = list(sel.itertuples(index=False))
    pairs, skipped = [], {}
    pbar = tqdm(total=len(rows), desc="plinder", unit="cplx")
    for start in range(0, len(rows), batch):
        chunk = rows[start: start + batch]
        if stream:
            free = _free_gb(ccp4_dir)
            if free < min_free_gb:
                print(f"\n[disk] free {free:.0f} GB < floor {min_free_gb:.0f} GB — "
                      f"stopping at {start}/{len(rows)} (keeping what's built).")
                break
            seen, tmap, tcif = set(), [], []
            for r in chunk:
                pid = str(r.entry_pdb_id).lower()
                if pid in seen:
                    continue
                seen.add(pid)
                if not (ccp4_dir / f"{pid}.ccp4").exists():
                    tmap.append((pid, MAP_URL.format(pid=pid), ccp4_dir / f"{pid}.ccp4"))
                if not (cif_dir / f"{pid}.cif").exists():
                    tcif.append((pid, CIF_URL.format(pid=pid), cif_dir / f"{pid}.cif"))
            if tcif:
                DL.run_download(tcif, workers, f"cif  {start//batch}")
            if tmap:
                DL.run_download(tmap, workers, f"ccp4 {start//batch}")

        chunk_pids = set()
        for r in chunk:
            pid = str(r.entry_pdb_id).lower()
            key = f"{pid}_{r.ligand_asym_id}"
            chunk_pids.add(pid)
            pbar.update(1)
            try:
                res, reason = _process_case(r, cif_dir / f"{pid}.cif", ccp4_dir / f"{pid}.ccp4",
                                            stage_dir, norm, pocket_radius)
                if res is None:
                    skipped[key] = reason
                else:
                    pairs.append(res)
            except Exception as e:  # never let one complex kill the build
                skipped[key] = f"error:{type(e).__name__}"

        # stream-and-discard: drop this batch's raw maps (the big artifact).
        if stream and not keep_maps:
            for pid in chunk_pids:
                m = ccp4_dir / f"{pid}.ccp4"
                if m.exists():
                    try:
                        m.unlink()
                    except OSError:
                        pass
        if stream:
            pbar.set_postfix(free_gb=int(_free_gb(ccp4_dir)), kept=len(pairs), refresh=False)
    pbar.close()
    return pairs, skipped


def run_resample_manifest(args, out_root: Path):
    """RESAMPLE mode: emit a train manifest + recipe (NO frozen crops) so density is cropped from the
    full map at the AUGMENTED pose at train time (voxbind.dataset.crossdocked_density.DatasetCrossDockedDensity).

    PLINDER is deposited-frame (mmCIF + 2Fo-Fc map share the crystal frame, transform=None) → R=I, t=0, so
    no Kabsch is needed. Reuses the existing data_train_plinder.pt tuples (built by `precompute`) and replicates
    the dataset's train laydown (shuffle(1234) → drop last VAL_SZ → filter max_len) so manifest row i aligns
    positionally with DatasetCrossDockedDensity index i. The retained plinder/ccp4 maps are read at TRAIN time.
    """
    pt = out_root / "data_train_plinder.pt"
    if not pt.exists():
        sys.exit(f"[error] resample needs {pt} — run 03c precompute once to build the tuples first.")
    v6 = json.loads(Path(args.v6_stats).read_text())
    s5, mu_a, sigma_a = float(v6["arcsinh_scale"]), float(v6["mu_a"]), float(v6["sigma_a"])

    tuples = torch.load(str(pt), weights_only=False)
    order = list(tuples)                            # mirror DatasetCrossDockedXray train laydown
    random.Random(1234).shuffle(order)
    order = order[: len(order) - VAL_SZ]
    order = [t for t in order if t[1]["max_len"] <= MAX_LEN]
    n = len(order)
    pdb_ids = [str(p["id"]).split("/")[-1].split("_")[0].lower() for p, _l in order]
    centroids = (np.stack([l["coords"].to(torch.float32).mean(dim=0).numpy() for _p, l in order]).astype(np.float32)
                 if n else np.zeros((0, 3), np.float32))

    out_dir = out_root / "xray_resample_plinder"
    out_dir.mkdir(parents=True, exist_ok=True)
    R = np.broadcast_to(np.eye(3, dtype=np.float32), (n, 3, 3)).copy()    # deposited frame → identity
    t = np.zeros((n, 3), dtype=np.float32)
    ok = np.ones(n, dtype=bool)                                          # every PLINDER case is RSCC-matched
    np.savez(out_dir / "train_manifest.npz",
             pdb_id=np.array(pdb_ids), centroid=centroids, R=R, t=t, ok=ok)
    np.save(str(out_dir / "train_available.npy"), ok)

    recipe = dict(
        kind="resample_manifest", grid_dim=GRID_DIM, resolution=RESOLUTION,
        ccp4_dir=str((PLINDER / "ccp4").resolve()), ccp4_ext=".ccp4",
        frame="deposited (transform=None): PLINDER mmCIF + 2Fo-Fc map share the crystal frame → R=I, t=0",
        norm_version="v6",
        normalization=dict(
            scheme="pocket-pool arcsinh soft-squash + z-score (v6 stats, PLINDER)",
            formula="x' = (arcsinh(x / s) - mu_a) / sigma_a",
            arcsinh_scale=s5, mu_a=mu_a, sigma_a=sigma_a),
        train_time_note=(
            "Per sample i where ok[i]: sample the RAW map plinder/ccp4/{pdb_id[i]}.ccp4 at "
            "cart = centroid[i] + R_aug.T @ (o - t_aug)  (R=I, t=0 deposited frame), o = 64^3 box offsets (A); "
            "then x' = (arcsinh(raw/arcsinh_scale) - mu_a)/sigma_a; derive gradmag on the fly."),
    )
    (out_dir / "resample.json").write_text(json.dumps(recipe, indent=2))

    n_maps = len(list((PLINDER / "ccp4").glob("*.ccp4")))
    print("=== PLINDER RESAMPLE-manifest mode (on-the-fly aug prep) ===")
    print(f"  tuples    : {pt}  ({len(tuples)} pre-shuffle)")
    print(f"  manifest  : {n} train positions → {out_dir / 'train_manifest.npz'}")
    print(f"  norm (v6) : arcsinh(x/{s5}) z-score  mu_a {mu_a:+.6f}  sigma_a {sigma_a:.6f}")
    print(f"  maps      : {PLINDER / 'ccp4'}  ({n_maps} .ccp4 on disk)"
          + ("" if n_maps else "  <- 0! re-acquire via `s2 --dataset plinder` (03b) before training."))
    print(f"  train: dset.resample_dir={out_dir} dset.data_file=data_train_plinder.pt "
          f"dset.subset_n={max(n - VAL_SZ, 0)} dset.subset_val_n={VAL_SZ} dset.subset_xray_only=true")


def run_diverse_build(args, out_root: Path):
    """DIVERSE mode (Phase-2 role-split): re-parse the SAME PLINDER selection keeping
    EVERY heavy atom (role channel 0 + per-atom vdW radius; noble-gas/dummy artifacts
    excluded) → data_train_plinder_diverse.pt + an aligned resample manifest.

    No frozen crops + no density work: the role encoder uses the OTF resampler, which
    reads the full deposited-frame maps (R=I, t=0) at train time. Mirrors the
    precompute → resample laydown (shuffle(1234) → drop VAL_SZ → max_len filter) so
    manifest row i aligns with DatasetCrossDockedDensity index i. atoms_channel is 0 for
    every atom, so the existing <7/<4 mask + role-channel sum admit all elements; the
    dataset reads the stored per-atom radius (see crossdocked_density.__getitem__).
    """
    cif_dir = PLINDER / "cif"
    sel = pd.read_csv(args.selection)
    if args.limit:
        sel = sel.head(args.limit)
    print("=== build PLINDER DIVERSE role-split tuples (Phase 2; atoms-only, OTF density) ===")
    print(f"  selection : {args.selection}  ({len(sel)} rows)")
    print(f"  cifs      : {cif_dir}")

    tuples, skipped = [], {}
    for r in tqdm(list(sel.itertuples(index=False)), desc="diverse", unit="cplx"):
        pid = str(r.entry_pdb_id).lower()
        key = f"{pid}_{r.ligand_asym_id}"
        cif = cif_dir / f"{pid}.cif"
        if not cif.exists():
            skipped[key] = "missing_cif"; continue
        try:
            parsed = parse_cif_system(cif, r.ligand_asym_id, r.ligand_ccd_code,
                                      args.pocket_radius, diverse=True)
        except Exception as e:
            skipped[key] = f"error:{type(e).__name__}"; continue
        if parsed is None:
            skipped[key] = "parse_failed"; continue
        lig_xyz, lig_ch, poc_xyz, poc_ch, _lu, _pu, lig_rad, poc_rad = parsed
        max_len = round(float((lig_xyz.max(0).values - lig_xyz.min(0).values).max()), 2)
        pocket_ = {"id": f"plinder/{key}", "coords": poc_xyz.to(torch.float32),
                   "atoms_channel": poc_ch.to(torch.uint8), "radius": poc_rad.to(torch.float32)}
        ligand_ = {"id": f"plinder/{key}", "coords": lig_xyz.to(torch.float32),
                   "atoms_channel": lig_ch.to(torch.uint8), "radius": lig_rad.to(torch.float32),
                   "max_len": max_len}
        tuples.append((pocket_, ligand_))
    print(f"  built {len(tuples)} tuples  (skipped {len(skipped)})")
    if skipped:
        for reason, n in Counter(skipped.values()).most_common():
            print(f"     skip {reason:20s} {n}")
    if not tuples:
        print("  [abort] no tuples built."); return

    pt = out_root / "data_train_plinder_diverse.pt"
    out_root.mkdir(parents=True, exist_ok=True)
    torch.save(tuples, str(pt))
    print(f"  {pt.name} : {len(tuples)} tuples")

    # Reuse the EXISTING PLINDER resample.json normalization so the diverse density is
    # normalized byte-identically to the per-element run (the v6 stats.json may be gone).
    src_recipe = out_root / "xray_resample_plinder" / "resample.json"
    if src_recipe.exists():
        norm_block = json.loads(src_recipe.read_text())["normalization"]
    else:
        v6 = json.loads(Path(args.v6_stats).read_text())
        norm_block = dict(scheme="pocket-pool arcsinh soft-squash + z-score (v6 stats, PLINDER)",
                          formula="x' = (arcsinh(x / s) - mu_a) / sigma_a",
                          arcsinh_scale=float(v6["arcsinh_scale"]),
                          mu_a=float(v6["mu_a"]), sigma_a=float(v6["sigma_a"]))

    # Aligned manifest (mirror run_resample_manifest laydown) for the diverse tuples.
    order = list(tuples)
    random.Random(1234).shuffle(order)
    order = order[: len(order) - VAL_SZ]
    order = [t for t in order if t[1]["max_len"] <= MAX_LEN]
    n = len(order)
    pdb_ids = [str(p["id"]).split("/")[-1].split("_")[0].lower() for p, _l in order]
    centroids = (np.stack([l["coords"].to(torch.float32).mean(dim=0).numpy() for _p, l in order]).astype(np.float32)
                 if n else np.zeros((0, 3), np.float32))
    out_dir = out_root / "xray_resample_plinder_diverse"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "train_manifest.npz",
             pdb_id=np.array(pdb_ids),
             centroid=centroids,
             R=np.broadcast_to(np.eye(3, dtype=np.float32), (n, 3, 3)).copy(),
             t=np.zeros((n, 3), dtype=np.float32),
             ok=np.ones(n, dtype=bool))
    np.save(str(out_dir / "train_available.npy"), np.ones(n, dtype=bool))
    (out_dir / "resample.json").write_text(json.dumps(dict(
        kind="resample_manifest", grid_dim=GRID_DIM, resolution=RESOLUTION,
        ccp4_dir=str((PLINDER / "ccp4").resolve()), ccp4_ext=".ccp4",
        frame="deposited (transform=None): R=I, t=0",
        norm_version="v6",
        normalization=norm_block,
        train_time_note="DIVERSE role-split tuples (data_train_plinder_diverse.pt): per-atom "
                        "vdW radius stored, atoms_channel=0 → single role channel admits all elements.",
    ), indent=2))
    print(f"  manifest  : {n} train positions → {out_dir / 'train_manifest.npz'}")
    print(f"  launch: dset.resample_dir={out_dir} dset.data_file=data_train_plinder_diverse.pt "
          f"dset.subset_n={max(n - VAL_SZ, 0)} dset.subset_val_n={VAL_SZ} dset.subset_xray_only=true "
          f"dset.ligand_radius=-1 dset.pocket_radius=-1")


def _resolve_selection(cli):
    """Prefer the committed FROZEN selection (splits/plinder/) — cross-server stable; verify its
    sha256 vs plinder_inputs.json and fail loudly on drift. Fall back to the LOCAL (unfrozen)
    selection with a warning, or honour an explicit --selection override."""
    if cli:
        return Path(cli)
    fz_dir = VOX_ROOT / "splits" / "plinder"
    fz_csv = fz_dir / "plinder_selected.csv"
    if fz_csv.exists():
        inp = fz_dir / "plinder_inputs.json"
        if inp.exists():
            import hashlib
            meta = json.loads(inp.read_text()); want = meta.get("selection_sha256")
            got = hashlib.sha256(fz_csv.read_bytes()).hexdigest()
            if want and got != want:
                raise SystemExit(
                    f"[selection] FROZEN selection hash MISMATCH: {fz_csv}\n"
                    f"  expected {want}\n  got      {got}\n"
                    f"  re-pin deliberately via 03a_plinder_select.py --rederive --freeze")
        print(f"  [selection] FROZEN committed (hash-verified): {fz_csv}")
        return fz_csv
    local = PLINDER / "plinder_selected.csv"
    print(f"  [selection] WARNING: no frozen selection in splits/plinder/ — using LOCAL {local} "
          f"(NOT cross-server stable; run 03a --freeze to pin)")
    return local


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--selection", default=None,
                    help="Selection CSV. Default: the committed frozen selection "
                         "(splits/plinder/plinder_selected.csv), hash-verified; else local fallback.")
    ap.add_argument("--out_root", default=str(VOXDATA))
    ap.add_argument("--v6_stats", default=str(VOXDATA / "xray_crops_aligned_v6" / "stats.json"),
                    help="density normalization stats to reuse (shared scale with v6/v7)")
    ap.add_argument("--pocket_radius", type=float, default=10.0, help="pocket cutoff (A) around ligand")
    ap.add_argument("--limit", type=int, default=0, help="only the first N selected rows (pilot)")
    ap.add_argument("--stream", action="store_true",
                    help="download maps+cifs per batch and DELETE raw .ccp4 after cropping "
                         "(peak disk ~ batch*map_size; for the full set without holding ~90 GB of maps)")
    ap.add_argument("--batch", type=int, default=400, help="stream batch size (cases per download+crop+purge cycle)")
    ap.add_argument("--workers", type=int, default=16, help="download threads (stream mode)")
    ap.add_argument("--min_free_gb", type=float, default=120.0, help="abort streaming if free disk drops below this")
    ap.add_argument("--keep_maps", action="store_true", help="stream mode: do NOT delete raw .ccp4 after cropping")
    ap.add_argument("--mode", choices=["precompute", "resample", "diverse"], default="precompute",
                    help="precompute = write frozen normalized crops (default). resample = write ONLY a train "
                         "manifest (pdb_id, centroid, R=I, t=0, ok) + resample.json recipe, so density is cropped "
                         "from the full map at the AUGMENTED pose at train time (DatasetCrossDockedDensity). Reuses "
                         "data_train_plinder.pt; needs the plinder/ccp4 maps RETAINED — re-acquire via `s2 --dataset plinder`.")
    args = ap.parse_args()
    args.selection = _resolve_selection(args.selection)   # frozen-first (cross-server stable)

    out_root = Path(args.out_root)
    cif_dir = PLINDER / "cif"
    ccp4_dir = PLINDER / "ccp4"
    out_dir = out_root / "xray_crops_aligned_plinder"
    stage_dir = out_dir / ".stage"

    if args.mode == "resample":
        run_resample_manifest(args, out_root)
        return

    if args.mode == "diverse":
        run_diverse_build(args, out_root)
        return

    sel = pd.read_csv(args.selection)
    if args.limit:
        sel = sel.head(args.limit)

    v6 = json.loads(Path(args.v6_stats).read_text())
    s5, mu_a, sigma_a = float(v6["arcsinh_scale"]), float(v6["mu_a"]), float(v6["sigma_a"])

    def norm(raw):
        return ((np.arcsinh(raw.astype(np.float64) / s5) - mu_a) / sigma_a).astype(np.float16)

    print("=== build PLINDER standalone density pretraining set (Strategy A, deposited frame) ===")
    print(f"  selection : {args.selection}  ({len(sel)} rows)")
    print(f"  out       : {out_dir}")
    print(f"  v6 norm   : arcsinh(x/{s5}) z-score  μ_a {mu_a:+.6f}  σ_a {sigma_a:.6f}")
    print(f"  pocket    : heavy protein atoms within {args.pocket_radius} A of ligand")
    print(f"  mode      : {'STREAM (download→crop→purge maps, batch=%d, floor=%.0f GB)' % (args.batch, args.min_free_gb) if args.stream else 'pre-downloaded (03b)'}"
          f"  | free now {_free_gb(out_root):.0f} GB")

    # ── 1. parse + crop every selected case (streamed + disk-bounded if --stream) ──
    pairs, skipped = build_pairs(sel, cif_dir, ccp4_dir, stage_dir, norm, args.pocket_radius,
                                 stream=args.stream, batch=args.batch, workers=args.workers,
                                 keep_maps=args.keep_maps, min_free_gb=args.min_free_gb)
    print(f"  built {len(pairs)} pairs  (skipped {len(skipped)})")
    if skipped:
        for reason, n in Counter(skipped.values()).most_common():
            print(f"     skip {reason:20s} {n}")

    if not pairs:
        print("  [abort] no pairs built — check 03b downloads / asym_id matching.")
        return

    # ── 2. combined (pre-shuffle) data file ──────────────────────────────────────
    tuples = [t for t, _ in pairs]
    torch.save(tuples, str(out_root / "data_train_plinder.pt"))
    print(f"  data_train_plinder.pt : {len(tuples)} tuples")

    # ── 3. replicate the dataset's train order, lay crops at those positions ─────
    # shuffle(1234) → drop last 100 → filter max_len  (mirrors DatasetCrossDockedXray / 00h)
    order = list(pairs)
    random.Random(1234).shuffle(order)
    order = order[: len(order) - VAL_SZ]
    order = [(t, k) for t, k in order if t[1]["max_len"] <= MAX_LEN]
    n = len(order)
    print(f"  train positions (post shuffle/split/filter): {n}")

    train_out = out_dir / "train"
    if train_out.exists():
        shutil.rmtree(train_out)
    train_out.mkdir(parents=True, exist_ok=True)
    for i, (_t, key) in enumerate(tqdm(order, desc="write crops", unit="file")):
        shutil.copyfile(stage_dir / f"{key}.npy", train_out / f"{i:06d}.npy")
    np.save(str(out_dir / "train_available.npy"), np.ones(n, dtype=bool))

    # ── 4. stats.json + cleanup ──────────────────────────────────────────────────
    (out_dir / "stats.json").write_text(json.dumps({
        "scheme": "PLINDER standalone: ligand-matched X-ray 2Fo-Fc (arcsinh+z, v6 stats)",
        "formula": "x' = (arcsinh(x / s) - mu_a) / sigma_a",
        "arcsinh_scale": s5, "mu_a": mu_a, "sigma_a": sigma_a,
        "n_train": n, "n_built": len(pairs), "n_skipped": len(skipped),
        "pocket_radius": args.pocket_radius, "frame": "deposited (transform=None)",
        "selection": str(args.selection),
        "note": "PLINDER selected via 03a (RSCC>=0.8 ligand-matched density, leakage-filtered). "
                "Density cropped from deposited 2Fo-Fc map at supported-atom ligand centroid. "
                "Train data file: data_train_plinder.pt (load via dset.data_file). "
                "Pretrain sizing: dset.subset_n=%d dset.subset_val_n=%d." % (n - VAL_SZ, VAL_SZ),
    }, indent=2))
    shutil.rmtree(stage_dir, ignore_errors=True)

    print(f"\nDone. PLINDER train crops: {n}  →  {out_dir}")
    print(f"  pretrain sizing: dset.subset_n={n - VAL_SZ} dset.subset_val_n={VAL_SZ}")
    print(f"  launch (Phase 4): dset.data_file=data_train_plinder.pt "
          f"dset.crops_dir={out_dir} dset.subset_xray_only=true")


if __name__ == "__main__":
    main()
