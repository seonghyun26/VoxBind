"""04_build_perelem.py — v2 110K corpus, PER-ELEMENT + "other" channel (hybrid).

Same selection / density / OTF laydown as 04_build.py, but the ligand is encoded with
explicit per-element channels (C,N,O,S,P,F,Cl → 0..6) plus an "other" channel (idx 7)
catching the 0.25% rare tail (Br,I,B,Se,metals) — so NOTHING is dropped (unlike the old
per-element build) while the common 99.75% keep one-hot element identity. Pocket stays
C,N,O,S (0..3). Per-atom vdW radius is also stored so blobs voxelize at true size.

    out: pretrain/data_train_plinder_v2_perelem.pt        (8-lig / 4-poc channel tuples)
         pretrain/xray_resample_plinder_v2_perelem/        (OTF manifest + resample.json → data/ccp4)

Channels: 8 lig + 4 poc + density + gradmag = 14  → ChannelViT [8,4,2].

    cd voxbind && python dataset/plinder/04_build_perelem.py --limit 300   # smoke
    cd voxbind && python dataset/plinder/04_build_perelem.py               # full build
"""
import gemmi  # noqa: F401 — before torch
import os; os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse, hashlib, importlib.util, json, random
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, torch
from tqdm import tqdm
torch.set_num_threads(2)

VOX_ROOT = Path(__file__).resolve().parents[2]
DATA = VOX_ROOT / "dataset" / "data"
CIF_DIR = DATA / "cif"
CCP4_DIR = DATA / "ccp4"
PRETRAIN = DATA / "pretrain"


def version_paths(version):
    """v2 → the original v2 names (bit-identical); other versions get a suffix. All
    reuse the SHARED data/cif + data/ccp4 caches (no downloads)."""
    suff = "" if version == "v2" else f"_{version}"
    return (VOX_ROOT / "splits" / "plinder" / version,
            PRETRAIN / f"data_train_plinder_{version}_perelem.pt",
            PRETRAIN / f"xray_resample_plinder_{version}_perelem")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod); return mod


M = load_module(VOX_ROOT / "dataset" / "legacy" / "03c_plinder_preprocess.py", "p03c")


def resolve_selection(freeze: Path) -> Path:
    csv = freeze / "plinder_selected.csv"
    inp = freeze / "plinder_inputs.json"
    if inp.exists():
        want = json.loads(inp.read_text()).get("selection_sha256")
        if want and hashlib.sha256(csv.read_bytes()).hexdigest() != want:
            raise SystemExit("[fatal] selection hash MISMATCH")
    return csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2", help="selection freeze / output suffix (v2, v2p1, …)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pocket_radius", type=float, default=10.0)
    ap.add_argument("--pose_dedup_tau", type=float, default=0.0,
                    help=">0 enables DENSITY-AWARE pose dedup: within each (PDB, ligand) group keep one "
                         "copy per density cluster (aligned density-crop corr < tau); drops symmetry copies.")
    args = ap.parse_args()

    global FREEZE, OUT_TUPLES, OUT_DIR
    FREEZE, OUT_TUPLES, OUT_DIR = version_paths(args.version)

    sel = pd.read_csv(resolve_selection(FREEZE))
    if args.limit:
        sel = sel.head(args.limit)
    print(f"=== v2 PER-ELEMENT+other build (8 lig / 4 poc) ===\n  selection: {len(sel):,} rows\n  cifs: {CIF_DIR}")

    tuples, meta, skipped, ch_hist = [], [], {}, Counter()
    for r in tqdm(list(sel.itertuples(index=False)), desc="build", unit="cplx"):
        pid = str(r.entry_pdb_id).lower(); key = f"{pid}_{r.ligand_asym_id}"
        cif = CIF_DIR / f"{pid}.cif"
        if not cif.exists():
            skipped[key] = "missing_cif"; continue
        try:
            parsed = M.parse_cif_system(cif, r.ligand_asym_id, r.ligand_ccd_code,
                                        args.pocket_radius, diverse=False, other_channel=True)
        except Exception as e:
            skipped[key] = f"error:{type(e).__name__}"; continue
        if parsed is None:
            skipped[key] = "parse_failed"; continue
        lig_xyz, lig_ch, poc_xyz, poc_ch, _lu, _pu, lig_rad, poc_rad = parsed
        for c in lig_ch.tolist(): ch_hist[c] += 1
        max_len = round(float((lig_xyz.max(0).values - lig_xyz.min(0).values).max()), 2)
        pocket_ = {"id": f"plinder/{key}", "coords": poc_xyz.to(torch.float32),
                   "atoms_channel": poc_ch.to(torch.uint8), "radius": poc_rad.to(torch.float32)}
        ligand_ = {"id": f"plinder/{key}", "coords": lig_xyz.to(torch.float32),
                   "atoms_channel": lig_ch.to(torch.uint8), "radius": lig_rad.to(torch.float32),
                   "max_len": max_len}
        tuples.append((pocket_, ligand_))
        meta.append((pid, str(r.ligand_ccd_code)))

    print(f"  built {len(tuples):,} tuples  (skipped {len(skipped):,})")
    for reason, n in Counter(skipped.values()).most_common():
        print(f"     skip {reason:20s} {n}")
    tot = sum(ch_hist.values())
    print("  ligand channel histogram (0-6=C/O/N/S/F/Cl/P, 7=other):")
    for c in range(8):
        print(f"     ch {c}: {ch_hist.get(c,0):>10,}  ({100*ch_hist.get(c,0)/max(tot,1):5.2f}%)")
    if not tuples:
        raise SystemExit("  [abort] no tuples built.")

    # ── DENSITY-AWARE POSE DEDUP (opt-in, --pose_dedup_tau>0) ────────────────────────────
    if args.pose_dedup_tau > 0:
        from collections import defaultdict
        cdm = load_module(VOX_ROOT / "dataset" / "crossdocked_density.py", "cdm_pd")
        pdd = load_module(VOX_ROOT / "dataset" / "plinder" / "pose_dedup.py", "pose_dedup")
        tau = float(args.pose_dedup_tau)
        # Group by PID → CCD → indices so each CCP4 map is loaded ONCE and FREED before the next
        # PID (caching every map blows up host RAM → OOM at ~40k PDBs).
        pid_groups = defaultdict(lambda: defaultdict(list))
        for i, (pid, ccd) in enumerate(meta):
            pid_groups[pid][ccd].append(i)
        keep = np.zeros(len(tuples), dtype=bool)
        n_nomap = 0
        for pid, ccd_map in tqdm(pid_groups.items(), desc=f"pose-dedup(t={tau})", unit="pdb"):
            g = None
            if any(len(idxs) > 1 for idxs in ccd_map.values()):
                p = CCP4_DIR / f"{pid}.ccp4"
                g = cdm._load_raw_grid(p) if p.exists() else None
            for ccd, idxs in ccd_map.items():
                if len(idxs) == 1:
                    keep[idxs[0]] = True; continue
                if g is None:                               # no map → keep all copies (rare)
                    keep[idxs] = True; n_nomap += 1; continue
                arr, fMT, nu, nv, nw = g
                rf = lambda center, R: cdm._resample_density(arr, fMT, nu, nv, nw, center, R_aug=R, G=40, res=0.35)
                by_n = defaultdict(list)                     # only equal-atom copies are Kabsch-comparable
                for i in idxs:
                    by_n[len(tuples[i][1]["coords"])].append(i)
                for _n, sub in by_n.items():
                    if len(sub) == 1:
                        keep[sub[0]] = True; continue
                    coords = [tuples[i][1]["coords"].numpy().astype(np.float64) for i in sub]
                    for k in pdd.pose_dedup_group(coords, rf, tau=tau, cap=16):
                        keep[sub[k]] = True
            g = None                                         # free this PID's map before the next
        tuples = [t for t, k in zip(tuples, keep) if k]
        print(f"  pose-dedup(tau={tau}): {len(keep):,} → {len(tuples):,} kept "
              f"({int(keep.sum())} distinct-pose crops; {n_nomap} groups had no map)")
        if not tuples:
            raise SystemExit("  [abort] pose-dedup removed everything.")

    PRETRAIN.mkdir(parents=True, exist_ok=True)
    torch.save(tuples, str(OUT_TUPLES))
    print(f"  {OUT_TUPLES.name}: {len(tuples):,} tuples")

    # manifest (own, to guarantee alignment) — same laydown as 04_build.py
    norm_block = json.loads((DATA / "plinder" / "xray_resample_plinder" / "resample.json").read_text())["normalization"]
    order = list(tuples); random.Random(1234).shuffle(order)
    order = order[: len(order) - M.VAL_SZ]
    order = [t for t in order if t[1]["max_len"] <= M.MAX_LEN]
    n = len(order)
    pdb_ids = [str(p["id"]).split("/")[-1].split("_")[0].lower() for p, _l in order]
    centroids = (np.stack([l["coords"].to(torch.float32).mean(0).numpy() for _p, l in order]).astype(np.float32)
                 if n else np.zeros((0, 3), np.float32))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "train_manifest.npz", pdb_id=np.array(pdb_ids), centroid=centroids,
             R=np.broadcast_to(np.eye(3, dtype=np.float32), (n, 3, 3)).copy(),
             t=np.zeros((n, 3), dtype=np.float32), ok=np.ones(n, dtype=bool))
    np.save(str(OUT_DIR / "train_available.npy"), np.ones(n, dtype=bool))
    (OUT_DIR / "resample.json").write_text(json.dumps(dict(
        kind="resample_manifest", grid_dim=M.GRID_DIM, resolution=M.RESOLUTION,
        ccp4_dir=str(CCP4_DIR.resolve()), ccp4_ext=".ccp4",
        frame="deposited (transform=None): R=I, t=0", norm_version="v6", normalization=norm_block,
        train_time_note="v2 PER-ELEMENT+other (data_train_plinder_v2_perelem.pt): ligand atoms_channel 0..7 "
                        "(7=other rare tail), pocket 0..3 (C/N/O/S); per-atom vdW radius stored; reads SHARED data/ccp4.",
    ), indent=2))
    print(f"  manifest: {n:,} train positions → {OUT_DIR}")
    print(f"  launch: dset.resample_dir={OUT_DIR} dset.data_file=pretrain/data_train_plinder_v2_perelem.pt "
          f"dset.subset_n={max(n - M.VAL_SZ, 0)} dset.subset_val_n={M.VAL_SZ} dset.n_lig_ch=8")


if __name__ == "__main__":
    main()
