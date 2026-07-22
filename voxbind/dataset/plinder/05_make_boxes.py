"""05_make_boxes.py — precompute 96³ canonical-pose density boxes for fast training.

Why: OTF training reads + resamples a full ccp4 map PER sample PER epoch → GPU starves
(~40% util, data-pipeline bound). This does that resample ONCE, offline, into a small
96³ box per sample. At train time the box loader resamples the 64³ at the augmented pose
from the box (cheap) instead of the full map — same aug semantics, ~85% util expected.

The box is the RAW (un-normalized) density at the CANONICAL pose (R_aug=None), produced by
the EXACT OTF resampler (`_resample_density`, G=96) using the manifest's centroid + Kabsch
(R=I, t=0 for the deposited-frame v2 build). 96³ vs the 64³ train crop gives an 8-voxel
(2 Å) margin per side that fills rotated-in voxels with real density instead of zeros (up
to moderate rotations). Stored as a single fp16 memmap [N,96,96,96] (~199 GB).

Parallel by pdb_id (each map loaded once), writes disjoint rows of the shared memmap.

    cd voxbind && python dataset/plinder/05_make_boxes.py --jobs 100      # full
    cd voxbind && python dataset/plinder/05_make_boxes.py --limit 200     # smoke
"""
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, importlib.util, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import multiprocessing as mp

VOX = Path(__file__).resolve().parents[2]
DATA = VOX / "dataset" / "data"
RESAMPLE_DIR = DATA / "pretrain" / "xray_resample_plinder_v2"
MANIFEST = RESAMPLE_DIR / "train_manifest.npz"
RECIPE = RESAMPLE_DIR / "resample.json"
OUT_DAT = RESAMPLE_DIR / "box96.dat"
OUT_META = RESAMPLE_DIR / "box96_meta.json"
G_BOX = 96
RES = 0.25


def _load_cd():
    """Load _load_raw_grid + _resample_density from crossdocked_density.py without
    triggering the dataset package's heavy __init__ side effects."""
    p = VOX / "dataset" / "crossdocked_density.py"
    spec = importlib.util.spec_from_file_location("cd_box", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


# ── per-worker globals (set in initializer) ──────────────────────────────────
_CD = None; _MM = None; _CCP4 = None; _EXT = None
_CENTROID = None; _R = None; _T = None; _PID = None


def _init(ccp4_dir, ext, n, centroid, Rm, Tm, pid):
    global _CD, _MM, _CCP4, _EXT, _CENTROID, _R, _T, _PID
    _CD = _load_cd()
    _CCP4 = Path(ccp4_dir); _EXT = ext
    _MM = np.memmap(OUT_DAT, dtype=np.float16, mode="r+", shape=(n, G_BOX, G_BOX, G_BOX))
    _CENTROID, _R, _T, _PID = centroid, Rm, Tm, pid


def _do_pid(args):
    """Resample every sample of one pdb_id (map loaded once) → write boxes to memmap."""
    pid, idxs = args
    path = _CCP4 / f"{pid}{_EXT}"
    grid = _CD._load_raw_grid(path) if path.exists() else None
    if grid is None:
        for i in idxs:
            _MM[i] = 0.0
        return (pid, 0, len(idxs))   # (pid, n_ok, n_fail)
    arr, frac_T, nu, nv, nw = grid
    ok = 0
    for i in idxs:
        box = _CD._resample_density(
            arr, frac_T, nu, nv, nw,
            center_cart=_CENTROID[i],
            R_kab=_R[i], t_kab=_T[i],
            R_aug=None, t_aug=None,            # CANONICAL pose
            G=G_BOX, res=RES,
        )
        _MM[i] = box.astype(np.float16)
        ok += 1
    return (pid, ok, 0)


def main():
    global RESAMPLE_DIR, MANIFEST, RECIPE, OUT_DAT, OUT_META
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="first N samples (smoke)")
    ap.add_argument("--resample_dir", default=str(RESAMPLE_DIR),
                    help="dir holding train_manifest.npz + resample.json; box96.dat is written here "
                         "(default: the v2 set). Pool workers fork after this is set → inherit it.")
    args = ap.parse_args()
    RESAMPLE_DIR = Path(args.resample_dir)
    MANIFEST = RESAMPLE_DIR / "train_manifest.npz"
    RECIPE = RESAMPLE_DIR / "resample.json"
    OUT_DAT = RESAMPLE_DIR / "box96.dat"
    OUT_META = RESAMPLE_DIR / "box96_meta.json"

    man = np.load(MANIFEST, allow_pickle=True)
    pid = np.asarray(man["pdb_id"]).astype(str)
    centroid = np.asarray(man["centroid"], dtype=np.float32)
    Rm = np.asarray(man["R"], dtype=np.float32)
    Tm = np.asarray(man["t"], dtype=np.float32)
    ok_mask = np.asarray(man["ok"], dtype=bool)
    n = len(pid)
    recipe = json.loads(RECIPE.read_text())
    ccp4_dir = recipe["ccp4_dir"]; ext = recipe.get("ccp4_ext", ".ccp4")

    sel = np.arange(args.limit if args.limit else n)
    print(f"[05_make_boxes] N={n:,}  building {len(sel):,}  G_box={G_BOX}  res={RES}")
    print(f"  ccp4: {ccp4_dir}  →  {OUT_DAT}  (~{n*G_BOX**3*2/1e9:.0f} GB fp16)")

    # allocate the memmap (sparse file; rows written as workers finish)
    mm = np.memmap(OUT_DAT, dtype=np.float16, mode="w+", shape=(n, G_BOX, G_BOX, G_BOX))
    del mm

    # group selected, available samples by pid → one map load per task
    groups = defaultdict(list)
    for i in sel:
        if ok_mask[i]:
            groups[pid[i]].append(int(i))
    tasks = list(groups.items())
    print(f"  {len(tasks):,} unique pids over {sum(len(v) for _, v in tasks):,} samples")

    t0 = time.time(); done = 0; nfail = 0
    with mp.Pool(args.jobs, initializer=_init,
                 initargs=(ccp4_dir, ext, n, centroid, Rm, Tm, pid)) as pool:
        for k, (pp, nok, nf) in enumerate(pool.imap_unordered(_do_pid, tasks, chunksize=4)):
            done += nok; nfail += nf
            if (k + 1) % 500 == 0 or (k + 1) == len(tasks):
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                print(f"  pids {k+1:,}/{len(tasks):,}  boxes {done:,}  fail {nfail}  "
                      f"{rate:.0f}/s  eta {(len(sel)-done)/max(rate,1e-9)/60:.0f}m", flush=True)

    OUT_META.write_text(json.dumps(dict(
        n=int(n), g_box=G_BOX, res=RES, dtype="float16",
        raw=True, norm_recipe=recipe.get("normalization"),
        note="canonical-pose RAW density boxes; train-time resamples 64^3 at aug pose then "
             "applies recipe norm + derives gradmag (mirrors DatasetCrossDockedDensity).",
    ), indent=2))
    print(f"[done] {done:,} boxes ({nfail} failed) in {(time.time()-t0)/60:.1f}m → {OUT_DAT}")


if __name__ == "__main__":
    main()
