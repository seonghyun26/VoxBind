"""04c_build_v2p1_atomblob7.py — PLINDER v2.1: 7-ch atomblob, drop out-of-vocab ligands.

v2.1 = v2 minus every complex whose ligand has an out-of-vocab heavy atom (i.e. an atom that
atomblob8 routed to the 8th "other" channel). The survivors use ONLY the 7-element vocab
(C/O/N/S/F/Cl/P → 0..6), so no 8th channel is needed. Built by filtering the atomblob8 tuples
(channel-7-free), then replaying 04_build's laydown so a fresh manifest stays aligned, and a
fresh 96³ box set is resampled (05_make_boxes --resample_dir ...). Isolates: does modelling the
rare out-of-vocab atoms (8th channel + the 5% complexes) help vs. a clean 7-vocab corpus?

    cd voxbind && python dataset/plinder/04c_build_v2p1_atomblob7.py
"""
import importlib.util, json, random, shutil
from pathlib import Path
import numpy as np, torch

VOX = Path(__file__).resolve().parents[2]
DATA = VOX / "dataset" / "data"
PRETRAIN = DATA / "pretrain"
SRC_TUPLES = PRETRAIN / "data_train_plinder_v2_atomblob8.pt"      # atomblob8 (all 113,874)
SRC_RESAMPLE = PRETRAIN / "xray_resample_plinder_v2"             # for resample.json
OUT_TUPLES = PRETRAIN / "data_train_plinder_v2p1_atomblob7.pt"
OUT_DIR = PRETRAIN / "xray_resample_plinder_v2p1"


def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m


M = _load(VOX / "dataset" / "legacy" / "03c_plinder_preprocess.py", "p03c_v2p1")  # VAL_SZ/MAX_LEN/GRID_DIM/RESOLUTION


def main():
    src = torch.load(str(SRC_TUPLES), weights_only=False)
    # clean = ligand uses only channels 0..6 (no "other" ch7 atom). These tuples are already
    # 7-ch; we just drop the complexes that have any ch7 atom. Keep CSV order (drop in place).
    tuples = [(p, l) for (p, l) in src if int(l["atoms_channel"].max()) < 7]
    n_drop = len(src) - len(tuples)
    print(f"=== v2.1 atomblob7 build ===\n  atomblob8 tuples: {len(src):,}  → clean (7-vocab): {len(tuples):,}  (dropped {n_drop:,})")
    # sanity: every kept ligand channel must be in 0..6
    mx = max(int(l["atoms_channel"].max()) for _p, l in tuples)
    assert mx < 7, f"clean filter failed: max ligand channel {mx}"

    PRETRAIN.mkdir(parents=True, exist_ok=True)
    torch.save(tuples, str(OUT_TUPLES))
    print(f"  saved → {OUT_TUPLES} ({len(tuples):,} tuples)")

    # ── laydown: IDENTICAL to 04_build (shuffle 1234 → drop VAL_SZ → max_len filter) so the
    #    dataset's load-time shuffle aligns row i ↔ manifest row i ↔ box row i ──────────────
    order = list(tuples)
    random.Random(1234).shuffle(order)
    order = order[: len(order) - M.VAL_SZ]
    order = [t for t in order if t[1]["max_len"] <= M.MAX_LEN]
    n = len(order)
    pdb_ids = [str(p["id"]).split("/")[-1].split("_")[0].lower() for p, _l in order]
    centroids = np.stack([l["coords"].to(torch.float32).mean(0).numpy() for _p, l in order]).astype(np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "train_manifest.npz",
             pdb_id=np.array(pdb_ids), centroid=centroids,
             R=np.broadcast_to(np.eye(3, dtype=np.float32), (n, 3, 3)).copy(),
             t=np.zeros((n, 3), dtype=np.float32), ok=np.ones(n, dtype=bool))
    np.save(str(OUT_DIR / "train_available.npy"), np.ones(n, dtype=bool))
    # reuse v2's resample recipe verbatim (same maps / norm / grid)
    shutil.copyfile(SRC_RESAMPLE / "resample.json", OUT_DIR / "resample.json")
    print(f"  manifest: {n:,} train positions → {OUT_DIR/'train_manifest.npz'}")
    print(f"  subset_n = {max(n - M.VAL_SZ, 0):,}  subset_val_n = {M.VAL_SZ}")
    print(f"  NEXT: python dataset/plinder/05_make_boxes.py --resample_dir {OUT_DIR} --jobs 80")


if __name__ == "__main__":
    main()
