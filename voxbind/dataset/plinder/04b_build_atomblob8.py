"""04b_build_atomblob8.py — v2 ATOMBLOB tuples with an 8th "other" ligand channel.

Same complexes / same order as 04_build's DIVERSE tuples (so the existing
xray_resample_plinder_v2 manifest + box96.dat stay aligned by position), but parsed in
PER-ELEMENT mode with other_channel=True:
    ligand  C/O/N/S/F/Cl/P → 0..6 , every other heavy (is_diverse_role_atom) → 7   (8 ch)
    pocket  C/O/N/S         → 0..3                                                   (4 ch)
The kept-atom SET is identical to the diverse build (both keep is_diverse_role_atom
heavies), only the channel assignment differs → tuple order/count match → reuses the
density boxes. Verified against data_train_plinder_v2.pt before use.

    cd voxbind && python dataset/plinder/04b_build_atomblob8.py --jobs 80
"""
import gemmi  # noqa: F401 — before torch
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, importlib.util, json
from collections import defaultdict, Counter
from pathlib import Path
import multiprocessing as mp
import pandas as pd, torch
torch.set_num_threads(1)

VOX = Path(__file__).resolve().parents[2]
DATA = VOX / "dataset" / "data"
CIF_DIR = DATA / "cif"
PRETRAIN = DATA / "pretrain"
FREEZE = VOX / "splits" / "plinder" / "v2"
OUT_TUPLES = PRETRAIN / "data_train_plinder_v2_atomblob8.pt"
DIVERSE_REF = PRETRAIN / "data_train_plinder_v2.pt"   # alignment reference


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


_M = None
def _init():
    global _M
    _M = _load(VOX / "dataset" / "legacy" / "03c_plinder_preprocess.py", "p03c_ab")


def _parse_pid(args):
    """Parse every row of one pdb_id (cif loaded once) in atomblob+other mode."""
    pid, rows = args
    cif = CIF_DIR / f"{pid}.cif"
    out = []
    if not cif.exists():
        return [(idx, None) for idx, _ in rows]
    for idx, r in rows:
        try:
            parsed = _M.parse_cif_system(cif, r["ligand_asym_id"], r["ligand_ccd_code"],
                                         10.0, diverse=False, other_channel=True)
        except Exception:
            out.append((idx, None)); continue
        if parsed is None:
            out.append((idx, None)); continue
        lig_xyz, lig_ch, poc_xyz, poc_ch, _lu, _pu, lig_rad, poc_rad = parsed
        key = f"{pid}_{r['ligand_asym_id']}"
        max_len = round(float((lig_xyz.max(0).values - lig_xyz.min(0).values).max()), 2)
        pocket_ = {"id": f"plinder/{key}", "coords": poc_xyz.to(torch.float32),
                   "atoms_channel": poc_ch.to(torch.uint8), "radius": poc_rad.to(torch.float32)}
        ligand_ = {"id": f"plinder/{key}", "coords": lig_xyz.to(torch.float32),
                   "atoms_channel": lig_ch.to(torch.uint8), "radius": lig_rad.to(torch.float32),
                   "max_len": max_len}
        out.append((idx, (pocket_, ligand_)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=80)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sel = pd.read_csv(FREEZE / "plinder_selected.csv")
    if args.limit:
        sel = sel.head(args.limit)
    print(f"=== v2 ATOMBLOB-8ch build (per-element + other_channel) — {len(sel):,} rows ===")

    # group rows by pid, preserving original row index for in-order assembly
    groups = defaultdict(list)
    for idx, r in enumerate(sel.to_dict("records")):
        groups[str(r["entry_pdb_id"]).lower()].append((idx, r))
    tasks = list(groups.items())

    results = {}
    with mp.Pool(args.jobs, initializer=_init) as pool:
        for k, chunk in enumerate(pool.imap_unordered(_parse_pid, tasks, chunksize=4)):
            for idx, tup in chunk:
                results[idx] = tup
            if (k + 1) % 2000 == 0 or (k + 1) == len(tasks):
                print(f"  pids {k+1:,}/{len(tasks):,}  parsed {sum(v is not None for v in results.values()):,}", flush=True)

    # assemble in CSV row order, dropping failures (== 04_build's append-on-success order)
    tuples = [results[i] for i in range(len(sel)) if results.get(i) is not None]
    skipped = sum(1 for i in range(len(sel)) if results.get(i) is None)
    print(f"  built {len(tuples):,} tuples (skipped {skipped:,})")

    # channel histogram sanity (ligand): expect 0..7, with 7 = the rare 'other' tail
    ch_hist = Counter()
    for _p, l in tuples[:5000]:
        ch_hist.update(l["atoms_channel"].tolist())
    print(f"  ligand channel hist (first 5k tuples): {dict(sorted(ch_hist.items()))}")

    PRETRAIN.mkdir(parents=True, exist_ok=True)
    torch.save(tuples, str(OUT_TUPLES))
    print(f"  saved → {OUT_TUPLES} ({len(tuples):,} tuples)")

    # ── ALIGNMENT VERIFY vs the diverse tuples (must match position-by-position) ──
    if DIVERSE_REF.exists():
        ref = torch.load(str(DIVERSE_REF), weights_only=False)
        same_n = len(ref) == len(tuples)
        mism = [i for i in range(min(len(ref), len(tuples))) if ref[i][1]["id"] != tuples[i][1]["id"]]
        print(f"  [ALIGN] ref={len(ref):,} new={len(tuples):,} same_n={same_n}  id_mismatches={len(mism)}")
        if same_n and not mism:
            print("  [ALIGN] ✓ identical id order → reuses xray_resample_plinder_v2 manifest + box96.dat")
        else:
            print(f"  [ALIGN] ✗ MISALIGNED — do NOT train (first mismatch idx {mism[0] if mism else 'n/a'})")
    else:
        print("  [ALIGN] no diverse ref found to compare")


if __name__ == "__main__":
    main()
