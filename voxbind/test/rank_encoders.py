"""rank_encoders.py — aggregate base/_casf/encoder_search/*.json into a ranked table.

Ranks pre-trained encoders by frozen-probe metrics on four affinity test sets
(lp_edrscc_v2, cl123, cl123_novel60, cl123_novel30). Default sort = mean Spearman rho
across the four sets; --by picks a single test set. Champion row (260705_ar_cvit_100m_v2_mask075)
is marked. Prints Pearson r / Spearman rho / RMSE (mean +/- std, 5 seeds).
"""
import argparse
import glob
import json
import os

REPO = "/home/shpark/prj-denovo/VoxBind"
D = f"{REPO}/base/_casf/encoder_search"
SETS = ["lp_edrscc_v2", "cl123", "cl123_novel60", "cl123_novel30"]
CHAMP = "atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_mask075"


def short(tag):
    return tag.replace("atomblob_density_gradmag_e49_v5_", "").replace("atomblob_e49_v5_", "C:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", default="mean", choices=["mean"] + SETS)
    ap.add_argument("--metric", default="spearman", choices=["spearman", "pearson", "rmse"])
    ap.add_argument("--top", type=int, default=0, help="show only top N (0=all)")
    args = ap.parse_args()

    rows = []
    for f in glob.glob(f"{D}/*.json"):
        d = json.load(open(f))
        if not all(s in d for s in SETS):
            continue
        rho = {s: d[s][args.metric][0] for s in SETS}
        key = (sum(rho.values()) / len(SETS)) if args.by == "mean" else rho[args.by]
        rows.append((key, d))
    higher = args.metric != "rmse"
    rows.sort(key=lambda x: x[0], reverse=higher)
    if args.top:
        rows = rows[: args.top]

    champ_key = next((k for k, d in rows if d["tag"] == CHAMP), None)
    hdr = f"{'#':>3} {'encoder':44} " + " ".join(f"{s.replace('cl123_',''):>16}" for s in SETS) + f"   {'mean_rho':>8}"
    print(f"sorted by {args.by} {args.metric} ({len(rows)} encoders, 5 seeds)\n")
    print(hdr); print("-" * len(hdr))
    for rank, (key, d) in enumerate(rows, 1):
        cells = []
        for s in SETS:
            m, sd = d[s]["spearman"]
            cells.append(f"{m:.3f}±{sd:.3f}")
        mean_rho = sum(d[s]["spearman"][0] for s in SETS) / len(SETS)
        star = " *CHAMP" if d["tag"] == CHAMP else ""
        print(f"{rank:>3} {short(d['tag'])[:44]:44} " + " ".join(f"{c:>16}" for c in cells) + f"   {mean_rho:8.3f}{star}")

    if champ_key is not None:
        best = rows[0][1]
        if best["tag"] != CHAMP:
            print(f"\nBest = {short(best['tag'])}  (mean rho {rows[0][0]:.3f})  vs CHAMP mean rho {champ_key:.3f}"
                  f"  Δ={rows[0][0]-champ_key:+.3f}")
        else:
            print(f"\nChampion is #1 (mean rho {champ_key:.3f}).")


if __name__ == "__main__":
    main()
