#!/usr/bin/env python3
"""pose_eval_summary.py — coverage + headline numbers for the pose-quality eval.

Reads the per-target metrics.json of the four runs that live on this box (the two
baselines and the two "Ours" arms, the same roots build_posecheck_analysis.py plots)
and reports, per run:

  * COVERAGE — how many pockets carry a real PoseCheck block and a real PoseBusters
    block. "Real" means a measurement, not an in-band {"error": ...}: a timed-out chunk
    leaves the key present but empty, and counting those as done is exactly how the
    2026-07-22 gaps stayed invisible for six weeks.
  * POSECHECK — clashes and strain, MEDIAN over all molecules pooled across pockets.
    Median, not mean: strain is a UFF energy difference with a long positive tail (a
    single failed relax lands in the tens of thousands), so its mean is a tail statistic.
  * POSEBUSTERS — the share of molecules passing all of PoseBusters' dock-mode checks,
    plus the individual checks that fail most often, which is what actually says WHY a
    method's validity rate is what it is.

    /opt/conda/envs/voxbind/bin/python voxbind/scripts/tools/pose_eval_summary.py
    ... --json out.json      # also write the numbers out
"""
import argparse
import collections
import glob
import json
import os
import statistics as st

E = "/home1/irteam/VoxBind/voxbind/exps"
RUNS = [
    ("TargetDiff",       "/home1/irteam/base_drug/eval/targetdiff"),
    ("VoxBind sigma=0.9", f"{E}/_vanilla_ep923/samples/full_eval_ep923"),
    ("Ours v1",          f"{E}/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"),
    ("Ours v2",          f"{E}/samples_reference_receptor_ed_ep350"),
]


def _real_pc(block):
    return isinstance(block, dict) and block.get("clashes") is not None


def _real_pb(block):
    return isinstance(block, dict) and isinstance(block.get("valid"), bool)


def summarise(root):
    dirs = sorted(glob.glob(os.path.join(root, "target_*")))
    out = {
        "pockets": len(dirs), "pc_pockets": 0, "pb_pockets": 0,
        "clashes": [], "strain": [], "pb_valid": [], "fails": collections.Counter(),
        "missing": [],
    }
    for d in dirs:
        p = os.path.join(d, "metrics.json")
        if not os.path.exists(p):
            out["missing"].append(os.path.basename(d))
            continue
        try:
            rows = json.load(open(p)).get("samples", [])
        except (json.JSONDecodeError, OSError):
            out["missing"].append(os.path.basename(d))
            continue
        pc = [r["posecheck"] for r in rows if _real_pc(r.get("posecheck"))]
        pb = [r["posebusters"] for r in rows if _real_pb(r.get("posebusters"))]
        out["pc_pockets"] += bool(pc)
        out["pb_pockets"] += bool(pb)
        if not pc or not pb:
            out["missing"].append(
                f"{os.path.basename(d)}({'no-pc' if not pc else ''}{'no-pb' if not pb else ''})")
        out["clashes"] += [b["clashes"] for b in pc if b.get("clashes") is not None]
        out["strain"] += [b["strain"] for b in pc
                          if isinstance(b.get("strain"), (int, float))]
        out["pb_valid"] += [b["valid"] for b in pb]
        for b in pb:
            for name, passed in (b.get("checks") or {}).items():
                if passed is False:
                    out["fails"][name] += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="also write the numbers to this path")
    args = ap.parse_args()

    print(f"{'run':20s} {'pockets':>8s} {'PoseCheck':>10s} {'PoseBusters':>12s} "
          f"{'clash med':>10s} {'strain med':>11s} {'PB-valid':>9s} {'mols':>7s}")
    payload = {}
    for label, root in RUNS:
        s = summarise(root)
        pb_rate = 100 * sum(s["pb_valid"]) / len(s["pb_valid"]) if s["pb_valid"] else float("nan")
        print(f"{label:20s} {s['pockets']:8d} "
              f"{s['pc_pockets']:>6d}/{s['pockets']:<3d} {s['pb_pockets']:>8d}/{s['pockets']:<3d} "
              f"{st.median(s['clashes']) if s['clashes'] else float('nan'):10.1f} "
              f"{st.median(s['strain']) if s['strain'] else float('nan'):11.1f} "
              f"{pb_rate:8.1f}% {len(s['pb_valid']):7d}")
        payload[label] = {
            "root": root, "pockets": s["pockets"],
            "posecheck_pockets": s["pc_pockets"], "posebusters_pockets": s["pb_pockets"],
            "clashes_median": st.median(s["clashes"]) if s["clashes"] else None,
            "strain_median": st.median(s["strain"]) if s["strain"] else None,
            "pb_valid_rate": (sum(s["pb_valid"]) / len(s["pb_valid"])) if s["pb_valid"] else None,
            "n_mols_pb": len(s["pb_valid"]), "n_mols_pc": len(s["clashes"]),
            "top_failed_checks": s["fails"].most_common(6),
            "incomplete_pockets": s["missing"],
        }
        if s["missing"]:
            print(f"{'':20s} incomplete: {', '.join(s['missing'][:12])}"
                  + (f" (+{len(s['missing']) - 12} more)" if len(s["missing"]) > 12 else ""))

    print("\nmost-failed PoseBusters checks (count of molecules failing each):")
    for label, _ in RUNS:
        top = payload[label]["top_failed_checks"]
        print(f"  {label:20s} " + (", ".join(f"{k}={v}" for k, v in top) if top else "-"))

    if args.json:
        json.dump(payload, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
