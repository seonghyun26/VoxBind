"""CoDE (VoxBind + CDG) vs CoE (VoxBind + C) — one table, straight from the bundle.

Both rows come from results/task2-drugdesign/<arm>/eval/<evaluation>/results.json, i.e. the
same files collect_task2_eval.py writes and the reports read, so the table cannot drift from
the bundle.

KEY LAYOUT GOTCHA: the collector keys a native arm's Vina set "density79" (collect_native's
eval_docking_results_full79.json fallback) while sample_quality / posecheck / posebusters key
theirs "all". Reading only "all" is how vina_* silently came back None once already.

CONVENTIONS (voxbind-vina-aggregation-convention): Vina mean/median pool every molecule;
high affinity is a per-pocket share, then averaged over pockets with >= 50 docked molecules.

    python voxbind/scripts/tools/compare_code_vs_coe.py [--arms VoxBind-Ours VoxBind+CoE]
"""
import argparse
import json
import os

T2 = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                  "results", "task2-drugdesign")
LABEL = {"VoxBind-Ours": "CoDE (VoxBind + CDG)", "VoxBind+CoE": "CoE (VoxBind + C)"}


def ev(arm, name):
    p = os.path.normpath(os.path.join(T2, arm, "eval", name, "results.json"))
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    ps = d.get("pocket_sets") or {}
    return ps.get("all") or ps.get("density79") or d


def fmt(v, nd=2, pct=False):
    if v is None:
        return "—"
    return f"{100 * v:.1f}" if pct else f"{v:.{nd}f}"


def pair(block, key, nd=2):
    b = (block.get(key) or {})
    return f"{fmt(b.get('mean'), nd)} / {fmt(b.get('median'), nd)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["VoxBind-Ours", "VoxBind+CoE"])
    a = ap.parse_args()

    rows = []
    for arm in a.arms:
        q, v = ev(arm, "sample_quality"), ev(arm, "vina_docking")
        pc, pb = ev(arm, "posecheck"), ev(arm, "posebusters")
        n_p = q.get("n_pockets") or v.get("n_pockets")
        n_m = q.get("n_molecules") or v.get("n_molecules")
        rows.append({
            "arm": LABEL.get(arm, arm),
            "pockets": n_p,
            "mols": n_m,
            "per_pocket": (n_m / n_p) if (n_p and n_m) else None,
            "vina_pockets": v.get("n_pockets"),
            "vina_mols": v.get("n_molecules"),
            "score": pair(v, "vina_score"),
            "min": pair(v, "vina_min"),
            "dock": pair(v, "vina_dock"),
            "ha": fmt(v.get("high_affinity"), pct=True),
            "qed": fmt(q.get("qed_mean")),
            "sa": fmt(q.get("sa_mean")),
            "div": fmt(q.get("diversity")),
            "atoms": fmt(q.get("n_atoms_mean"), 1),
            "sim_ref": fmt(q.get("sim_to_ref_mean")),
            "pb": fmt(pb.get("pb_valid_rate"), pct=True),
            "clash": fmt(pc.get("clashes_median"), 1),
            "strain": fmt(pc.get("strain_median"), 1),
        })

    hdr = ["metric"] + [r["arm"] for r in rows]
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "---|" * len(hdr)]

    def row(label, key):
        lines.append("| " + " | ".join([label] + [str(r[key]) for r in rows]) + " |")

    row("pockets", "pockets")
    row("molecules", "mols")
    lines.append("| molecules / pocket | " +
                 " | ".join(fmt(r["per_pocket"], 1) for r in rows) + " |")
    row("Vina Score ↓ (mean / median)", "score")
    row("Vina Min ↓", "min")
    row("Vina Dock ↓", "dock")
    row("High affinity ↑ (%)", "ha")
    row("QED ↑", "qed")
    row("SA ↑", "sa")
    row("Diversity ↑", "div")
    row("Atoms / mol", "atoms")
    row("Tanimoto to ref", "sim_ref")
    row("PoseBusters valid ↑ (%)", "pb")
    row("Clashes ↓ (median)", "clash")
    row("Strain ↓ (median)", "strain")

    print("\n".join(lines))
    print()
    for r in rows:
        print(f"- {r['arm']}: Vina over {r['vina_pockets']} pockets / {r['vina_mols']} molecules")
        # A docking run that covered only part of the arm still produces a full-looking row;
        # 2026-09-14 the driver lost 72 of 79 targets to one Vina abort and exited 0. Say so
        # loudly rather than letting the number be read as the arm's Vina result.
        if r["vina_pockets"] and r["pockets"] and r["vina_pockets"] < r["pockets"]:
            print(f"  !! PARTIAL: Vina covers {r['vina_pockets']}/{r['pockets']} pockets "
                  f"({r['vina_mols']}/{r['mols']} molecules) — NOT comparable yet")
    print("- Vina: full CrossDocked receptor, exhaustiveness 32, TargetDiff VinaDockingTask "
          "(vina 1.2.2); mean/median pool molecules, high affinity averages per-pocket shares.")


if __name__ == "__main__":
    main()
