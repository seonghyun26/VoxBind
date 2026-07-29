#!/usr/bin/env python3
"""Summarise gen_stats.json across a VoxBind sampling run.

Each pocket's `sample_molecules` writes <target_dir>/gen_stats.json recording how
many voxel->mol attempts were made (n_generated_total), how many unique valid
molecules were kept (n_valid_unique), and how many attempts were rejected as
invalid or as duplicate SMILES. This aggregates them so you can see the
duplication / invalidity rate of a whole run at a glance.

Usage:
    python scripts/tools/gen_stats_summary.py <sample_dir>

<sample_dir> is the directory that holds the per-pocket target_* folders
(e.g. exps/exp_sig0.9/samples/res). Only runs sampled AFTER the gen_stats
instrumentation was added will have the files (rejected mols are not saved, so
duplicates cannot be recovered from an old run's SDFs).
"""
import glob
import json
import os
import sys


def main(root: str) -> None:
    files = sorted(glob.glob(os.path.join(root, "target_*", "gen_stats.json")))
    if not files:
        print(f"no target_*/gen_stats.json under {root!r} — re-sample with the "
              f"instrumented sample_molecules (old runs discarded rejected mols).")
        return

    keys = ("n_generated_total", "n_valid_unique", "n_invalid", "n_duplicate")
    tot = {k: 0 for k in keys}
    capped = []
    print(f"{'pocket':<16}{'gen':>6}{'kept':>6}{'dup':>6}{'inval':>7}"
          f"{'valid%':>8}{'uniq%':>7}")
    print("-" * 56)
    for f in files:
        d = json.load(open(f))
        name = os.path.basename(os.path.dirname(f))
        for k in keys:
            tot[k] += d.get(k, 0)
        if d.get("hit_cap"):
            capped.append(name)
        print(f"{name:<16}{d['n_generated_total']:>6}{d['n_valid_unique']:>6}"
              f"{d['n_duplicate']:>6}{d['n_invalid']:>7}"
              f"{100 * d['validity']:>7.1f}{100 * d['uniqueness_raw']:>7.1f}")

    g, k, inv, dup = (tot["n_generated_total"], tot["n_valid_unique"],
                      tot["n_invalid"], tot["n_duplicate"])
    proc = g - inv
    print("-" * 56)
    print(f"{'TOTAL':<16}{g:>6}{k:>6}{dup:>6}{inv:>7}"
          f"{100 * proc / g if g else 0:>7.1f}{100 * k / proc if proc else 0:>7.1f}")
    print(
        f"\n{len(files)} pockets | generated {g} attempts -> kept {k} unique valid | "
        f"{dup} duplicates ({100 * dup / g if g else 0:.1f}% of attempts), "
        f"{inv} invalid ({100 * inv / g if g else 0:.1f}%)"
    )
    if capped:
        print(f"hit the 500-attempt cap (under-filled, high duplication): "
              f"{len(capped)} pockets -> {', '.join(capped)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
