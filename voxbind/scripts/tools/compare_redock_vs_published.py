#!/usr/bin/env python
"""Diff a Vina re-dock against the published run it is meant to reproduce.

Written for the results.html Table 4 "Ours" re-check (see
scripts/82_redock_ours_crop_exh16_79.sh), but works for any two
eval_docking_results*.json produced by frozenenc_probes/run_docking_eval.py.

The comparison is asymmetric on purpose, because the pipeline is:
  score_only / minimize  deterministic -> ANY drift is a pipeline defect (wrong vina
                         build, wrong receptor, wrong box) and must be exactly 0.
  dock                   Vina runs with seed=0 = a RANDOM seed, so a re-run only has to
                         agree within search noise; the paired per-molecule delta and its
                         spread are the honest way to report that.

Usage:
  python scripts/tools/compare_redock_vs_published.py OLD.json NEW.json [--md OUT.md]
"""
import argparse
import json
import statistics as st


METRICS = ("vina_score", "vina_min", "vina_dock", "high_affinity",
           "qed", "sa", "diversity", "validity")


def load(path):
    d = json.load(open(path))
    return d.get("summary", {}), {e["target"]: e for e in d.get("per_target", [])}


def agg(per, key):
    vals = [e[key] for e in per.values() if e.get(key) is not None]
    return st.mean(vals) if vals else None


def paired_mol_deltas(old, new):
    """Per-molecule old->new deltas, matched on (target, molecule index).

    Matching on index alone would silently pair different molecules if a re-run
    dropped one, so the SMILES is checked too and mismatches are counted, not hidden.
    """
    out = {k: [] for k in ("vina_score", "vina_min", "vina_dock")}
    mismatched = 0
    for t, ne in new.items():
        oe = old.get(t)
        if oe is None:
            continue
        obyi = {m["idx"]: m for m in oe.get("per_mol", [])}
        for nm in ne.get("per_mol", []):
            om = obyi.get(nm["idx"])
            if om is None:
                continue
            if om.get("smiles") != nm.get("smiles"):
                mismatched += 1
                continue
            for k in out:
                if om.get(k) is not None and nm.get(k) is not None:
                    out[k].append(nm[k] - om[k])
    return out, mismatched


def fmt(x, nd=4):
    return "  n/a  " if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--md", default=None, help="also write a markdown report here")
    args = ap.parse_args()

    _, old = load(args.old)
    _, new = load(args.new)
    shared = sorted(set(old) & set(new))

    lines = []
    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say(f"published : {args.old}")
    say(f"re-run    : {args.new}")
    say(f"targets   : old {len(old)}  new {len(new)}  compared {len(shared)}")
    if set(old) - set(new):
        say(f"  missing from re-run: {sorted(set(old) - set(new))}")
    say()

    o_sub = {t: old[t] for t in shared}
    n_sub = {t: new[t] for t in shared}
    say("## Pocket-averaged summary (mean over compared targets)")
    say()
    say(f"| metric | published | re-run | delta |")
    say(f"|---|---|---|---|")
    for k in METRICS:
        a, b = agg(o_sub, k), agg(n_sub, k)
        d = None if (a is None or b is None) else b - a
        say(f"| {k} | {fmt(a)} | {fmt(b)} | {fmt(d)} |")
    say()

    deltas, mismatched = paired_mol_deltas(o_sub, n_sub)
    say("## Per-molecule paired deltas (re-run minus published)")
    say()
    if mismatched:
        say(f"WARNING: {mismatched} molecule(s) skipped on SMILES mismatch")
        say()
    say("| mode | n | exact match | mean | median | sd | max abs |")
    say("|---|---|---|---|---|---|---|")
    for k in ("vina_score", "vina_min", "vina_dock"):
        d = deltas[k]
        if not d:
            say(f"| {k} | 0 | - | - | - | - | - |")
            continue
        ex = sum(1 for x in d if abs(x) < 1e-9)
        sd = st.pstdev(d) if len(d) > 1 else 0.0
        say(f"| {k} | {len(d)} | {ex}/{len(d)} ({100*ex/len(d):.1f}%) | "
            f"{st.mean(d):+.4f} | {st.median(d):+.4f} | {sd:.4f} | "
            f"{max(abs(x) for x in d):.3f} |")
    say()

    det_bad = sum(1 for k in ("vina_score", "vina_min")
                  for x in deltas[k] if abs(x) >= 1e-9)
    say("## Verdict")
    say()
    if det_bad == 0 and deltas["vina_score"]:
        say("- score_only and minimize reproduce EXACTLY -> receptor prep, docking box, "
            "scoring function and Vina build are identical to the published run.")
    else:
        say(f"- FAIL: {det_bad} deterministic-mode value(s) drifted. The two runs did NOT "
            "use the same pipeline; do not compare their dock numbers.")
    if deltas["vina_dock"]:
        d = deltas["vina_dock"]
        say(f"- dock differs by {st.mean(d):+.3f} kcal/mol on average "
            f"(sd {st.pstdev(d):.3f}), which is Vina's random-seed search noise: "
            "seed=0 means a random seed, so dock is not bit-reproducible by design.")

    if args.md:
        open(args.md, "w").write("\n".join(lines) + "\n")
        print(f"\nwrote {args.md}", flush=True)


if __name__ == "__main__":
    main()
