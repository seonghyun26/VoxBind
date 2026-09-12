#!/usr/bin/env python3
"""Merge per-target metrics.json files into one eval_docking_results.json.

fig-qual.ipynb ranks molecules with the score file each method root exposes as
`eval_docking_results.json` (schema: {summary, per_target:[{target, per_mol:[
{idx, smiles, n_atoms, vina_score, vina_min, vina_dock}]}]}). The bundled
VoxBind base (ep350) run only ships per-target `target_XX/metrics.json` files,
whose per-molecule scores live under `samples[].vina.{score_only,minimize,dock}`
-- a different name and schema -- so the notebook shows it as unscored.

This converter reads every `target_*/metrics.json` under a samples directory and
writes the notebook-compatible `eval_docking_results.json` next to them. The
top-level `summary` block is reused from `summary.json` when present. Molecule
matching in the notebook is by canonical SMILES, so `idx` is just the sample's
position in its target's `samples` list.

Usage:
  python build_base_docking_json.py [SAMPLES_DIR] [--receptor-from OURS_JSON] [--dry-run]
Default SAMPLES_DIR: results/task2-drugdesign/VoxBind-base-ep350/samples
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

VINA_MAP = {"vina_score": "score_only", "vina_min": "minimize", "vina_dock": "dock"}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "notebook/figures/qual_utils.py").is_file():
            return p
    return here.parents[2]


def receptor_lookup(ours_json: Path | None) -> dict[str, str]:
    if not ours_json or not ours_json.is_file():
        return {}
    doc = json.loads(ours_json.read_text())
    out = {}
    for row in doc.get("per_target", []):
        if row.get("target") and row.get("receptor"):
            out[row["target"]] = row["receptor"]
    return out


def build(samples_dir: Path, receptors: dict[str, str]) -> dict:
    target_files = sorted(samples_dir.glob("target_*/metrics.json"))
    if not target_files:
        raise FileNotFoundError(f"no target_*/metrics.json under {samples_dir}")

    per_target = []
    n_mols = 0
    for tf in target_files:
        target = tf.parent.name  # e.g. "target_03"
        d = json.loads(tf.read_text())
        agg = d.get("aggregates", {})
        per_mol = []
        for idx, s in enumerate(d.get("samples", [])):
            vina = s.get("vina") or {}
            row = {
                "idx": idx,
                "smiles": s.get("smiles"),
                "n_atoms": s.get("n_atoms"),
            }
            for out_key, in_key in VINA_MAP.items():
                v = vina.get(in_key)
                row[out_key] = float(v) if isinstance(v, (int, float)) else None
            per_mol.append(row)
        n_mols += len(per_mol)
        entry = {
            "target": target,
            "receptor": receptors.get(target),
            "n_total": agg.get("n_total", len(per_mol)),
            "n_valid": agg.get("n_valid"),
            "validity": agg.get("validity"),
            "uniqueness": agg.get("uniqueness"),
            "diversity": agg.get("diversity"),
            "vina_score_mean": agg.get("vina_score_mean"),
            "vina_min_mean": agg.get("vina_min_mean"),
            "vina_dock_mean": agg.get("vina_dock_mean"),
            "high_affinity": agg.get("high_affinity"),
            "per_mol": per_mol,
        }
        per_target.append(entry)

    summary_path = samples_dir / "summary.json"
    summary = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text()).get("summary", {})

    return {
        "_generated_by": "build_base_docking_json.py",
        "_source": "merged per-target metrics.json (vina.{score_only,minimize,dock})",
        "summary": summary,
        "per_target": per_target,
        "_n_targets": len(per_target),
        "_n_molecules": n_mols,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("samples_dir", nargs="?", default=None)
    ap.add_argument("--receptor-from", default=None,
                    help="an Ours-style eval_docking_results.json to borrow receptor names by target")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = repo_root()
    samples_dir = Path(args.samples_dir) if args.samples_dir else (
        root / "results/task2-drugdesign/VoxBind-base-ep350/samples")
    samples_dir = samples_dir.resolve()

    receptor_src = Path(args.receptor_from) if args.receptor_from else (
        root / "results/task2-drugdesign/VoxBind-Ours/samples/eval_docking_results.json")
    receptors = receptor_lookup(receptor_src)

    doc = build(samples_dir, receptors)
    out = samples_dir / "eval_docking_results.json"
    finite = sum(1 for t in doc["per_target"] for m in t["per_mol"]
                 if isinstance(m.get("vina_dock"), float))
    print(f"targets={doc['_n_targets']}  molecules={doc['_n_molecules']}  "
          f"with_finite_vina_dock={finite}  receptors_mapped={sum(bool(t['receptor']) for t in doc['per_target'])}")
    if args.dry_run:
        print(f"[dry-run] would write {out}")
        return
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
