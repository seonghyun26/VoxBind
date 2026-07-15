#!/usr/bin/env python
"""Generate a static per-target 3Dmol inspection page (the `viz_target92` pattern).

Each page is a self-contained folder under this directory:

    voxbind/notebook/html/viz_target<N>/
        index.html               (copied template, title swapped)
        3Dmol-min.js             (copied viewer lib)
        pocket.pdb  ref.sdf
        samples_<tag>.sdf         (valid generated poses, per model)
        scores.json               (per-mol Vina + meta, index-aligned to the sdf)

Because every asset is fetched with a RELATIVE path, the page works unchanged
behind the ncloud notebook proxy — serve this directory with

    python -m http.server 8092 --directory notebook/html

and open  .../proxy/8092/viz_target<N>/

Data source: each run's samples/<run>/target_*/samples.sdf (coordinates) paired
position-for-position with eval_docking_results.json per_mol (Vina scores). The
pairing is valid-filter → per_mol[i]; verified same-molecule by formula, so it is
robust to RDKit's 3D→SMILES perception differences.

Usage:
    python build_viz_target.py 92 58 74      # frozen-enc target indices
    python build_viz_target.py --all         # every target with x-ray density
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors as rd

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent                 # voxbind/notebook/html
VOXBIND = HERE.parent.parent                           # voxbind/
EXPS = VOXBIND / "exps"
TEMPLATE_DIR = HERE / "viz_target92"                   # reuse its index.html + 3Dmol-min.js

# The three models the comparison page shows, in display order.
RUNS = [
    ("frozenenc_sig0.9", "Frozen-enc σ0.9", "exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"),
    ("vanilla_sig0.9",   "Vanilla σ0.9",    "exps/exp_sig0.9_v2/samples/full_eval_ep350"),
    ("vanilla_sig1.0",   "Vanilla σ1.0",    "exps/exp_sig1.0_350ep/samples/full_eval_ep349"),
]
BASE_TAG = RUNS[0][0]  # frozen-enc is the reference run for target indexing / pocket + ref files


def load_run(root: str) -> dict:
    """Index a run's eval_docking_results.json by target name and by receptor."""
    d = json.load(open(VOXBIND / root / "eval_docking_results.json"))
    by_target = {t["target"]: t for t in d["per_target"]}
    by_receptor = {t["receptor"]: t for t in d["per_target"]}
    return {"by_target": by_target, "by_receptor": by_receptor}


def valid_molblocks(sdf: Path) -> list[tuple[str, str, int]]:
    """Sanitizable + connected mols, in file order (mirrors data.py load_samples).

    Returns (molblock_text, canonical_smiles, formula_hash) so callers can both
    render and cross-check identity against per_mol.
    """
    out: list[tuple[str, str, int]] = []
    for m in Chem.SDMolSupplier(str(sdf), sanitize=True):
        if m is None:
            continue
        try:
            smi = Chem.MolToSmiles(m)
        except Exception:
            continue
        if "." in smi:
            continue
        out.append((Chem.MolToMolBlock(m), smi, rd.CalcMolFormula(m)))
    return out


def formula_of(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return rd.CalcMolFormula(m) if m else None


def find_pocket_pdb(target_dir: Path) -> Path | None:
    return next(iter(sorted(target_dir.glob("*_pocket10.pdb"))), None)


def find_ref_sdf(target_dir: Path) -> Path | None:
    # the reference ligand: a *_lig_tt_*.sdf that isn't the samples file
    cands = [p for p in target_dir.glob("*_lig_tt_*.sdf") if p.name != "samples.sdf"]
    if cands:
        return cands[0]
    return next((p for p in target_dir.glob("*.sdf") if p.name != "samples.sdf"), None)


def parse_ids(receptor: str) -> tuple[str, str]:
    """Pull (receptor_pdb, ligand_code) out of a CrossDocked receptor filename."""
    rec_pdb, lig = "?", "?"
    m = re.search(r"_rec_([0-9a-zA-Z]{4})_([0-9a-zA-Z]{2,3})_lig", receptor)
    if m:
        rec_pdb, lig = m.group(1), m.group(2)
    else:
        m = re.search(r"__([0-9a-zA-Z]{4})_", receptor)
        if m:
            rec_pdb = m.group(1)
    return rec_pdb, lig


def build_target(idx: int, runs_data: dict) -> bool:
    tname = f"target_{idx:02d}"
    base = runs_data[BASE_TAG]["by_target"].get(tname)
    if base is None:
        print(f"  {tname}: not in {BASE_TAG} run — skip")
        return False
    receptor = base["receptor"]
    rec_pdb, lig = parse_ids(receptor)

    base_root = VOXBIND / dict((t, r) for t, _, r in RUNS)[BASE_TAG]
    base_tdir = base_root / tname
    pocket = find_pocket_pdb(base_tdir)
    ref = find_ref_sdf(base_tdir)
    if pocket is None:
        print(f"  {tname}: no pocket pdb — skip")
        return False

    scores: dict[str, list] = {}
    meta: dict[str, dict] = {}
    models: list[str] = []
    sdf_texts: dict[str, str] = {}

    for tag, label, root in RUNS:
        t = runs_data[tag]["by_receptor"].get(receptor)
        if t is None:
            print(f"  {tname}: {tag} has no matching receptor — omitted")
            continue
        tdir = VOXBIND / root / t["target"]
        vb = valid_molblocks(tdir / "samples.sdf")
        pm = t["per_mol"]
        n = min(len(vb), len(pm))
        if len(vb) != len(pm):
            print(f"  {tname}: {tag} valid({len(vb)}) != per_mol({len(pm)}) — pairing first {n}")
        # identity guard: formulas must agree position-for-position
        bad = sum(1 for i in range(n) if vb[i][2] != formula_of(pm[i]["smiles"]))
        if bad:
            print(f"  {tname}: {tag} ⚠ {bad}/{n} formula mismatches — samples.sdf may be out of sync with docking JSON")
        rows = []
        blocks = []
        for i in range(n):
            mb, smi, _ = vb[i]
            p = pm[i]
            blocks.append(mb)
            rows.append({
                "dock":  p.get("vina_dock"),
                "min":   p.get("vina_min"),
                "score": p.get("vina_score"),
                "na":    p.get("n_atoms"),
                "smi":   p.get("smiles"),
            })
        models.append(tag)
        scores[tag] = rows
        # RDKit MolToMolBlock omits the SDF record separator; the viewer splits on
        # `$$$$\n`, so rejoin each molblock with an explicit `$$$$` line.
        sdf_texts[tag] = "".join(
            (mb if mb.endswith("\n") else mb + "\n") + "$$$$\n" for mb in blocks
        )
        meta[tag] = {
            "label": label,
            "mean_dock": t.get("vina_dock"),
            "ref_dock": t.get("ref_vina_dock"),
            "high_aff": t.get("high_affinity"),
            "qed": t.get("qed"),
            "sa": t.get("sa"),
            "div": t.get("diversity"),
            "n": n,
        }

    if not models:
        print(f"  {tname}: no models available — skip")
        return False

    out = HERE / f"viz_target{idx}"
    out.mkdir(exist_ok=True)
    # copy viewer lib
    shutil.copy(TEMPLATE_DIR / "3Dmol-min.js", out / "3Dmol-min.js")
    # data files
    shutil.copy(pocket, out / "pocket.pdb")
    if ref is not None:
        shutil.copy(ref, out / "ref.sdf")
    else:
        (out / "ref.sdf").write_text("")
    for tag in models:
        (out / f"samples_{tag}.sdf").write_text(sdf_texts[tag])
    (out / "scores.json").write_text(json.dumps({"scores": scores, "meta": meta, "models": models}))

    # index.html: copy template, swap the title/header/subtitle
    html = (TEMPLATE_DIR / "index.html").read_text()
    n_ref = meta[models[0]]["n"]
    html = html.replace(
        "<title>VoxBind · target 92 (4z2g) — generated samples</title>",
        f"<title>VoxBind · target {idx} ({rec_pdb}) — generated samples</title>",
    )
    html = re.sub(
        r'<h1>Target 92 · <span style="color:var\(--acc\)">4z2g</span></h1>',
        f'<h1>Target {idx} · <span style="color:var(--acc)">{rec_pdb}</span></h1>',
        html,
    )
    html = re.sub(
        r'<div class="sub">Serralysin \(m6v\) pocket.*?</div>',
        f'<div class="sub">{rec_pdb} ({lig}) pocket — {n_ref} generated ligands/model, Vina-docked</div>',
        html,
    )
    (out / "index.html").write_text(html)

    print(f"  {tname} ({rec_pdb}/{lig}): wrote viz_target{idx}/  "
          f"[{', '.join(f'{t}:{meta[t][chr(110)]}' for t in models)}]")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="frozen-enc target indices, e.g. 92 58 74")
    ap.add_argument("--all", action="store_true", help="build every target in the frozen-enc run")
    args = ap.parse_args()

    runs_data = {tag: load_run(root) for tag, _, root in RUNS}

    if args.all:
        idxs = sorted(int(t.split("_")[1]) for t in runs_data[BASE_TAG]["by_target"])
    else:
        idxs = [int(x) for x in args.targets]
    if not idxs:
        ap.error("give target indices or --all")

    print(f"Building {len(idxs)} target page(s) → {HERE}")
    ok = 0
    for i in idxs:
        if build_target(i, runs_data):
            ok += 1
    print(f"Done: {ok}/{len(idxs)} pages. Serve with the 8092 http.server; open .../proxy/8092/viz_target<N>/")


if __name__ == "__main__":
    main()
