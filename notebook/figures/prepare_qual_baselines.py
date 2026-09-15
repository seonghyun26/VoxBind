"""Stage the downloaded baseline meta bundles for the qualitative viewer.

Preserve bundle order and coordinates; verify pocket identity and score SMILES.
The generated SDFs and score indexes live under .cache/fig-qual/baselines.
"""

import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from rdkit import Chem, rdBase


BASELINE_STEMS = {
    "AR": "AR",
    "Pocket2Mol": "Pocket2Mol",
    "DiffSBDD": "DiffSBDD",
    "DecompDiff": "DecompDiff_ref_prior",
    "FuncBind": "FuncBind",
}


def _identity(mol):
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False)


def prepare_baselines(repo, catalog, methods):
    import torch

    targets = catalog.common_targets(["VoxBind", "VoxBind + Ours"])
    for method in methods:
        if method not in BASELINE_STEMS:
            continue
        source = repo / "results/task2-drugdesign" / method
        stem = BASELINE_STEMS[method]
        bundles = [source / "samples/meta" / f"{stem}{part}.pt"
                   for part in ("", "_part2", "_gap")]
        bundles = [path for path in bundles if path.is_file()]
        if not bundles:
            if catalog.files.get(method):
                continue
            raise FileNotFoundError(
                f"Missing {method} samples. Run: "
                "bash results/dropbox_pull_baselines.sh --task task2-drugdesign")
        score_csv = source / "eval/vina_docking/per_molecule.csv"
        inputs = [*bundles, *([score_csv] if score_csv.is_file() else [])]
        for target in targets:
            inputs.extend(catalog.reference_paths("VoxBind + Ours", target))
        signature = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in inputs]
        digest = hashlib.sha256(json.dumps([1, targets, signature]).encode()).hexdigest()
        destination = repo / ".cache/fig-qual/baselines" / method
        manifest = destination / "source.json"
        if (manifest.is_file() and json.loads(manifest.read_text()).get("digest") == digest
                and all((destination / target / "samples.sdf").is_file() for target in targets)):
            continue

        print(f"Preparing {method} qualitative samples...", flush=True)
        merged = [[] for _ in range(100)]
        for path in bundles:
            for index, entries in enumerate(torch.load(path, map_location="cpu", weights_only=False)):
                merged[index].extend(entries)
        scores = {}
        if score_csv.is_file():
            with score_csv.open() as handle:
                for record in csv.DictReader(handle):
                    key = (record["target"], int(record["mol_idx"]))
                    if key in scores:
                        raise ValueError(f"Duplicate score identity: {method}, {key}")
                    scores[key] = record

        rows = []
        count = 0
        for target in targets:
            reference, pocket = catalog.reference_paths("VoxBind + Ours", target)
            entries = merged[int(target.split("_")[1])]
            if not entries:
                raise ValueError(f"No {method} samples for {target}")
            directory = destination / target
            directory.mkdir(parents=True, exist_ok=True)
            scored = []
            with Chem.SDWriter(str(directory / "samples.sdf")) as writer, rdBase.BlockLogs():
                for index, entry in enumerate(entries):
                    identity = entry["ligand_filename"].replace("/", "__")
                    if identity != reference.name:
                        raise ValueError(f"Pocket identity mismatch: {method}, {target}, {identity}")
                    mol = entry.get("mol")
                    if mol is None or not mol.GetNumConformers() or not mol.GetNumHeavyAtoms():
                        continue
                    if not np.isfinite(mol.GetConformer().GetPositions()).all():
                        continue
                    mol = Chem.Mol(mol)
                    mol.SetIntProp("source_mol_idx", index)
                    writer.write(mol)
                    count += 1
                    score = scores.get((target, index))
                    if score is None:
                        continue
                    score_mol = Chem.MolFromSmiles(score["smiles"])
                    if (score["ligand_filename"].replace("/", "__") != reference.name
                            or score_mol is None or _identity(score_mol) != _identity(mol)):
                        raise ValueError(f"Score identity mismatch: {method}, {target}, {index}")
                    values = {}
                    for key in ("vina_score", "vina_min", "vina_dock"):
                        value = float(score[key]) if score.get(key) else None
                        values[key] = value if value is not None and np.isfinite(value) else None
                    scored.append(dict(idx=index, smiles=score["smiles"], **values))
            shutil.copy2(reference, directory / reference.name)
            shutil.copy2(pocket, directory / pocket.name)
            rows.append(dict(target=target, receptor=str(pocket), per_mol=scored))
        (destination / "eval_docking_results.json").write_text(json.dumps({
            "source": str(score_csv), "per_target": rows,
        }))
        manifest.write_text(json.dumps({"digest": digest, "source": str(source),
                                       "targets": targets, "molecules": count}, indent=2))
        print(f"  {method}: {len(targets)} pockets, {count} molecules", flush=True)
