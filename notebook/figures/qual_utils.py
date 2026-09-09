"""Data loading and fig1-style interactive panels for fig-qual.ipynb."""

import ast
from copy import deepcopy
from functools import lru_cache
import html
import hashlib
import tempfile
import json
from pathlib import Path
import re

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display
from rdkit import Chem, rdBase


def natural_key(path):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", str(path))]


def fig1_style(path):
    """Reuse only fig1's molecular meshes/constants; never execute its data/UI cells."""
    names = {
        "LIGAND_COLOR", "POCKET_COLOR", "BALL_RADIUS_A", "STICK_RADIUS_A",
        "SPHERE_LATITUDES", "SPHERE_LONGITUDES", "CYLINDER_SIDES", "DEFAULT_CAMERA",
        "mesh_trace", "atom_color", "append_uv_sphere", "atom_trace",
        "append_cylinder", "bond_trace",
    }
    nodes = []
    notebook = json.loads(Path(path).read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                nodes.append(node)
            elif isinstance(node, ast.Assign) and all(
                isinstance(t, ast.Name) and t.id in names for t in node.targets
            ):
                nodes.append(node)
    namespace = {"np": np, "go": go}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    missing = names - namespace.keys()
    if missing:
        raise ValueError(f"fig1 style definitions missing: {sorted(missing)}")
    original_mesh_trace = namespace["mesh_trace"]

    def uniform_mesh_trace(vertices, faces, vertex_colors, *args, **kwargs):
        # fig1 uses one color per role. A scalar color avoids repeating it at every vertex.
        if vertex_colors and len(set(vertex_colors)) == 1:
            trace = original_mesh_trace(vertices, faces, [], *args, **kwargs)
            trace.vertexcolor = None
            trace.color = vertex_colors[0]
            return trace
        return original_mesh_trace(vertices, faces, vertex_colors, *args, **kwargs)

    namespace["mesh_trace"] = uniform_mesh_trace
    return namespace


def method_specs(repo, baseline):
    exps = repo / "voxbind/exps"
    vanilla = exps / "_vanilla_ep923/samples/full_eval_ep923"
    if not vanilla.is_dir():
        bundled_vanilla = repo / "results/task2-drugdesign/VoxBind-base-ep350/samples"
        if bundled_vanilla.is_dir():
            vanilla = bundled_vanilla
    ours = exps / "voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350"
    ours_layout = "eval"
    ours_extra = {}
    if not ours.is_dir():
        ours = repo / "voxbind/model_zoo/generated_samples/ours_v1_frozenenc_atomblob7_v2p1_sig0.9_ep350/samples"
    if not ours.is_dir():
        bundled_ours = repo / "results/task2-drugdesign/VoxBind-Ours/samples"
        if bundled_ours.is_dir():
            ours = bundled_ours
            ours_extra["energy_file"] = ours / "eval_docking_results_full79.json"
    if not ours.is_dir():
        mcp_results = repo / "results/task3-mcp/Ours-receptorED/samples"
        if mcp_results.is_dir():
            ours = mcp_results
            ours_layout = "mcp_results"
            ours_extra["reference_root"] = (
                repo / "voxbind/dataset/data/pdbbind/structures/pbpp-2020"
            )
    return {
        "VoxBind": dict(root=vanilla, layout="eval"),
        "VoxBind + Ours": dict(root=ours, layout=ours_layout, **ours_extra),
        "TargetDiff": dict(root=baseline / "eval/targetdiff", layout="eval"),
        "FuncBind": dict(root=repo.parent / "funcbind/artifacts/reproduction/crossdocked/paper_run", layout="eval_shards"),
        "AR": dict(root=baseline / "samples/ar", layout="sweep"),
        "Pocket2Mol": dict(root=baseline / "samples/pocket2mol", layout="sweep"),
        "VoxBind + Ours v2": dict(root=exps / "samples_reference_receptor_ed_ep350", layout="eval"),
        "VoxBind σ=1.0": dict(root=exps / "exp_sig1.0_350ep/samples/full_eval_ep349", layout="eval"),
        "DecompDiff": dict(root=baseline / "samples/decompdiff", layout="sweep"),
    }


class Catalog:
    def __init__(self, specs):
        self.specs = specs
        self.files = {}
        for method, spec in specs.items():
            found = {}
            root = Path(spec["root"])
            if spec["layout"] in ("eval", "eval_shards"):
                pattern = "gpu*/samples/target_*/samples.sdf" if spec["layout"] == "eval_shards" else "target_*/samples.sdf"
                for p in sorted(root.glob(pattern)):
                    if p.stat().st_size:
                        if p.parent.name in found:
                            raise ValueError(f"Duplicate target across shards: {method}, {p.parent.name}")
                        found[p.parent.name] = [p]
            elif spec["layout"] == "sweep":
                # id_N == target_N in the server's common CrossDocked split;
                # see base_drug/export_targetdiff_sdf.py. Do not pool reruns.
                for p in sorted(root.glob("id_*"), key=natural_key):
                    if not re.fullmatch(r"id_\d+", p.name):
                        continue
                    runs = [r for r in sorted(p.glob("*/SDF")) if any(r.glob("*.sdf"))]
                    if runs:
                        files = [f for f in sorted(runs[-1].glob("*.sdf"), key=natural_key)
                                 if f.stat().st_size]
                        if files:
                            found[f"target_{int(p.name[3:]):02d}"] = files
            elif spec["layout"] == "mcp_results":
                # Exported MCP bundles use mcpp_<pdb>_<run>/pooled_<n>.sdf.
                for run in sorted(root.glob("mcpp_*"), key=natural_key):
                    match = re.match(r"mcpp_([A-Za-z0-9]{4})(?:_|$)", run.name)
                    files = [p for p in sorted(run.glob("pooled_*.sdf"), key=natural_key)
                             if p.stat().st_size]
                    if not match or not files:
                        continue
                    target = match.group(1).lower()
                    if target in found:
                        raise ValueError(f"Duplicate MCP result target: {method}, {target}")
                    found[target] = [files[-1]]
            else:
                raise ValueError(f"Unknown layout: {spec['layout']}")
            self.files[method] = found

    def common_targets(self, methods):
        return sorted(set.intersection(*(set(self.files[m]) for m in methods)), key=natural_key)

    def coverage(self):
        return [dict(method=m, pockets_with_sdf=len(v), root=str(self.specs[m]["root"]))
                for m, v in self.files.items()]

    @lru_cache(maxsize=128)
    def records(self, method, target):
        records, invalid = [], []
        with rdBase.BlockLogs():
            for path in self.files[method][target]:
                for index, mol in enumerate(Chem.SDMolSupplier(str(path), removeHs=False)):
                    if mol is None or not mol.GetNumConformers() or not mol.GetNumHeavyAtoms():
                        invalid.append(f"{path.name} [{index}]")
                        continue
                    if not np.isfinite(mol.GetConformer().GetPositions()).all():
                        invalid.append(f"{path.name} [{index}] (nonfinite coordinates)")
                        continue
                    records.append(dict(path=path, index=index, mol=mol,
                                        label=f"{path.name} [{index}] · {mol.GetNumHeavyAtoms()} atoms"))
        if not records:
            raise ValueError(f"No readable 3D ligands: {method}, {target}")
        return records, invalid


    @lru_cache(maxsize=128)
    def ranked_records(self, method, target, energy_key="vina_dock"):
        """Match cached evaluation scores to unique molecular identities, not filtered indices."""
        if energy_key not in ("vina_score", "vina_min", "vina_dock"):
            raise ValueError(f"Unsupported energy field: {energy_key}")
        raw, invalid = self.records(method, target)
        records = [dict(r, energy=None, energy_key=energy_key, energy_source=None) for r in raw]
        score_file = Path(self.specs[method].get(
            "energy_file", Path(self.specs[method]["root"]) / "eval_docking_results.json"))
        if not score_file.is_file():
            return records, invalid
        results = self._energy_results(score_file)
        row = results.get(target)
        if row is None:
            return records, invalid

        def identity(mol):
            return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False)

        by_identity = {}
        for record in records:
            by_identity.setdefault(identity(record["mol"]), []).append(record)
        scored = {}
        for item in row.get("per_mol", []):
            value = item.get(energy_key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                continue
            mol = Chem.MolFromSmiles(item.get("smiles", ""))
            if mol is not None and mol.GetNumAtoms():
                scored.setdefault(identity(mol), []).append(item)
        for key, items in scored.items():
            matches = by_identity.get(key, [])
            # Duplicate structures may have different poses. Do not guess their score mapping.
            if len(items) != 1 or len(matches) != 1:
                continue
            record, item = matches[0], items[0]
            record.update(energy=float(item[energy_key]), energy_source=str(score_file),
                          energy_eval_index=item.get("idx"),
                          energy_receptor=row.get("receptor"))
            record["label"] += f" · {record['energy']:.3f} kcal/mol"
        records.sort(key=lambda r: (r["energy"] is None, r["energy"] if r["energy"] is not None else 0))
        return records, invalid


    def _score_file(self, method):
        return Path(self.specs[method].get(
            "energy_file", Path(self.specs[method]["root"]) / "eval_docking_results.json"))

    def selection_index(self, method, target, energy_key="vina_dock"):
        """Persist score-to-SDF mappings once; subsequent loads do not sanitize all molecules."""
        score_file = self._score_file(method)
        if not score_file.is_file():
            return None
        paths = [*self.files[method][target], score_file]
        signature = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        payload = json.dumps([2, rdBase.rdkitVersion, energy_key, signature], sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        cache_dir = Path(__file__).resolve().parents[2] / ".cache/fig-qual/selection-index"
        cache_file = cache_dir / f"{digest}.json"
        if cache_file.is_file():
            return json.loads(cache_file.read_text())
        print(f"Indexing {method} / {target} scores once...", flush=True)
        # A new fingerprint must not reuse stale in-memory source data.
        self.records.cache_clear()
        self.ranked_records.cache_clear()
        self._energy_results.cache_clear()
        records, invalid = self.ranked_records(method, target, energy_key)
        entries = []
        for r in records:
            entry = {k: v for k, v in r.items() if k != "mol"}
            entry["path"] = str(r["path"])
            entries.append(entry)
        index = dict(records=entries, invalid=invalid)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=cache_dir, suffix=".tmp",
                                         delete=False) as handle:
            json.dump(index, handle)
            temp_path = Path(handle.name)
        temp_path.replace(cache_file)
        return index

    @staticmethod
    @lru_cache(maxsize=256)
    def _read_one(path, record_index, size, modified_ns):
        # size/mtime are cache keys so edited source SDFs cannot return stale molecules.
        with rdBase.BlockLogs():
            supplier = Chem.SDMolSupplier(str(path), removeHs=False)
            try:
                mol = supplier[record_index]
            except IndexError:
                return None
        if (mol is None or not mol.GetNumConformers() or not mol.GetNumHeavyAtoms()
                or not np.isfinite(mol.GetConformer().GetPositions()).all()):
            return None
        return mol

    def initial_record(self, method, target, energy_key="vina_dock", saved=None):
        """Read only the selected molecule; unscored methods stop at the first readable record."""
        index = self.selection_index(method, target, energy_key)
        if index is not None:
            entries = index["records"]
            if saved is not None:
                entries = [r for r in entries if Path(r["path"]).resolve() == Path(saved["path"]).resolve()
                           and r["index"] == saved["record_index"]]
            for entry in entries:
                path = Path(entry["path"])
                stat = path.stat()
                mol = self._read_one(str(path), entry["index"], stat.st_size, stat.st_mtime_ns)
                if mol is not None:
                    return dict(entry, path=path, mol=mol), index["invalid"]
            raise ValueError(f"No readable indexed selection: {method}, {target}")
        invalid = []
        for path in self.files[method][target]:
            if saved is not None and path.resolve() != Path(saved["path"]).resolve():
                continue
            with rdBase.BlockLogs():
                supplier = Chem.SDMolSupplier(str(path), removeHs=False)
                indices = [saved["record_index"]] if saved else range(len(supplier))
                for i in indices:
                    stat = path.stat()
                    mol = self._read_one(str(path), i, stat.st_size, stat.st_mtime_ns)
                    if mol is None:
                        invalid.append(f"{path.name} [{i}]")
                        continue
                    return dict(path=path, index=i, mol=mol, energy=None,
                                energy_key=energy_key, energy_source=None,
                                label=f"{path.name} [{i}] · {mol.GetNumHeavyAtoms()} atoms"), invalid
        raise ValueError(f"No readable 3D selection: {method}, {target}")

    @staticmethod
    @lru_cache(maxsize=16)
    def _energy_results(path):
        document = json.loads(Path(path).read_text())
        return {r["target"]: r for r in document.get("per_target", [])}

    def reference_paths(self, method, target):
        spec = self.specs[method]
        if spec["layout"] == "mcp_results":
            folder = Path(spec["reference_root"]) / target.lower()
            ligand = folder / f"{target.lower()}_ligand.sdf"
            pocket = folder / f"{target.lower()}_pocket.pdb"
            if not ligand.is_file() or not pocket.is_file():
                raise ValueError(f"Missing MCP reference ligand or pocket in {folder}")
            return ligand, pocket
        folder = self.files[method][target][0].parent
        ligands = [p for p in sorted(folder.glob("*.sdf")) if p.name != "samples.sdf"]
        pockets = sorted(folder.glob("*_pocket10.pdb"))
        if len(ligands) != 1 or len(pockets) != 1:
            raise ValueError(f"Expected one reference SDF and pocket10 PDB in {folder}")
        return ligands[0], pockets[0]

    @lru_cache(maxsize=64)
    def reference(self, target):
        ligand_path, pocket_path = self.reference_paths("VoxBind + Ours", target)
        mol = next(iter(Chem.SDMolSupplier(str(ligand_path), removeHs=False)), None)
        pocket = Chem.MolFromPDBFile(str(pocket_path), sanitize=False, removeHs=False,
                                     proximityBonding=True)
        if mol is None or pocket is None:
            raise ValueError(f"Cannot parse reference for {target}")
        center = geometry(mol)[0].mean(axis=0)
        return dict(mol=mol, pocket=pocket, center=center,
                    ligand_path=ligand_path, pocket_path=pocket_path,
                    label=target + " · " + ligand_path.stem.split("__")[-1][:4].upper())

    def validate_target(self, target, methods):
        """Fail on mismatched references, rather than silently aligning unrelated pockets."""
        ref = self.reference(target)
        canonical = ref["ligand_path"].stem.split("__")[-1].removesuffix("_ref")
        for method in methods:
            if self.specs[method]["layout"] in ("eval", "eval_shards"):
                ligand_path, pocket_path = self.reference_paths(method, target)
                identity = ligand_path.stem.split("__")[-1].removesuffix("_ref")
                if identity != canonical:
                    raise ValueError(f"Reference identity mismatch: {target}, {method}")
                other = next(iter(Chem.SDMolSupplier(str(ligand_path), removeHs=False)), None)
                other_pocket = Chem.MolFromPDBFile(str(pocket_path), sanitize=False,
                                                  removeHs=False, proximityBonding=True)
                for expected, actual in ((ref["mol"], other), (ref["pocket"], other_pocket)):
                    if actual is None:
                        raise ValueError(f"Unreadable reference: {target}, {method}")
                    xyz, elements, _ = geometry(expected)
                    other_xyz, other_elements, _ = geometry(actual)
                    if (xyz.shape != other_xyz.shape or not np.array_equal(elements, other_elements)
                            or not np.allclose(xyz, other_xyz, atol=0.02)):
                        raise ValueError(f"Reference coordinate mismatch: {target}, {method}")
            elif self.specs[method]["layout"] == "sweep":
                info = self.files[method][target][0].parent.parent / "pocket_info.txt"
                if info.exists() and canonical not in info.read_text():
                    raise ValueError(f"Pocket identity mismatch: {info}")


def geometry(mol, center=None, extent=None):
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    if center is not None:
        coords = coords - center
    elements = np.array([a.GetSymbol() for a in mol.GetAtoms()])
    keep = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
    if extent is not None:
        keep &= np.all(np.abs(coords) <= extent + 0.15, axis=1)
    indices = np.flatnonzero(keep)
    remap = np.full(len(coords), -1, dtype=int)
    remap[indices] = np.arange(len(indices))
    bonds = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()],
                     dtype=int).reshape(-1, 2)
    bonds = remap[bonds[np.all(keep[bonds], axis=1)]]
    return coords[keep], elements[keep], bonds


def scene_name(index):
    return "scene" if index == 0 else f"scene{index + 1}"


def make_table(rows):
    if not rows:
        return widgets.HTML("No entries")
    keys = list(rows[0])
    esc = lambda value: html.escape(str(value))
    table = '<table style="border-collapse:collapse;text-align:left;font-size:12px">'
    table += "<tr>" + "".join(f"<th style='padding:5px'>{esc(k)}</th>" for k in keys) + "</tr>"
    for row in rows:
        table += "<tr>" + "".join(f"<td style='padding:5px;vertical-align:top'>{esc(row[k])}</td>"
                                  for k in keys) + "</tr>"
    return widgets.HTML(table + "</table>")


class QualitativeGrid:
    def __init__(self, catalog, style, methods, targets, panel_px=310, pocket_extent=8.0,
                 selections=None, cameras=None, energy_key="vina_dock"):
        self.catalog, self.style, self.methods = catalog, style, list(methods)
        self.energy_key = energy_key
        self.targets = list(targets)
        self.panel_px, self.pocket_extent = panel_px, pocket_extent
        self.selections = deepcopy(selections or {})
        self.cameras = deepcopy(cameras or {})
        self.columns = ["Reference", *self.methods]
        self.rows = []
        common = catalog.common_targets(methods)
        if len(set(targets)) != len(targets):
            raise ValueError("Choose distinct pockets for the rows.")
        for target in targets:
            if target not in common:
                raise ValueError(f"{target} is not available for all selected methods")
        for row_index, target in enumerate(targets):
            state = dict(target=target, syncing=False)
            target_menu = widgets.Dropdown(options=[(t, t) for t in common],
                                          value=target, description=f"Pocket {row_index + 1}",
                                          layout=widgets.Layout(width="320px"))
            controls, output, error = widgets.HBox(), widgets.Output(), widgets.HTML()
            state.update(menu=target_menu, controls=controls, output=output, error=error)
            self.rows.append(state)
            self._load_row(row_index)

            def change_target(change, row=row_index):
                old = self.rows[row]["target"]
                if change["new"] == old:
                    return
                try:
                    if change["new"] in [s["target"] for i, s in enumerate(self.rows) if i != row]:
                        raise ValueError("Select a different pocket for each row.")
                    self.rows[row]["target"] = change["new"]
                    self._load_row(row)
                    self.rows[row]["error"].value = ""
                except Exception as exc:
                    self.rows[row]["target"] = old
                    self.rows[row]["menu"].value = old
                    self.rows[row]["error"].value = html.escape(str(exc))

            target_menu.observe(change_target, names="value")
        self.widget = widgets.VBox([
            widgets.VBox([s["menu"], s["controls"], s["error"], s["output"]],
                         layout=widgets.Layout(overflow="auto", width="100%")) for s in self.rows
        ])

    def close(self):
        """Release old widgets before rebuilding the comparison cell."""
        seen = set()

        def close_widget(widget):
            if id(widget) in seen:
                return
            seen.add(id(widget))
            for child in getattr(widget, "children", ()):
                close_widget(child)
            widget.close()

        for state in self.rows:
            close_widget(state["figure"])
        close_widget(self.widget)

    def _model_traces(self, geom, role):
        coords, elements, bonds = geom
        color = self.style["LIGAND_COLOR" if role == "Ligand" else "POCKET_COLOR"]
        return [self.style["bond_trace"](coords, elements, bonds, role, role, color),
                self.style["atom_trace"](coords, elements, role, role, color)]

    def _load_row(self, row_index):
        state = self.rows[row_index]
        target = state["target"]
        self.catalog.validate_target(target, self.methods)
        ref = self.catalog.reference(target)
        state["ref"] = ref
        state["pocket_traces"] = self._model_traces(
            geometry(ref["pocket"], ref["center"], self.pocket_extent), "Pocket")
        state["records"], state["menus"] = {}, {}
        controls = []
        for method in self.methods:
            saved = self.selections.get(target, {}).get(method)
            record, invalid = self.catalog.initial_record(method, target, self.energy_key, saved)
            records = [record]
            state["records"][method] = records
            menu = widgets.Dropdown(options=[(record["label"], 0)], value=0,
                                    layout=widgets.Layout(width=f"{self.panel_px - 12}px"))
            state["menus"][method] = menu
            score_note = (f"{self.energy_key} = {record['energy']:.3f} kcal/mol"
                          if record.get("energy") is not None else "No verified energy score")
            summary = widgets.HTML(f"<small>{html.escape(score_note)} · 1 ligand loaded</small>")
            load_all = widgets.Button(description="Load all ligands", icon="list",
                                      layout=widgets.Layout(width="150px"))
            controls.append(widgets.VBox([
                widgets.HTML(f"<b>{html.escape(method)}</b>"), menu, summary, load_all,
            ]))

            def select_ligand(change, row=row_index, method=method):
                if change["new"] is not None:
                    self._refresh(row, method)

            menu.observe(select_ligand, names="value")

            def load_remaining(button, method=method, menu=menu, summary=summary,
                               callback=select_ligand, row=row_index):
                button.disabled = True
                button.description = "Loading..."
                current = self.rows[row]
                selected = self.selections[current["target"]][method]
                try:
                    all_records, skipped = self.catalog.ranked_records(method, current["target"], self.energy_key)
                    chosen = next(i for i, r in enumerate(all_records)
                                  if str(r["path"]) == selected["path"] and r["index"] == selected["record_index"])
                    menu.unobserve(callback, names="value")
                    try:
                        current["records"][method] = all_records
                        menu.options = [(r["label"], i) for i, r in enumerate(all_records)]
                        menu.value = chosen
                    finally:
                        menu.observe(callback, names="value")
                    summary.value = f"<small>{len(all_records)} ligands · {len(skipped)} skipped</small>"
                    button.description = "All ligands loaded"
                except Exception as exc:
                    current["error"].value = html.escape(str(exc))
                    button.description = "Retry loading"
                    button.disabled = False

            load_all.on_click(load_remaining)
        inspect_button = widgets.Button(description="Inspect selected ligands", icon="search")
        inspect_button.on_click(lambda _: self.inspect(row_index))
        state["controls"].children = [widgets.VBox([inspect_button], layout=widgets.Layout(
            width=f"{self.panel_px - 12}px", min_width=f"{self.panel_px - 12}px")), *controls]
        self._refresh(row_index)

    def _refresh(self, row_index, changed_method=None):
        state = self.rows[row_index]
        ref, target = state["ref"], state["target"]
        molecules = [ref["mol"]]
        self.selections.setdefault(target, {})
        for method, menu in state["menus"].items():
            record = state["records"][method][menu.value]
            molecules.append(record["mol"])
            self.selections[target][method] = dict(path=str(record["path"]), record_index=record["index"])
        geometries = [geometry(m, ref["center"]) for m in molecules]
        # Every column shares a center, camera and scale. Never recenter a generated ligand.
        # Keep the reference-centered pocket crop fixed; expand the view for large ligands.
        extent = max(self.pocket_extent + 0.5,
                     max(float(np.abs(g[0]).max()) for g in geometries) + 0.8)
        state["extent"] = extent
        axis = dict(range=[-extent, extent], visible=False, showgrid=False,
                    zeroline=False, showbackground=False)
        if changed_method is not None and "figure" in state:
            # A ligand change updates only that column's two meshes; pocket geometry stays put.
            figure = state["figure"]
            column = self.columns.index(changed_method)
            traces = self._model_traces(geometries[column], "Ligand")
            with figure.batch_update():
                for offset, trace in enumerate(traces):
                    payload = trace.to_plotly_json()
                    payload.pop("type", None)
                    figure.data[column * 4 + 2 + offset].update(payload)
                for i in range(len(self.columns)):
                    scene = figure.layout[scene_name(i)]
                    scene.xaxis.range = [-extent, extent]
                    scene.yaxis.range = [-extent, extent]
                    scene.zaxis.range = [-extent, extent]
            return

        fig = make_subplots(rows=1, cols=len(self.columns),
                            specs=[[{"type": "scene"}] * len(self.columns)],
                            subplot_titles=self.columns, horizontal_spacing=0.006)
        camera = self.cameras.get(target, self.style["DEFAULT_CAMERA"])
        for column, geom in enumerate(geometries):
            for trace in [*state["pocket_traces"], *self._model_traces(geom, "Ligand")]:
                fig.add_trace(trace, row=1, col=column + 1)
            fig.update_layout({scene_name(column): dict(
                xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube", camera=deepcopy(camera),
                dragmode="orbit", bgcolor="white")})
        fig.update_layout(width=self.panel_px * len(self.columns), height=self.panel_px + 35,
                          margin=dict(l=0, r=0, t=35, b=0), showlegend=False, hovermode=False,
                          paper_bgcolor="white", plot_bgcolor="white",
                          font=dict(family="Arial", size=15, color="#514F52"))
        old = state.get("figure")
        figure = go.FigureWidget(fig)
        state["figure"] = figure

        def sync_camera(scene, camera):
            if state["syncing"]:
                return
            state["syncing"] = True
            try:
                saved = camera.to_plotly_json()
                self.cameras[target] = saved
                with figure.batch_update():
                    for column in range(len(self.columns)):
                        other = figure.layout[scene_name(column)]
                        if other is not scene:
                            other.camera = deepcopy(saved)
            finally:
                state["syncing"] = False

        for column in range(len(self.columns)):
            figure.layout[scene_name(column)].on_change(sync_camera, "camera")
        with state["output"]:
            state["output"].clear_output(wait=True)
            display(figure)
        if old is not None:
            old.close()

    def inspect(self, row_index):
        """Show 2D structures and exact source records without modifying 3D coordinates."""
        from rdkit.Chem import Draw, rdDepictor
        state = self.rows[row_index]
        molecules = [state["ref"]["mol"]] + [state["records"][m][menu.value]["mol"]
                                              for m, menu in state["menus"].items()]
        copies = [Chem.Mol(m) for m in molecules]
        for mol in copies:
            rdDepictor.Compute2DCoords(mol)
        with state["output"]:
            state["output"].clear_output(wait=True)
            display(state["figure"])
            display(Draw.MolsToGridImage(copies, legends=self.columns, molsPerRow=3,
                                        subImgSize=(280, 210), useSVG=True))
            display(make_table([r for r in self.selection_report() if r["target"] == state["target"]]))

    def selection_report(self):
        rows = []
        for state in self.rows:
            ref = state["ref"]
            for method, menu in state["menus"].items():
                record = state["records"][method][menu.value]
                xyz = geometry(record["mol"], ref["center"])[0]
                offset = float(np.linalg.norm(xyz.mean(axis=0)))
                rows.append(dict(target=state["target"], method=method,
                                 source=str(record["path"]), record_index=record["index"],
                                 heavy_atoms=len(xyz), energy=record.get("energy"),
                                 energy_key=self.energy_key, energy_source=record.get("energy_source"),
                                 energy_eval_index=record.get("energy_eval_index"),
                                 centroid_offset_A=round(offset, 2),
                                 note=("CHECK coordinate frame / displaced pose" if offset > 10 else
                                       ("No verified energy score" if record.get("energy") is None else "")),
                                 smiles=Chem.MolToSmiles(Chem.RemoveHs(record["mol"]))))
        return rows

    def figure(self):
        """Snapshot the live widgets, including the cameras currently shown."""
        nrows, ncols = len(self.rows), len(self.columns)
        fig = make_subplots(rows=nrows, cols=ncols,
                            specs=[[{"type": "scene"}] * ncols for _ in self.rows],
                            horizontal_spacing=0.005, vertical_spacing=0.075)
        for row, state in enumerate(self.rows):
            live = state["figure"]
            for col in range(ncols):
                src_name = scene_name(col)
                for trace in live.data:
                    if trace.scene == src_name:
                        fig.add_trace(trace, row=row + 1, col=col + 1)
                scene = live.layout[src_name].to_plotly_json()
                scene.pop("domain", None)
                dest_name = scene_name(row * ncols + col)
                fig.update_layout({dest_name: scene})
                domain = fig.layout[dest_name].domain
                if row == 0:
                    fig.add_annotation(text=self.columns[col], x=sum(domain.x) / 2, y=1.045,
                                       xref="paper", yref="paper", showarrow=False, font=dict(size=17))
            first = fig.layout[scene_name(row * ncols)].domain
            fig.add_annotation(text=state["ref"]["label"], x=-0.018, y=sum(first.y) / 2,
                               xref="paper", yref="paper", textangle=-90, showarrow=False,
                               font=dict(size=16))
        fig.update_layout(width=ncols * self.panel_px + 65, height=nrows * self.panel_px + 85,
                          margin=dict(l=65, r=5, t=55, b=20), showlegend=False,
                          paper_bgcolor="white", plot_bgcolor="white",
                          font=dict(family="Arial", color="#514F52"))
        return fig

    def manifest(self):
        return dict(methods=self.methods, energy_key=self.energy_key, targets=[s["target"] for s in self.rows],
                    selections={s["target"]: self.selections[s["target"]] for s in self.rows},
                    cameras={s["target"]: s["figure"].layout.scene.camera.to_plotly_json()
                             for s in self.rows},
                    pocket_half_extent_A=self.pocket_extent, panel_px=self.panel_px,
                    pose="original SDF coordinates; reference-heavy-atom centroid subtracted from all models",
                    style_source=self.style.get("_source_name", "fig1.ipynb"), ligand_color=self.style["LIGAND_COLOR"],
                    pocket_color=self.style["POCKET_COLOR"],
                    references=[dict(target=s["target"], ligand=str(s["ref"]["ligand_path"]),
                                     pocket=str(s["ref"]["pocket_path"]),
                                     shared_view_half_extent_A=s["extent"]) for s in self.rows],
                    selected_ligands=self.selection_report())
