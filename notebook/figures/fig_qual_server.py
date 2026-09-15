#!/usr/bin/env python3
"""Standalone local web viewer for the fig-qual comparison.

Run with:
    bash notebook/figures/run_fig_qual.sh
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from io import BytesIO
from pathlib import Path
import re
import sys
import tempfile
import threading
import traceback
from urllib.parse import parse_qs, urlparse
import zlib

import numpy as np
from rdkit import Chem
from PIL import Image


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qual_setup import initialize  # noqa: E402
from qual_utils import POCKET_OCCUPANCY_OPACITY  # noqa: E402
from prepare_qual_baselines import prepare_baselines  # noqa: E402


DEFAULT_METHODS = ["AR", "Pocket2Mol", "DiffSBDD", "TargetDiff", "DecompDiff", "FuncBind", "VoxBind", "VoxBind + Ours"]
DEFAULT_TARGETS = ["target_33", "target_79"]
POCKET_EXTENT_A = 8.0
POCKET_SURFACE_COLOR = "#FFFFFF"
ENERGY_KEY = "vina_dock"
DATA_LOCK = threading.RLock()
CAPTURE_LOCK = threading.Lock()
CAPTURE_ROOT = HERE.parent / "results/qual"
CAMERA_PATH = HERE / "fig_qual_cameras.json"
NGL_PATH = HERE / "assets" / "ngl.js"

# Camera directions look through the largest atom-silhouette-free opening.
# Example selection compares pocket energy minima independently of this angle.
OPEN_POCKET_METADATA = {
    "target_79": {"open_cone_deg": 30.7,
                  "direction": [0.2086627, 0.9745194, -0.0822917]},
    "target_04": {"open_cone_deg": 22.2,
                  "direction": [-0.8793810, -0.4692569, 0.0805417]},
    "target_57": {"open_cone_deg": 32.3, "energy_delta": -5.791,
                  "direction": [0.1120, 0.8709, 0.4786]},
    "target_33": {"open_cone_deg": 31.0, "energy_delta": -5.356,
                  "direction": [0.9650, 0.2360, 0.1145]},
    "target_65": {"open_cone_deg": 30.9, "energy_delta": -4.605,
                  "direction": [-0.7916, -0.0650, 0.6076]},
    "target_05": {"open_cone_deg": 85.0, "energy_delta": -2.998,
                  "direction": [-0.9886, -0.1411, -0.0532]},
    "target_96": {"open_cone_deg": 24.8, "energy_delta": -0.291,
                  "direction": [0.1374871, 0.6714238, 0.7282083]},
}


def _open_side_camera(target, fallback):
    metadata = OPEN_POCKET_METADATA.get(target)
    if metadata is None:
        return deepcopy(fallback)
    direction = np.asarray(metadata["direction"], dtype=float)
    direction /= np.linalg.norm(direction)
    world_up = np.array([0.0, 0.0, 1.0])
    up = world_up - direction * np.dot(world_up, direction)
    if np.linalg.norm(up) < 0.15:
        world_up = np.array([0.0, 1.0, 0.0])
        up = world_up - direction * np.dot(world_up, direction)
    up /= np.linalg.norm(up)
    eye = 2.05 * direction
    return dict(
        eye=dict(zip(("x", "y", "z"), eye)),
        center=dict(x=0.0, y=0.0, z=0.0),
        up=dict(zip(("x", "y", "z"), up)),
        projection=dict(type="perspective"),
    )


def _normalized_camera(camera):
    """Validate and reduce a browser camera payload to reproducible fields."""
    if not isinstance(camera, dict):
        raise ValueError("Camera must be a JSON object.")
    if "rotation" in camera:
        rotation = camera["rotation"]
        position = camera.get("position")
        distance = camera.get("distance")
        if not isinstance(rotation, list) or len(rotation) != 4:
            raise ValueError("NGL camera rotation must contain four quaternion values.")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("NGL camera position must contain three values.")
        rotation = [float(value) for value in rotation]
        position = [float(value) for value in position]
        distance = float(distance)
        if not np.isfinite([*rotation, *position, distance]).all() or distance <= 0:
            raise ValueError("NGL camera contains an invalid value.")
        norm = float(np.linalg.norm(rotation))
        if norm < 1e-8:
            raise ValueError("NGL camera rotation quaternion is zero.")
        rotation = [value / norm for value in rotation]
        return {"rotation": rotation, "position": position, "distance": distance}
    if "orientation" in camera:
        orientation = camera["orientation"]
        if not isinstance(orientation, list) or len(orientation) != 16:
            raise ValueError("NGL camera orientation must contain 16 values.")
        orientation = [float(value) for value in orientation]
        if not np.isfinite(orientation).all():
            raise ValueError("NGL camera orientation contains a non-finite value.")
        return {"orientation": orientation}
    normalized = {}
    for group in ("eye", "center", "up"):
        source = camera.get(group)
        if not isinstance(source, dict):
            raise ValueError(f"Camera is missing {group}.")
        values = {axis: float(source[axis]) for axis in ("x", "y", "z")}
        if not np.isfinite(list(values.values())).all():
            raise ValueError(f"Camera {group} contains a non-finite value.")
        normalized[group] = values
    projection = camera.get("projection", {"type": "perspective"})
    projection_type = projection.get("type") if isinstance(projection, dict) else None
    if projection_type not in ("perspective", "orthographic"):
        raise ValueError("Camera projection must be perspective or orthographic.")
    normalized["projection"] = {"type": projection_type}
    return normalized


def _centered_molecule(molecule, center, extent=None):
    """Copy an RDKit molecule, center it, and optionally crop it to the data cube."""
    editable = Chem.RWMol(Chem.Mol(molecule))
    conformer = editable.GetConformer()
    center = np.asarray(center, dtype=float)
    keep = []
    for atom in editable.GetAtoms():
        index = atom.GetIdx()
        point = np.asarray(conformer.GetAtomPosition(index), dtype=float) - center
        conformer.SetAtomPosition(index, point)
        keep.append(
            atom.GetAtomicNum() > 1
            and (extent is None or bool(np.all(np.abs(point) <= extent + 0.15)))
        )
    if extent is not None:
        for index in reversed(np.flatnonzero(np.logical_not(keep)).tolist()):
            editable.RemoveAtom(int(index))
    return editable.GetMol()


def _pocket_pdb(molecule, center, extent):
    return Chem.MolToPDBBlock(_centered_molecule(molecule, center, extent))


def _ligand_sdf(molecule, center):
    return Chem.MolToMolBlock(_centered_molecule(molecule, center))


class ViewerData:
    def __init__(self, methods):
        context = initialize()
        prepare_baselines(context["REPO_ROOT"], context["catalog"], methods)
        context = initialize()
        self.catalog = context["catalog"]
        self.style = context["STYLE"]
        self.methods = list(methods)
        unknown = [method for method in self.methods if method not in self.catalog.files]
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}")
        # Reserve TargetDiff's panel while its samples are being transferred.
        required = [method for method in self.methods if method != "TargetDiff"]
        self.targets = self.catalog.common_targets(required or self.methods)
        if not self.targets:
            raise ValueError("No pockets are shared by the selected methods.")

    def available_methods(self, target):
        return [method for method in self.methods if target in self.catalog.files[method]
                and self.catalog.selection_index(method, target, ENERGY_KEY, pb_only=True)["records"]]

    def filter_counts(self, target, method):
        if target not in self.catalog.files[method]:
            return None
        return self.catalog.selection_index(method, target, ENERGY_KEY, pb_only=True)["counts"]

    def unavailable_reason(self, target, method):
        return ("Awaiting results" if target not in self.catalog.files[method]
                else "No verified PB-valid ligands")

    @lru_cache(maxsize=64)
    def _pocket_block(self, target):
        ref = self.catalog.reference(target)
        return _pocket_pdb(ref["pocket"], ref["center"], POCKET_EXTENT_A)

    def _record_summary(self, record):
        return {
            "label": record["label"],
            "path": str(record["path"]),
            "record_index": record["index"],
            "energy": record.get("energy"),
            "ring_count": record["mol"].GetRingInfo().NumRings(),
            **{key: value for key, value in record.items() if key.startswith("pb_")},
        }

    def config(self):
        defaults = [target for target in DEFAULT_TARGETS if target in self.targets]
        defaults += [target for target in self.targets if target not in defaults]
        return {
            "methods": self.methods,
            "columns": ["Reference", *self.methods],
            "targets": self.targets,
            "default_targets": defaults[:2],
            "energy_key": ENERGY_KEY,
            "selection_filter": "pb_valid",
            "camera_file": str(CAMERA_PATH),
            "renderer": "NGL 2.4.0 (nglview engine)",
            "pocket_surface_opacity": POCKET_OCCUPANCY_OPACITY,
            "cube_extent": POCKET_EXTENT_A,
            "colors": {
                "ligand": "#9EA3AA",
                "pocket_surface": POCKET_SURFACE_COLOR,
            },
        }

    def saved_camera(self, target):
        if not CAMERA_PATH.is_file():
            return None
        document = json.loads(CAMERA_PATH.read_text())
        camera = document.get("cameras", {}).get(target)
        return _normalized_camera(camera) if camera is not None else None

    def save_camera(self, target, camera):
        if target not in self.targets:
            raise ValueError(f"Unavailable pocket: {target}")
        camera = _normalized_camera(camera)
        document = json.loads(CAMERA_PATH.read_text()) if CAMERA_PATH.is_file() else {
            "version": 2, "cameras": {},
        }
        document["version"] = 2
        document.setdefault("cameras", {})[target] = camera
        document["updated_at"] = datetime.now(timezone.utc).isoformat()
        with tempfile.NamedTemporaryFile(
            mode="w", dir=CAMERA_PATH.parent, suffix=".tmp", delete=False,
        ) as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(CAMERA_PATH)
        return {"target": target, "camera": camera, "camera_file": str(CAMERA_PATH)}

    def row(self, target):
        if target not in self.targets:
            raise ValueError(f"Unavailable pocket: {target}")
        available = self.available_methods(target)
        self.catalog.validate_target(target, [m for m in self.methods if target in self.catalog.files[m]])
        ref = self.catalog.reference(target)
        records = {}
        molecules = [ref["mol"]]
        for method in self.methods:
            if method not in available:
                records[method] = None
                molecules.append(None)
                continue
            record, invalid = self.catalog.initial_record(method, target, ENERGY_KEY, pb_only=True)
            records[method] = {**self._record_summary(record), "invalid_count": len(invalid)}
            molecules.append(record["mol"])
        columns = ["Reference", *self.methods]
        energies = {method: record["energy"] for method, record in records.items()
                    if record is not None and record["energy"] is not None}
        ours = records.get("VoxBind + Ours")
        comparison = None
        selection_note = None
        if ours and ours["energy"] is not None:
            rank = 1 + sum(energy < ours["energy"] for energy in energies.values())
            baselines = {method: energy for method, energy in energies.items()
                         if method != "VoxBind + Ours"}
            comparison = dict(method_minima=energies, ours_rank=rank, selection_filter="pb_valid",
                              best_baseline=min(baselines.values()) if baselines else None,
                              complete=len(energies) == len(self.methods),
                              unavailable_methods=[m for m in self.methods if m not in available])
            coverage = " available" if not comparison["complete"] else ""
            selection_note = f"PB-valid minima · Ours #{rank}/{len(energies)}{coverage} · {ours['ring_count']} rings"
        camera = self.saved_camera(target) or _open_side_camera(
            target, self.style["DEFAULT_CAMERA"])
        return {
            "target": target,
            "label": ref["label"],
            "pocket_code": ref["label"].split(" · ")[-1],
            "reference_energy": self.catalog.reference_energy(target, ENERGY_KEY),
            "pocket_pdb": self._pocket_block(target),
            "panels": [
                {"column": column, "available": molecule is not None,
                 "unavailable_reason": self.unavailable_reason(target, column) if molecule is None else None,
                 "ligand_sdf": _ligand_sdf(molecule, ref["center"]) if molecule is not None else None}
                for column, molecule in zip(columns, molecules)
            ],
            "camera": camera,
            "records": records,
            "filter_counts": {method: self.filter_counts(target, method) for method in self.methods},
            "selection_metadata": OPEN_POCKET_METADATA.get(target),
            "selection_note": selection_note,
            "comparison": comparison,
        }

    def records(self, target, method):
        self._validate_method_target(target, method)
        records, invalid, counts = self.catalog.pb_records(method, target, ENERGY_KEY)
        return {
            "records": [dict(choice=i, **self._record_summary(record))
                        for i, record in enumerate(records)],
            "invalid_count": len(invalid),
            "filter_counts": counts,
        }

    def ligand(self, target, method, choice):
        self._validate_method_target(target, method)
        records, _, counts = self.catalog.pb_records(method, target, ENERGY_KEY)
        if choice < 0 or choice >= len(records):
            raise ValueError(f"Ligand choice out of range: {choice}")
        record = records[choice]
        ref = self.catalog.reference(target)
        return {
            "record": self._record_summary(record),
            "filter_counts": counts,
            "ligand_sdf": _ligand_sdf(record["mol"], ref["center"]),
        }

    def _validate_method_target(self, target, method):
        if target not in self.targets:
            raise ValueError(f"Unavailable pocket: {target}")
        if method not in self.methods:
            raise ValueError(f"Unavailable method: {method}")
        if method not in self.available_methods(target):
            raise ValueError(f"{method}: {self.unavailable_reason(target, method)} ({target}).")

    def capture_metadata(self, target, panels):
        reference = self.catalog.reference(target)
        captured_at = datetime.now(timezone.utc).isoformat()
        methods = {}
        for panel in panels:
            method = panel["method"]
            if method in self.methods and method not in self.available_methods(target):
                if panel.get("status") != "unavailable" or panel.get("png") is not None:
                    raise ValueError(f"No capture is available for {method}.")
                methods[method] = dict(status="unavailable", energy=None, camera=None,
                                       reason=self.unavailable_reason(target, method),
                                       selection_filter="pb_valid", filter_counts=self.filter_counts(target, method),
                                       files={}, captured_at=None)
                continue
            if panel.get("status") == "unavailable":
                raise ValueError(f"Capture is required for {method}.")
            if method == "Reference":
                record = dict(path=reference["ligand_path"], index=0, mol=reference["mol"],
                              label=reference["label"],
                              energy=self.catalog.reference_energy(target, ENERGY_KEY))
            else:
                self._validate_method_target(target, method)
                selection = panel.get("record")
                if not isinstance(selection, dict):
                    raise ValueError(f"Missing captured ligand identity: {method}")
                record, _ = self.catalog.initial_record(method, target, ENERGY_KEY, saved=selection, pb_only=True)
            camera = _capture_camera(panel["camera"])
            stem = _capture_stem(reference["label"].split(" · ")[-1], method)
            methods[method] = {
                "status": "captured",
                "selection_filter": "reference" if method == "Reference" else "pb_valid",
                "filter_counts": None if method == "Reference" else self.filter_counts(target, method),
                "background": "transparent",
                **self._record_summary(record),
                "energy_source": record.get("energy_source"),
                "energy_eval_index": record.get("energy_eval_index"),
                "smiles": Chem.MolToSmiles(Chem.RemoveHs(record["mol"])),
                "camera": camera,
                "captured_at": captured_at,
                "files": {"png": f"{stem}.png", "pdf": f"{stem}.pdf"},
            }
        return {
            "schema_version": 2,
            "target": target,
            "pocket_code": reference["label"].split(" · ")[-1],
            "updated_at": captured_at,
            "energy_key": ENERGY_KEY,
            "energy_unit": "kcal/mol",
            "selection_filter": "pb_valid",
            "default_selection": "lowest verified vina_dock among PB-valid source ligands",
            "pb_pose": "original source SDF pose; per-method native evaluation and receptor scope",
            "image_size_px": [1600, 1600],
            "coordinate_origin_angstrom": reference["center"].tolist(),
            "camera_convention": "NGL viewerControls; reference-centered scene rotation; quaternion [x,y,z,w]; intrinsic XYZ Euler angles in degrees; position and distance in angstrom",
            "methods": methods,
        }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VoxBind qualitative comparison</title>
  <script src="/assets/ngl.js"></script>
  <style>
    :root { --ink:#514F52; --muted:#858187; --line:#D8D4D9; --soft:#F6F4F7;
            --ligand:#9EA3AA; --pocket:#8291E8; --occupancy:#FFFFFF; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); font-family:Arial,sans-serif; }
    main { width:min(1440px, calc(100vw - 32px)); margin:24px auto 56px; }
    header { display:flex; align-items:flex-end; justify-content:space-between; gap:24px;
             padding-bottom:14px; border-bottom:1px solid var(--line); }
    h1 { margin:0 0 5px; font-size:24px; font-weight:650; letter-spacing:-.02em; }
    .subtitle { color:var(--muted); font-size:13px; }
    .legend { display:flex; flex-wrap:wrap; gap:12px; font-size:12px; white-space:nowrap; }
    .swatch { display:inline-block; width:11px; height:11px; border-radius:3px;
              vertical-align:-1px; margin-right:5px; }
    .row { margin-top:26px; padding-bottom:22px; border-bottom:2px solid var(--ink); }
    .row.capturing .viewer { pointer-events:none; }
    .capture-note { min-height:16px; margin-top:8px; color:var(--muted);
                    font-size:12px; overflow-wrap:anywhere; }
    .panel { min-width:0; }
    .control { min-width:0; padding:10px; border:1px solid var(--line); background:var(--soft); }
    .control label { display:block; margin-bottom:6px; font-size:11px; font-weight:700;
                     letter-spacing:.04em; text-transform:uppercase; }
    .control-line { display:flex; flex-wrap:wrap; gap:7px; }
    .control-line select { flex:1 1 100%; }
    select, button { min-width:0; height:32px; border:1px solid #BEB9C0; background:#fff;
                     color:var(--ink); font:12px Arial,sans-serif; }
    select { width:100%; padding:0 8px; }
    button { flex:0 0 auto; padding:0 10px; cursor:pointer; }
    button:hover { background:#ECE9EE; }
    button:disabled { cursor:wait; color:#999; }
    .note { height:15px; margin-top:5px; overflow:hidden; text-overflow:ellipsis;
            white-space:nowrap; color:var(--muted); font-size:11px; }
    .viewer-grid { display:grid; grid-template-columns:repeat(var(--display-columns,3),minmax(0,1fr)); gap:18px 10px;
                   width:100%; margin-inline:auto; }
    .viewer { position:relative; width:100%; aspect-ratio:1; overflow:hidden; }
    .viewer canvas { display:block; width:100% !important; height:100% !important; }
    .viewer-snapshot { position:absolute; inset:0; width:100%; height:100%;
                       object-fit:contain; pointer-events:none; }
    .placeholder { height:100%; display:flex; flex-direction:column; align-items:center;
                   justify-content:center; gap:8px; color:var(--muted); font-size:13px; }
    .placeholder strong { color:var(--ink); font-size:16px; font-weight:500; }
    .status { display:grid; place-items:center; min-height:260px; border:1.5px solid var(--ink);
              color:var(--muted); font-size:13px; }
    .error { color:#A03E45; }
    footer { margin-top:20px; color:var(--muted); font-size:11px; }
    @media (max-width:640px) {
      main { width:calc(100vw - 18px); margin-top:16px; }
      header { align-items:flex-start; flex-direction:column; }
      .viewer-grid { grid-template-columns:1fr; }
      .legend { white-space:normal; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Qualitative ligand · pocket comparison</h1>
      <div class="subtitle">PB-valid ligands only · lowest dock score per method · synchronized cameras within each pocket</div>
    </div>
    <div class="legend">
      <span><i class="swatch" style="background:#9EA3AA"></i>C</span>
      <span><i class="swatch" style="background:#8FA9D7"></i>N</span>
      <span><i class="swatch" style="background:#E59A9A"></i>O</span>
      <span><i class="swatch" style="background:var(--occupancy); border:1px solid var(--line)"></i>Protein pocket surface</span>
    </div>
  </header>
  <section id="rows"></section>
  <footer>Drag any panel to rotate all views of that pocket. Scroll to zoom. Captures save PNG, PDF, and energy/camera JSON to notebook/results/qual/{pocket}/.</footer>
</main>
<script type="text/plain" id="legacy-plotly-app">
const app = { config:null, rows:[], syncing:false };
const EXPORT_SIZE_PX = 1600;
const scenes = i => i === 0 ? 'scene' : `scene${i+1}`;
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

async function api(path, options={}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

function graphHeight(graph) {
  const columns = app.config.columns.length;
  const sceneWidth = graph.parentElement.clientWidth * (1 - (columns - 1) * 0.006) / columns;
  return Math.max(320, Math.min(520, sceneWidth));
}

function controlMarkup(rowIndex) {
  const targetOptions = app.config.targets.map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join('');
  const methodControls = app.config.methods.map((method, methodIndex) => `
    <div class="control">
      <label>${esc(method)}</label>
      <div class="control-line">
        <select id="ligand-${rowIndex}-${methodIndex}" disabled></select>
        <button id="load-${rowIndex}-${methodIndex}" type="button">Load all</button>
        <button id="png-${rowIndex}-${methodIndex + 1}" type="button">PNG</button>
      </div>
      <div class="note" id="note-${rowIndex}-${methodIndex}">Loading initial ligand…</div>
    </div>`).join('');
  return `<div class="controls">
    <div class="control">
      <label id="pocket-label-${rowIndex}">Pocket ${rowIndex + 1}</label>
      <div class="control-line">
        <select id="target-${rowIndex}">${targetOptions}</select>
        <button id="camera-${rowIndex}" type="button">Save camera</button>
        <button id="png-${rowIndex}-0" type="button">PNG</button>
      </div>
      <div class="note" id="pocket-note-${rowIndex}">Reference</div>
    </div>${methodControls}</div>`;
}

function makeRow(rowIndex, target) {
  const section = document.createElement('article');
  section.className = 'row';
  section.innerHTML = controlMarkup(rowIndex) +
    `<div id="plot-${rowIndex}" class="plot"><div class="status">Loading ${esc(target)}…</div></div>`;
  document.querySelector('#rows').appendChild(section);
  document.querySelector(`#target-${rowIndex}`).value = target;
  const row = { index:rowIndex, target, section, camera:null, extents:{}, payload:null, syncing:false };
  app.rows.push(row);
  document.querySelector(`#target-${rowIndex}`).addEventListener('change', async event => {
    const next = event.target.value;
    if (app.rows.some(other => other !== row && other.target === next)) {
      event.target.value = row.target;
      window.alert('Choose a different pocket for each row.');
      return;
    }
    row.target = next;
    row.camera = null;
    await loadRow(row);
  });
  app.config.methods.forEach((method, methodIndex) => {
    document.querySelector(`#load-${rowIndex}-${methodIndex}`).addEventListener('click', () => loadRecords(row, method, methodIndex));
    document.querySelector(`#png-${rowIndex}-${methodIndex + 1}`).addEventListener('click', event => savePanel(row, methodIndex + 1, method, event.currentTarget));
    document.querySelector(`#ligand-${rowIndex}-${methodIndex}`).addEventListener('change', event => {
      if (event.target.dataset.loaded === 'true') updateLigand(row, method, methodIndex, Number(event.target.value));
    });
  });
  document.querySelector(`#camera-${rowIndex}`).addEventListener('click', event => saveCamera(row, event.currentTarget));
  document.querySelector(`#png-${rowIndex}-0`).addEventListener('click', event => savePanel(row, 0, 'Reference', event.currentTarget));
  return row;
}

function setInitialControl(row, method, methodIndex, record) {
  const select = document.querySelector(`#ligand-${row.index}-${methodIndex}`);
  select.innerHTML = `<option>${esc(record.label)}</option>`;
  select.disabled = true;
  select.dataset.loaded = 'false';
  select.dataset.path = record.path;
  select.dataset.recordIndex = record.record_index;
  const energy = record.energy == null ? 'No verified score' : `${app.config.energy_key} = ${record.energy.toFixed(3)} kcal/mol`;
  document.querySelector(`#note-${row.index}-${methodIndex}`).textContent = `${energy} · 1 ligand loaded`;
}

async function loadRow(row) {
  const graph = document.querySelector(`#plot-${row.index}`);
  graph.innerHTML = `<div class="status">Loading ${esc(row.target)}…</div>`;
  try {
    const payload = await api(`/api/row?target=${encodeURIComponent(row.target)}`);
    row.payload = payload;
    row.extents = {...payload.required_extents};
    document.querySelector(`#pocket-label-${row.index}`).textContent =
      `Pocket ${row.index + 1} · ${payload.label}`;
    const meta = payload.selection_metadata;
    document.querySelector(`#pocket-note-${row.index}`).textContent = meta
      ? `open cone ${meta.open_cone_deg.toFixed(1)}° · best Δ ${meta.energy_delta.toFixed(2)} kcal/mol`
      : 'Reference';
    app.config.methods.forEach((method, i) => setInitialControl(row, method, i, payload.records[method]));
    payload.figure.layout.height = graphHeight(graph);
    payload.figure.layout.autosize = true;
    if (row.camera) app.config.columns.forEach((_, i) => payload.figure.layout[scenes(i)].camera = row.camera);
    graph.innerHTML = '';
    await Plotly.newPlot(graph, payload.figure.data, payload.figure.layout, {
      responsive:true, displaylogo:false, scrollZoom:true,
      modeBarButtonsToRemove:['select2d','lasso2d','toImage'],
    });
    row.camera = graph.layout.scene.camera;
    graph.on('plotly_relayout', event => syncCamera(row, event));
  } catch (error) {
    graph.innerHTML = `<div class="status error">${esc(error.message)}</div>`;
  }
}

async function syncCamera(row, event) {
  if (row.syncing) return;
  const entry = Object.entries(event).find(([key]) => /^scene\d*\.camera$/.test(key));
  if (!entry) return;
  row.camera = entry[1];
  row.syncing = true;
  const update = {};
  app.config.columns.forEach((_, i) => update[`${scenes(i)}.camera`] = row.camera);
  try { await Plotly.relayout(document.querySelector(`#plot-${row.index}`), update); }
  finally { row.syncing = false; }
}

async function loadRecords(row, method, methodIndex) {
  const button = document.querySelector(`#load-${row.index}-${methodIndex}`);
  const select = document.querySelector(`#ligand-${row.index}-${methodIndex}`);
  const note = document.querySelector(`#note-${row.index}-${methodIndex}`);
  button.disabled = true;
  button.textContent = 'Loading…';
  try {
    const payload = await api(`/api/records?target=${encodeURIComponent(row.target)}&method=${encodeURIComponent(method)}`);
    const currentPath = select.dataset.path;
    const currentIndex = Number(select.dataset.recordIndex);
    select.innerHTML = payload.records.map(record => `<option value="${record.choice}">${esc(record.label)}</option>`).join('');
    const current = payload.records.find(record => record.path === currentPath && record.record_index === currentIndex);
    if (current) select.value = String(current.choice);
    select.disabled = false;
    select.dataset.loaded = 'true';
    note.textContent = `${payload.records.length} ligands · ${payload.invalid_count} skipped`;
    button.textContent = 'Loaded';
  } catch (error) {
    note.textContent = error.message;
    note.classList.add('error');
    button.disabled = false;
    button.textContent = 'Retry';
  }
}

async function updateLigand(row, method, methodIndex, choice) {
  const select = document.querySelector(`#ligand-${row.index}-${methodIndex}`);
  const note = document.querySelector(`#note-${row.index}-${methodIndex}`);
  select.disabled = true;
  note.textContent = 'Updating ligand…';
  try {
    const payload = await api(`/api/ligand?target=${encodeURIComponent(row.target)}&method=${encodeURIComponent(method)}&choice=${choice}`);
    const graph = document.querySelector(`#plot-${row.index}`);
    const panel = methodIndex + 1;
    const start = panel * row.payload.traces_per_panel + row.payload.ligand_trace_offset;
    payload.traces.forEach(trace => trace.scene = scenes(panel));
    await Plotly.deleteTraces(graph, [start + 1, start]);
    await Plotly.addTraces(graph, payload.traces, [start, start + 1]);
    row.extents[method] = payload.required_extent;
    const extent = Math.max(...Object.values(row.extents));
    const ranges = {};
    app.config.columns.forEach((_, i) => ['xaxis','yaxis','zaxis'].forEach(axis => {
      ranges[`${scenes(i)}.${axis}.range`] = [-extent, extent];
    }));
    if (row.camera) app.config.columns.forEach((_, i) => ranges[`${scenes(i)}.camera`] = row.camera);
    await Plotly.relayout(graph, ranges);
    select.dataset.path = payload.record.path;
    select.dataset.recordIndex = payload.record.record_index;
    const energy = payload.record.energy == null ? 'No verified score' : `${app.config.energy_key} = ${payload.record.energy.toFixed(3)} kcal/mol`;
    note.textContent = energy;
  } catch (error) {
    note.textContent = error.message;
    note.classList.add('error');
  } finally {
    select.disabled = false;
  }
}

async function saveCamera(row, button) {
  const graph = document.querySelector(`#plot-${row.index}`);
  if (!graph?.layout?.scene?.camera) return;
  const original = button.textContent;
  button.disabled = true;
  button.textContent = 'Saving…';
  try {
    const payload = await api('/api/camera', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({target:row.target, camera:graph.layout.scene.camera}),
    });
    row.camera = payload.camera;
    button.textContent = 'Saved';
  } catch (error) {
    button.textContent = 'Failed';
    window.alert(error.message);
  } finally {
    setTimeout(() => { button.textContent = original; button.disabled = false; }, 1200);
  }
}

function cleanTrace(trace) {
  const copy = {};
  for (const [key, value] of Object.entries(trace)) {
    if (key !== 'uid' && !key.startsWith('_')) copy[key] = value;
  }
  copy.scene = 'scene';
  return copy;
}

async function savePanel(row, panelIndex, label, button) {
  const graph = document.querySelector(`#plot-${row.index}`);
  const sourceScene = scenes(panelIndex);
  const traces = graph.data
    .filter(trace => (trace.scene || 'scene') === sourceScene)
    .map(cleanTrace);
  const source = graph.layout[sourceScene];
  const scene = {};
  for (const key of ['xaxis','yaxis','zaxis','aspectmode','camera','dragmode','bgcolor']) {
    if (source[key] !== undefined) scene[key] = source[key];
  }
  scene.domain = {x:[0,1], y:[0,1]};
  scene.aspectmode = 'cube';
  const exportNode = document.createElement('div');
  Object.assign(exportNode.style, {
    position:'fixed', left:'-10000px', top:'0',
    width:`${EXPORT_SIZE_PX}px`, height:`${EXPORT_SIZE_PX}px`, opacity:'0',
  });
  document.body.appendChild(exportNode);
  const original = button.textContent;
  button.disabled = true;
  button.textContent = 'Saving…';
  try {
    await Plotly.newPlot(exportNode, traces, {
      scene, width:EXPORT_SIZE_PX, height:EXPORT_SIZE_PX, autosize:false,
      margin:{l:0,r:0,t:0,b:0},
      showlegend:false, hovermode:false, paper_bgcolor:'white', plot_bgcolor:'white',
    }, {staticPlot:true, displayModeBar:false});
    const filename = `${row.target}-${label}`.toLowerCase().replace(/[^a-z0-9._-]+/g, '-');
    await Plotly.downloadImage(exportNode, {
      format:'png', filename, width:EXPORT_SIZE_PX, height:EXPORT_SIZE_PX, scale:1,
    });
    button.textContent = 'Saved';
  } catch (error) {
    button.textContent = 'Failed';
    window.alert(error.message);
  } finally {
    Plotly.purge(exportNode);
    exportNode.remove();
    setTimeout(() => { button.textContent = original; button.disabled = false; }, 1200);
  }
}

async function start() {
  try {
    app.config = await api('/api/config');
    document.documentElement.style.setProperty('--ligand', app.config.colors.ligand);
    document.documentElement.style.setProperty('--pocket', app.config.colors.pocket);
    document.documentElement.style.setProperty('--occupancy', app.config.colors.pocket_occupancy);
    document.documentElement.style.setProperty('--ink', app.config.colors.data_cube);
    app.config.default_targets.forEach((target, index) => makeRow(index, target));
    await Promise.all(app.rows.map(loadRow));
    new ResizeObserver(() => app.rows.forEach(row => {
      const graph = document.querySelector(`#plot-${row.index}`);
      if (graph && graph.data) Plotly.relayout(graph, {height:graphHeight(graph)});
    })).observe(document.querySelector('main'));
  } catch (error) {
    document.querySelector('#rows').innerHTML = `<div class="status error">${esc(error.message)}</div>`;
  }
}
start();
</script>
<script src="/assets/fig_qual_app.js"></script>
</body>
</html>
"""


def _png_pdf(png):
    """Embed RGB and alpha losslessly in a square, 300-dpi PDF page."""
    with Image.open(BytesIO(png)) as image:
        if image.format != "PNG" or image.size != (1600, 1600):
            raise ValueError("Expected a 1600 × 1600 PNG capture.")
        pixels = zlib.compress(image.convert("RGB").tobytes())
        alpha = zlib.compress(image.convert("RGBA").getchannel("A").tobytes())
    content = b"q 384 0 0 384 0 0 cm /Im0 Do Q\n"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 384 384] "
        b"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /XObject /Subtype /Image /Width 1600 /Height 1600 "
        b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /SMask 6 0 R /Filter /FlateDecode /Length "
        + str(len(pixels)).encode() + b" >>\nstream\n" + pixels + b"\nendstream",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
        + content + b"endstream",
        b"<< /Type /XObject /Subtype /Image /Width 1600 /Height 1600 "
        b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode /Length "
        + str(len(alpha)).encode() + b" >>\nstream\n" + alpha + b"\nendstream",
    ]
    document = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(document))
        document.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(document)
    document.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode())
    document.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(document)


def _capture_camera(payload):
    camera = _normalized_camera(payload)
    if "rotation" not in camera:
        raise ValueError("Captures require the NGL camera quaternion, position, and distance.")
    fov = float(payload["fov_degrees"])
    if not np.isfinite(fov) or not 0 < fov < 180:
        raise ValueError("Invalid camera field of view.")
    x, y, z, w = camera["rotation"]
    m11, m12, m13 = 1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)
    m22, m23 = 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)
    m32, m33 = 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)
    # Intrinsic XYZ decomposition, including the pitch +/-90 degree singularity.
    angles = [np.arctan2(-m23, m33), np.arcsin(np.clip(m13, -1, 1)), np.arctan2(-m12, m11)]
    if abs(m13) >= 0.9999999:
        angles[0], angles[2] = np.arctan2(m32, m22), 0.0
    return {**camera, "euler_xyz_degrees": np.degrees(angles).tolist(),
            "projection": "perspective", "fov_degrees": fov}


def _capture_stem(pocket_code, method):
    method = re.sub(r"[^A-Za-z0-9.+_-]+", "-", method).strip("-")
    return f"Pocket{pocket_code}-{method}"


def _capture_files(pocket_code, panels, columns):
    if re.fullmatch(r"[A-Z0-9]{4}", pocket_code) is None:
        raise ValueError("Invalid pocket code.")
    if (not isinstance(panels, list) or not all(isinstance(panel, dict) for panel in panels)
            or [panel.get("method") for panel in panels] != columns):
        raise ValueError("Capture must include each requested method exactly once.")
    files = {}
    for panel in panels:
        if panel.get("status") == "unavailable":
            continue
        stem = _capture_stem(pocket_code, panel["method"])
        png = base64.b64decode(panel["png"], validate=True)
        files[f"{stem}.pdf"] = _png_pdf(png)
        files[f"{stem}.png"] = png
    return files


def _save_capture_files(pocket_code, files, metadata):
    directory = CAPTURE_ROOT / pocket_code
    with CAPTURE_LOCK:
        directory.mkdir(parents=True, exist_ok=True)
        metadata_name = f"Pocket{pocket_code}.json"
        metadata_path = directory / metadata_name
        if metadata_path.is_file():
            previous = json.loads(metadata_path.read_text())
            if previous.get("target") != metadata["target"]:
                raise ValueError("Existing capture metadata belongs to a different target.")
            metadata["methods"] = {**previous.get("methods", {}), **metadata["methods"]}
        files = {**files, metadata_name: (json.dumps(metadata, indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n").encode()}
        for name, content in files.items():
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, prefix=".capture-", delete=False) as handle:
                    handle.write(content)
                    temporary = Path(handle.name)
                temporary.replace(directory / name)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return {"directory": str(directory), "files": sorted(files), "file_count": len(files)}


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "VoxBindQual/1.0"

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send(HTML.encode(), "text/html; charset=utf-8")
                return
            if parsed.path == "/assets/ngl.js":
                self._send(self.server.ngl_js, "text/javascript; charset=utf-8",
                           cache="public, max-age=86400")
                return
            if parsed.path == "/assets/fig_qual_app.js":
                self._send(self.server.app_js, "text/javascript; charset=utf-8",
                           cache="no-cache")
                return
            query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
            with DATA_LOCK:
                if parsed.path == "/api/config":
                    payload = self.server.viewer.config()
                elif parsed.path == "/api/row":
                    payload = self.server.viewer.row(query["target"])
                elif parsed.path == "/api/records":
                    payload = self.server.viewer.records(query["target"], query["method"])
                elif parsed.path == "/api/ligand":
                    payload = self.server.viewer.ligand(
                        query["target"], query["method"], int(query["choice"]))
                else:
                    self._json({"error": "Not found"}, status=404)
                    return
            self._json(payload)
        except (KeyError, ValueError) as exc:
            self._json({"error": str(exc)}, status=400)
        except Exception as exc:  # Keep the browser readable; retain traceback in terminal.
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path not in ("/api/camera", "/api/capture"):
                self._json({"error": "Not found"}, status=404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            limit = 64 * 1024 * 1024 if parsed.path == "/api/capture" else 65536
            if length <= 0 or length > limit:
                raise ValueError("Invalid request size.")
            request = json.loads(self.rfile.read(length))
            if parsed.path == "/api/capture":
                with DATA_LOCK:
                    if request["target"] not in self.server.viewer.targets:
                        raise ValueError("Unavailable capture pocket.")
                    reference = self.server.viewer.catalog.reference(request["target"])
                    code = reference["label"].split(" · ")[-1]
                    columns = ["Reference", *self.server.viewer.methods]
                    method = request.get("method")
                    if method is not None:
                        if method not in ["Reference", *self.server.viewer.available_methods(request["target"])]:
                            raise ValueError("Unavailable capture method.")
                        columns = [method]
                files = _capture_files(code, request["panels"], columns)
                with DATA_LOCK:
                    metadata = self.server.viewer.capture_metadata(request["target"], request["panels"])
                saved = _save_capture_files(code, files, metadata)
                self._json(dict(target=request["target"], pocket_code=code, **saved))
                return
            with DATA_LOCK:
                payload = self.server.viewer.save_camera(
                    request["target"], request["camera"])
            self._json(payload)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, status=400)
        except Exception as exc:
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

    def _json(self, payload, status=200):
        self._send(json.dumps(payload, separators=(",", ":")).encode(),
                   "application/json", status=status)

    def _send(self, body, content_type, status=200, cache="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    args = parser.parse_args()

    viewer = ViewerData(args.methods)
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    server.viewer = viewer
    if not NGL_PATH.is_file():
        raise FileNotFoundError(f"NGL.js bundle not found: {NGL_PATH}")
    app_path = HERE / "assets" / "fig_qual_app.js"
    if not app_path.is_file():
        raise FileNotFoundError(f"Fig-qual browser app not found: {app_path}")
    server.ngl_js = NGL_PATH.read_bytes()
    server.app_js = app_path.read_bytes()
    print(f"VoxBind qualitative viewer: http://{args.host}:{args.port}/", flush=True)
    print(f"Methods: {', '.join(args.methods)} · pockets: {len(viewer.targets)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
