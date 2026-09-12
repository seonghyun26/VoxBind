#!/usr/bin/env python3
"""Standalone local web viewer for the fig-qual comparison.

Run with:
    bash notebook/figures/run_fig_qual.sh
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import traceback
from urllib.parse import parse_qs, urlparse

import numpy as np
import plotly.io as pio
from plotly.subplots import make_subplots


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from qual_setup import initialize  # noqa: E402
from qual_utils import (  # noqa: E402
    DATA_CUBE_BORDER_COLOR,
    POCKET_OCCUPANCY_COLOR,
    POCKET_OCCUPANCY_OPACITY,
    atom_occupancy_mesh,
    data_cube_trace,
    geometry,
    scene_name,
)


DEFAULT_METHODS = ["VoxBind", "VoxBind + Ours"]
DEFAULT_TARGETS = ["target_57", "target_33", "target_65", "target_05"]
PANEL_PX = 440
POCKET_EXTENT_A = 8.0
ENERGY_KEY = "vina_dock"
DATA_LOCK = threading.RLock()
CAMERA_PATH = HERE / "fig_qual_cameras.json"

# Audited over all 79 common pockets using the largest atom-silhouette-free
# cone from the reference-ligand center. To avoid visibly clipped pocket
# fragments, candidates have an open-cone half-angle >= 30 degrees and at most
# four crop-created singleton atoms, then rank by best-energy improvement.
OPEN_POCKET_METADATA = {
    "target_57": {"open_cone_deg": 32.3, "energy_delta": -5.791,
                  "direction": [0.1120, 0.8709, 0.4786]},
    "target_33": {"open_cone_deg": 31.0, "energy_delta": -5.356,
                  "direction": [0.9650, 0.2360, 0.1145]},
    "target_65": {"open_cone_deg": 30.9, "energy_delta": -4.605,
                  "direction": [-0.7916, -0.0650, 0.6076]},
    "target_05": {"open_cone_deg": 85.0, "energy_delta": -2.998,
                  "direction": [-0.9886, -0.1411, -0.0532]},
}


def _trace_json(trace):
    """Return Plotly.js-ready JSON, including Plotly 6 typed arrays."""
    return json.loads(pio.to_json({"data": [trace], "layout": {}}, validate=False))["data"][0]


def _figure_json(figure):
    return json.loads(pio.to_json(figure, validate=False, remove_uids=True))


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


class ViewerData:
    def __init__(self, methods):
        context = initialize()
        self.catalog = context["catalog"]
        self.style = context["STYLE"]
        self.methods = list(methods)
        unknown = [method for method in self.methods if method not in self.catalog.files]
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}")
        self.targets = self.catalog.common_targets(self.methods)
        if not self.targets:
            raise ValueError("No pockets are shared by the selected methods.")

    def _model_traces(self, geom, role):
        coords, elements, bonds = geom
        color = self.style["LIGAND_COLOR" if role == "Ligand" else "POCKET_COLOR"]
        return [
            self.style["bond_trace"](coords, elements, bonds, role, role, color),
            self.style["atom_trace"](coords, elements, role, role, color),
        ]

    @lru_cache(maxsize=64)
    def _pocket_traces(self, target):
        ref = self.catalog.reference(target)
        pocket_geom = geometry(ref["pocket"], ref["center"], POCKET_EXTENT_A)
        vertices, faces = atom_occupancy_mesh(
            pocket_geom[0], pocket_geom[1], POCKET_EXTENT_A)
        occupancy = self.style["mesh_trace"](
            vertices, faces, [POCKET_OCCUPANCY_COLOR] * len(vertices),
            "Pocket atom occupancy", "pocket-occupancy",
            opacity=POCKET_OCCUPANCY_OPACITY,
        )
        return (
            occupancy,
            data_cube_trace(POCKET_EXTENT_A),
            *self._model_traces(pocket_geom, "Pocket"),
        )

    def _record_summary(self, record):
        return {
            "label": record["label"],
            "path": str(record["path"]),
            "record_index": record["index"],
            "energy": record.get("energy"),
        }

    def config(self):
        defaults = [target for target in DEFAULT_TARGETS if target in self.targets]
        defaults += [target for target in self.targets if target not in defaults]
        return {
            "methods": self.methods,
            "columns": ["Reference", *self.methods],
            "targets": self.targets,
            "default_targets": defaults[:4],
            "energy_key": ENERGY_KEY,
            "camera_file": str(CAMERA_PATH),
            "colors": {
                "ligand": self.style["LIGAND_COLOR"],
                "pocket": self.style["POCKET_COLOR"],
                "pocket_occupancy": POCKET_OCCUPANCY_COLOR,
                "data_cube": DATA_CUBE_BORDER_COLOR,
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
            "version": 1, "cameras": {},
        }
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
        self.catalog.validate_target(target, self.methods)
        ref = self.catalog.reference(target)
        records = {}
        molecules = [ref["mol"]]
        for method in self.methods:
            record, invalid = self.catalog.initial_record(method, target, ENERGY_KEY)
            records[method] = {**self._record_summary(record), "invalid_count": len(invalid)}
            molecules.append(record["mol"])

        geometries = [geometry(molecule, ref["center"]) for molecule in molecules]
        required_extents = {
            column: max(POCKET_EXTENT_A + 0.5, float(np.abs(geom[0]).max()) + 0.8)
            for column, geom in zip(["Reference", *self.methods], geometries)
        }
        extent = max(required_extents.values())
        axis = dict(range=[-extent, extent], visible=False, showgrid=False,
                    zeroline=False, showbackground=False)
        columns = ["Reference", *self.methods]

        figure = make_subplots(
            rows=1, cols=len(columns), specs=[[{"type": "scene"}] * len(columns)],
            horizontal_spacing=0.006,
        )
        pocket_traces = self._pocket_traces(target)
        camera = self.saved_camera(target) or _open_side_camera(
            target, self.style["DEFAULT_CAMERA"])
        for column, geom in enumerate(geometries):
            traces = [*pocket_traces, *self._model_traces(geom, "Ligand")]
            for trace in traces:
                figure.add_trace(deepcopy(trace), row=1, col=column + 1)
            figure.update_layout({scene_name(column): dict(
                xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube",
                camera=deepcopy(camera), dragmode="orbit",
                bgcolor="white", uirevision=target,
            )})
        figure.update_layout(
            width=PANEL_PX * len(columns), height=PANEL_PX,
            margin=dict(l=0, r=0, t=0, b=0), showlegend=False, hovermode=False,
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Arial, sans-serif", size=15, color=DATA_CUBE_BORDER_COLOR),
        )
        return {
            "target": target,
            "label": ref["label"],
            "figure": _figure_json(figure),
            "records": records,
            "required_extents": required_extents,
            "traces_per_panel": len(pocket_traces) + 2,
            "ligand_trace_offset": len(pocket_traces),
            "selection_metadata": OPEN_POCKET_METADATA.get(target),
        }

    def records(self, target, method):
        self._validate_method_target(target, method)
        records, invalid = self.catalog.ranked_records(method, target, ENERGY_KEY)
        return {
            "records": [dict(choice=i, **self._record_summary(record))
                        for i, record in enumerate(records)],
            "invalid_count": len(invalid),
        }

    def ligand(self, target, method, choice):
        self._validate_method_target(target, method)
        records, _ = self.catalog.ranked_records(method, target, ENERGY_KEY)
        if choice < 0 or choice >= len(records):
            raise ValueError(f"Ligand choice out of range: {choice}")
        record = records[choice]
        ref = self.catalog.reference(target)
        geom = geometry(record["mol"], ref["center"])
        traces = self._model_traces(geom, "Ligand")
        return {
            "record": self._record_summary(record),
            "traces": [_trace_json(trace) for trace in traces],
            "required_extent": max(
                POCKET_EXTENT_A + 0.5, float(np.abs(geom[0]).max()) + 0.8),
        }

    def _validate_method_target(self, target, method):
        if target not in self.targets:
            raise ValueError(f"Unavailable pocket: {target}")
        if method not in self.methods:
            raise ValueError(f"Unavailable method: {method}")


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VoxBind qualitative comparison</title>
  <script src="/assets/plotly.min.js"></script>
  <style>
    :root { --ink:#514F52; --muted:#858187; --line:#D8D4D9; --soft:#F6F4F7;
            --ligand:#F5B27E; --pocket:#8291E8; --occupancy:#B4BEF0; }
    * { box-sizing:border-box; }
    body { margin:0; background:#fff; color:var(--ink); font-family:Arial,sans-serif; }
    main { width:min(1440px, calc(100vw - 32px)); margin:24px auto 56px; }
    header { display:flex; align-items:flex-end; justify-content:space-between; gap:24px;
             padding-bottom:14px; border-bottom:1px solid var(--line); }
    h1 { margin:0 0 5px; font-size:24px; font-weight:650; letter-spacing:-.02em; }
    .subtitle { color:var(--muted); font-size:13px; }
    .legend { display:flex; flex-wrap:wrap; gap:12px; font-size:12px; white-space:nowrap; }
    .swatch { display:inline-block; width:11px; height:11px; border-radius:3px;
              vertical-align:-1px; margin-right:5px; }
    .row { margin-top:26px; }
    .controls { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.6%;
                margin-bottom:8px; }
    .control { min-width:0; padding:10px; border:1px solid var(--line); background:var(--soft); }
    .control label { display:block; margin-bottom:6px; font-size:11px; font-weight:700;
                     letter-spacing:.04em; text-transform:uppercase; }
    .control-line { display:flex; gap:7px; }
    select, button { min-width:0; height:32px; border:1px solid #BEB9C0; background:#fff;
                     color:var(--ink); font:12px Arial,sans-serif; }
    select { width:100%; padding:0 8px; }
    button { flex:0 0 auto; padding:0 10px; cursor:pointer; }
    button:hover { background:#ECE9EE; }
    button:disabled { cursor:wait; color:#999; }
    .note { height:15px; margin-top:5px; overflow:hidden; text-overflow:ellipsis;
            white-space:nowrap; color:var(--muted); font-size:11px; }
    .plot { width:100%; min-height:280px; margin-inline:auto; }
    .plot .plot-container, .plot .svg-container { margin-inline:auto !important; }
    .status { display:grid; place-items:center; min-height:260px; border:1.5px solid var(--ink);
              color:var(--muted); font-size:13px; }
    .error { color:#A03E45; }
    footer { margin-top:20px; color:var(--muted); font-size:11px; }
    @media (max-width:760px) {
      main { width:calc(100vw - 18px); margin-top:16px; }
      header { align-items:flex-start; flex-direction:column; }
      .controls { grid-template-columns:1fr; }
      .legend { white-space:normal; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Qualitative ligand · pocket comparison</h1>
      <div class="subtitle">Reference-centered coordinates · original generated poses · synchronized cameras within each pocket</div>
    </div>
    <div class="legend">
      <span><i class="swatch" style="background:var(--ligand)"></i>Ligand</span>
      <span><i class="swatch" style="background:var(--pocket)"></i>Pocket atoms</span>
      <span><i class="swatch" style="background:var(--occupancy)"></i>Pocket occupancy</span>
      <span><i class="swatch" style="background:var(--ink)"></i>Data cube</span>
    </div>
  </header>
  <section id="rows"></section>
  <footer>Drag any panel to rotate all three views in that row. Scroll to zoom. Ligand lists load on demand.</footer>
</main>
<script>
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
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "VoxBindQual/1.0"

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send(HTML.encode(), "text/html; charset=utf-8")
                return
            if parsed.path == "/assets/plotly.min.js":
                self._send(self.server.plotly_js, "text/javascript; charset=utf-8",
                           cache="public, max-age=86400")
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
            if parsed.path != "/api/camera":
                self._json({"error": "Not found"}, status=404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                raise ValueError("Invalid camera request size.")
            request = json.loads(self.rfile.read(length))
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
    plotly_path = Path(pio.__file__).resolve().parent.parent / "package_data" / "plotly.min.js"
    if not plotly_path.is_file():
        raise FileNotFoundError(f"Plotly.js bundle not found: {plotly_path}")
    server.plotly_js = plotly_path.read_bytes()
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
