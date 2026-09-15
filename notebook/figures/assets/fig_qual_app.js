"use strict";

const app = {config: null, rows: [], ready: false, activationQueue: Promise.resolve()};
const EXPORT_SIZE_PX = 1600;
const STICK_RADIUS_A = 0.24;
const LIGAND_ELEMENT_COLORS = {
  C: 0x9ea3aa, H: 0xf8f8f6, N: 0x8fa9d7, O: 0xe59a9a,
  F: 0xadcba6, P: 0xe6b485, S: 0xe3d382, CL: 0xa3c79e,
  BR: 0xcaa28a, I: 0xb8a0d1,
};
const esc = value => String(value).replace(
  /[&<>"']/g,
  char => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[char],
);

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

function colorNumber(hex) {
  return Number.parseInt(hex.replace("#", ""), 16);
}

function controlMarkup(rowIndex) {
  const targetOptions = app.config.targets
    .map(target => `<option value="${esc(target)}">${esc(target)}</option>`)
    .join("");
  const methodControls = app.config.methods.map((method, methodIndex) => `
    <div class="control">
      <label>${esc(method)}</label>
      <div class="control-line">
        <select id="ligand-${rowIndex}-${methodIndex}" disabled></select>
        <button id="load-${rowIndex}-${methodIndex}" type="button">Load all</button>
        <button id="png-${rowIndex}-${methodIndex + 1}" type="button">PNG / PDF</button>
      </div>
      <div class="note" id="note-${rowIndex}-${methodIndex}">Loading initial ligand…</div>
    </div>`);
  return [
    `<div class="control">
      <label id="pocket-label-${rowIndex}">Pocket ${rowIndex + 1}</label>
      <div class="control-line">
        <select id="target-${rowIndex}">${targetOptions}</select>
        <button id="camera-${rowIndex}" type="button">Save camera</button>
        <button id="png-${rowIndex}-0" type="button">PNG / PDF</button>
        <button id="capture-${rowIndex}" type="button" title="Save PNG, PDF, and energy/camera JSON in notebook/results/qual" disabled>Capture all</button>
      </div>
      <div class="note" id="pocket-note-${rowIndex}">Reference</div>
    </div>`,
    ...methodControls,
  ];
}

function viewerMarkup(rowIndex, target) {
  const controls = controlMarkup(rowIndex);
  return `<div class="viewer-grid">${app.config.columns.map((_, panelIndex) =>
    `<div class="panel">${controls[panelIndex]}<div id="viewer-${rowIndex}-${panelIndex}" class="viewer"><div class="status">Loading ${esc(target)}…</div></div></div>`,
  ).join("")}</div>`;
}

function makeRow(rowIndex, target) {
  const section = document.createElement("article");
  section.className = "row";
  section.innerHTML = viewerMarkup(rowIndex, target)
    + `<div id="capture-note-${rowIndex}" class="capture-note" role="status" aria-live="polite"></div>`;
  document.querySelector("#rows").appendChild(section);
  document.querySelector(`#target-${rowIndex}`).value = target;
  const row = {
    index: rowIndex,
    target,
    section,
    panels: [],
    camera: null,
    syncing: false,
    syncFrame: null,
    loadToken: 0,
  };
  app.rows.push(row);
  for (const event of ["pointerdown", "wheel"]) {
    section.addEventListener(event, input => {
      if (app.ready && input.target.closest(".viewer")) activateRow(row).catch(error => {
        document.querySelector(`#capture-note-${row.index}`).textContent = error.message;
      });
    }, {passive: true});
  }

  document.querySelector(`#target-${rowIndex}`).addEventListener("change", async event => {
    const next = event.target.value;
    if (app.rows.some(other => other !== row && other.target === next)) {
      event.target.value = row.target;
      window.alert("Choose a different pocket for each row.");
      return;
    }
    if (!await activateRow(row)) return;
    row.target = next;
    row.camera = null;
    await loadRow(row);
  });
  app.config.methods.forEach((method, methodIndex) => {
    document.querySelector(`#load-${rowIndex}-${methodIndex}`).addEventListener(
      "click", () => loadRecords(row, method, methodIndex),
    );
    document.querySelector(`#png-${rowIndex}-${methodIndex + 1}`).addEventListener(
      "click", event => savePanel(row, methodIndex + 1, method, event.currentTarget),
    );
    document.querySelector(`#ligand-${rowIndex}-${methodIndex}`).addEventListener("change", event => {
      if (event.target.dataset.loaded === "true") {
        updateLigand(row, method, methodIndex, Number(event.target.value));
      }
    });
  });
  document.querySelector(`#camera-${rowIndex}`).addEventListener(
    "click", event => saveCamera(row, event.currentTarget),
  );
  document.querySelector(`#png-${rowIndex}-0`).addEventListener(
    "click", event => savePanel(row, 0, "Reference", event.currentTarget),
  );
  document.querySelector(`#capture-${rowIndex}`).addEventListener(
    "click", event => savePocket(row, event.currentTarget),
  );
  return row;
}

function disposePanels(row) {
  if (row.syncFrame !== null) cancelAnimationFrame(row.syncFrame);
  row.syncFrame = null;
  for (const panel of row.panels) {
    if (panel.snapshotUrl) URL.revokeObjectURL(panel.snapshotUrl);
    panel.stage?.viewer.renderer.forceContextLoss();
    panel.stage?.dispose();
  }
  row.panels = [];
}

// Keep a rendered image for the other pocket while its GPU contexts are idle.
// Eighteen live NGL stages exceed the browser's usual sixteen-context limit.
async function suspendRow(row) {
  if (row.suspended || !row.panels.length) return;
  await Promise.all(row.panels.filter(panel => panel.stage).map(panel =>
    new Promise(resolve => panel.stage.tasks.onZeroOnce(resolve)),
  ));
  row.panels.forEach(panel => panel.stage?.viewer.requestRender());
  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  for (const [index, panel] of row.panels.entries()) {
    if (!panel.stage) continue;
    // NGL preserves the drawing buffer, so copy the displayed frame directly.
    const blob = await new Promise((resolve, reject) => panel.stage.viewer.renderer.domElement.toBlob(
      value => value ? resolve(value) : reject(new Error("Could not preserve the pocket view.")), "image/png",
    ));
    if (panel.snapshotUrl) URL.revokeObjectURL(panel.snapshotUrl);
    panel.snapshotUrl = URL.createObjectURL(blob);
    const snapshot = document.createElement("img");
    snapshot.className = "viewer-snapshot";
    snapshot.alt = app.config.columns[index];
    snapshot.src = panel.snapshotUrl;
    await snapshot.decode();
    document.querySelector(`#viewer-${row.index}-${index}`).appendChild(snapshot);
    panel.snapshot = snapshot;
  }
  row.camera = cameraState(row.panels[0].stage);
  row.panels.forEach(panel => {
    if (!panel.stage) return;
    const renderer = panel.stage.viewer.renderer;
    // A lost WebGL canvas can paint an opaque fallback behind the snapshot.
    renderer.domElement.style.visibility = "hidden";
    renderer.forceContextLoss();
  });
  row.suspended = true;
}

async function restorePanel(panel) {
  if (!panel.stage) return;
  const stage = panel.stage;
  const renderer = stage.viewer.renderer;
  if (renderer.getContext().isContextLost()) {
    await new Promise((resolve, reject) => {
      const canvas = renderer.domElement;
      const restored = () => { clearTimeout(timer); resolve(); };
      const timer = setTimeout(() => {
        canvas.removeEventListener("webglcontextrestored", restored);
        reject(new Error("Could not restore the pocket view. Reload the page to retry."));
      }, 10000);
      canvas.addEventListener("webglcontextrestored", restored, {once: true});
      renderer.forceContextRestore();
    });
  }
  setTransparentBackground(stage);
  stage.viewer.requestRender();
  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  renderer.domElement.style.removeProperty("visibility");
  panel.snapshot?.remove();
  panel.snapshot = null;
  if (panel.snapshotUrl) URL.revokeObjectURL(panel.snapshotUrl);
  panel.snapshotUrl = null;
}

function activateRow(row) {
  const pending = app.activationQueue.then(async () => {
    if (!app.reuseContexts || !row.suspended) return true;
    if (app.rows.some(other => other.capturing || other.updating)) return false;
    const unlock = lockCaptureRow(row);
    try {
      for (const other of app.rows) if (other !== row) await suspendRow(other);
      await Promise.all(row.panels.map(restorePanel));
      row.suspended = false;
      return true;
    } finally {
      unlock();
    }
  });
  app.activationQueue = pending.catch(() => {});
  return pending;
}

function setTransparentBackground(stage) {
  // NGL adds a canvas CSS background; WebGL restoration also resets clear alpha.
  const renderer = stage.viewer.renderer;
  renderer.domElement.style.removeProperty("background-color");
  // Use a fully transparent clear value for every render target.
  renderer.setClearColor(0, 0);
}

function stageFor(element) {
  element.innerHTML = "";
  const stage = new NGL.Stage(element, {
    cameraType: "perspective",
    quality: "high",
    sampleLevel: 1,
  });
  stage.setParameters({
    cameraType: "perspective",
    clipNear: 0,
    clipFar: 100,
    fogNear: 100,
    fogFar: 100,
    lightIntensity: 1.0,
    ambientIntensity: 0.55,
    hoverTimeout: 0,
  });
  setTransparentBackground(stage);
  return stage;
}

function loadText(stage, text, extension, name) {
  const file = new File([text], `${name}.${extension}`, {type: "text/plain"});
  return stage.loadFile(file, {ext: extension, firstModelOnly: true});
}

function addPocketSurface(component) {
  component.addRepresentation("surface", {
    sele: "not hydrogen",
    surfaceType: "ms",
    probeRadius: 1.4,
    smooth: 2,
    scaleFactor: 2.0,
    colorScheme: "uniform",
    colorValue: colorNumber(app.config.colors.pocket_surface),
    opacity: app.config.pocket_surface_opacity,
    opaqueBack: false,
    side: "double",
    depthWrite: false,
    useWorker: true,
  });
}

function addLigandSticks(component) {
  component.addRepresentation("licorice", {
    sele: "not hydrogen",
    colorScheme: app.ligandColorScheme,
    radiusType: "size",
    radiusSize: STICK_RADIUS_A,
    radiusScale: 1.0,
    // Licorice uses the same radius at atoms and bonds, without enlarged balls.
    multipleBond: "off",
    quality: "high",
  });
}

function framePocket(stage) {
  // Keep the common reference-centered framing without rendering box edges.
  const extent = app.config.cube_extent;
  const bounds = new NGL.Box3(
    new NGL.Vector3(-extent, -extent, -extent),
    new NGL.Vector3(extent, extent, extent),
  );
  stage.animationControls.zoomMove(new NGL.Vector3(0, 0, 0), stage.getZoomForBox(bounds), 0);
}

async function buildPanel(row, panelIndex, payload) {
  const element = document.querySelector(`#viewer-${row.index}-${panelIndex}`);
  if (!payload.panels[panelIndex].available) {
    element.innerHTML = `<div class="placeholder"><strong>${esc(app.config.columns[panelIndex])}</strong><span>${esc(payload.panels[panelIndex].unavailable_reason || "Awaiting results")}</span></div>`;
    return {stage: null, ligand: null, record: null};
  }
  const stage = stageFor(element);
  const panel = {stage, ligand: null, record: payload.records[app.config.columns[panelIndex]] || null};
  try {
    const pocket = await loadText(stage, payload.pocket_pdb, "pdb", `${row.target}-pocket`);
    addPocketSurface(pocket);
    panel.ligand = await loadText(
      stage,
      payload.panels[panelIndex].ligand_sdf,
      "sdf",
      `${row.target}-ligand-${panelIndex}`,
    );
    addLigandSticks(panel.ligand);
    stage.handleResize();
    return panel;
  } catch (error) {
    stage.dispose();
    element.innerHTML = `<div class="status error">${esc(error.message)}</div>`;
    throw error;
  }
}

function setInitialControl(row, method, methodIndex, record) {
  const select = document.querySelector(`#ligand-${row.index}-${methodIndex}`);
  const button = document.querySelector(`#load-${row.index}-${methodIndex}`);
  const capture = document.querySelector(`#png-${row.index}-${methodIndex + 1}`);
  const note = document.querySelector(`#note-${row.index}-${methodIndex}`);
  capture.disabled = !record;
  note.classList.remove("error");
  if (!record) {
    select.innerHTML = "<option>No PB-valid ligands available</option>";
    select.disabled = true;
    select.dataset.loaded = "false";
    delete select.dataset.path;
    delete select.dataset.recordIndex;
    button.disabled = true;
    button.textContent = "Pending";
    note.textContent = "No verified PB-valid selection";
    return;
  }
  select.innerHTML = `<option>${esc(record.label)}</option>`;
  select.disabled = true;
  select.dataset.loaded = "false";
  select.dataset.path = record.path;
  select.dataset.recordIndex = record.record_index;
  button.disabled = false;
  button.textContent = "Load all";
  const energy = record.energy == null
    ? "No verified score"
    : `${app.config.energy_key} = ${record.energy.toFixed(3)} kcal/mol`;
  document.querySelector(`#note-${row.index}-${methodIndex}`).textContent = `${energy} · ${record.ring_count} rings · PB-valid`;
}

function cameraState(stage) {
  const controls = stage.viewerControls;
  return {
    rotation: controls.rotation.toArray(),
    position: controls.position.toArray(),
    distance: controls.getCameraDistance(),
  };
}

function setCameraState(stage, camera) {
  const controls = stage.viewerControls;
  controls.rotation.set(...camera.rotation);
  controls.position.set(...camera.position);
  controls.distance(camera.distance);
}

function applyInitialCamera(stage, camera) {
  if (camera?.rotation) {
    setCameraState(stage, camera);
    return;
  }
  if (camera?.orientation) {
    stage.viewerControls.orient(camera.orientation);
    return;
  }
  if (camera?.eye && camera?.center && camera?.up) {
    const eye = new NGL.Vector3(camera.eye.x, camera.eye.y, camera.eye.z);
    const center = new NGL.Vector3(camera.center.x, camera.center.y, camera.center.z);
    const up = new NGL.Vector3(camera.up.x, camera.up.y, camera.up.z);
    stage.viewerControls.align(new NGL.Matrix4().lookAt(eye, center, up));
  }
}

function connectCameras(row) {
  row.panels.forEach((panel, sourceIndex) => {
    if (!panel.stage) return;
    panel.stage.viewerControls.signals.changed.add(() => {
      if (row.syncing) return;
      row.camera = cameraState(panel.stage);
      if (row.syncFrame !== null) cancelAnimationFrame(row.syncFrame);
      row.syncFrame = requestAnimationFrame(() => {
        row.syncFrame = null;
        row.syncing = true;
        try {
          row.panels.forEach((other, index) => {
            if (index !== sourceIndex && other.stage) setCameraState(other.stage, row.camera);
          });
        } finally {
          row.syncing = false;
        }
      });
    });
  });
}

async function loadRow(row) {
  const token = ++row.loadToken;
  document.querySelector(`#capture-${row.index}`).disabled = true;
  document.querySelector(`#capture-note-${row.index}`).textContent = "";
  disposePanels(row);
  row.section.querySelectorAll(".viewer").forEach(element => {
    element.innerHTML = `<div class="status">Loading ${esc(row.target)}…</div>`;
  });
  try {
    const payload = await api(`/api/row?target=${encodeURIComponent(row.target)}`);
    if (token !== row.loadToken) return;
    row.pocketCode = payload.pocket_code;
    document.querySelector(`#pocket-label-${row.index}`).textContent =
      `Pocket ${row.index + 1} · ${payload.label}`;
    const meta = payload.selection_metadata;
    const selectionNote = payload.selection_note || (meta
      ? `open cone ${meta.open_cone_deg.toFixed(1)}°`
      : "Reference");
    const referenceEnergy = payload.reference_energy == null
      ? "No verified reference score"
      : `${app.config.energy_key} = ${payload.reference_energy.toFixed(3)} kcal/mol`;
    const pocketNote = document.querySelector(`#pocket-note-${row.index}`);
    pocketNote.textContent = `${referenceEnergy} · ${selectionNote}`;
    pocketNote.title = pocketNote.textContent;
    app.config.methods.forEach((method, index) => setInitialControl(row, method, index, payload.records[method]));

    const results = await Promise.allSettled(payload.panels.map((_, index) => buildPanel(row, index, payload)));
    const panels = results.filter(result => result.status === "fulfilled").map(result => result.value);
    const failed = results.find(result => result.status === "rejected");
    if (failed) {
      panels.forEach(panel => panel.stage?.dispose());
      throw failed.reason;
    }
    if (token !== row.loadToken) {
      panels.forEach(panel => panel.stage?.dispose());
      return;
    }
    row.panels = panels;
    row.suspended = false;
    framePocket(panels[0].stage);
    applyInitialCamera(panels[0].stage, row.camera || payload.camera);
    row.camera = cameraState(panels[0].stage);
    panels.slice(1).forEach(panel => { if (panel.stage) setCameraState(panel.stage, row.camera); });
    connectCameras(row);
    await Promise.all(app.config.methods.map((method, index) =>
      payload.records[method] ? loadRecords(row, method, index) : Promise.resolve(),
    ));
    if (token === row.loadToken) document.querySelector(`#capture-${row.index}`).disabled = false;
  } catch (error) {
    if (token !== row.loadToken) return;
    disposePanels(row);
    row.section.querySelectorAll(".viewer").forEach(element => {
      element.innerHTML = `<div class="status error">${esc(error.message)}</div>`;
    });
  }
}

async function loadRecords(row, method, methodIndex) {
  const token = row.loadToken;
  const button = document.querySelector(`#load-${row.index}-${methodIndex}`);
  const select = document.querySelector(`#ligand-${row.index}-${methodIndex}`);
  const note = document.querySelector(`#note-${row.index}-${methodIndex}`);
  button.disabled = true;
  button.textContent = "Loading…";
  try {
    const payload = await api(
      `/api/records?target=${encodeURIComponent(row.target)}&method=${encodeURIComponent(method)}`,
    );
    if (token !== row.loadToken) return;
    const currentPath = select.dataset.path;
    const currentIndex = Number(select.dataset.recordIndex);
    select.innerHTML = payload.records
      .map(record => `<option value="${record.choice}">${esc(record.label)}</option>`)
      .join("");
    const current = payload.records.find(
      record => record.path === currentPath && record.record_index === currentIndex,
    );
    if (current) select.value = String(current.choice);
    select.disabled = false;
    select.dataset.loaded = "true";
    note.textContent = current
      ? `${current.energy == null ? "No verified score" : `${app.config.energy_key} = ${current.energy.toFixed(3)}`} · ${current.ring_count} rings · PB-valid ${payload.records.length}/${payload.filter_counts.total_count}`
      : `PB-valid ${payload.records.length}/${payload.filter_counts.total_count}`;
    note.title = `${payload.filter_counts.pb_failed_count} PB failed; ${payload.filter_counts.pb_unknown_count} without verified PB result; ${payload.filter_counts.scored_count} PB-valid ligands with dock scores`;
    button.textContent = "Loaded";
  } catch (error) {
    if (token !== row.loadToken) return;
    note.textContent = error.message;
    note.classList.add("error");
    button.disabled = false;
    button.textContent = "Retry";
  }
}

async function updateLigand(row, method, methodIndex, choice) {
  const select = document.querySelector(`#ligand-${row.index}-${methodIndex}`);
  const note = document.querySelector(`#note-${row.index}-${methodIndex}`);
  if (!await activateRow(row)) return;
  const unlock = lockCaptureRow(row);
  row.updating = true;
  note.textContent = "Updating ligand…";
  try {
    const payload = await api(
      `/api/ligand?target=${encodeURIComponent(row.target)}&method=${encodeURIComponent(method)}&choice=${choice}`,
    );
    const panel = row.panels[methodIndex + 1];
    const next = await loadText(
      panel.stage,
      payload.ligand_sdf,
      "sdf",
      `${row.target}-${method}-${choice}`,
    );
    addLigandSticks(next);
    if (panel.ligand) panel.stage.removeComponent(panel.ligand);
    panel.ligand = next;
    panel.record = payload.record;
    if (row.camera) setCameraState(panel.stage, row.camera);
    select.dataset.path = payload.record.path;
    select.dataset.recordIndex = payload.record.record_index;
    const energy = payload.record.energy == null
      ? "No verified score"
      : `${app.config.energy_key} = ${payload.record.energy.toFixed(3)} kcal/mol`;
    note.textContent = `${energy} · ${payload.record.ring_count} rings · PB-valid ${payload.filter_counts.pb_valid_count}/${payload.filter_counts.total_count}`;
  } catch (error) {
    note.textContent = error.message;
    note.classList.add("error");
  } finally {
    row.updating = false;
    unlock();
  }
}

async function saveCamera(row, button) {
  if (!row.panels.length) return;
  if (!await activateRow(row)) return;
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Saving…";
  try {
    row.camera = cameraState(row.panels[0].stage);
    const payload = await api("/api/camera", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({target: row.target, camera: row.camera}),
    });
    row.camera = payload.camera;
    button.textContent = "Saved";
  } catch (error) {
    button.textContent = "Failed";
    window.alert(error.message);
  } finally {
    setTimeout(() => {
      button.textContent = original;
      button.disabled = false;
    }, 1200);
  }
}

async function resizePng(blob, size) {
  const bitmap = await createImageBitmap(blob);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  context.drawImage(bitmap, 0, 0, size, size);
  bitmap.close();
  return new Promise((resolve, reject) => canvas.toBlob(
    result => result ? resolve(result) : reject(new Error("PNG encoding failed.")),
    "image/png",
  ));
}

async function capturePng(panel) {
  const source = await panel.stage.makeImage({
    factor: Math.max(1, Math.ceil(EXPORT_SIZE_PX / Math.min(panel.stage.viewer.width, panel.stage.viewer.height))),
    antialias: true,
    trim: false,
    transparent: true,
  });
  return resizePng(source, EXPORT_SIZE_PX);
}

function pngBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

function showCaptureResult(row, result) {
  const note = document.querySelector(`#capture-note-${row.index}`);
  note.textContent = `Saved ${result.file_count} files to ${result.directory}/`;
}

function lockCaptureRow(row) {
  const rows = app.reuseContexts ? app.rows : [row];
  const controls = rows.flatMap(item => [...item.section.querySelectorAll("select, button")])
    .map(element => ({element, disabled: element.disabled}));
  controls.forEach(({element}) => { element.disabled = true; });
  rows.forEach(item => item.section.classList.add("capturing"));
  return () => {
    rows.forEach(item => item.section.classList.remove("capturing"));
    controls.forEach(({element, disabled}) => { element.disabled = disabled; });
  };
}

function captureMetadata(panel, method) {
  return {
    method,
    record: panel.record ? {path: panel.record.path, record_index: panel.record.record_index} : null,
    camera: {...cameraState(panel.stage), fov_degrees: panel.stage.getParameters().cameraFov},
  };
}

async function savePocket(row, button) {
  if (!row.panels.length || row.capturing || row.updating) return;
  if (!await activateRow(row)) return;
  const original = button.textContent;
  const unlock = lockCaptureRow(row);
  row.capturing = true;
  try {
    row.camera = cameraState(row.panels[0].stage);
    row.panels.forEach(panel => { if (panel.stage) setCameraState(panel.stage, row.camera); });
    const panels = [];
    for (let index = 0; index < row.panels.length; index += 1) {
      if (!row.panels[index].stage) {
        panels.push({method: app.config.columns[index], status: "unavailable"});
        continue;
      }
      button.textContent = `Capturing ${index + 1}/${row.panels.length}…`;
      const metadata = captureMetadata(row.panels[index], app.config.columns[index]);
      const png = await capturePng(row.panels[index]);
      panels.push({...metadata, png: await pngBase64(png)});
    }
    button.textContent = "Creating PDFs…";
    const result = await api("/api/capture", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({target: row.target, panels}),
    });
    showCaptureResult(row, result);
    button.textContent = "Saved";
  } catch (error) {
    button.textContent = "Failed";
    window.alert(error.message);
  } finally {
    row.capturing = false;
    unlock();
    setTimeout(() => { button.textContent = original; }, 1200);
  }
}

async function savePanel(row, panelIndex, label, button) {
  const panel = row.panels[panelIndex];
  if (!panel?.stage || row.capturing || row.updating) return;
  if (!await activateRow(row)) return;
  const original = button.textContent;
  const token = row.loadToken;
  const target = row.target;
  const unlock = lockCaptureRow(row);
  row.capturing = true;
  button.textContent = "Saving…";
  try {
    const metadata = captureMetadata(panel, label);
    const image = await capturePng(panel);
    const result = await api("/api/capture", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({target, method: label, panels: [{...metadata, png: await pngBase64(image)}]}),
    });
    if (token === row.loadToken) showCaptureResult(row, result);
    button.textContent = "Saved";
  } catch (error) {
    button.textContent = "Failed";
    window.alert(error.message);
  } finally {
    row.capturing = false;
    unlock();
    setTimeout(() => {
      button.textContent = original;
    }, 1200);
  }
}

async function start() {
  const rowsElement = document.querySelector("#rows");
  rowsElement.inert = true;
  try {
    if (!window.NGL) throw new Error("NGL renderer failed to load.");
    app.ligandColorScheme = NGL.ColormakerRegistry.addScheme(function () {
      this.atomColor = atom => LIGAND_ELEMENT_COLORS[atom.element.toUpperCase()] ?? LIGAND_ELEMENT_COLORS.C;
    }, "qual-pastel-elements");
    app.config = await api("/api/config");
    app.reuseContexts = app.config.columns.length * app.config.default_targets.length > 16;
    document.documentElement.style.setProperty("--display-columns", Math.min(3, app.config.columns.length));
    document.documentElement.style.setProperty("--ligand", app.config.colors.ligand);
    document.documentElement.style.setProperty("--occupancy", app.config.colors.pocket_surface);
    app.config.default_targets.forEach((target, index) => makeRow(index, target));
    for (const row of app.rows) {
      if (app.reuseContexts) {
        for (const other of app.rows) if (other !== row) await suspendRow(other);
      }
      await loadRow(row);
    }
    await activateRow(app.rows[0]);
    app.ready = true;
    new ResizeObserver(() => app.rows.forEach(row => row.panels.forEach(panel => {
      panel.stage?.handleResize();
    }))).observe(document.querySelector("main"));
  } catch (error) {
    document.querySelector("#rows").innerHTML = `<div class="status error">${esc(error.message)}</div>`;
  } finally {
    rowsElement.inert = false;
  }
}

start();
