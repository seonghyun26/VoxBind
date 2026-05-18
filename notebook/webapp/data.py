"""Directory scanning, structure loading, and NGL.js HTML generation."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


HERE = Path(__file__).resolve()
VOXBIND = next((p for p in HERE.parents if p.name == "VoxBind"), None)
if VOXBIND is None:
    raise RuntimeError(f"Could not locate VoxBind project root from {HERE}")
TARGETDIFF = VOXBIND.parent / "TargetDIff"
EXPS_ROOT = VOXBIND / "voxbind" / "exps"

for _p in (str(VOXBIND), str(TARGETDIFF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def list_sample_runs(exp_dir: Path) -> list[str]:
    """Return the names of sampling-run subdirs under <exp>/samples/ that
    actually contain target_* directories (e.g. 'res', 'res_ep5007_test')."""
    samples = exp_dir / "samples"
    if not samples.exists():
        return []
    return sorted(
        d.name for d in samples.iterdir()
        if d.is_dir() and any(d.glob("target_*"))
    )


def list_experiments() -> list[Path]:
    """Experiments that have at least one non-empty sampling run."""
    if not EXPS_ROOT.exists():
        return []
    return sorted(
        p for p in EXPS_ROOT.iterdir()
        if p.is_dir() and list_sample_runs(p)
    )


def list_targets(exp_dir: Path, run: str = "res") -> list[Path]:
    """List target_* dirs for a given sampling run (defaults to 'res')."""
    res = exp_dir / "samples" / run
    if not res.exists():
        return []
    return sorted(t for t in res.glob("target_*") if t.is_dir())


def find_pocket_pdb(target_dir: Path) -> Path | None:
    return next(iter(sorted(target_dir.glob("*_pocket10.pdb"))), None)


def find_gt_sdf(target_dir: Path) -> Path | None:
    return next(
        (p for p in target_dir.glob("*.sdf") if p.name != "samples.sdf"),
        None,
    )


def load_gt_mols(target_dir: Path) -> list[Chem.Mol]:
    """Return all valid reference ligands from the target's GT SDF.

    The TargetDiff convention is one ligand per pocket, but the SDF format
    supports multiple — we surface all of them so the caller can decide.
    """
    sdf = find_gt_sdf(target_dir)
    if sdf is None:
        return []
    return [m for m in Chem.SDMolSupplier(str(sdf), sanitize=True) if m is not None]


def parse_pocket_pdb(pdb_path: Path) -> tuple[np.ndarray, list[str]]:
    coords: list[list[float]] = []
    elems: list[str] = []
    for line in open(pdb_path):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        elem = line[76:78].strip() if len(line) > 76 else ""
        if not elem:
            elem = re.sub(r"[^A-Za-z]", "", line[12:16]).strip()[:2].capitalize()
        if elem in ("H", "D"):
            continue
        coords.append([x, y, z])
        elems.append(elem)
    return np.array(coords, dtype=float), elems


def load_samples(target_dir: Path) -> list[Chem.Mol]:
    """Return valid, connected RDKit mols (Hs stripped, matches notebook §5)."""
    sdf = target_dir / "samples.sdf"
    if not sdf.exists():
        return []
    out: list[Chem.Mol] = []
    for m in Chem.SDMolSupplier(str(sdf), sanitize=True):
        if m is None:
            continue
        try:
            smi = Chem.MolToSmiles(m)
        except Exception:
            continue
        if "." in smi:
            continue
        out.append(m)
    return out


def count_raw_samples(target_dir: Path) -> int:
    sdf = target_dir / "samples.sdf"
    if not sdf.exists():
        return 0
    return sum(1 for _ in Chem.SDMolSupplier(str(sdf), sanitize=False))


def mol_to_sdf_text(mol: Chem.Mol) -> str:
    return Chem.MolToMolBlock(mol)


def ngl_html(
    pocket_pdb_text: str,
    gt_sdf_text: str,
    sample_sdf_text: str,
) -> str:
    """Embed NGL.js viewer with pocket + GT (#8C9ED9) + sample (element colours).

    The viewer fills its iframe (100% width/height) so it stays responsive on
    narrow displays — the host iframe controls the actual pixel size.

    The GT-visibility toggle lives inside the iframe (HTML checkbox driving
    `gtComp.setVisibility`) so flipping it does NOT trigger a Streamlit rerun
    and therefore doesn't reset the camera. State is persisted in
    `localStorage` so it survives sample-switches (which DO rebuild the iframe).
    """
    pdb_js = json.dumps(pocket_pdb_text)
    gt_js = json.dumps(gt_sdf_text or "")
    samp_js = json.dumps(sample_sdf_text)
    has_gt_js = "true" if gt_sdf_text else "false"
    return f"""
<style>
  html, body {{ margin:0; padding:0; height:100%; overflow:hidden; }}
  #layout {{
    display:flex; flex-direction:column; gap:10px;
    width:100%; height:100%; box-sizing:border-box;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  }}
  /* Representations: a long horizontal bar sitting above the viewer. */
  #toggle-panel {{
    flex:none; box-sizing:border-box;
    display:flex; flex-flow:row wrap; align-items:center; gap:6px 22px;
    background:#fff; padding:8px 14px;
    border-radius:6px; border:1px solid #d0d0d0;
    font-size:13px; user-select:none;
    box-shadow:0 1px 3px rgba(0,0,0,0.08);
  }}
  #toggle-panel .hdr {{
    font-weight:600; font-size:11px; letter-spacing:0.04em;
    text-transform:uppercase; color:#666;
  }}
  #toggle-panel label {{
    display:flex; align-items:center; cursor:pointer; white-space:nowrap;
  }}
  #toggle-panel input {{ margin:0 6px 0 0; }}
  #viewport {{
    flex:1; min-height:0; width:100%; box-sizing:border-box;
    border:3px solid #888; border-radius:8px;
  }}
</style>
<div id="layout">
  <div id="toggle-panel">
    <div class="hdr">Representations</div>
    <label><input type="checkbox" id="poc-bs-toggle">Pocket: ball &amp; stick</label>
    <label><input type="checkbox" id="poc-surf-toggle" checked>Pocket: surface</label>
    <label id="gt-toggle-wrap" style="display:none;">
      <input type="checkbox" id="gt-toggle" checked>Reference ligand
    </label>
  </div>
  <div id="viewport"></div>
</div>
<script src="https://unpkg.com/ngl@2.3.1/dist/ngl.js"></script>
<script>
(function() {{
  var stage = new NGL.Stage("viewport", {{ backgroundColor: "white" }});
  window.addEventListener("resize", function() {{ stage.handleResize(); }}, false);

  var pdb_text = {pdb_js};
  var gt_text  = {gt_js};
  var samp_text = {samp_js};
  var has_gt   = {has_gt_js};

  var pocketComp = null, gtComp = null, sampleComp = null;
  var pocketBSRep = null, pocketSurfRep = null;
  var GT_KEY = "voxbind-show-gt";
  var BS_KEY = "voxbind-poc-bs";
  var SURF_KEY = "voxbind-poc-surf";

  // Restore persisted toggle choices. Defaults: GT on, ball+stick off,
  // surface on (matches the prior always-on surface behaviour).
  function restore(key, dflt) {{
    try {{
      var s = window.localStorage.getItem(key);
      if (s !== null) return (s === "true");
    }} catch (e) {{}}
    return dflt;
  }}
  var initialShowGt   = restore(GT_KEY, true);
  var initialShowBS   = restore(BS_KEY, false);
  var initialShowSurf = restore(SURF_KEY, true);

  // Wire toggles. Changing any never re-mounts NGL, so the camera is kept.
  var gtWrap   = document.getElementById("gt-toggle-wrap");
  var gtBox    = document.getElementById("gt-toggle");
  var bsBox    = document.getElementById("poc-bs-toggle");
  var surfBox  = document.getElementById("poc-surf-toggle");

  bsBox.checked = initialShowBS;
  bsBox.addEventListener("change", function(ev) {{
    var v = ev.target.checked;
    try {{ window.localStorage.setItem(BS_KEY, v); }} catch (e) {{}}
    if (pocketBSRep) pocketBSRep.setVisibility(v);
  }});

  surfBox.checked = initialShowSurf;
  surfBox.addEventListener("change", function(ev) {{
    var v = ev.target.checked;
    try {{ window.localStorage.setItem(SURF_KEY, v); }} catch (e) {{}}
    if (pocketSurfRep) pocketSurfRep.setVisibility(v);
  }});

  if (has_gt) {{
    gtWrap.style.display = "";
    gtBox.checked = initialShowGt;
    gtBox.addEventListener("change", function(ev) {{
      var v = ev.target.checked;
      try {{ window.localStorage.setItem(GT_KEY, v); }} catch (e) {{}}
      if (gtComp) gtComp.setVisibility(v);
    }});
  }}

  var tasks = [];
  tasks.push(stage.loadFile(
    new Blob([pdb_text], {{ type: "text/plain" }}), {{ ext: "pdb" }}
  ).then(function(c) {{
    pocketComp = c;
    c.addRepresentation("cartoon", {{ color: "lightgrey", opacity: 0.7 }});
    // Ball & stick with aspectRatio 1.0 => ball radius == stick radius
    // (uniform-radius "tube" look), fixed 0.3 A size.
    pocketBSRep = c.addRepresentation("ball+stick", {{
      color: "lightgrey", aspectRatio: 1.0,
      radiusType: "size", radius: 0.3, multipleBond: "symmetric"
    }});
    pocketSurfRep = c.addRepresentation("surface", {{ color: "lightgrey", opacity: 0.10 }});
    pocketBSRep.setVisibility(initialShowBS);
    pocketSurfRep.setVisibility(initialShowSurf);
  }}));

  if (has_gt) {{
    tasks.push(stage.loadFile(
      new Blob([gt_text], {{ type: "text/plain" }}), {{ ext: "sdf" }}
    ).then(function(c) {{
      gtComp = c;
      c.addRepresentation("ball+stick", {{ color: "#8C9ED9", aspectRatio: 1.5 }});
      c.setVisibility(initialShowGt);
    }}));
  }}

  tasks.push(stage.loadFile(
    new Blob([samp_text], {{ type: "text/plain" }}), {{ ext: "sdf" }}
  ).then(function(c) {{
    sampleComp = c;
    c.addRepresentation("ball+stick", {{ aspectRatio: 1.5, multipleBond: "symmetric" }});
  }}));

  Promise.all(tasks).then(function() {{
    // Always frame the camera on pocket + sample, regardless of GT visibility.
    if (pocketComp && sampleComp) {{
      var box = pocketComp.getBox().clone();
      box.union(sampleComp.getBox());
      stage.animationControls.zoomMove(
        box.getCenter(new NGL.Vector3()),
        stage.getZoomForBox(box),
        0
      );
    }} else {{
      stage.autoView(0);
    }}
  }});
}})();
</script>
"""
