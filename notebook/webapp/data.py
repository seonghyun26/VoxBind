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


def list_experiments() -> list[Path]:
    if not EXPS_ROOT.exists():
        return []
    return sorted(
        p for p in EXPS_ROOT.iterdir()
        if p.is_dir() and (p / "samples" / "res").is_dir()
    )


def list_targets(exp_dir: Path) -> list[Path]:
    res = exp_dir / "samples" / "res"
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
    height: int = 600,
    show_gt: bool = True,
) -> str:
    """Embed NGL.js viewer with pocket + GT (magenta) + sample (element colours).

    If `show_gt` is False, the GT ligand is loaded but hidden so toggling back
    on doesn't require a refetch.
    """
    pdb_js = json.dumps(pocket_pdb_text)
    gt_js = json.dumps(gt_sdf_text or "")
    samp_js = json.dumps(sample_sdf_text)
    show_gt_js = "true" if show_gt else "false"
    return f"""
<div id="viewport" style="width:100%; height:{height}px; border:1px solid #d0d0d0; border-radius:6px;"></div>
<script src="https://unpkg.com/ngl@2.3.1/dist/ngl.js"></script>
<script>
(function() {{
  var stage = new NGL.Stage("viewport", {{ backgroundColor: "white" }});
  window.addEventListener("resize", function() {{ stage.handleResize(); }}, false);

  var pdb_text  = {pdb_js};
  var gt_text   = {gt_js};
  var samp_text = {samp_js};
  var show_gt   = {show_gt_js};

  var pocketComp = null;
  var sampleComp = null;

  var tasks = [];
  tasks.push(stage.loadFile(
    new Blob([pdb_text], {{ type: "text/plain" }}), {{ ext: "pdb" }}
  ).then(function(c) {{
    pocketComp = c;
    c.addRepresentation("cartoon", {{ color: "lightgrey", opacity: 0.7 }});
    c.addRepresentation("surface", {{ color: "lightgrey", opacity: 0.10 }});
  }}));

  if (gt_text.length > 0) {{
    tasks.push(stage.loadFile(
      new Blob([gt_text], {{ type: "text/plain" }}), {{ ext: "sdf" }}
    ).then(function(c) {{
      c.addRepresentation("ball+stick", {{ color: "magenta", aspectRatio: 1.5 }});
      c.setVisibility(show_gt);
    }}));
  }}

  tasks.push(stage.loadFile(
    new Blob([samp_text], {{ type: "text/plain" }}), {{ ext: "sdf" }}
  ).then(function(c) {{
    sampleComp = c;
    c.addRepresentation("ball+stick", {{ aspectRatio: 1.5, multipleBond: "symmetric" }});
  }}));

  Promise.all(tasks).then(function() {{
    // Center the camera on pocket + sample only (ignore GT framing).
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
