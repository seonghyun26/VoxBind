"""Reusable initialization for the qualitative-results notebook."""

from pathlib import Path
import json
import importlib
import os
import sys


def initialize():
    """Build a complete notebook context, independent of cell execution order."""
    figure_dir = Path(__file__).resolve().parent
    repo_root = figure_dir.parent.parent
    runtime = repo_root / ".cache/fig-qual"
    if sys.version_info[:2] == (3, 12) and (runtime / "python").is_dir():
        local_python = str(runtime / "python")
        if local_python not in sys.path:
            sys.path.insert(0, local_python)
        # Kaleido's Python subprocess also needs the isolated dependencies.
        if local_python not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
            os.environ["PYTHONPATH"] = os.pathsep.join(
                p for p in (local_python, os.environ.get("PYTHONPATH", "")) if p
            )
    chrome = runtime / "chrome/chrome-linux64/chrome"
    if chrome.is_file():
        os.environ.setdefault("BROWSER_PATH", str(chrome))
        libs = str(runtime / "browser-libs/usr/lib/x86_64-linux-gnu")
        if libs not in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
                p for p in (libs, os.environ.get("LD_LIBRARY_PATH", "")) if p
            )
    fonts = runtime / "fonts.conf"
    if fonts.is_file():
        os.environ.setdefault("FONTCONFIG_FILE", str(fonts))

    if str(figure_dir) not in sys.path:
        sys.path.insert(0, str(figure_dir))
    import plotly.io as pio
    from IPython.display import display
    import qual_utils
    importlib.reload(qual_utils)
    from qual_utils import Catalog, QualitativeGrid, fig1_style, make_table, method_specs

    # The overview was renamed from fig1.ipynb; support both repository layouts.
    style_path = next((figure_dir / name for name in ("fig-overview.ipynb", "fig1.ipynb")
                       if (figure_dir / name).is_file()), None)
    if style_path is None:
        raise FileNotFoundError(
            f"No overview style notebook found in {figure_dir}; "
            "expected fig-overview.ipynb or fig1.ipynb."
        )
    style = fig1_style(style_path)
    style["_source_name"] = style_path.name
    baseline_root = repo_root.parent / "base_drug"
    specs = method_specs(repo_root, baseline_root)
    catalog = Catalog(specs)
    return dict(
        Path=Path, json=json, pio=pio, display=display,
        REPO_ROOT=repo_root, FIGURE_DIR=figure_dir, BASELINE_ROOT=baseline_root,
        STYLE_PATH=style_path, STYLE=style, METHOD_SPECS=specs, catalog=catalog,
        Catalog=Catalog, QualitativeGrid=QualitativeGrid, make_table=make_table,
        method_specs=method_specs, fig1_style=fig1_style,
    )
