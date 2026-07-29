#!/usr/bin/env python3
"""Embed the original results.html Figure 1 chart in the 260730 meeting page.

The chart remains single-sourced in 260715/build_appendixB_bar.py, which is also
used by build_results.py. This wrapper injects that generator's SVG directly
into 260730_meeting.html, matching the self-contained prior meeting slides.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "260715" / "build_appendixB_bar.py"
MEETING = HERE / "260730_meeting.html"
START = "      <!-- AFFINITY_BAR_START -->"
END = "      <!-- AFFINITY_BAR_END -->"


def load_source():
    spec = importlib.util.spec_from_file_location("affinity_bar_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load chart source: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    bar = load_source()
    svg = bar.svg(rank_labels=True, value_decimals=2)
    html = MEETING.read_text(encoding="utf-8")
    start = html.index(START) + len(START)
    end = html.index(END, start)
    embedded = "\n      " + svg + "\n"
    html = html[:start] + embedded + html[end:]
    MEETING.write_text(html, encoding="utf-8")
    print(f"embedded Figure 1 in {MEETING} ({MEETING.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
