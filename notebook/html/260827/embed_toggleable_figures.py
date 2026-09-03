"""embed_toggleable_figures.py — make Figures 5-7 of results_drug_design.html toggleable.

The three PoseCheck figures are re-emitted as INLINE svg instead of <img>, so a checkbox
row can show or hide one method without re-rendering anything.  Each series artist
carries a gid of the form ``sr-<key>-<part>`` (set in plot_posecheck_fig6_fig7.py and
build_strain_by_size_curve.py); the SVG backend writes that out as ``id=``, and the page
toggles a method by hiding every element under its id prefix.

Why the ids are namespaced: matplotlib emits glyphs once into <defs> and references them
with <use xlink:href="#DejaVuSans-xx">.  Three figures in one document share 114 such ids,
so an un-namespaced inline embed makes every <use> resolve against the FIRST figure's
defs and the later figures render the wrong characters.  Each figure therefore gets its
own id prefix, references included.

Re-runnable: the block between the two markers is replaced wholesale on each run, so the
figures can be regenerated and re-embedded without the file accumulating copies.

    /opt/conda/envs/voxbind/bin/python notebook/html/260827/embed_toggleable_figures.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(os.path.dirname(HERE), "results_drug_design.html")

# Swatch colours match the curves; "reference" is the crystal-ligand series, drawn as a
# dark bar in Figure 6 and grey stars in Figures 5 and 7.
SERIES = [
    ("reference", "Reference ligand", "#2b3440"),
    ("targetdiff", "TargetDiff", "#3a7bd5"),
    ("vanilla", "VoxBind σ=0.9", "#e08a1e"),
    ("ours", "Ours", "#1e7a4d"),
    ("ours_v2", "Ours · v2", "#7fc7a2"),
]

FIGURES = [
    ("fig5", "denovo78_posecheck_fig6_strain_ecdf.svg"),
    ("fig6", "denovo78_posecheck_fig7_steric_clashes.svg"),
    ("fig7", "strain_by_size_curve.svg"),
]


def namespace_svg(svg: str, token: str) -> str:
    """Prefix every id and every #reference so three SVGs can share one document."""
    ids = set(re.findall(r'id="([^"]+)"', svg))

    def sub_id(m):
        return f'id="{token}-{m.group(1)}"'

    svg = re.sub(r'id="([^"]+)"', sub_id, svg)

    # url(#x), href="#x" and xlink:href="#x" all have to follow their target.
    def sub_ref(m):
        name = m.group(2)
        return f"{m.group(1)}#{token}-{name}" if name in ids else m.group(0)

    svg = re.sub(r'(url\(|href=")#([^)"]+)', sub_ref, svg)

    # Strip the XML prolog and DOCTYPE: valid in a standalone file, not inside HTML.
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg)
    # Let CSS drive the size instead of the hard-coded pt width/height.
    svg = re.sub(r'(<svg[^>]*?)\swidth="[^"]*"\s+height="[^"]*"', r"\1", svg, count=1)
    return svg.strip()


def toggle_row(fig_token: str, svg: str) -> str:
    """A checkbox per series actually present in this figure. Doubles as the legend."""
    items = []
    for key, label, color in SERIES:
        if f'id="{fig_token}-sr-{key}-' not in svg:
            continue
        items.append(
            f'<label class="fig-toggle-item"><input type="checkbox" checked '
            f'data-series="{key}">'
            f'<span class="fig-swatch" style="background:{color}"></span>{label}</label>'
        )
    return (
        f'<div class="fig-toggle" role="group" aria-label="show or hide a method">'
        + "".join(items)
        + "</div>"
    )


CSS_JS = """
<style id="fig-toggle-style">
  .fig-toggle{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;
    margin:10px 0 2px;padding:9px 12px;border:1px solid #e3e0d8;border-radius:6px;
    background:#fbfaf7;font-size:13px}
  .fig-toggle-item{display:inline-flex;align-items:center;gap:7px;cursor:pointer;
    user-select:none;color:#31404d}
  .fig-toggle-item input{cursor:pointer;margin:0}
  .fig-toggle-item:has(input:not(:checked)){opacity:.45}
  .fig-swatch{width:22px;height:4px;border-radius:2px;display:inline-block}
  .fig-inline-svg svg{width:100%;height:auto;display:block}
</style>
<script id="fig-toggle-script">
(function () {
  // A series owns every element whose id starts with "<fig>-sr-<key>-". The trailing
  // hyphen is what keeps "ours" from also matching "ours_v2".
  function apply(box) {
    var block = box.closest(".fig-block");
    if (!block) return;
    var svg = block.querySelector("svg");
    if (!svg) return;
    // The namespace token lives on the container: matplotlib's root <svg> carries no id.
    var prefix = block.dataset.fig + "-sr-" + box.dataset.series + "-";
    var hits = svg.querySelectorAll('[id^="' + prefix + '"]');
    for (var i = 0; i < hits.length; i++) {
      hits[i].style.display = box.checked ? "" : "none";
    }
  }
  document.addEventListener("change", function (ev) {
    var box = ev.target;
    if (box && box.matches && box.matches(".fig-toggle input[type=checkbox]")) apply(box);
  });
})();
</script>
"""


def main():
    doc = open(DOC, encoding="utf-8").read()

    for token, filename in FIGURES:
        svg = namespace_svg(open(os.path.join(HERE, filename), encoding="utf-8").read(), token)
        png = filename.replace(".svg", ".png")
        pattern = re.compile(
            r'<div class="table-wrap" style="padding:16px"><img src="260827/'
            + re.escape(png)
            + r'".*?></div>',
            re.S,
        )
        if not pattern.search(doc):
            print(f"  {token}: <img> block not found (already embedded?) — skipped")
            continue
        block = (
            f'<div class="fig-block" data-fig="{token}">'
            + toggle_row(token, svg)
            + '<div class="table-wrap fig-inline-svg" style="padding:16px">'
            + svg
            + "</div></div>"
        )
        doc = pattern.sub(lambda _m: block, doc, count=1)
        print(f"  {token}: embedded {filename} ({len(svg):,} chars)")

    if 'id="fig-toggle-style"' not in doc:
        doc = doc.replace("</body>", CSS_JS + "\n</body>", 1)
        print("  injected toggle CSS + JS")

    open(DOC, "w", encoding="utf-8").write(doc)
    print(f"  wrote {DOC} ({len(doc):,} chars)")


if __name__ == "__main__":
    main()
