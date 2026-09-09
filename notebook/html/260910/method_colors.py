"""method_colors.py — one colour per generative method, shared by every 260910 figure.

A reader moving between the Vina figures, the PoseBusters figures and the PoseCheck
figures should never have to re-learn the key. So the assignment lives here and the
builders import it; nothing defines a method's colour locally.

THE FIXED POINTS (set 2026-09-09, and the reason the rest moved):

  VoxBind + Ours   #4363D8  blue        — OUR blue: the saturated blue the CDG charts and
                                          the PoseCheck figures already carry (the Vina
                                          3-line figures use #8291E8, a lighter tint of
                                          the same family, and still read as ours)
  VoxBind          #F5B27E  sand        — VoxBind's existing orange, same figures
  DecompDiff       #3CB44B  green
  TargetDiff       #B58FDB  violet

Two of those forced changes elsewhere, which is the whole point of centralising this:

  * TargetDiff used to be #f58231 orange in the PoseCheck figures, which put it a few
    degrees of hue from VoxBind's sand -- two BASELINES reading as one family. Violet ends
    that, and it is far from both the blue and the green.
  * AR used to be #9b59b6 purple, which violet now sits on top of. AR takes a teal-cyan
    instead: no other method is anywhere near that hue.
  * Ours v1 used to be green #3cb44b in the PoseCheck figures. It hands the green to
    DecompDiff and takes our blue, so "ours" is one colour everywhere.

Ours v2 is a deeper shade of the same blue rather than a hue of its own, because it IS the
same family as Ours v1 and should read as one. It is deliberately DARKER and not lighter:
TargetDiff's violet is a light tint, and a light blue beside it is the one pairing in this
palette that genuinely does not separate. Dark blue / mid blue / light violet is a
lightness ladder, and that reads at a glance even where the hues are neighbours.

The five published baselines keep a second identity channel (line style) in the figures
that draw them, so their colours only have to be distinguishable, not maximally separated.

    from method_colors import color
    color("Ours · v1")     # -> "#4363D8", via the alias table
"""

COLORS = {
    "Reference ligand": "#9AA0A6",   # grey, and dashed wherever it is drawn
    "AR":               "#17A2B8",   # teal-cyan — moved off purple for TargetDiff
    "Pocket2Mol":       "#E87BA4",   # pink
    "DiffSBDD":         "#E34948",   # red
    "DecompDiff":       "#3CB44B",   # green
    "FuncBind":         "#A9744F",   # brown
    "TargetDiff":       "#B58FDB",   # violet
    "VoxBind":          "#F5B27E",   # sand
    "VoxBind + Ours":   "#4363D8",   # blue — ours
    "Ours v2":          "#2B3A8C",   # deep indigo — same family as Ours v1
}

# The same method is spelled several ways across the builders and the HTML. Every spelling
# resolves here rather than each builder carrying its own hard-coded hex.
ALIASES = {
    "Reference": "Reference ligand",
    "reference": "Reference ligand",
    "DecompDiff_ref_prior": "DecompDiff",
    "DecompDiff (ref-informed)": "DecompDiff",
    "targetdiff": "TargetDiff",
    "vanilla": "VoxBind",
    "VoxBind σ=0.9": "VoxBind",
    "VoxBind σ0.9": "VoxBind",
    "VoxBind sigma=0.9": "VoxBind",
    "ours_v1": "VoxBind + Ours",
    "Ours": "VoxBind + Ours",
    "Ours · v1": "VoxBind + Ours",
    "Ours &middot; v1": "VoxBind + Ours",
    "Ours v1": "VoxBind + Ours",
    "ours_v2": "Ours v2",
    "Ours · v2": "Ours v2",
    "Ours &middot; v2": "Ours v2",
}


def color(label):
    """Colour for a method, by any of its spellings. Raises rather than returning a
    default: a silently grey method in a figure is worse than a failed build."""
    key = ALIASES.get(label, label)
    if key not in COLORS:
        raise KeyError(f"no colour for method {label!r} "
                       f"(known: {', '.join(sorted(COLORS))})")
    return COLORS[key]
