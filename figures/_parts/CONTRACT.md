# Porting contract — one figure family per part file

We are consolidating `notebook/html/260910/*/build_*.py` into a single
`/home1/irteam/VoxBind/figures/draw.py`. You port ONE family.

## What you write

`/home1/irteam/VoxBind/figures/_parts/<family>.py` — its contents are **appended verbatim**
into `draw.py`. Therefore it must contain **only**:

* a `# ═══ ... ═══` banner comment naming the figure ids,
* family-local constants, prefixed so they cannot collide (`PB_`, `PC_`, `RF_`, ...),
* private helpers, prefixed `_<family>_`,
* the `@figure("fig-<eval>-<name>", needs=(...))`-decorated draw functions.

It must **NOT** contain: `import` statements, `if __name__ == "__main__"`, a re-definition
of any shared name below, or `sys.path` juggling. It is not a runnable script.

## Shared names already defined above your part in draw.py

Modules: `argparse collections csv json math os statistics as st sys time`, `Path`,
`matplotlib`, `plt`, `np`, `Line2D`, `Patch`, `MaxNLocator`, `MultipleLocator`.

Paths: `HERE` (= VoxBind/figures), `REPO`, `OUT_ROOT` (= figures/260910), `LEGACY`
(= notebook/html/260910, where the exported CSV/JSON artifacts live), `E` (= voxbind/exps).

Palette: `COLORS ALIASES DISPLAY color(label) display(label)`, and the lighter steps
`SOFT PALE soft(label) pale(label)`. Use these rather than writing a tint's hex in a part.

Style: `INK GRID AXIS LEGEND_EDGE SOLID DASH DOT MODEL_LW REF_LW AXIS_LW GRID_LW DIST_LW
DIST_FILL WIDE TALL FIG_W PANEL_H STACK_H HEIGHT_RATIOS H_PAD XTICK_STEP X_LABEL RC
use_style() furniture(ax,...) legend(target,handles,...) arm_handles(arms,...) fit(fig,...)
plot_width(fig,ax)`.

Pose data: `ARMS CORE REF_LABEL REF_COLOR REF_ROOT P79 EDGES BIN_LABELS MIN_N REF_WIN
MIN_REF rows_of(dir,reference=) pose_data() arms_for(field,data,arms=) variants(field,data)
by_size(rows,field,key=) bin_of(n) model_curve(per,xs,f,win=) reference_curve(per,xs,f)
x_range(per_arm,arms) share(per,xs) rolled(values,xs) size_distribution(ax,xs,per,ref,arms)`.

Registry: `figure(fig_id, needs=())` decorator, `save(fig, out, stem)`,
`write_csv(out, stem, header, rows)`, `legacy(*parts)`.

## Translation rules

| original | port to |
|---|---|
| `pc.load_arms()` / `DATA, P79_ROWS, REFROWS = ...` at module level | `data, p79_rows, refrows = pose_data()` **inside** the draw function |
| `pc.save(fig, HERE, stem, variant)` | `save(fig, out, f"{stem}_{variant}")` |
| `pc.<anything>` / `from method_colors import ...` | the bare shared name |
| writing a CSV/JSON beside the figure | `write_csv(out, stem, header, rows)`; drop JSON exports (the recompute scripts still own those) |
| reading an exported artifact next to the builder | `legacy("<folder>", "<file>")` |
| module-level heavy work | move inside the draw function |

Each draw function has signature `def draw_x(out):` where `out` is a `Path` to
`figures/260910/<fig_id>/`, already created by the caller. Keep the original's `print()`
progress lines — indent them two spaces. Keep the original's design-rationale comments;
they are the reason the figures look the way they do. Do not restyle or "improve" a figure.

## Acceptance test — non-negotiable

A regenerated PNG must be **byte-identical** to the one the original builder produced in
`notebook/html/260910/<family>/`. Verify like this:

```bash
cd /home1/irteam/VoxBind
cat figures/draw.py figures/_parts/<family>.py > /tmp/_t_<family>.py
/opt/conda/envs/voxbind/bin/python - <<'PY'
import sys; sys.path.insert(0, "/tmp")
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("t", "/tmp/_t_<family>.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for fid, fn in m.FIGURES.items():
    if not fid.startswith("fig-<eval>"): continue
    out = pathlib.Path("/tmp/_out")/fid; out.mkdir(parents=True, exist_ok=True); fn(out)
PY
# then compare every PNG against notebook/html/260910/<family>/<stem>.png with cmp
```

If a PNG differs, find out why and fix it — a difference means the port changed the figure.
Report any stem you could NOT make identical, and why. Do not silently drop a figure.

The python is `/opt/conda/envs/voxbind/bin/python` (matplotlib 3.10.9, numpy 2.4.6).
