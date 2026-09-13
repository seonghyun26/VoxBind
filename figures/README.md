# figures/ — the paper's figures, and the one command that draws them

```bash
/opt/conda/envs/voxbind/bin/python figures/draw.py --list          # what there is
/opt/conda/envs/voxbind/bin/python figures/draw.py fig-vina-3line-dock
/opt/conda/envs/voxbind/bin/python figures/draw.py posecheck       # a whole evaluation
/opt/conda/envs/voxbind/bin/python figures/draw.py --all --keep-going
```

Figures are grouped by **evaluation** — one folder per question, named by it:

```
figures/fig-vina/              dock/score/min per pocket, and Vina against ligand size
figures/fig-posebusters/       validity, which checks fail, SuCOS
figures/fig-posecheck/         strain, clashes and interaction fingerprints
figures/fig-rigid-consistency/ rigid-fragment RMSD against fragment size
figures/fig-molweight/         where each arm puts its molecules
figures/fig-similarity/        similarity and novelty against the crystal ligand
figures/fig-mcp/               the macrocyclic-peptide fine-tune ladder
figures/fig-ensemble/          champion + partner encoder cohorts
```

A file is named by the figure, minus whatever its folder already says — so
`fig-posecheck/` holds `clash-per-atom-mean-all.png`, `strain-ecdf-by-size-le15.png`,
`strain-box-per-rotbond-core.png`, and `fig-rigid-consistency/` holds `violin-all.png`
rather than `rigid_violin_all.png`. The `core` / `all` suffix is the arm set and stays: two
figures that differ in which arms they draw must not be able to share a name. Each folder
also carries the CSVs behind its figures; where two figures export the same table (the pose
per-atom table, the interaction table) it is now written once instead of once per figure.

The interaction fingerprints live in `fig-posecheck/` under an `interaction-` prefix,
because that is what produces them: PoseCheck runs ProLIF over the same pocket10 crop and
writes the fingerprint into the same `posecheck` block of the same `metrics.json` that
strain and clashes come from (`compute_baseline_interactions.py` imports PoseCheck directly
to make it for the five baselines, which carry no posecheck block of their own).
`draw.py interaction` still selects exactly those six.

A name on the command line may be a **figure id** (`fig-posecheck-clash-per-atom`) or a
whole **evaluation** (`posecheck`) — ids stay `fig-<eval>-<name>` so one figure is still
addressable on its own; only the output folder is the evaluation.

PNG to look at, SVG and PDF to place; both vector formats keep live text (`svg.fonttype:
none`, `pdf.fonttype: 42` — Type 3 is what journals reject and Illustrator mangles).
`--formats png` skips the vector pair when you only want to look.

The names on disk are not the names the original builders used. `STEMS` in `draw.py` is the
map between the two trees — which is how the port was verified figure by figure. Families
call `save()` with the original name; the rename happens once, at the writing edge.

## Where the code is

`draw.py` is one file, on purpose: the palette, the axis furniture and the per-molecule
loader used to be copied into ten `build_*.py` scripts that could drift apart, and a
reader moving between the Vina, PoseBusters and PoseCheck figures should never have to
re-learn the key. It is *edited* in pieces under `_parts/` and concatenated:

```bash
bash figures/_parts/assemble.sh      # _parts/*.py -> draw.py
```

**Never hand-edit `draw.py`** — it is generated, and assemble.sh will overwrite you. Edit
the family's part file. `_parts/00_core.py` holds everything shared (palette, style, the
pose-data loader, the registry, the Vina 3-line family); `_parts/zz_cli.py` is the runner
and always goes last. `_parts/CONTRACT.md` says what a part file may and may not contain.

Adding a figure is one function:

```python
@figure("fig-posecheck-something", needs=("metrics.json",))
def draw_something(out):
    """One line, shown by --list."""
    ...
    save(fig, out, "stem_name")
```

## What this does NOT do

It draws; it does not re-score. The per-molecule inputs — `metrics.json` under each run
tree, `eval_docking_results_full79.json`, the PoseCheck/PoseBusters exports — come from
`voxbind/scripts/73_evaluate_samples.sh`, `85_fill_pose_eval_4runs.sh`,
`86_pose_eval_baselines.sh` and the docking driver. The original builders under
`notebook/html/260910/*/build_*.py` remain the source of truth for **data**; this file is
the source of truth for **pictures**.

The expensive per-molecule load (~60k molecules over eight run trees) happens **once per
run** no matter how many figures ask for it — which is the other reason the families share
one process.

## How the port was verified (2026-09-13)

Every figure here was checked against the one its original `build_*.py` left in
`notebook/html/260910/`, by sha256. The bar was byte-identical: a difference would mean the
consolidation changed a figure, which is the one thing it was not allowed to do. Result:

**69 byte-identical, 16 stale-committed, 0 differing, 2 not drawn** — every figure that can
be drawn on this box reproduces its original exactly.

For the 16 "stale-committed" the CHECKED-IN PNG is the wrong reference: running the
original builder today reproduces this tree byte for byte. Four verified reasons, and three
of them are latent problems in the originals worth knowing about:

| n | reason |
|---|---|
| 6 | `sucos_*` — inputs re-scored after the PNG was committed (its SVG already legends "CoDE", so this is not the rename below) |
| 6 | `strain_*rotbond*` — **pre-2026-09-10 legend labels.** Every differing pixel is inside the legend frame; the committed SVGs still say `>VoxBind + Ours<` and a plain `>VoxBind<` where the palette now renders `CoDE` and `VoxBind$_{\sigma=0.9}$` |
| 3 | `similarity_{b,c,d}` — these sit behind `--all` in the builder, labelled in its source as "the three treatments that were not chosen", so a bare run leaves them untouched and still looks successful |
| 1 | `mcp_finetune_size` — **committed under matplotlib 3.7.3**, the only file in the tree that is. `bbox_inches="tight"` sizes the raster from freetype's text extents, so the 3.10.9 stack cannot reproduce the old pixel height |

Two carry an action:

* **`build_strain_per_rotbond.py` no longer runs at all on its `all` variant.** It calls
  `pc.variants()` without a field, asking for all 8 arms when only TargetDiff / VoxBind /
  CoDE carry PoseCheck strain, and matplotlib rejects the boxplot length mismatch — it
  predates the five published baselines being prepended to `pose_common.ARMS`. The port
  passes the field, which is what `arms_for`'s own docstring prescribes, so there the port
  is the correction, not a regression.
* **`mcp_finetune_size` reproduces exactly under the old stack**, which doubles as a check
  that `draw.py` still runs on Python 3.8 / matplotlib 3.7.3:

  ```bash
  /opt/conda/envs/voxdock/bin/python figures/draw.py fig-mcp-finetune-size -o /tmp/x
  ```

If you ever need to re-run that comparison: a PNG's `tEXt` chunk names the matplotlib that
wrote it, which makes renderer drift a one-second triage before chasing a port bug (census
of the 97 committed figures: 84 say 3.10.9, ten `_baselines_only/` say 3.10.8, the two
ensemble ones 3.10.1, `mcp_finetune_size` alone 3.7.3). And read the committed side out of
git (`git cat-file blob HEAD:<path>`), never off disk — these builders set `OUT = HERE` and
write into the folder they live in, so an in-place re-run overwrites the very file you are
comparing against and every stem then looks unchanged.

`STEMS` in `draw.py` is the map between the two trees' filenames, which is what made this
comparison possible at all.

## Known gap

`fig-ensemble-6set-3metric` and `fig-ensemble-vs-alone` cannot be drawn on this box.
`gen_ensemble_data.py` wrote `ensemble_results.json` under `/home/shpark/prj-denovo/`, and
neither that JSON nor the frozen-encoder feature files it derives from are here or on
Dropbox — only the finished PNG/PDF travelled. The two functions are registered and fail
with the missing path rather than silently drawing nothing; recover the JSON from the box
that made it, drop it in `notebook/html/260910/fig-ensemble/`, and they will draw.

## The other figures in this folder

These are not part of `draw.py` — they are notebooks and a viewer, kept here so every
figure in the paper lives under one roof:

* `fig-overview/` — `fig-overview.ipynb`, the real-data 3D voxel viewer (ligand · pocket ·
  density in one 64³ frame, default sample `6ay5`), and the `overview_*.png` it exports.
  Moved from `notebook/figures/` on 2026-09-13.
* `fig-qual/` — the qualitative side-by-side comparison: `fig-qual.ipynb`, the standalone
  viewer `fig_qual_server.py` (launch with `bash figures/fig-qual/run_fig_qual.sh`;
  `VOXBIND_PYTHON` overrides the interpreter), and `qual_utils.py` / `qual_setup.py`.
  Moved from `notebook/figures/` on the same day. `qual_setup.initialize()` resolves the
  repo as `<this file>/../..`, which is why these live one level under `figures/`.
* `overview/` — the architecture/scheme drawings (`make_paper_overviews.py`). A different
  thing from `fig-overview/` despite the name: schematic diagrams of the model, not renders
  of real data.
