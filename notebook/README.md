# notebook/

Analysis notebooks and their figure/report helpers, grouped by topic. Loose files
that used to sit at this top level were moved into the folders below (2026-09-07).

| folder | contents |
|--------|----------|
| `dataset/`       | `01a_dataset_viewer`, `01b_dataset_sanity_check`, `01c_density_distribution` + `check_xray_data.py`, `dataset_stats.py` |
| `results/`       | `02a_results_docking`, `02b_results_pdbbind`, `04_cliff`, `results2latex` + `docking_autoresearch.py`, `make_hbond_scatter_figure.py`, `make_table2_pearson_figure.py`, `moleculeace_cliffs.py` |
| `ablation/`      | `03_ablation_density` + `make_vit_impl_report.py` |
| `visualization/` | `00_dashboard` (sample-metrics), `00_visualization` (generated-ligand viewer) |
| `html/`          | HTML report generators + their JSON/CSV inputs and the dated `2606xx/` report snapshots (unchanged) |
| `figures/`, `webapp/`, `archive/` | figures, the sample-viewer web app, and archived material (unchanged) |

**Note on the two `results` folders.** `notebook/results/` (here) holds the
results *notebooks*. The published results *bundle* — the rendered HTML pages,
generated samples, metrics, and Docker env — lives at the repo-level
`VoxBind/results/` (git-ignored, Dropbox-backed).

Helpers find the repo root via `Path(__file__).resolve().parents[2]`; run them as
`python notebook/<topic>/<script>.py`. Notebooks use absolute paths, so they run
regardless of kernel working directory.
