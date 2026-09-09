# VoxBind environment — conda-lock files

Reproducible YAML locks (via [conda-lock](https://github.com/conda/conda-lock)).
Two envs are locked, because the paper's docking build (vina 1.2.2, py3.8) cannot
share the py3.10 pipeline env:

| Env | Python | Runs | Input spec | Lock (YAML) |
|-----|--------|------|-----------|-------------|
| `voxbind` | 3.10 | GPU train/sample + chem/geometry eval + **vina 1.2.7** | `voxbind.environment.yml` | `voxbind.conda-lock.yml` |
| `voxdock` | 3.8 | **paper-faithful Vina 1.2.2 docking** | `voxdock.environment.yml` | `voxdock.conda-lock.yml` |

Each `*.conda-lock.yml` is a hash-pinned, solver-free YAML lock covering **both
conda and pip** deps (torch/cuda, vina, meeko, pdb2pqr, AutoDockTools, …),
rendered for **linux-64**. The `*.environment.yml` is the human-editable source
spec you re-lock from. PoseCheck/PoseBusters *pose* eval lives in a third
`moleval` env (build with `bash script/00_setup_env.sh moleval`).

## Rebuild (needs conda-lock: `pip install conda-lock` or `conda install -c conda-forge conda-lock`)

```bash
# voxbind (pipeline + chem eval + docking @ 1.2.7)
conda-lock install -n voxbind env/voxbind.conda-lock.yml
conda run -n voxbind pip install -e .          # from the repo root

# voxdock (paper-faithful docking @ vina 1.2.2)
conda-lock install -n voxdock env/voxdock.conda-lock.yml
```

`conda-lock install` restores the exact conda **and** pip packages from the lock.
`script/00_setup_env.sh` and `env/Dockerfile` automate this.

## Update a lock (after editing an `environment.yml`)

```bash
conda-lock lock -f env/voxbind.environment.yml -p linux-64 --lockfile env/voxbind.conda-lock.yml
```

## Notes

- **Vina split is forced by Python.** vina 1.2.2 has **no py3.10 wheel** (only
  cp36–cp39), so it can't live in the py3.10 `voxbind` env. Paper-faithful 1.2.2
  is in the py3.8 `voxdock` env; `05_evaluate.sh` runs docking there by default.
  `meeko 0.1.dev3` matches the 1.2.2 OBMol API.
- **Platform**: locks are rendered for `linux-64`. For another arch, add it to
  the `-p` list when re-locking (some CUDA/vina deps are linux-only).
- The input `environment.yml` files are curated *functional* specs (pipeline +
  eval), not a byte-copy of the live env's jupyter/streamlit cruft.
