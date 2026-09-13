# VoxBind environment — conda-lock files

Reproducible YAML locks (via [conda-lock](https://github.com/conda/conda-lock)).
The two pipeline envs are locked separately, because the paper's docking build
(vina 1.2.2, py3.8) cannot share the py3.10 pipeline env. A third, optional env
(`voxel-bind`) adds FuncBind on top of the VoxBind runtime:

| Env | Python | Runs | Input spec | Lock (YAML) |
|-----|--------|------|-----------|-------------|
| `voxbind` | 3.10 | GPU train/sample + chem/geometry eval + **vina 1.2.7** | `voxbind.environment.yml` | `voxbind.conda-lock.yml` |
| `voxdock` | 3.8 | **paper-faithful Vina 1.2.2 docking** | `voxdock.environment.yml` | `voxdock.conda-lock.yml` |
| `voxel-bind` | 3.12 | optional — VoxBind runtime **+ FuncBind's density branch** in one interpreter | `voxel-bind.environment.yml` | explicit pair (below) |

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

## `voxel-bind` — one env for VoxBind *and* FuncBind

Optional, not part of the paper pipeline, not built by default. It exists so FuncBind's
density-conditioning branch can `import voxbind.models.density_vit` in its own process
instead of reaching across two interpreters. Build it by name:

```bash
# on a box whose default envs dir survives a restart
bash script/00_setup_env.sh voxel-bind

# here, /opt/conda/envs is a container overlay that a restart wipes — install onto the PVC
VOXELBIND_PREFIX=$HOME/.conda/envs/voxel-bind bash script/00_setup_env.sh voxel-bind
```

Locked as an **explicit pair**, not a `conda-lock.yml`: the heavy stack (torch, rdkit,
numpy, scipy) is pip in this env, so the conda layer is thin and the byte-exact pair needs
no conda-lock tool at all.

```bash
conda create -p $HOME/.conda/envs/voxel-bind --file env/voxel-bind.conda-linux-64.lock
ENV=$HOME/.conda/envs/voxel-bind
PATH="$ENV/bin:$PATH" CC="$ENV/bin/gcc" $ENV/bin/pip install -r env/voxel-bind.pip.lock.txt
$ENV/bin/pip install -e .
```

- **It is not the paper env.** `voxbind` above is py3.10 / cu118 / numpy 1.26 and is where
  the published numbers come from. `voxel-bind` mirrors the *live* H200 box — python 3.12,
  torch 2.5.1 + cu124 wheels, numpy 2.x, rdkit 2025.3.6 — because that is the stack the
  FuncBind smoke test (`test/mcp_default_fusion_smoke.py`, 11/11) was validated on.
- **`CC` is load-bearing.** `cpdb-protein` is source-only on PyPI and FuncBind's MCP
  sampling calls it at runtime (`save_sdf_pdb` → `extract_sequences_from_pdb`), so it
  cannot be skipped. The compiler comes from the conda layer (`c-compiler`); this box has
  no system `cc`.
- **`mamba run -n voxel-bind` does not work here, `conda run -n` does.** mamba 2.x resolves
  a name only against `$MAMBA_ROOT_PREFIX/envs` (= `/opt/conda/envs`), so an env under
  `$HOME/.conda/envs` is nameless to it. Prefer the absolute interpreter,
  `$HOME/.conda/envs/voxel-bind/bin/python`.
- **FuncBind itself is not installed into it.** `import funcbind.train_fb` still pulls
  pyrosetta / anarci / posecheck / easydict / meeko / vina / AutoDockTools / plotly at
  import time; that is an import-hygiene fix in the FuncBind repo, not a package to add
  here. The density branch itself runs fine without them.

## Notes

- **Vina split is forced by Python.** vina 1.2.2 has **no py3.10 wheel** (only
  cp36–cp39), so it can't live in the py3.10 `voxbind` env. Paper-faithful 1.2.2
  is in the py3.8 `voxdock` env; `05_evaluate.sh` runs docking there by default.
  `meeko 0.1.dev3` matches the 1.2.2 OBMol API.
- **Platform**: locks are rendered for `linux-64`. For another arch, add it to
  the `-p` list when re-locking (some CUDA/vina deps are linux-only).
- The input `environment.yml` files are curated *functional* specs (pipeline +
  eval), not a byte-copy of the live env's jupyter/streamlit cruft.
- **Explicit locks are also kept** as a byte-exact alternative (no conda-lock
  tool needed): `voxbind.conda-linux-64.lock` + `voxbind.pip.lock.txt` and the
  voxdock pair. Install with `conda create -n <env> --file <env>.conda-linux-64.lock`
  then `pip install -r <env>.pip.lock.txt`. (These capture the full live env,
  jupyter cruft included; note voxbind's explicit pip lock pins vina 1.2.7.)
