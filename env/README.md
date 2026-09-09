# VoxBind environment — lock files

Exact, reproducible pins of the working conda envs. Two envs are locked, because
the paper's docking build (vina 1.2.2, py3.8) cannot share the py3.10 pipeline env:

| Env | Python | Runs | Conda lock | Pip lock |
|-----|--------|------|-----------|----------|
| `voxbind` | 3.10 | GPU train/sample + chem/geometry eval + **vina 1.2.7** | `voxbind.conda-linux-64.lock` (245) | `voxbind.pip.lock.txt` (~109) |
| `voxdock` | 3.8 | **paper-faithful Vina 1.2.2 docking** | `voxdock.conda-linux-64.lock` (129) | `voxdock.pip.lock.txt` (12) |

Each env = a conda layer (byte-exact URLs+md5, **linux-64 only**) applied first,
then a pip layer. PoseCheck/PoseBusters *pose* eval lives in a third `moleval`
env (not locked here — build with `bash script/00_setup_env.sh moleval`).

## Rebuild

```bash
# voxbind (pipeline + chem eval + docking @ 1.2.7)
conda create -n voxbind --file env/voxbind.conda-linux-64.lock
conda run -n voxbind pip install -r env/voxbind.pip.lock.txt
conda run -n voxbind pip install -e .          # from the repo root

# voxdock (paper-faithful docking @ vina 1.2.2)
conda create -n voxdock --file env/voxdock.conda-linux-64.lock
conda run -n voxdock pip install -r env/voxdock.pip.lock.txt
```

`script/00_setup_env.sh` and `env/Dockerfile` use these automatically when present
(falling back to `env.yaml` otherwise). `05_evaluate.sh` runs docking under
`voxdock` (vina 1.2.2) by default.

## Notes

- **Vina is 1.2.7 here, and that is forced by Python.** This env is py3.10 and
  vina 1.2.2 ships **no py3.10 wheel** (only cp36–cp39 + sdist), so 1.2.2 cannot
  be installed here without a source build. The **paper-faithful vina 1.2.2**
  therefore needs a **separate Python 3.8 env** (`voxdock`, built by
  `script/00_setup_env.sh voxdock`). Use this py3.10 env's 1.2.7 for convenience
  docking; use `voxdock` (1.2.2) when the numbers must match the paper.
- **Docking can run in this env** (vina 1.2.7 + meeko + pdb2pqr + AutoDockTools +
  openbabel are all here). To run `05_evaluate.sh` from this env, set
  `VOXDOCK_ENV=voxbind` (but note it will be scored with 1.2.7, not 1.2.2).
- **torch/triton are pip wheels** (2.5.1 / 3.1.0), not the conda `pytorch-cuda`
  build. If the default PyPI wheel's CUDA runtime mismatches the coworker's
  driver, add `--extra-index-url https://download.pytorch.org/whl/cu121` (or the
  matching cuXX channel) to the pip step.
- **Platform**: the conda lock is `linux-64`. On a different OS/arch, regenerate
  with `conda list -n voxbind --explicit --md5` on that box, or fall back to
  `conda env create -f env.yaml`.
- Pose stack (`moleval`) is intentionally *not* in this lock. Build it with
  `bash script/00_setup_env.sh moleval` if you need PoseCheck/PoseBusters.
