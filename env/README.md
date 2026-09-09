# voxbind environment — lock files

Exact, reproducible pin of the working `voxbind` conda env on the source server.
This single env runs **most of the pipeline and evaluation**: the GPU train/sample
stack **and** AutoDock **Vina docking** + chemical/geometry metrics. (Only
PoseCheck/PoseBusters *pose-quality* eval lives in a separate `moleval` env.)

Two layers, applied in order — the conda layer, then the pip layer:

| File | What |
|------|------|
| `voxbind.conda-linux-64.lock` | 245 conda packages, byte-exact URLs+md5 (`conda list --explicit`). **linux-64 only.** |
| `voxbind.pip.lock.txt` | 109 pip packages (torch, vina, meeko, pdb2pqr, AutoDockTools, …). |

## Rebuild

```bash
conda create -n voxbind --file env/voxbind.conda-linux-64.lock
conda run -n voxbind pip install -r env/voxbind.pip.lock.txt
conda run -n voxbind pip install -e .          # from the repo root
```

`script/00_setup_env.sh` and `script/Dockerfile` use these automatically when
present (falling back to `env.yaml` otherwise).

## Notes

- **Vina is pinned to 1.2.2**, not the live env's 1.2.7. Every VoxBind paper
  affinity comes from 1.2.2; `05_evaluate.sh` refuses any other build. `meeko
  0.1.dev3` matches the 1.2.2 OBMol API.
- **Docking runs *in* this env** (vina + meeko + pdb2pqr + AutoDockTools +
  openbabel are all here). To run `05_evaluate.sh` entirely from this env +
  the existing pose env, set `VOXDOCK_ENV=voxbind MOLEVAL_ENV=moleval`.
- **torch/triton are pip wheels** (2.5.1 / 3.1.0), not the conda `pytorch-cuda`
  build. If the default PyPI wheel's CUDA runtime mismatches the coworker's
  driver, add `--extra-index-url https://download.pytorch.org/whl/cu121` (or the
  matching cuXX channel) to the pip step.
- **Platform**: the conda lock is `linux-64`. On a different OS/arch, regenerate
  with `conda list -n voxbind --explicit --md5` on that box, or fall back to
  `conda env create -f env.yaml`.
- Pose stack (`moleval`) is intentionally *not* in this lock. Build it with
  `bash script/00_setup_env.sh moleval` if you need PoseCheck/PoseBusters.
```
