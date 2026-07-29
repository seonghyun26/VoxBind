# VoxBind scripts

The top-level scripts are the stable public entry points:

- `00_data_density_process.sh` — prepare density data.
- `01_downstream_voxbind.sh` — train the downstream generative model.
- `02_sample.sh` — sample molecules.
- `03_pretrain.sh` — pretrain an encoder from a Hydra experiment preset.
- `04_probe.sh` — evaluate a frozen encoder.
- `99_chain.sh` — wait for GPU resources and launch an arbitrary command.

## Pretraining

Experiment settings belong in `configs/experiment/*.yaml`; temporary changes can
be supplied as Hydra overrides:

```bash
bash scripts/03_pretrain.sh \
  --experiment cha_gradmag \
  --gpus 0-3 \
  -- num_epochs=200 seed=43
```

For a complete pretrain-then-probe run:

```bash
bash scripts/workflows/pretrain_then_probe.sh \
  --experiment cha_gradmag \
  --condition atomblob_density_gradmag \
  --gpus 0-3
```

To queue either command until GPUs are free, omit its `--gpus` option and place
it behind `99_chain.sh`:

```bash
bash scripts/99_chain.sh --count 4 --timeout 24h -- \
  bash scripts/workflows/pretrain_then_probe.sh \
    --experiment cha_gradmag \
    --condition atomblob_density_gradmag
```

`99_chain.sh` uses advisory per-GPU locks, several idle samples, and direct
argument execution. It does not use `eval`, kill existing processes, or assume
that a process name identifies ownership. It can also combine the GPU gate with
`--after-file`, `--after-pid`, or a side-effect-free `--condition-script`:

```bash
bash scripts/99_chain.sh \
  --after-file dataset/data/pretrain/READY \
  --count 4 \
  -- \
  bash scripts/03_pretrain.sh --experiment cha_gradmag
```

## Layout

- `lib/` contains sourced launcher helpers.
- `workflows/` contains reusable multi-step workflows and multi-GPU sampling.
- `tools/` contains environment/container utilities.
- `archive/` contains historical one-off chains, watchers, and dated experiment
  launchers. Archived scripts are retained for provenance and may contain
  server-specific paths; new experiments should use Hydra presets instead.
- `archive/arxiv/` contains the separately maintained original-paper workflow.

Do not add a new top-level script for a hyperparameter variant. Add or extend a
Hydra experiment preset and invoke it through `03_pretrain.sh`.
