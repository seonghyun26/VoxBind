# Encoder pretraining presets

Each file in this directory is a named Hydra preset composed on top of
`configs/pretrain.yaml`. Presets select the reusable dataset, model, and MAE
groups and provide a stable `exp_name` and WandB tags.

Launch a preset through the stable wrapper:

```bash
bash scripts/03_pretrain.sh \
  --experiment cha_gradmag \
  --gpus 0-3
```

Override values without creating another shell script:

```bash
bash scripts/03_pretrain.sh \
  --experiment cha_gradmag \
  --name cha_gradmag_depth24 \
  --gpus 0-3 \
  -- model.depth=24 num_epochs=200 seed=43
```

Queue the same run on the first four available GPUs:

```bash
bash scripts/99_chain.sh --count 4 --timeout 24h -- \
  bash scripts/03_pretrain.sh \
    --experiment cha_gradmag \
    --name cha_gradmag_depth24 \
    -- model.depth=24 num_epochs=200 seed=43
```

Create a new preset when a configuration represents a repeatable experiment
recipe. Use command-line overrides for temporary debugging and small sweeps.
Do not create a new `chain_*.sh` or dated launch script for hyperparameters.

When migrating a legacy monolithic config, verify resolved parity:

```bash
python configs/verify_parity.py <legacy_config_name> <experiment_name>
```
