# MCP density-conditioned FuncBind

The runnable MCP workflow lives in the FuncBind submodule:

```bash
# VoxBind host: build the image
bash FuncBind/scripts/0_env_setup.sh

# Inside the container
bash scripts/1_data_process.sh
SMOKE=1 bash scripts/2_train.sh
bash scripts/2_train.sh
SMOKE=1 bash scripts/3_generate.sh
bash scripts/3_generate.sh
```

The default condition is CDG v2 epoch 25 with `default` density fusion. The
host `.repro-env` is not required; the Docker image creates its own environment.

See
[`FuncBind/scripts/README_mcp_density.md`](../../../FuncBind/scripts/README_mcp_density.md)
for the Docker mounts and model-link variables, and
[`script/README_mcp.md`](../../../script/README_mcp.md) for the SB handoff links.

`2_train.sh` targets H100 80GB x 8 with `bf16-mixed`, batch 1, accumulation 95,
eager execution, non-foreach AdamW with ZeRO-1 state sharding, CPU EMA, and
activation checkpointing. Trainable static state is estimated at ~43.1 GiB/GPU,
excluding activations and temporary buffers. Reduced-model distributed tests
pass; full-model H100 capacity still needs a smoke test on the SB node.
