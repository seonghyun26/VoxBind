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
for the Docker mounts and Dropbox shared-link variables.

`2_train.sh` preserves `bf16-mixed`, with batch 1, accumulation 95, eager
execution, and non-foreach AdamW. **H100 80GB remains blocked under plain DDP:**
FP32 weights, gradients, Adam moments, and EMA need about 95.8 GiB per GPU before
activations. FSDP/ZeRO sharding or CPU offload must be implemented to run this
mixed-precision configuration on H100; the current preflight rejects it.
