# Archived launch scripts

These scripts are retained as experiment provenance, not as the supported
launcher interface. Most encode a single historical run, server-specific path,
or temporary resource-sharing policy.

- `chains/` — one-off multi-stage experiment chains.
- `watchers/` — process- or artifact-specific wait loops.
- `experiments/` — dated autoresearch and sweep launchers.
- `arxiv/` — the original-paper workflow and its POC20 experiments.
- `launchers/` — superseded queue and relaunch utilities.
- `workflows/` — server-specific multi-stage build and evaluation workflows.
- `log-scripts/` — shell scripts recovered from the generated `log/` directory.

For new work, use:

- `scripts/99_chain.sh` for GPU waiting and command launch.
- `scripts/03_pretrain.sh` with `configs/experiment/*.yaml` for encoder
  pretraining.
- `scripts/workflows/pretrain_then_probe.sh` for the common two-stage workflow.

Archived scripts may still be runnable, but they are not expected to be
portable. Promote generally useful behavior into a stable launcher or workflow
instead of copying an archived script back to the top level.
