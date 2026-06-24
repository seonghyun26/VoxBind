# Frozen PLINDER pretraining selection (`plinder_v1`)

`plinder_selected.csv` — 17721 ligand-instances (RSCC≥0.8, leakage-filtered, ECOD-deduped), the committed source-of-truth for the PLINDER density pretraining corpus.

- `plinder_inputs.json` — pinned bucket / filters / build params + sha256 of the selection and the three leakage holdout files.
- `plinder_funnel.json` — full provenance funnel (1,357,906 → 17,721).

**Consumed** by `dataset/legacy/03b,03c` (and the `build/` pipeline) via `voxbind.splits.frozen_plinder_selection()`, which verifies the sha256 and loud-fails on drift.
**Re-pin** only deliberately: `python dataset/legacy/03a_plinder_select.py --rederive --freeze` (bump the name, e.g. plinder_v2).
