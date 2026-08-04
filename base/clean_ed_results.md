# clean_ed_v1 benchmark results (test set)

Splits: clean_ed_v1 (CASF-2016, test=214) / clean_ed_v1_indep (CASF-2016 independent, test=109)
Shared train 4099 / val 1000. Filter = GEMS CleanSplit ∩ ED ∩ RSCC≥0.8 ∩ Kd/Ki.

| model | clean_ed_v1 (214) ρ / r / RMSE | clean_ed_v1_indep (109) ρ / r / RMSE |
|---|---|---|
| **C+D+G ChannelViT[7,4,2] (ours, best)** | 0.748±0.003 / 0.755 / 1.436 | 0.792±0.002 / 0.791 / 1.430 |
| ProFSA (ICLR 2024, pretrained+probe) | 0.751 / 0.765 / 1.435 | **0.819 / 0.818 / 1.344** |
| GET (ICML 2024, 3-model ens) | 0.675 / 0.664 / 1.623 | 0.771 / 0.742 / 1.540 |

Density probe (ours) per-seed:
- clean_ed_v1: ρ 0.7459/0.7509/0.7479, r 0.7527/0.7578/0.7547, RMSE 1.4359/1.4534/1.4178
- clean_ed_v1_indep: ρ 0.7899/0.7937/0.7929, r 0.7894/0.7911/0.7937, RMSE 1.4483/1.4276/1.4124

GET per-seed (best-val ckpt, 3 seeds):
- clean_ed_v1: ρ 0.673/0.676/0.666 (mean 0.6715±0.004), r 0.662/0.667/0.656 (0.6619±0.005); ensemble ρ 0.675 / r 0.664 / RMSE 1.623
- clean_ed_v1_indep: ρ 0.774/0.777/0.758 (mean 0.7693±0.009), r 0.743/0.749/0.728 (0.7398±0.009); ensemble ρ 0.771 / r 0.742 / RMSE 1.540

ProFSA (Lightning trainer.test, best-val ckpt, single run):
- clean_ed_v1 (214): Pearson 0.765 / Spearman 0.751 / RMSE 1.435 / MAE 1.128
- clean_ed_v1_indep (109): Pearson 0.818 / Spearman 0.819 / RMSE 1.344 / MAE 1.065

## Notes / caveats
- CASF-2016 is small (214) and curated → all structure-based models score much higher here than on `lp_edrscc_v2` (e.g. ProFSA ρ 0.597 → 0.751; ours 0.637 → 0.748). CleanSplit removes train↔test leakage, but the benchmark's high quality inflates absolute numbers vs the harder LP split.
- ProFSA (pretrained pocket encoder, frozen + probe) edges ours on this split. Its ScPDB/PDBbind-style pretraining may overlap the CASF targets — a leakage vector orthogonal to CleanSplit's train↔test control — so treat the ProFSA≳ours gap with that caveat.
- GET is trained from scratch on the 4,099 train complexes (no pretraining); it trails both, consistent with its `lp_edrscc_v2` standing.
- indep (109, leakage-independent CASF subset) ranks all three the same way and is the stricter comparison.
