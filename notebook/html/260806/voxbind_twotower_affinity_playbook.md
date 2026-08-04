# Two-Tower Affinity — If It Underperforms: Diagnosis & Playbook

Companion to `voxbind_pocket_ed_implementation_handoff.md`. What to try if the two-tower
cross-attention probe (`260806_tt_pocket_protein_vdw_mc` + `260806_tt_ligdens_protein_vdw_mc`,
lp_edrscc_v2) lands well below the single-encoder champion **ρ 0.644**, e.g. under ~0.60.

## What "underperforms" means, and what is already ruled out

- **Bar to proceed as-is:** two-tower affinity *near* champion 0.644.
- **The real success criterion is not the joint affinity number** — it is a *transferable pocket
  encoder* for VoxBind. A modest joint-affinity gap is expected (the two-tower deliberately
  handicaps itself: separate encoders, late token-level fusion, pocket usable without a ligand).
- **Ruled out as causes** (so don't spend budget here):
  - *Capacity* — 10M already reached near-champion in prior single-encoder runs; the towers are
    ~45M each, well past the plateau. Bigger will not help (and has hurt before).
  - *Data* — both towers train on the **same** corpus as champion (PLINDER v2, 112k, holo,
    ligand-matched 2Fo-Fc, RSCC≥0.8). No data-parity gap vs champion.
- **So a gap, if any, comes from:** the split / late-fusion handicap, the spec knobs
  (protein_vdw mask · mask-as-channel · ligand-density), the frozen-probe *lower bound*, or the
  fusion head — in roughly that order of suspicion.

## Step 0 — Diagnose first (cheap, do before any fix)

1. **Baseline delta.** Re-probe the old `260803` two-tower (ligand_footprint pocket, atoms-only
   ligand, no mask). If 260803 ≥ new spec → the spec knobs *hurt*; if new ≥ 260803 → spec helps
   and the gap is intrinsic to two-tower. (One probe run each; towers already exist.)
2. **Pocket-only probe.** Regress affinity from the pocket tower *alone* (no ligand). If it beats
   a coords-only pocket, the pocket representation is healthy and the deficit is on the
   fusion / ligand side, not the pocket. This is the single most decision-relevant check.
3. **Over/underfit read.** Compare val ρ vs test ρ of the head. Large val≫test → head overfits
   (→ regularize / shrink head, group A). val≈test but both low → representation-limited
   (→ fine-tune / spec, groups B–C).

## The ladder (cheap → expensive) — exhaust cheap first

### A. Head / fusion — minutes per trial, NO re-pretrain (start here)
The head trains in minutes on cached tokens, so sweep freely.
- Head hparams: layers 2→4/6, n_heads, dropout, lr, wd, epochs, batch.
- **Fusion variant:** try **joint self-attention** (concat pocket+ligand tokens + a modality
  embedding → shared transformer) vs the current bidirectional cross-attention. Handoff allows
  either; they can differ a lot.
- **Contact bias:** add a distance/adjacency bias to the pocket↔ligand attention on the shared
  8³ grid, so tokens that are physically close attend more (injects interaction geometry).
- **Pooling:** mean vs attention-pool vs a learned CLS/interaction token (current head mean-pools
  *after* interaction — vary this).
- **Pairwise features:** bilinear / outer-product pooling of pocket×ligand tokens before the MLP.

### B. Fine-tune the towers — hours, highest ROI, transfer-compatible (do next)
The frozen probe measures a **lower bound**. Unfreezing usually gives the biggest single jump.
- Unfreeze the pocket (± ligand) encoder — last N blocks first, then full — jointly with the head.
- **Fully compatible with the transfer plan:** VoxBind receives the *fine-tuned* pocket encoder.
- Watch the val/test gap; use LLRD (layer-wise lr decay) + small lr on the encoder.

### C. Spec / tower ablation — ~10h re-pretrain each, run only the informative ones
Each is a fresh tower pretrain, so pick deliberately (guided by Step 0 + A/B):
- **Mask basis:** `protein_vdw` (keeps only protein-interior ED, drops solvent/cavity — may be too
  aggressive near the pocket surface) vs `ligand_footprint` (keeps everything but the ligand) vs
  `none` (holo). Flip via `mae.pocket_ed_mask`.
- **mask_as_channel** off (the explicit M_P channel may add noise more than signal).
- **Ligand tower:** `ligand_density` vs atoms-only `ligand` (does the ligand branch benefit from
  ρ_L at all).
All are single config flags in `scripts/exps/260806_twotower.sh` (`BASIS`, `MASKCH`, `TOWER`).

### D. Pretraining objective — expensive, only if A–C stall
- Add an **affinity-aware / contrastive auxiliary** to the MAE pretext (pull complexes with
  similar pK together), or a cross-modal `mae.density_visible` term (reconstruct atoms from
  visible density → forces density→chemistry decoding).
- Mask-ratio / schedule tweaks; longer pretraining.

### E. Larger architecture changes — last resort
- **Partial weight sharing:** shared early encoder blocks, separate late blocks + heads (keeps a
  transferable pocket path while sharing low-level features).
- **Explicit pair module:** a small pairwise-representation block (AlphaFold-style) between the
  towers instead of plain attention.
- (Do **not** collapse to a single shared encoder — that breaks the pocket-only transfer.)

## The real go / no-go for the adapter stage

Judge the **pocket encoder directly**, not just the joint affinity number:
- **Pocket-only affinity probe** — does pocket alone carry affinity-relevant ED (> coords-only)?
- **B-factor / H-bond probes** on the pocket encoder — density/gradmag historically dominate
  coords on these; a healthy signal confirms the pocket ED representation is intact.

Decision tree:
- Joint affinity low **but** pocket-only + B-factor/H-bond healthy → the deficit is a
  fusion/ligand-side artifact; **proceed to the VoxBind adapter** (transfer the pocket encoder),
  and improve the joint head in parallel (group A/B).
- Pocket-only **also** weak → fix the pocket representation first: fine-tune (B), then spec
  ablation (C, especially the mask basis) — before the adapter.

## Priority summary

1. Diagnose (Step 0): baseline delta + pocket-only probe + over/underfit read.
2. Group A (head/fusion sweep) — cheap, may close a fusion-limited gap outright.
3. Group B (fine-tune towers) — highest ROI, transfer-compatible.
4. Group C (spec ablation) — only the runs Step 0/A–B point at.
5. Reframe on the pocket encoder → decide adapter go/no-go independent of the joint number.

Not on the list, deliberately: scaling model size, scaling/ changing the data — both ruled out above.
