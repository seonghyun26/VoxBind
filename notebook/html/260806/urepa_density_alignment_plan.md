# Distilling X-ray Density into VoxBind via U-REPA Alignment

**Track:** Alignment (density used at training time only; inference is density-free)
**Status:** Design fixed, implementation pending

---

## 1. Core idea

VoxBind is finetuned with an auxiliary **representation alignment loss** whose target is a
frozen CDG encoder (3D ViT, coordinates + experimental X-ray density) applied to the **clean
holo complex**. At inference, no CDG encoder and no density — plain VoxBind.

```
L = L_denoise + λ · L_align
```

**Why this framing:** affinity regression and generation are asymmetric w.r.t. density. In
regression the ligand is an *input*, so ligand density in the map is legitimate signal. In
generation the ligand is the *output*, so the same signal is label leakage. Moving density
from the inference path to a training-time target dissolves the asymmetry.

**Consequence — leakage stops being a problem.** The alignment target never touches the
generation path, so the holo map (ligand included) can be used as-is. No α-masking, no omit
maps, no cross-docked splitting required *on this track*.

**Resulting claim:** *"Experimental density supervision yields better generative
representations"* — not *"density informs generation at inference time."* Weaker on paper,
but deployable to targets with no crystal structure.

---

## 2. Architecture

```
TARGET (training only, frozen):
    CDG_ViT(pocket coords + ligand coords + experimental density)  →  tokens

ALIGNED:
    VoxBind U-Net middle-stage features
    (input: noisy ligand voxel σ=0.9 + pocket voxel, NO density)
      → MLP projector → upsample → manifold loss against CDG tokens

INFERENCE:
    stock VoxBind. No CDG, no density.
```

Both encoders already share the 64³ grid, voxel size, and coordinate frame — so no
resampling or frame reconciliation is needed.

**Two privileged signals are being distilled:**
1. the **clean ligand** (U-Net only sees it at σ=0.9) — this is standard REPA, effect near-certain
2. the **density** — the novel contribution

The coords-only-CDG control (§5) is what separates them.

---

## 3. U-REPA specifics

VoxBind is a U-Net, not a DiT, so follow **U-REPA (arXiv 2503.18414, NeurIPS 2025)**, not
vanilla REPA. Its three prescriptions:

| Prescription | Reason | Action here |
|---|---|---|
| Align at the **middle stage** (bottleneck) | skip connections couple shallow/deep layers, distorting shallow alignment | single alignment point at U-Net bottleneck; do not start with shallow layers |
| **MLP projection first, then upsample** | upscale the U-Net feature, never downscale the ViT target | MLP on bottleneck feature → upsample to CDG token grid |
| **Manifold loss**, not tokenwise cosine | U-Net/ViT feature spaces have a large gap; strict correspondence is too rigid | relational loss on inter-sample similarity structure |

**Why the manifold loss matters conceptually.** Hard alignment would demand the U-Net
*become* the CDG feature, which destroys the extra capacity we want it to keep. Relational
alignment instead demands *"pockets CDG considers similar, you should also consider
similar"* — absolute positions stay free.

**Practical:** the manifold loss operates over inter-sample relations, so **batch size is a
real hyperparameter**. 3D voxels constrain it; gradient accumulation does not substitute
(the similarity matrix needs actual co-resident samples).

**Simplification vs. image diffusion:** walk-jump uses a single fixed σ = 0.9, so there is no
noise-level schedule to tune for the alignment loss.

---

## 4. Training recipe

### Data split of the two losses

- **`L_denoise`: full CrossDocked**, every step. Never restricted.
- **`L_align`: only samples with matching experimental density.**

This is the correction to the earlier failed N=20 experiment, where finetuning on a
20-pocket subset *alone* collapsed generation (Vina median → −0.06). Here the main loss
stays on the full distribution, so the cause of that forgetting is structurally absent.
Alignment is a regularizer, not a replacement training signal.

### Unfreezing ladder — start at (1)

1. **Projector + adapter/LoRA only; U-Net frozen.** ← start here
2. Open blocks around the bottleneck.
3. Full finetuning.

Stage (1) is chosen for *diagnostic* value, not cost: **how far the alignment loss falls
with the U-Net frozen measures how compatible the two representations already are.** If it
barely moves, wider unfreezing is unlikely to rescue it and the plan should pivot.

### Efficiency

CDG is frozen → **precompute and cache all target tokens**. The density subset is small
enough to fit on disk. Removes the ViT forward from the training loop entirely.

### Split hygiene

Constructing the density subset can silently break the CrossDocked split. **Explicitly
exclude any PDB entry overlapping the test pockets.**

---

## 5. Experiments

### Primary: λ sweep

`λ ∈ {0, 0.1, 0.5, 1, 5}` → plot Vina / QED / diversity / validity.

Lower alignment loss is **not** the objective. Perfect alignment means CDG becomes a
function of what the U-Net already computes — the added information vanishes. Known failure
mode in the literature (capacity mismatch / gradient conflict; overly rigid alignment to a
lower-capacity teacher restricts generative power, especially late in training; cf. HASTE).

The **λ vs. Vina curve is the headline result**: architecture, parameter count, and data are
all held fixed while one axis moves. Far stronger than binary present/absent comparison, and
it replaces the CFG dose-response evidence that is unavailable on this track.

Optional: schedule λ → 0 late in training (HASTE-style stage-wise termination).

### Controls (mandatory)

| Control | Isolates |
|---|---|
| λ = 0, same data, same finetuning | effect of alignment itself |
| **coords-only CDG as target** | **contribution of density — decides the paper** |
| computed-density CDG as target | whether experimental data is needed at all |

Without control #2 the claim collapses to "our pocket encoder is better." All three controls
must share architecture and pretraining recipe with the main CDG encoder.

---

## 6. Evidence already collected

Probe on 12,674 matched samples, pooled 640-d, identical architecture and training for both
encoders (input differs only):

| Metric | Value | Reading |
|---|---|---|
| linear CKA(CDG, C) | 0.602 | random baseline 0.004 → substantial overlap, clearly not identical |
| R²: C ← CDG | 0.912 | CDG nearly contains the coords representation |
| R²: CDG ← C | 0.517 | ~48% of CDG variance is not linearly recoverable from coords |

**The asymmetry is the result.** Symmetric high values would have meant "same content,
different encoding" — nothing to gain. CKA at 0.602 sits in the useful band: close enough to
align, far enough to be worth aligning.

### Still to measure before implementation

- **CKA profile across U-Net depths** (is the bottleneck actually optimal here, as U-REPA claims?)
- U-Net features must be computed from **noisy input at σ = 0.9**, matching training conditions
- **R²(U-Net ← CDG) vs. R²(CDG ← U-Net)** — same asymmetry analysis; CDG ⊃ U-Net confirms the alignment direction
- **Token-level CKA** in addition to pooled — pooling discards spatial structure, and the manifold loss operates on tokens

All forward-pass only. Roughly a day.

---

## 7. What success can and cannot look like

**Information-theoretic ceiling.** With coords-only input, U-Net features are a deterministic
function of coordinates. Alignment cannot open a channel for density-specific information.
The best attainable is `E[CDG features | coords]` — and `R²(CDG←C) = 0.517` estimates that
ceiling. The remaining ~48% will not transfer.

**So alignment is an inductive bias, not an information channel.** What it can deliver:
(a) coords-derivable structure the denoising loss alone never surfaced, and (b) the standard
privileged-information effect.

**Expect the alignment loss to plateau above zero.** In the closest precedents (3D Infomax,
GraphMVP, D&D), the teacher modality is largely *determined* by the student's input —
conformers are computable from a 2D graph — so distillation loss can approach zero.
Experimental density is *not* determined by coordinates (measurement noise, ordered water,
disorder, partial occupancy). A loss floor is the expected outcome, and where it plateaus is
an empirical measurement of that 48%.

**What the U-Net actually acquires** is a learned prior — *"given this pocket geometry,
density probably looks like this"* — not instance-specific measurements. That is precisely
what makes it deployable to targets with no crystal structure.

---

## 8. Implementation checklist

Built + verified on the analysis server (ckpt-independent Phase A):

- [x] Confirm bottleneck size / CDG patch size → **U-Net `(B,128,8³)` ↔ CDG `(B,512,640)`, grids match → projector is a pure 128→640 map, upsample = identity**
- [x] Implement MLP projector → upsample → manifold loss → **`models/urepa.py`** (`UREPAAlignment`, `manifold_alignment_loss` relkl/gram/pool, `BottleneckTap`; smoke-passed)
- [x] Build density subset w/ explicit test-pocket exclusion → **`dataset/00i_urepa_subset.py` → `urepa_subset.pt` (6,181 native+density; 0 test-leak — CrossDocked split already pocket-clean)**
- [x] Precompute + cache CDG target tokens → **`dataset/00j_urepa_cache_cdg.py`** (load→assemble→encode→cache self-test OK; full run needs the density crops in the finetuning env)
- [x] Wire dual-loss training → **`urepa_train_integration.md`** (reference patch: pid-carrying loader, `BottleneckTap`, Stage-1 frozen-U-Net, λ block, aug caveat)
- [x] Controls: coords-only-CDG target → rebuild `00j` with `model_zoo/coords_100m_v2_mask075` (n_in=11). Computed-density-CDG control **dropped** (synthetic density not needed).

Needs the VoxBind generator ckpt (Phase B, finetuning server):

- [ ] CKA-vs-depth profile (noisy input, σ = 0.9) to validate the bottleneck as the alignment point
- [ ] Token-level CKA + directional R²(U-Net ← CDG) — pooled CDG↔C already measured: CKA 0.60, R²(C←CDG) 0.97 / R²(CDG←C) 0.64
- [ ] Freeze U-Net; train projector only; log alignment-loss floor → **decision point**
- [ ] λ sweep {0, 0.1, 0.5, 1, 5} + verify batch size gives a meaningful manifold matrix

---

## 9. Relation to the other track

The **conditioning track** (replace VoxBind's pocket branch with the CDG encoder, density as
live input) is the only way to actually exploit the unrecoverable ~48% at inference. It
requires α-masking of the density (protein-atom-derived attenuation field, no ligand
coordinates in mask construction) and only applies to targets with crystal structures.

The two tracks are complementary, not exclusive:

- **Alignment track** → general-purpose, deployable everywhere
- **Conditioning track** → upper bound when a holo structure exists

Presenting both as tiers makes the story complete.

---

## 10. Key references

| Paper | Why |
|---|---|
| **U-REPA: Aligning Diffusion U-Nets to ViTs** (arXiv 2503.18414, NeurIPS 2025) | **read first** — bottleneck alignment, MLP-then-upsample, manifold loss |
| REPA (Yu et al., ICLR 2025) | original formulation |
| VideoREPA / CREPA | soft relational alignment for finetuning *pretrained* models |
| SoftREPA | lightweight finetuning, <1M added parameters |
| Align & Invert | post-hoc REPA on models not pretrained with it |
| HASTE | capacity mismatch, stage-wise alignment termination |
| VoxBind (Pinheiro et al., ICML 2024) | base model; conditional walk-jump, single σ |
| 3D Infomax (ICML 2022), GraphMVP (ICLR 2022), D&D | molecular-domain precedent: train with richer modality, deploy without it |
