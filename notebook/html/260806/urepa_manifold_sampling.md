# U-REPA manifold loss — how the similarity matrix is sampled

Companion to `urepa_density_alignment_plan.md` and `urepa_train_integration.md`.
Concerns `manifold_alignment_loss` in `voxbind/models/urepa.py`.

**TL;DR** — with flat token sampling the fraction of *intra-sample* pairs in the
similarity matrix is exactly `1/B`, set by the batch size and by nothing else.
`n_sample` cannot change it. That makes batch size a hidden knob on *what the loss
optimizes*, not just on how well it estimates it, and it means a tuned `λ` does not
transfer across batch sizes. Fix by sampling a fixed number of tokens **per sample**.

---

## 1. The mechanism

```python
S = student.reshape(b * n, d)            # (B·N, D) — all tokens of all samples, flattened
idx = torch.randperm(m)[:n_sample]       # flat random subset
S = S[idx]
sim_s = (S @ S.t()) / tau                # (n_sample, n_sample)
```

Points are drawn uniformly from the flattened `B·N` pool, so each sample contributes
about `m/B` of the `m = n_sample` points. Counting pairs:

```
intra-sample pairs ≈ B · C(m/B, 2) ≈ B · (m/B)² / 2 = m² / (2B)
total pairs        ≈ C(m, 2)       ≈ m² / 2

intra fraction     ≈ 1 / B                    (independent of n_sample)
```

| bsz | intra-sample pair fraction |
|---:|---:|
| 8 | 12.5 % |
| 16 | 6.3 % |
| **32** (current plan) | **3.1 %** |
| 64 | 1.6 % |

`n_sample` is purely a cost / variance knob — it changes how well the matrix is
estimated, never its composition.

## 2. Why this matters

1. **Batch size silently sets the objective's composition.** The plan treats batch size
   as "make sure the relation matrix is not too small" (checklist C5). It has a second,
   unstated role: it fixes the intra/inter mix at `1/B`. The two pull in opposite
   directions — a bigger batch gives a richer matrix but a *smaller* intra-sample share.
2. **`λ` does not transfer.** A `λ` tuned at bsz 8 is scaling a materially different loss
   at bsz 32. Any λ sweep must be re-run if the batch size changes.
3. **DDP does not help and can hurt.** The loss is computed per rank, so `B` is the
   per-rank batch (32), not the effective 128. All-gathering tokens to 128 would push the
   intra fraction *down* to 1.6 %.
4. **The knob is currently implicit.** Nobody chose 3.1 %; it fell out of the batch size.

## 3. Which axis actually needs the work?

Measured on this exact model pair — VoxBind `exp_sig0.9` ep923 bottleneck ↔ frozen
`efficient_60m_v3_mask085` — with `voxbind/test/repa_cka_profile.py`, n = 1024 pockets.
`token` rows = every (pocket, patch) pair (524,288 rows); `pooled` rows = per-pocket
mean-pooled vectors (1,024 rows).

Bottleneck, linear CKA:

| student | teacher | token | pooled | pooled − token |
|---|---|---:|---:|---:|
| `noise` (ligand-free) | `atoms0_dm` (apo) | 0.254 | 0.527 | **+0.272** |
| `noise` | `holo` | 0.298 | 0.487 | **+0.190** |
| **`sig0.9`** (training condition) | `atoms0_dm` | 0.236 | 0.277 | +0.042 |
| **`sig0.9`** | `holo` | 0.325 | 0.305 | **−0.020** |

**The answer depends on the pairing, and it is not the one a quick look suggests.**

- With a **ligand-free student** (the *generation* condition), pooling raises CKA a lot
  (+0.19 … +0.27): the intra-sample variation — which patch is which, spatially — is the
  poorly aligned component.
- With the **σ=0.9 student** (the condition the finetune actually trains under), the two
  axes are comparable: +0.042 for the apo teacher and **−0.020** for the holo teacher.
  There is no strong evidence here that the intra-sample axis is the weaker one.

So the earlier framing — "the loss spends 97 % of its pairs on the axis that is already
aligned" — holds for the generation-condition pairing but **not** for the training-condition
pairing. It should not be used as the motivation.

Two caveats on the numbers:

- The `pooled` estimate is the shakier one: 1,024 rows against 512-dim features. Linear CKA
  is biased upward when rows ≲ dim (at n = 32 the same measurement returned 0.82–0.96). The
  `token` column, with ~½ M rows, is solid. Treat pooled values as soft upper bounds.
- Depth matters: at shallow depths the `noise`/apo pairing shows a very large pooled−token
  gap (L0: 0.818 vs 0.548), but shallow layers are not the alignment point.

## 4. Recommendation

Block sampling is still worth doing — the justification is **controllability, not a proven
misallocation**:

**(a) Per-sample block sampling.** Draw a fixed `n_tok` tokens from each sample instead of
a flat subset. The intra fraction becomes `(n_tok − 1)/(m − 1)`, independent of `B`, and
explicitly tunable. `λ` then transfers across batch sizes. This is a small change to the
`randperm` line and is worth making regardless of which ratio turns out to be best.

**(b) Split the two terms.** `L = λ_tok · L_intra + λ_batch · L_inter`, swept separately.
Given a λ sweep is planned anyway, this costs little extra and answers *which axis carries
the effect* instead of assuming it.

**(c) `pool_samples=True` is a control, not a default.** It is the pure inter-sample
extreme. Under the training-condition pairing it targets an axis that is *not* better
aligned than the token axis, so it is not obviously wasteful — but it does discard the
spatial structure entirely, which is the thing U-REPA's manifold framing is about. Useful
to answer "does sample-level relational structure alone do anything?"

Suggested order: implement (a), sweep `n_tok` over a few values, and use (c) as one
endpoint of that sweep. Escalate to (b) only if the sweep shows a real effect.

## 5. Two secondary issues

**`tau = 0.5` may be too soft.** Cosine similarities lie in [−1, 1], so logits lie in
[−2, 2] and the largest/smallest softmax ratio is `e⁴ ≈ 55` spread over `n_sample` points.
The teacher's target distribution can end up close to uniform, which weakens the signal.
Before trusting a λ sweep, log the **row entropy** of `p = softmax(sim_t/tau)`; if it sits
near `log(m)`, lower `tau`.

**`gram` mode has no centering.** `relkl`'s row-wise softmax is invariant to a constant
shift of a row, so a large shared component in the similarity matrix cancels automatically.
`gram` matches raw cosine matrices with MSE and has no such protection — a common offset
would dominate the loss. `relkl` is the right default; if `gram` is used, subtract the row
mean first.

## 6. What to log during training

The alignment loss alone cannot tell you whether the representation moved.

| signal | reading |
|---|---|
| **token CKA** (start 0.236 for `sig0.9`→apo, 0.325 → holo) | rising ⇒ spatial arrangement is moving |
| **pooled CKA** (start 0.277 / 0.305) | rising alone ⇒ only sample-level relations are moving |
| loss ↓, both CKA flat | only the projector is learning — backbone untouched |
| Stage-1 (U-Net frozen) loss floor | how much the projector can absorb on its own; Stage 2 must beat it |

Context from `voxbind/test/repa_b1_linear.py` (pocket-level split, ridge, n = 1024): at the
bottleneck **R²(teacher ← student) = 0.08–0.10**, R²(student ← teacher) = 0.17–0.21, and
only 8–11 of 512 CCA directions exceed 0.9. The near-zero forward R² is why the loss must be
relational at all — a tokenwise cosine/MSE term would chase an unreachable target — and it
also means the projector has limited room to absorb the loss by itself, so a Stage-2 gain
over the Stage-1 floor is a meaningful signal.

`test/repa_cka_profile.py` doubles as the monitor; point its student/teacher conditions at
the training configuration.

## 7. Provenance

| number | source |
|---|---|
| CKA table | `voxbind/test/repa_cka_profile.py` → `test/results/repa_cka_profile.csv` (n=1024 pockets, lp_edrscc_v2 order, PDBbind voxels_v5) |
| R² / CCA | `voxbind/test/repa_b1_linear.py` → `test/results/repa_b1_linear.csv` (60/20/20 pocket split, ridge, λ on val) |
| bottleneck = (B,512,8³) | `unet3d.py` cumulative `ch_mults`; verified on `exp_sig0.9` ep923 `unet3d.middle.res1.conv1.weight` = (512,512,3,3,3) |

Measurements were taken on PDBbind voxels (`voxels_v5`), whose arcsinh stats differ slightly
from the CrossDocked/PLINDER constants the finetune will use; treat the values as indicative
of the representation geometry rather than exact for the finetuning corpus.
