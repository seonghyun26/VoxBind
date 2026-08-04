# Coordinate + Experimental ED Conditioning for VoxBind

## Goal

Pretrain separate pocket and ligand representations with binding-affinity regression, then transfer the pretrained pocket encoder to VoxBind as an additional experimental electron-density (ED) condition.

## Voxel masks and ED channels

For each fixed protein–ligand crop, construct a protein van der Waals occupancy mask:

$\boxed{
  M_P(\mathbf r)=\mathbb{1}\left[\mathbf r\text{ is inside the vdW radius of a protein atom}\right].
}$

Use opposite masks for the two regions:

$\boxed{
M_{\mathrm{pocket}}=M_P,
\qquad
M_{\mathrm{ligand}}=1-M_P.
}$

Apply them to the aligned experimental ED map:

$\boxed{
\rho_P=M_P\odot\rho_{\mathrm{exp}},
\qquad
\rho_L=(1-M_P)\odot\rho_{\mathrm{exp}}.
}$

Here, \(\rho_L\) is the non-protein or ligand-accessible ED within the selected crop. It may also contain solvent, ions, or empty cavity space; it is not assumed to be ligand-only ED.

All coordinate voxels, masks, and ED grids must share the same center, orientation, crop size, and voxel resolution.

## Binding-affinity model

Encode pocket and ligand with separate early encoders, keep their **spatial** structure, and fuse at the **token level**. This is not "pool each branch, then concatenate two vectors" — pooling before interaction discards where the ligand contacts the pocket.

1. **Shared pocket-centered frame.** Pocket and ligand inputs use the same pocket-centered coordinate frame, crop size, orientation, and voxel resolution. Do **not** independently recenter the two inputs — their relative spatial geometry (how the ligand sits in the pocket) must be preserved.
2. **Separate early encoders**, coordinate and ED channels jointly encoded within each branch:
   - Pocket: $[\,V_P^{\mathrm{coord}},\ M_P\odot\rho_{\mathrm{exp}},\ M_P\,]$
   - Ligand: $[\,V_L^{\mathrm{coord}},\ (1-M_P)\odot\rho_{\mathrm{exp}},\ 1-M_P\,]$
3. **Retain spatial tokens / feature grids.** Each encoder emits per-token spatial features $T_P,\ T_L$ (tokens on the shared patch grid, or feature volumes). Do not globally pool either branch before interaction modeling.
4. **Token-level fusion.** Fuse $T_P$ and $T_L$ with joint self-attention or cross-attention blocks over their tokens, so a ligand token can attend to the pocket tokens it physically contacts.
5. **Regress affinity** from the fused representation.
6. **Transfer.** Only the pocket representation *before* fusion, $T_P$, is carried to VoxBind (see below).

$\boxed{T_P = E_P\left([\,V_P^{\mathrm{coord}},\ M_P\odot\rho_{\mathrm{exp}},\ M_P\,]\right)}$

$\boxed{T_L = E_L\left([\,V_L^{\mathrm{coord}},\ (1-M_P)\odot\rho_{\mathrm{exp}},\ 1-M_P\,]\right)}$

$\boxed{T_{PL} = \operatorname{JointFusion}(T_P,\ T_L)}$

$\boxed{\hat a = \operatorname{RegressionHead}(T_{PL})}$

```text
Pocket C+D ── Pocket spatial encoder ── T_P ──┐
                                              ├─ Token-level joint fusion ─ Affinity
Ligand C+D ── Ligand spatial encoder ── T_L ──┘

T_P before fusion ── Spatial adapter ── VoxBind conditioning
```

Do not mix pocket and ligand input channels before their respective encoders. The pre-fusion pocket representation $T_P$ must be independently usable without a ligand.

## VoxBind transfer

Transfer only the pretrained pocket encoder:

```text
Pocket coordinate + masked pocket ED + pocket mask
                         |
              Frozen pretrained pocket encoder
                         |
                Trainable spatial adapter
                         |
                  Frozen VoxBind
```

The initial implementation should:

- preserve the original VoxBind pocket coordinate channels;
- freeze the pretrained pocket encoder and the original VoxBind model;
- inject spatial pocket features through trainable residual adapters;
- avoid using ligand coordinates, ligand masks, or ligand ED as VoxBind test-time inputs.

The retained pocket ED is masked holo-pocket ED. Direct ligand density is largely removed from the pocket channel, although ligand-induced protein conformation and map/refinement effects may remain.
