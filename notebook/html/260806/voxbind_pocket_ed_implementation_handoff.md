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

Keep the pocket and ligand encoders independent until the interaction stage. Coordinate and ED channels may be jointly encoded within each branch.

$\boxed{
z_P=E_P\left([V_P^{\mathrm{coord}},\rho_P,M_P]\right),
}$

$\boxed{
z_L=E_L\left([V_L^{\mathrm{coord}},\rho_L,1-M_P]\right).
}$

Then predict affinity only after combining the independently encoded representations:

$\boxed{
\hat a=R\left(\operatorname{Fuse}(z_P,z_L)\right).
}$

```text
Pocket coordinate + masked pocket ED + pocket mask -> Pocket encoder --+
                                                                    +-> Fusion -> Affinity
Ligand coordinate + non-protein ED + inverse mask -> Ligand encoder --+
```

Do not mix pocket and ligand input channels before their respective encoders. The pre-fusion pocket representation must be independently usable without a ligand.

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
