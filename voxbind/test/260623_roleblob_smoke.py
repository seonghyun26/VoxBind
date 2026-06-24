"""Smoke test for the role-split (roleblob) representation — Phase 1.

Validates the 4-channel [ligand, pocket, density, gradmag] role-split path end to end
WITHOUT a GPU or the dataset:
  1. _channel_layout for roleblob / roleblob_density
  2. _build_role_atoms == single-channel voxelization (sum of per-element channels)
  3. DensityViTMAE forward in BOTH fused and channel_group([1,1,1,1]) modes at n_in=4
  4. compute_losses on the 4-ch reconstruction (mae path) — per-modality split
  5. the per-channel loss-weight vector (uniform atoms + 0.1 density + 0.1 gradmag)
  6. ChannelViT group_attention → 4x4 ligand/pocket/density/gradmag map

Run:  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python voxbind/test/260623_roleblob_smoke.py
"""
import gemmi  # noqa: F401  (import before torch — silent _load_grid failures otherwise)
import torch

torch.set_num_threads(2)
torch.manual_seed(0)

from voxbind.train_common import _build_role_atoms, _channel_layout
from voxbind.train_density import compute_losses
from voxbind.models.density_vit import DensityViTMAE

G, P, B = 32, 8, 2
gp = G // P


def check(cond, msg):
    assert cond, f"FAIL: {msg}"
    print(f"  ok: {msg}")


# ── 1. layout ────────────────────────────────────────────────────────────────
print("[1] _channel_layout")
lay = _channel_layout("roleblob_density", with_gradmag=True, gradmag_reconstruct=True)
check(lay == {"n_atom": 2, "n_density": 1, "n_gradmag": 1, "n_in": 4, "n_recon": 4,
              "density_idx": 2, "gradmag_idx": 3}, f"roleblob_density+gradmag → {lay}")
lay2 = _channel_layout("roleblob", with_gradmag=False)
check(lay2["n_atom"] == 2 and lay2["n_in"] == 2 and lay2["n_recon"] == 2
      and lay2["gradmag_idx"] is None, f"roleblob (atoms only) → {lay2}")

# ── 2. _build_role_atoms == sum of per-element channels ───────────────────────
print("[2] _build_role_atoms")
v_lig = torch.rand(B, 7, G, G, G)
v_poc = torch.rand(B, 4, G, G, G)
role = _build_role_atoms(v_lig, v_poc)
ref = torch.cat([v_lig.sum(1, keepdim=True), v_poc.sum(1, keepdim=True)], dim=1)
check(role.shape == (B, 2, G, G, G), f"role shape {tuple(role.shape)}")
check(torch.allclose(role, ref), "role == [sum(lig), sum(poc)] (order [ligand, pocket])")

# ── 3 + 4 + 6. model forward + loss + group attention ─────────────────────────
x = torch.randn(B, 4, G, G, G)              # [ligand, pocket, density, gradmag]
mask = (torch.rand(B, 1, G, G, G) < 0.5).float()
# per-channel weight vector: uniform atoms (2) + density 0.1 + gradmag 0.1, sum→n_recon
parts = [torch.ones(2), torch.tensor([0.1]), torch.tensor([0.1])]
raw = torch.cat(parts)
ch_weight = raw * (4.0 / float(raw.sum()))
check(ch_weight.shape[0] == lay["n_recon"], f"ch_weight len {ch_weight.shape[0]} == n_recon 4")

for mode, groups in [("fused", None), ("channel_group", (1, 1, 1, 1))]:
    print(f"[3] DensityViTMAE forward — patch_embed_mode={mode} groups={groups}")
    model = DensityViTMAE(
        grid_dim=G, patch_size=P, n_in_channels=4, n_recon_channels=4,
        n_channels=32, dim=64, depth=2, n_heads=4, mlp_ratio=4, dropout=0.0,
        n_struct_channels=0, pretext_style="mae", head_style="patch_mlp",
        patch_embed_mode=mode, channel_groups=groups,
    ).eval()
    with torch.no_grad():
        out_pretext, out_struct = model(x)
    check(out_pretext.shape == (B, 4, G, G, G), f"out_pretext {tuple(out_pretext.shape)}")
    check(out_struct is None, "no struct head (n_struct_channels=0)")

    print("[4] compute_losses (mae, roleblob_density + gradmag)")
    losses = compute_losses(
        out_pretext, None, x, None, mask,
        pretext_style="mae", patch_size=P, input_mode="roleblob_density",
        ch_weight=ch_weight, atom_pos_weight=10.0, atom_pos_thresh=0.05,
        with_gradmag=True, gradmag_reconstruct=True,
    )
    check("L_dens" in losses and torch.isfinite(losses["L_dens"]), f"L_dens={float(losses['L_dens']):.4f}")
    for k in ("L_dens_atom", "L_dens_density", "L_dens_gradmag"):
        check(k in losses and torch.isfinite(losses[k]), f"{k}={float(losses[k]):.4f}")

    if mode == "channel_group":
        print("[6] group_attention → 4x4 ligand/pocket/density/gradmag map")
        gg, recv = model.encoder.group_attention(x)
        check(gg.shape == (2, 4, 4), f"group→group {tuple(gg.shape)} (depth, nG, nG)")
        check(recv.shape == (2, 4, gp, gp, gp), f"received {tuple(recv.shape)}")

print("\nALL ROLEBLOB SMOKE CHECKS PASSED ✓")
