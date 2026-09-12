# MCP fine-tune — does more density fine-tuning buy binding, or just size?

Figure 1 of §1. The macrocyclic-peptide generator is being fine-tuned with receptor
electron density (a ControlNet with zero-init zero-convs on frozen FuncBind), and this
folder answers the only question that matters for that run: **is the fine-tune getting
better at binding?**

## Why bin by size

Vina rewards buried surface, so a larger ligand scores better without necessarily binding
better. Every headline number for these arms moves with peptide size — QED, SA and ligand
efficiency are all size proxies at this scale, and Vina Dock is size-sensitive too. Binning
by heavy-atom count removes that: inside a bin every arm draws the same-sized peptides, so
a gap that survives the bin is a binding difference. This is the same argument as §4 of the
260827 note, applied to a different model family.

The bottom panel is not decoration. The top panel is only trustworthy where an arm has
molecules, which is why a bin with fewer than 5 is left undrawn rather than plotted as a
point — the 8.21M arm has 4 and 2 molecules in the last two bins.

## Arms

One model at four amounts of fine-tuning, measured in `acc_iter` (fine-tuning samples;
the base model's 87.2M is not counted, and epoch numbers are meaningless here because
every resume restarts the counter):

| arm | checkpoint | acc_iter |
|---|---|---|
| vanilla | `fb_unified` | 0 (density-free base) |
| fine-tune 3.17M | `..._ga32_r3` | 3,170,876 |
| fine-tune 8.21M | `..._ga32_r10` | 8,206,956 |
| fine-tune 26.1M | `..._ga32_r14` | 26,112,876 |

Because they are one model at four training amounts and not four methods, they are drawn
as an **ordinal ramp** — one brown, darkening with training — rather than four hues, and
`../method_colors.py` owns the assignment. The light end is that file's FuncBind brown, so
the untouched base model reads the same here as it does in §2. Each arm also carries a line
style: at a line crossing, two adjacent steps of a lightness ladder are hard to separate,
and the published baselines in this note already use style as a second identity channel.

## Inputs

Vina results written by `voxbind/exps/frozenenc_probes/run_docking_eval_parallel.py` over
eval trees assembled by `funcbind/scripts/build_mcpp_eval_tree.py`:

    /home1/irteam/funcbind/artifacts/reproduction/mcpp/cmp10/_eval/{vanilla,finetuned}
    /home1/irteam/funcbind/artifacts/reproduction/mcpp/cmp10_r10/_eval/finetuned
    /home1/irteam/funcbind/artifacts/reproduction/mcpp/cmp10_r14/_eval/finetuned

Only the 9 targets every arm produced molecules for are used; 3dv5 yielded nothing from
vanilla or the 3.17M checkpoint.

## Run

    /opt/conda/envs/voxdock/bin/python build_mcp_finetune_size.py

Writes `mcp_finetune_size.{png,svg}` and `mcp_finetune_size.csv` (the numbers behind both
panels). To add the next checkpoint, append a row to `ARMS` and register its colour in
`../method_colors.py` as the next step of the ramp.
