# Reference-ligand similarity — which fingerprints to report, and why

Companion note to `build_reference_similarity.py`. The 260820 meeting table
(`260820/260820_meeting.html` §1.2) carried seven similarity columns; this note records
which two survive into the paper, which four move to the appendix, and the primary
source for each.

## Headline (report these)

| Metric | Definition used here | Primary source |
|---|---|---|
| **ECFP4 / Morgan Tanimoto** | Morgan fingerprint, radius 2, folded to 2048 bits; Tanimoto against the pocket's reference ligand, mean and max per pocket, then macro-averaged | Rogers & Hahn, *Extended-Connectivity Fingerprints*, **J. Chem. Inf. Model.** 50(5):742–754, 2010. [doi:10.1021/ci100050t](https://doi.org/10.1021/ci100050t) — building on Morgan's canonical-numbering algorithm, *J. Chem. Doc.* 5(2):107–113, 1965 |
| **Bemis–Murcko scaffold match** | exact canonical-SMILES agreement of the Bemis–Murcko scaffold; molecules with no ring system count as non-matches | Bemis & Murcko, *The Properties of Known Drugs. 1. Molecular Frameworks*, **J. Med. Chem.** 39(15):2887–2893, 1996. [doi:10.1021/jm9602928](https://doi.org/10.1021/jm9602928) |

Why exactly these two: ECFP4 Tanimoto is the fingerprint the ML-venue molecular
generation literature converged on — MOSES computes its nearest-neighbour similarity
(SNN) on Morgan ECFP4, GuacaMol computes its KL/nearest-neighbour similarity on ECFP4,
and the SBDD line (TargetDiff, DecompDiff, Pocket2Mol and everything benchmarking
against them) reports Morgan-fingerprint Tanimoto for diversity and reference
similarity. Bemis–Murcko is its scaffold-level counterpart and is what MOSES's `Scaff`
metric and every "scaffold novelty" number in that literature mean.

- Polykovskiy et al., *Molecular Sets (MOSES)*, **Front. Pharmacol.** 11:565644, 2020
  ([arXiv:1811.12823](https://arxiv.org/abs/1811.12823)) — SNN/Frag/Scaff/IntDiv/FCD.
- Brown et al., *GuacaMol: Benchmarking Models for de Novo Molecular Design*,
  **J. Chem. Inf. Model.** 59(3):1096–1108, 2019.
  [doi:10.1021/acs.jcim.8b00839](https://doi.org/10.1021/acs.jcim.8b00839)
- Guan et al., *3D Equivariant Diffusion for Target-Aware Molecule Generation and
  Affinity Prediction* (TargetDiff), **ICLR 2023**.
  [arXiv:2303.03543](https://arxiv.org/abs/2303.03543)

## Appendix only (`--full`)

Kept computable as a robustness check, but they are not what the venues report and
they all rank the methods the same way, so they do not earn a main-table column.

| Metric | Definition used here | Primary source |
|---|---|---|
| MACCS | 166 public structural keys (RDKit emits 167 bits, bit 0 unused); Tanimoto | Durant, Leland, Henry & Nourse, *Reoptimization of MDL Keys for Use in Drug Discovery*, **J. Chem. Inf. Comput. Sci.** 42(6):1273–1280, 2002. [doi:10.1021/ci010132r](https://doi.org/10.1021/ci010132r) |
| AtomPair | hashed atom-pair fingerprint, 2048 bits; Tanimoto | Carhart, Smith & Venkataraghavan, *Atom Pairs as Molecular Features in Structure–Activity Studies*, **J. Chem. Inf. Comput. Sci.** 25(2):64–73, 1985. [doi:10.1021/ci00046a002](https://doi.org/10.1021/ci00046a002) |
| RDKit | Daylight-style topological path fingerprint, 2048 bits; Tanimoto | No journal paper. Daylight theory manual (topological fingerprints) + the RDKit implementation; cite the RDKit toolkit, not a fingerprint paper |
| Dice | unfolded sparse **count** Morgan (radius 2); Dice coefficient | Rogers & Hahn 2010 as above; Dice, *Measures of the Amount of Ecologic Association Between Species*, **Ecology** 26(3):297–302, 1945 |
| 3D shape (`--with-3d`) | RDKit shape Tanimoto in the shared pocket frame, no re-alignment | Grant, Gallardo & Pickup, *A fast method of molecular shape comparison*, **J. Comput. Chem.** 17(14):1653–1666, 1996 — the Gaussian-overlap formulation behind ROCS |

The Tanimoto coefficient itself: Jaccard, *Étude comparative de la distribution florale
dans une portion des Alpes et des Jura*, **Bull. Soc. Vaudoise Sci. Nat.** 37:547–579,
1901; usually cited in cheminformatics as Rogers & Tanimoto, *A Computer Program for
Classifying Plants*, **Science** 132:1115–1118, 1960.

## One thing NOT to change

The **Diversity** column in the Vina tables is a different metric with a different
lineage. TargetDiff's `utils/evaluation/similarity.py` computes it on
`Chem.RDKFingerprint` (the RDKit path fingerprint), DecompDiff and DiffSBDD inherited
that code, and our `notebook/webapp/metrics.py` matches it deliberately. Every
published diversity number we print next to ours is an RDKit-fingerprint number.
Switching diversity to Morgan for tidiness would make our column silently
incomparable to the baselines. Leave it.

## Reproduction check

Running `--full --with-3d` on `voxbind_frozenenc_atomblob7_v2p1_sig0.9/full_eval_ep350`
reproduces the 260820 "Best trial" row exactly on every metric — ECFP4 0.099/0.248,
MACCS 0.362/0.631, AtomPair 0.212/0.394, RDKit 0.243/0.442, Dice 0.208/0.426, 3D shape
0.400/0.564 — and lands scaffold match at 0.81% against the table's 0.82%. So the
rewritten script is the same computation as the lost one.

## Which methods can be measured where

On this box (all read from `samples.sdf`): **TargetDiff** (`base_drug/eval/targetdiff`),
**FuncBind** (`funcbind/artifacts/reproduction/crossdocked/paper_run/gpu{0..3}/samples`,
sharded 25 pockets each), **VoxBind σ=0.9 / σ=1.0**, and our three arms. 79 pockets are
shared by all seven, which is what the table is scored on.

**AR, Pocket2Mol, DiffSBDD and DecompDiff are not here.** They were sampled on the
Blackwell (sm_120) box — see `260903/baseline.html` appendix A — and only their
aggregates came back, in `260903/baseline_vina.json`. To add them, copy
`build_reference_similarity.py` to that machine, point `METHODS` at its sample roots
(any directory of `target_*/` holding a `samples.sdf` plus the pocket's reference
ligand), and run it there; a sharded run can pass a list of roots. Then bring the
resulting `reference_similarity.json` back.

There is no Reference row: the crystal ligand's similarity to itself is 1.0 and its
scaffold always matches.

## Wiring it into results2latex.ipynb

Done. The notebook's last cell reads `reference_similarity_table.html` and
`reference_similarity_to_latex()` transposes it into the paper table — metrics as rows,
methods as columns, ECFP4 as a `\multirow` block over mean/median, Scaffold match
spanning both stub columns. It reads the groups off the fragment's two-row header, so
`--full` / `--with-3d` add rows without a notebook edit, and it renders method labels
from the fragment's markup (`<sub class="sc">` becomes
`\textsubscript{\scriptsize …}`, matching how the de novo table writes sigma).

`wrap=True` gives the 260820 layout instead — the same rows in a `\resizebox`'d
`wraptable` for sitting beside body text, which additionally needs `wrapfig`. The cell
prints both. Each is byte-identical to the corresponding builder output
(`reference_similarity.tex` and `_wrap.tex`); the two entry points exist so a regression
in either is a one-line diff to catch. Otherwise this needs `booktabs` and `multirow`
only — `\shortstack` is plain LaTeX, no `makecell`.
