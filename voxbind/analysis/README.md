# Sequence–representation similarity analysis

`representation_similarity.py` compares cached binding-affinity representations
without running any encoder or touching a GPU. It currently has a zero-argument
preset for the `lp_edrscc_v2` cohort shared by:

- VoxBind coordinates-only (`C`)
- VoxBind coordinates+density+gradient magnitude (`C+D+G`)
- frozen IPNet
- frozen DSMBind

All four caches cover the 5,987 PIDs in the split manifest. The affinity-pair
analysis uses the 5,864 rows that also have valid sequence, SMILES, and pK
metadata in the current LP-PDBBind index.

The primary similarity is cosine similarity after standardizing every feature
dimension over the common cohort. This prevents one high-variance feature (for
example DSMBind's appended energy scalar) from dominating the geometry. Raw
cosine, mean-centered cosine, and z-scored Euclidean distance are retained as
sensitivity checks.

## Run

From the repository root:

```bash
python voxbind/analysis/representation_similarity.py --dry-run

python voxbind/analysis/representation_similarity.py \
  --threads 4 \
  --output-dir voxbind/dataset/data/pdbbind/representation_similarity/current4
```

The full run uses the bundled CPU MMseqs2 for sequence identity, RDKit Morgan
similarity for ligands, and the structure-derived pocket masks cached by
HonestAffinity. Pocket residues are concatenated in parent-sequence order and
compared with a BLOSUM62-guided global alignment (gap open -10, extension
-0.5); the reported value is exact residue identity over aligned columns. The
pocket residues do not have to be a contiguous segment, which is important when
interpreting that metric.

In addition to random non-homolog controls, the run samples same-ligand /
non-homolog pairs so the remote-sequence/high-ligand quadrant is not lost through
class imbalance. Re-running reuses `sequence_hits.tsv` only when its saved cohort
and search signature still match; `--force-sequence-search` is available for an
explicit rebuild. Use `--no-pocket-similarity` only for a faster sensitivity
run without the pocket alignment.

For a fast end-to-end smoke test:

```bash
python voxbind/analysis/representation_similarity.py \
  --max-pids 200 --random-controls 500 --same-ligand-controls 500 \
  --max-homolog-pairs 2000 \
  --output-dir /tmp/voxbind-representation-smoke
```

## Outputs

- `summary.json`: cohort provenance, pair counts, correlations, and the
  smooth-minus-cliff separation diagnostic.
- `pair_metrics.csv.gz`: all analyzed pairs with sequence, ligand, affinity,
  and representation metrics.
- `case_pairs.csv.gz`: binding-relevant subsets, including near-sequence /
  near-ligand smooth pairs and affinity cliffs.
- `sequence_summary.csv`, `pocket_sequence_summary.csv`, `case_summary.csv`:
  plot-ready aggregate tables. The sequence tables contain both a marginal view
  and a ligand-similarity-controlled (`Tanimoto >= 0.8`) view.
- `sequence_similarity_curve.png`, `pocket_sequence_similarity_curve.png`,
  `case_similarity.png`: compact diagnostics.

`non_hit` controls are random pairs absent from the configured MMseqs2 search
(20% identity, 80% coverage by default). They are deliberately not reported as
having an exact sequence identity of zero. Same-ligand/non-homolog targeted pairs
are tagged separately and excluded from the general sequence-bin curve.

## Add another model

Supply one or more `NAME=PATH` caches. A PyTorch cache may use `features`,
`feats`, or `feat` for its `pid -> vector` mapping. An `.npz` cache may contain a
`pids` vector and a two-dimensional `features` matrix.

```bash
python voxbind/analysis/representation_similarity.py \
  --representation VoxBind-CDG=path/to/cdg.pt \
  --representation AnotherModel=path/to/another_model_features.npz \
  --output-dir /tmp/representation-comparison
```

The comparison cohort is always the intersection of the split manifest,
metadata, and every supplied cache, so model-to-model pair comparisons use
identical complexes.

## Dump affinity-trained baseline representations

Two CPU-only adapters export hidden vectors from the affinity-trained baseline
checkpoints already in the repository:

```bash
python voxbind/analysis/dump_cheapnet_representation.py \
  --batch-size 32 --threads 4

/home/shpark/.conda/envs/get/bin/python \
  voxbind/analysis/dump_get_representation.py \
  --batch-size 32 --threads 4
```

CheapNet exports the 256-dimensional `l2p + p2l` vector immediately before its
fully connected regression head. GET's energy head acts on a variable number of
block vectors, so its adapter exports the model's native fixed-width 64D
`graph_repr` (the normalized sum of encoder block representations). Both files
also retain scalar predictions for auditing, but the analysis only reads the
`features` mapping.

The generated caches are:

- `base/cheapnet/_edrscc/features/cheapnet_casf_seed0_prehead.pt` (5,959 PIDs)
- `base/get/_edrscc/features/get_v2_seed0_graph_repr.pt` (5,986 PIDs)

Pass those alongside the four preset caches to compare all six methods. Their
current common, metadata-valid cohort contains 5,839 complexes. The completed
six-model output is under
`voxbind/dataset/data/pdbbind/representation_similarity/current6_pocket`.

```bash
python voxbind/analysis/representation_similarity.py \
  --include-affinity-baselines \
  --threads 4 \
  --output-dir \
    voxbind/dataset/data/pdbbind/representation_similarity/current6_pocket
```

## Scope of the initial preset

The four-method zero-argument preset uses caches that existed before this
analysis. CheapNet and GET are opt-in because their caches must first be dumped
from checkpoints and their structural preprocessing has small, model-specific
failure sets. Other affinity baselines in `base/` still mostly retain
checkpoints, processed graphs, or predictions rather than a common per-PID
hidden-vector cache. A scalar affinity prediction is not treated as an encoded
representation.
