"""Unified dataset-build pipeline — one file per stage, --dataset if-branch.

See dataset/build/README.md. Stage entry points: s1_acquire, s2_density,
s3_crop, s4_voxelize, s5_targets. Per-dataset config lives in registry.py.
"""
