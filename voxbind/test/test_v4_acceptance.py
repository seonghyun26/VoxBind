import pandas as pd

from voxbind.dataset.v4.acceptance import coverage_report, validate_manifest


def sample(sample_id, pdb_id, pocket_id, cluster_id, split, state):
    return {
        "sample_id": sample_id,
        "pdb_id": pdb_id,
        "assembly": "1",
        "chains": "A",
        "uniprot_acc": "P00001",
        "pocket_id": pocket_id,
        "canonical_pocket_residue_set": f"A:1,A:2,{pocket_id}",
        "cluster_id": cluster_id,
        "split": split,
        "state": state,
        "source": "plinder",
        "structure_origin": "experimental",
        "mms_id": None,
        "sample_weight": 1.0,
        "resolution_A": 2.0,
        "has_structure_factors": True,
        "density_path": f"{pdb_id}.ccp4",
        "density_source_pdb_id": pdb_id,
        "grid_frame_id": f"grid:{pocket_id}",
        "density_map_R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "density_map_t": [0.0, 0.0, 0.0],
        "grid_spacing_A": 0.25,
        "grid_dim": 64,
        "density_registration_ok": True,
        "paired_holo_id": None,
        "paired_apo_id": None,
        "pocket_rmsd_apo_holo": None,
        "is_interface_pocket": False,
    }


def test_valid_manifest_and_pair_pass():
    manifest = pd.DataFrame(
        [
            sample("h1", "1abc", "p1", "c1", "train", "holo"),
            sample("a1", "2abc", "p1", "c1", "train", "apo"),
            sample("h2", "3abc", "p2", "c2", "test", "holo"),
        ]
    )
    pairs = pd.DataFrame(
        [{"sample_id_a": "h1", "sample_id_b": "a1", "relation": "apo_holo"}]
    )
    assert validate_manifest(manifest, pairs=pairs)["ok"]


def test_cluster_split_leakage_fails():
    manifest = pd.DataFrame(
        [
            sample("h1", "1abc", "p1", "c1", "train", "holo"),
            sample("a1", "2abc", "p2", "c1", "test", "apo"),
        ]
    )
    result = validate_manifest(manifest)
    assert not result["ok"]
    assert result["violations"]["A1_cluster_split_conflict_rows"] == 2


def test_density_must_come_from_same_experiment():
    row = sample("a1", "2abc", "p1", "c1", "train", "apo")
    row["density_source_pdb_id"] = "1abc"
    result = validate_manifest(pd.DataFrame([row]))
    assert not result["ok"]
    assert result["violations"]["A4_density_wrong_source_pdb"] == 1


def test_each_mms_token_must_stay_in_one_split():
    first = sample("h1", "1abc", "p1", "c1", "train", "holo")
    second = sample("h2", "2abc", "p2", "c2", "test", "holo")
    first["mms_id"] = "mms:a;mms:shared"
    second["mms_id"] = "mms:b;mms:shared"
    result = validate_manifest(pd.DataFrame([first, second]))
    assert not result["ok"]
    assert result["violations"]["L4_mms_split_conflict_rows"] == 2


def test_pair_transform_must_match_common_grid_transforms():
    manifest = pd.DataFrame(
        [
            sample("h1", "1abc", "p1", "c1", "train", "holo"),
            sample("a1", "2abc", "p1", "c1", "train", "apo"),
        ]
    )
    pairs = pd.DataFrame(
        [
            {
                "sample_id_a": "h1",
                "sample_id_b": "a1",
                "relation": "apo_holo",
                "R_a_to_b": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "t_a_to_b": [1.0, 0.0, 0.0],
            }
        ]
    )
    result = validate_manifest(manifest, pairs=pairs)
    assert not result["ok"]
    assert result["violations"]["A4_pair_transform_violations"] == 1


def test_coverage_report_records_frozen_methodology_and_sf_gate():
    manifest = pd.DataFrame(
        [sample("h1", "1abc", "p1", "c1", "train", "holo")]
    )
    report = coverage_report(manifest, validate_manifest(manifest))
    assert "within 10 Å" in report
    assert "64³ at 0.25 Å/voxel" in report
    assert "pocket_fident__50__weak__component" in report
    assert "Samples passing the structure-factor gate: **1**" in report
