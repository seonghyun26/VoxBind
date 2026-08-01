from voxbind.dataset.v4.leakage import (
    crossdocked_test_pdbs,
    parse_crossdocked_id,
    protected_rows,
    resolve_component_split,
)


POCKET_ID = (
    "BSD_ASPTE_1_130_0/"
    "2z3h_A_rec_1wn6_bst_lig_tt_docked_3_pocket10.pdb"
)
LIGAND_ID = (
    "BSD_ASPTE_1_130_0/"
    "2z3h_A_rec_1wn6_bst_lig_tt_docked_3.sdf"
)


def test_crossdocked_parser_reads_experimental_pdb_ids():
    assert parse_crossdocked_id(POCKET_ID) == ("2z3h", "1wn6")


def test_crossdocked_tuple_protects_receptor_and_ligand_source():
    receptors, ligand_sources = crossdocked_test_pdbs(
        [({"id": POCKET_ID}, {"id": LIGAND_ID})]
    )
    assert receptors == {"2z3h"}
    assert ligand_sources == {"1wn6"}


def test_crossdocked_test_promotes_affinity_val_to_test():
    rows = protected_rows(
        {"train": set(), "val": {"2wlz"}, "test": {"1abc"}},
        crossdocked_receptors={"2wlz"},
        crossdocked_ligand_sources={"3def"},
    )
    by_id = {row["pdb_id"]: row for row in rows}
    assert by_id["2wlz"]["required_split"] == "test"
    assert "binding_affinity_val" in by_id["2wlz"]["reasons"]
    assert "crossdocked2020_test_receptor" in by_id["2wlz"]["reasons"]


def test_merged_component_uses_conservative_precedence():
    assert resolve_component_split(["train", "val"]) == "val"
    assert resolve_component_split(["train", "test", "val"]) == "test"
