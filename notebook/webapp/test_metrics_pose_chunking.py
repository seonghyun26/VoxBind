from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rdkit import Chem


WEBAPP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEBAPP_DIR))
import metrics  # noqa: E402


class PoseEvalChunkingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mols = [Chem.MolFromSmiles(smiles) for smiles in (
            "C", "CC", "CCC", "CCCC", "CCCCC",
        )]

    @staticmethod
    def _write_success(cmd: list[str], start: int) -> int:
        ligand_path = Path(cmd[cmd.index("--ligands") + 1])
        output_path = Path(cmd[cmd.index("--out") + 1])
        n_mols = sum(
            1 for _ in Chem.SDMolSupplier(str(ligand_path), sanitize=False)
        )
        output_path.write_text(json.dumps([
            {
                "posecheck": {"clashes": start + i, "strain": float(start + i)},
                "posebusters": {"valid": True},
            }
            for i in range(n_mols)
        ]))
        return n_mols

    def test_preserves_order_across_chunks(self) -> None:
        chunk_sizes: list[int] = []

        def fake_run(cmd, **_kwargs):
            n_mols = self._write_success(cmd, sum(chunk_sizes))
            chunk_sizes.append(n_mols)
            return SimpleNamespace(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(metrics, "pose_eval_available", return_value=True), \
                patch.object(metrics.subprocess, "run", side_effect=fake_run):
            rows = metrics.run_pose_eval(
                self.mols, "pocket.pdb", mode="all", tmp_dir=tmp, chunk_size=2,
            )

        self.assertEqual(chunk_sizes, [2, 2, 1])
        self.assertEqual(
            [row["posecheck"]["clashes"] for row in rows], list(range(5))
        )
        self.assertTrue(all(row["posebusters"]["valid"] for row in rows))

    def test_timeout_is_local_to_one_chunk(self) -> None:
        call_index = 0
        successful_offset = 0

        def fake_run(cmd, **_kwargs):
            nonlocal call_index, successful_offset
            call_index += 1
            if call_index == 2:
                raise subprocess.TimeoutExpired(cmd, timeout=12)
            n_mols = self._write_success(cmd, successful_offset)
            successful_offset += n_mols
            return SimpleNamespace(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(metrics, "pose_eval_available", return_value=True), \
                patch.object(metrics, "_POSE_TIMEOUT_S", 12), \
                patch.object(metrics.subprocess, "run", side_effect=fake_run):
            rows = metrics.run_pose_eval(
                self.mols, "pocket.pdb", mode="all", tmp_dir=tmp, chunk_size=2,
            )

        self.assertNotIn("error", rows[0]["posecheck"])
        self.assertNotIn("error", rows[1]["posecheck"])
        for row in rows[2:4]:
            error = row["posecheck"]["error"]
            self.assertIn("chunk 2/3 (mols 3-4)", error)
            self.assertIn("timed out after 12s", error)
            self.assertIn("error", row["posebusters"])
        self.assertNotIn("error", rows[4]["posecheck"])

    def test_worker_errors_remain_retriable(self) -> None:
        row = {
            "posecheck": {"error": "chunk timed out"},
            "posebusters": {"error": "chunk timed out"},
        }
        self.assertFalse(metrics._has_pose_for_mode(row, "all"))


if __name__ == "__main__":
    unittest.main()
