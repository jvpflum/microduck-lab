from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "tools"))

from agent_runs import load_agent_runs, publish_receipt  # noqa: E402
from capability_catalog import load_catalog, task_spec, training_tasks  # noqa: E402


class AgentRunTests(unittest.TestCase):
    def test_catalog_has_hardened_skating_and_frontflip_adapters(self) -> None:
        catalog = load_catalog()
        self.assertIn("microduck-skating", catalog["programs"])
        self.assertEqual(task_spec("backflip")["display_name"], "Front flip")
        self.assertTrue(training_tasks()["backflip"]["train_script"].is_file())

    def test_receipt_publish_is_atomic_and_hashes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "evaluation.json"
            artifact.write_text("{}\n")
            receipt = root / "receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "new-robot-task-001",
                        "project": "New robot",
                        "title": "Novel maneuver",
                        "goal": "Test a new behavior.",
                        "status": "running",
                        "metrics": [{"name": "Progress", "value": 12, "unit": "%"}],
                        "artifacts": [{"kind": "evaluation", "path": str(artifact)}],
                        "actions": [{"label": "Sim", "url": "http://localhost:8080"}],
                    }
                )
            )
            state = root / "state"
            output = publish_receipt(receipt, state)
            self.assertEqual(output.name, "new-robot-task-001.json")
            published = load_agent_runs(state)
            self.assertEqual(len(published), 1)
            self.assertEqual(len(published[0]["artifacts"][0]["sha256"]), 64)
            self.assertFalse(output.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
