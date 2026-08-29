from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = LAB_ROOT / "tools" / "policy_bench.py"
SPEC = importlib.util.spec_from_file_location("policy_bench", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
policy_bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = policy_bench
SPEC.loader.exec_module(policy_bench)


class PolicyBenchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state = self.root / "state"
        self.run_dir = self.root / "logs" / "velocity_swizzle" / "run-a"
        (self.run_dir / "params").mkdir(parents=True)
        (self.run_dir / "model_250.pt").write_bytes(b"checkpoint-250")
        (self.run_dir / "run-a.onnx").write_bytes(b"policy-250")
        (self.run_dir / "params" / "agent.yaml").write_text("seed: 1\n")
        (self.run_dir / "params" / "env.yaml").write_text("envs: 64\n")
        self.bench = policy_bench.Bench(self.state)
        self.bench.initialize()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def register(self) -> dict:
        return self.bench.register(self.run_dir)

    def metrics(self, forward: float) -> Path:
        path = self.root / f"metrics-{forward}.json"
        path.write_text(json.dumps({"phases": {"forward": {"mean_forward_speed_mps": forward}}}))
        return path

    def test_register_snapshots_artifacts_and_hashes(self) -> None:
        manifest = self.register()
        self.assertEqual(manifest["task"], "swizzle")
        self.assertEqual(manifest["latest_iteration"], 250)
        snapshot = Path(manifest["artifacts"]["policy"]["path"])
        self.assertTrue(snapshot.is_file())
        self.assertTrue(snapshot.is_relative_to(self.state))
        self.assertIn("commit", manifest["source"]["upstream"])
        self.assertIn("branch", manifest["source"]["upstream"])
        self.assertIn("remote", manifest["source"]["upstream"])
        (self.run_dir / "run-a.onnx").write_bytes(b"changed-source")
        self.assertEqual(snapshot.read_bytes(), b"policy-250")

    def test_new_checkpoint_becomes_new_immutable_candidate(self) -> None:
        first = self.register()
        (self.run_dir / "model_500.pt").write_bytes(b"checkpoint-500")
        (self.run_dir / "run-a.onnx").write_bytes(b"policy-500")
        second = self.register()
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(second["latest_iteration"], 500)

    def test_discovery_does_not_pair_checkpoint_newer_than_export(self) -> None:
        policy = self.run_dir / "run-a.onnx"
        checkpoint = self.run_dir / "model_500.pt"
        checkpoint.write_bytes(b"checkpoint-500")
        policy_time = policy.stat().st_mtime
        os.utime(checkpoint, (policy_time + 10, policy_time + 10))
        manifest = self.register()
        self.assertEqual(manifest["latest_iteration"], 250)

    def test_attached_evaluation_advances_to_evaluated(self) -> None:
        manifest = self.register()
        self.bench.attach_evaluation(manifest["run_id"], self.metrics(0.2), "skating-v1")
        updated = self.bench.load_manifest(manifest["run_id"])
        self.assertEqual(updated["stage"], "evaluated")
        self.assertEqual(updated["evaluations"][0]["suite"], "skating-v1")

    def test_compare_emits_candidate_delta(self) -> None:
        baseline = self.register()
        self.bench.attach_evaluation(baseline["run_id"], self.metrics(0.2), "skating-v1")
        (self.run_dir / "model_500.pt").write_bytes(b"checkpoint-500")
        (self.run_dir / "run-a.onnx").write_bytes(b"policy-500")
        candidate = self.register()
        self.bench.attach_evaluation(candidate["run_id"], self.metrics(0.3), "skating-v1")
        result = self.bench.compare(candidate["run_id"], baseline["run_id"], "skating-v1")
        metric = next(row for row in result["metrics"] if row["metric"].endswith("mean_forward_speed_mps"))
        self.assertAlmostEqual(metric["delta"], 0.1)

    def test_promotion_is_sequential_and_hardware_gated(self) -> None:
        manifest = self.register()
        run_id = manifest["run_id"]
        with self.assertRaises(SystemExit):
            self.bench.promote(run_id, "sim-qualified", "tester", "looks good", False)
        self.bench.attach_evaluation(run_id, self.metrics(0.2), "skating-v1")
        promoted = self.bench.promote(run_id, "sim-qualified", "tester", "reviewed", False)
        self.assertEqual(promoted["stage"], "sim-qualified")
        with self.assertRaises(SystemExit):
            self.bench.promote(run_id, "hardware-candidate", "tester", "bench", False)
        promoted = self.bench.promote(run_id, "hardware-candidate", "tester", "bench", True)
        self.assertEqual(promoted["stage"], "hardware-candidate")

    def test_resolve_verifies_hash(self) -> None:
        manifest = self.register()
        run_id = manifest["run_id"]
        self.bench.attach_evaluation(run_id, self.metrics(0.2), "skating-v1")
        self.bench.promote(run_id, "sim-qualified", "tester", "reviewed", False)
        resolved = self.bench.resolve("swizzle", "sim-qualified", "checkpoint")
        self.assertTrue(resolved.is_file())
        resolved.write_bytes(b"tampered")
        with self.assertRaises(SystemExit):
            self.bench.resolve("swizzle", "sim-qualified", "checkpoint")

    def test_dashboard_is_self_contained(self) -> None:
        self.register()
        dashboard = self.bench.render_dashboard()
        content = dashboard.read_text()
        self.assertIn("MicroDuck Policy Bench", content)
        self.assertIn("Open simulation", content)
        self.assertIn("Checkpoint debuggers", content)
        self.assertIn("viewer-sessions", content)
        self.assertIn("reward-scope", content)
        self.assertIn("Entire run", content)
        self.assertIn("Drive arena", content)
        self.assertIn("Open checkpoint debugger", content)
        self.assertIn("Open factory playground", content)
        self.assertIn("http://localhost:8091/factory/?boot=1", content)
        self.assertIn("Deployment check", content)
        self.assertIn("Resource mode", content)
        self.assertIn("Pollen stack", content)
        self.assertIn("outliers clipped", content)
        self.assertNotIn("raw range", content)
        self.assertIn("MicroDuck Control Room", content)
        self.assertIn("finished-card", content)
        self.assertIn("saved-dropdown", content)
        self.assertNotIn("<th>Training run</th>", content)
        self.assertIn("DuckLab Assistant", content)
        self.assertIn("__CONTROL_TOKEN__", content)
        self.assertNotIn("https://", content)

    def test_heuristic_score_is_transparent_and_bounded(self) -> None:
        evaluation = {
            "phases": {
                "forward": {"mean_forward_speed_mps": 0.3, "both_blades_grounded_fraction": 1.0, "tilt_max_deg": 0.0, "mean_action_acceleration": 0.0, "mean_abs_lateral_speed_mps": 0.0, "estimated_swizzle_cycles": 8},
                "reverse": {"mean_forward_speed_mps": -0.3, "both_blades_grounded_fraction": 1.0, "tilt_max_deg": 0.0, "mean_action_acceleration": 0.0, "mean_abs_lateral_speed_mps": 0.0, "estimated_swizzle_cycles": 8},
                "stop_forward": {"end_abs_forward_speed_mps": 0.0},
                "stop_reverse": {"end_abs_forward_speed_mps": 0.0},
                "heading_left": {"mean_yaw_rate_rad_s": 0.25, "both_blades_grounded_fraction": 1.0, "tilt_max_deg": 0.0, "mean_action_acceleration": 0.0, "mean_abs_lateral_speed_mps": 0.0},
                "heading_right": {"mean_yaw_rate_rad_s": -0.25, "both_blades_grounded_fraction": 1.0, "tilt_max_deg": 0.0, "mean_action_acceleration": 0.0, "mean_abs_lateral_speed_mps": 0.0},
            }
        }
        score = policy_bench.score_evaluation(evaluation, "swizzle")
        self.assertGreaterEqual(score["overall"], 0.0)
        self.assertLessEqual(score["overall"], 100.0)
        self.assertIn("reverse_tracking", score["components"])
        self.assertTrue(score["qualified"])
        self.assertIn("stopping", score["qualification_gates"])
        self.assertTrue(policy_bench.score_evaluation(evaluation, "roller")["qualified"])


if __name__ == "__main__":
    unittest.main()
