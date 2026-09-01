from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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

    def sprint_metrics(self, speed_offset: float = 0.0) -> dict:
        phases = {}
        for suffix, command, speed in (
            ("030", 0.30, 0.36),
            ("040", 0.40, 0.44),
            ("050", 0.50, 0.48),
            ("055", 0.55, 0.51),
        ):
            phases[f"speed_{suffix}"] = {
                "command_x_mps": command,
                "steady_mean_forward_speed_mps": speed + speed_offset,
                "steady_mean_abs_lateral_speed_mps": 0.024,
                "peak_abs_forward_speed_mps": speed + speed_offset + 0.04,
                "acceleration_first_second_mps2": 0.42,
                "time_to_0_5_mps_s": 1.1 if suffix == "055" else None,
                "tilt_max_deg": 14.9,
                "trunk_height_mean_m": 0.114,
            }
            phases[f"stop_{suffix}"] = {"stop_time_below_0_05_mps_s": 1.2}
        return {"profile": "sprint", "phases": phases}

    def test_race5_record_requires_speed_and_straightness_together(self) -> None:
        incumbent = {
            "performance": {
                "finished_100ft": True,
                "elapsed_time_100ft_s": 44.06,
                "long_run_max_drift_ft": 1.06,
                "long_run_max_heading_error_deg": 7.30,
                "agility_score": 67.77,
                "auto_steering_percent": 13.79,
            }
        }
        balanced_win = {
            "performance": {
                "finished_100ft": True,
                "elapsed_time_100ft_s": 43.90,
                "long_run_max_drift_ft": 0.95,
                "long_run_max_heading_error_deg": 7.10,
                "agility_score": 66.0,
                "auto_steering_percent": 13.5,
            }
        }
        faster_but_drifting = json.loads(json.dumps(balanced_win))
        faster_but_drifting["performance"]["long_run_max_drift_ft"] = 1.25
        straighter_but_slower = json.loads(json.dumps(balanced_win))
        straighter_but_slower["performance"]["elapsed_time_100ft_s"] = 44.20

        self.assertTrue(policy_bench.race5_advances_incumbent(balanced_win, incumbent))
        self.assertFalse(policy_bench.race5_advances_incumbent(faster_but_drifting, incumbent))
        self.assertFalse(policy_bench.race5_advances_incumbent(straighter_but_slower, incumbent))

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

    def test_archive_hides_run_without_deleting_artifacts(self) -> None:
        manifest = self.register()
        snapshot = Path(manifest["artifacts"]["checkpoint"]["path"])
        archived = self.bench.archive(manifest["run_id"], "legacy experiment")
        self.assertTrue(archived["archived"])
        self.assertTrue(snapshot.is_file())
        self.assertNotIn(manifest["run_id"], self.bench.render_dashboard().read_text())
        restored = self.bench.unarchive(manifest["run_id"])
        self.assertFalse(restored["archived"])
        self.assertIn(manifest["run_id"], {item["run_id"] for item in self.bench.manifests()})
        self.assertNotIn(manifest["run_id"], self.bench.render_dashboard().read_text())

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
        self.assertIn("Dark Wing Duck Enterprise", content)
        self.assertIn("MicroDuck model results", content)
        self.assertIn("What each checkpoint is for", content)
        self.assertIn("RECOMMENDED NEXT MOVE", content)
        self.assertIn("Best overall vs Pollen", content)
        self.assertIn("DEFINITIVE RANKING", content)
        self.assertIn("All comparable results", content)
        self.assertIn("Every artifact, raw run, and evaluation remains immutable", content)
        self.assertNotIn("Open simulator sessions", content)
        self.assertNotIn("Top 3 comparison models", content)
        self.assertNotIn("Dark Wing Copilot", content)
        self.assertNotIn("Resource mode", content)
        self.assertNotIn("viewer-sessions", content)
        self.assertNotIn("reward-scope", content)
        self.assertIn("__CONTROL_TOKEN__", content)
        self.assertNotIn("https://", content)

    def test_test_roster_requires_hash_verified_policy_snapshots(self) -> None:
        manifests = []
        configured = []
        for index in range(3):
            run_id = f"race5-comparison-{index}"
            policy_path = self.state / "runs" / run_id / "artifacts" / "policy.onnx"
            policy_path.parent.mkdir(parents=True)
            policy_path.write_bytes(f"policy-{index}".encode())
            manifests.append(
                {
                    "run_id": run_id,
                    "task": "race5",
                    "experiment_label": f"comparison-{index}",
                    "source_run_dir": str(self.run_dir),
                    "artifacts": {
                        "policy": {
                            "path": str(policy_path),
                            "sha256": policy_bench.sha256(policy_path),
                        }
                    },
                }
            )
            configured.append(
                {
                    "run_id": run_id,
                    "role": f"role-{index}",
                    "physics": "wheel friction 0.003",
                    "sustained_mph": 2.0 + index,
                    "top_mph": 2.5 + index,
                }
            )
        registry = {"tasks": {"race5": {"test-roster": configured}}}
        rendered = policy_bench.render_test_roster(registry, manifests, "race5")
        self.assertEqual(rendered.count("Try in arena"), 3)
        self.assertIn("TEST #1", rendered)
        self.assertIn("TEST #3", rendered)
        self.assertIn("wheel friction 0.003", rendered)

        Path(manifests[1]["artifacts"]["policy"]["path"]).write_bytes(b"tampered")
        rendered = policy_bench.render_test_roster(registry, manifests, "race5")
        self.assertEqual(rendered.count("Try in arena"), 2)
        self.assertIn("Policy hash unavailable", rendered)

    def test_legacy_internal_flip_id_is_presented_as_front_flip(self) -> None:
        self.assertEqual(policy_bench.display_task_name("backflip"), "Front flip")
        self.assertEqual(
            policy_bench.display_experiment_label("backflip", "roller-backflip-v1"),
            "roller-frontflip-v1",
        )

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

    def test_hop_score_requires_takeoff_and_controlled_landing(self) -> None:
        evaluation = {
            "hop": {
                "takeoff_detected": True,
                "landing_detected": True,
                "peak_clearance_m": 0.08,
                "air_time_s": 0.20,
                "horizontal_drift_m": 0.01,
                "final_tilt_mean_deg": 3.0,
                "final_speed_mean_mps": 0.02,
                "final_both_grounded_fraction": 1.0,
            }
        }
        score = policy_bench.score_evaluation(evaluation, "hop")
        self.assertTrue(score["qualified"])
        self.assertGreater(score["overall"], 80.0)
        evaluation["hop"]["takeoff_detected"] = False
        score = policy_bench.score_evaluation(evaluation, "hop")
        self.assertFalse(score["qualified"])
        self.assertLessEqual(score["overall"], 49.0)

    def test_hop_score_rejects_contact_flicker_as_a_jump(self) -> None:
        evaluation = {
            "hop": {
                "takeoff_detected": True,
                "landing_detected": True,
                "peak_clearance_m": 0.0278,
                "air_time_s": 0.06,
                "horizontal_drift_m": 0.0155,
                "final_tilt_mean_deg": 0.37,
                "final_speed_mean_mps": 0.0042,
                "final_both_grounded_fraction": 1.0,
            }
        }
        score = policy_bench.score_evaluation(evaluation, "hop")
        self.assertFalse(score["qualified"])
        self.assertFalse(score["qualification_gates"]["clearance"]["passed"])
        self.assertFalse(score["qualification_gates"]["air_time"]["passed"])

    def test_sprint_score_has_explicit_qualification_gates(self) -> None:
        score = policy_bench.score_evaluation(self.sprint_metrics(), "sprint")
        self.assertTrue(score["qualified"])
        self.assertGreater(score["overall"], 50.0)
        self.assertEqual(score["label"], "Sprint-v1 qualification score")
        self.assertTrue(score["qualification_gates"]["speed_at_0_55"]["passed"])
        self.assertAlmostEqual(score["performance"]["sustained_speed_mph"], 1.141, places=3)
        self.assertEqual(score["performance"]["acceleration_first_second_mps2"], 0.42)
        self.assertIn("agility_score", score["performance"])
        failed = policy_bench.score_evaluation(self.sprint_metrics(-0.03), "sprint")
        self.assertFalse(failed["qualified"])
        self.assertLessEqual(failed["overall"], 49.0)

    def test_race5_uses_five_mph_as_a_goal_not_a_hard_gate(self) -> None:
        evaluation = {
            "profile": "race-5mph",
            "phases": {
                "cruise": {
                    "steady_mean_forward_speed_mps": 0.34,
                    "yaw_change_deg": 12.0,
                    "tilt_max_deg": 11.0,
                    "trunk_height_mean_m": 0.115,
                },
                "stop_cruise": {
                    "stop_time_below_0_05_mps_s": 1.3,
                    "end_abs_forward_speed_mps": 0.01,
                    "tilt_max_deg": 12.0,
                    "trunk_height_mean_m": 0.115,
                },
                "turn_left": {
                    "yaw_change_deg": 90.0,
                    "tilt_max_deg": 10.0,
                    "trunk_height_mean_m": 0.115,
                },
                "turn_right": {
                    "yaw_change_deg": -90.0,
                    "tilt_max_deg": 10.0,
                    "trunk_height_mean_m": 0.115,
                },
                "max_speed": {
                    "peak_horizontal_speed_mps": 2.6,
                    "finished_100ft": True,
                    "finish_time_100ft_s": 15.0,
                    "trap_speed_100ft_mph": 5.4,
                    "distance_remaining_100ft_ft": 0.0,
                    "max_heading_error_deg": 18.0,
                    "max_lateral_drift_ft": 1.5,
                    "tilt_max_deg": 14.0,
                    "trunk_height_mean_m": 0.11,
                },
                "race": {
                    "finished_5m": True,
                    "finish_time_5m_s": 2.2,
                    "forward_progress_m": 5.01,
                    "duration_s": 2.9,
                    "mean_world_forward_speed_mps": 2.25,
                    "steady_mean_world_forward_speed_mps": 2.25,
                    "peak_world_forward_speed_mps": 2.30,
                    "acceleration_first_second_mps2": 1.6,
                    "time_to_0_5_mps_s": 0.4,
                    "yaw_change_deg": 8.0,
                    "tilt_max_deg": 12.0,
                    "steady_mean_abs_lateral_speed_mps": 0.01,
                    "trunk_height_mean_m": 0.11,
                },
            },
        }
        score = policy_bench.score_evaluation(evaluation, "race5")
        self.assertTrue(score["qualified"])
        self.assertTrue(score["five_mph_goal_reached"])
        self.assertGreater(score["overall"], 80.0)
        self.assertGreaterEqual(score["performance"]["top_speed_mph"], 5.0)
        self.assertAlmostEqual(score["performance"]["ten_mph_stretch_percent"], 50.3, places=1)
        evaluation["phases"]["race"]["steady_mean_world_forward_speed_mps"] = 1.5
        failed = policy_bench.score_evaluation(evaluation, "race5")
        self.assertTrue(failed["qualified"])
        self.assertFalse(failed["five_mph_goal_reached"])
        self.assertFalse(failed["record_qualified"])
        self.assertTrue(failed["simulation_champion_eligible"])
        self.assertGreater(failed["overall"], 49.0)
        self.assertEqual(failed["label"], "Race5 speed development score")
        self.assertNotIn("finish_5m", failed["qualification_gates"])
        self.assertNotIn("finish_time", failed["qualification_gates"])
        self.assertNotIn("five_mph_sustained", failed["qualification_gates"])
        self.assertNotIn("finish", failed["components"])

    def test_race5_speed_cannot_replace_lost_basic_skills(self) -> None:
        evaluation = {
            "profile": "race-5mph",
            "phases": {
                "race": {
                    "duration_s": 8.0,
                    "steady_mean_world_forward_speed_mps": 2.5,
                    "peak_world_forward_speed_mps": 2.8,
                    "acceleration_first_second_mps2": 1.8,
                    "yaw_change_deg": 5.0,
                    "tilt_max_deg": 10.0,
                    "steady_mean_abs_lateral_speed_mps": 0.01,
                    "trunk_height_mean_m": 0.115,
                },
                "cruise": {
                    "steady_mean_forward_speed_mps": 0.10,
                    "yaw_change_deg": 100.0,
                    "tilt_max_deg": 10.0,
                    "trunk_height_mean_m": 0.115,
                },
                "stop_cruise": {
                    "stop_time_below_0_05_mps_s": None,
                    "end_abs_forward_speed_mps": 0.2,
                    "tilt_max_deg": 10.0,
                    "trunk_height_mean_m": 0.115,
                },
                "turn_left": {"yaw_change_deg": -60.0, "tilt_max_deg": 10.0, "trunk_height_mean_m": 0.115},
                "turn_right": {"yaw_change_deg": 60.0, "tilt_max_deg": 10.0, "trunk_height_mean_m": 0.115},
                "max_speed": {
                    "peak_horizontal_speed_mps": 2.8,
                    "finished_100ft": True,
                    "finish_time_100ft_s": 14.0,
                    "trap_speed_100ft_mph": 5.8,
                    "distance_remaining_100ft_ft": 0.0,
                    "max_heading_error_deg": 10.0,
                    "max_lateral_drift_ft": 1.0,
                    "tilt_max_deg": 10.0,
                    "trunk_height_mean_m": 0.115,
                },
            },
        }
        score = policy_bench.score_evaluation(evaluation, "race5")
        self.assertTrue(score["five_mph_goal_reached"])
        self.assertFalse(score["qualified"])
        self.assertFalse(score["simulation_champion_eligible"])
        self.assertFalse(score["qualification_gates"]["cruise_speed"]["passed"])
        self.assertFalse(score["qualification_gates"]["braking"]["passed"])
        self.assertFalse(score["qualification_gates"]["turn_left"]["passed"])
        self.assertFalse(score["qualification_gates"]["turn_right"]["passed"])

    def test_race5_rejects_sideways_speed_on_long_course(self) -> None:
        evaluation = {
            "profile": "race-5mph",
            "phases": {
                "race": {
                    "duration_s": 8.0,
                    "steady_mean_world_forward_speed_mps": 1.0,
                    "peak_world_forward_speed_mps": 1.2,
                    "acceleration_first_second_mps2": 0.5,
                    "yaw_change_deg": 10.0,
                    "tilt_max_deg": 12.0,
                    "steady_mean_abs_lateral_speed_mps": 0.01,
                    "trunk_height_mean_m": 0.11,
                },
                "cruise": {"steady_mean_forward_speed_mps": 0.3, "yaw_change_deg": 5.0, "tilt_max_deg": 10.0, "trunk_height_mean_m": 0.11},
                "stop_cruise": {"stop_time_below_0_05_mps_s": 1.0, "end_abs_forward_speed_mps": 0.0, "tilt_max_deg": 10.0, "trunk_height_mean_m": 0.11},
                "turn_left": {"yaw_change_deg": 90.0, "tilt_max_deg": 10.0, "trunk_height_mean_m": 0.11},
                "turn_right": {"yaw_change_deg": -90.0, "tilt_max_deg": 10.0, "trunk_height_mean_m": 0.11},
                "max_speed": {
                    "peak_horizontal_speed_mps": 3.0,
                    "finished_100ft": False,
                    "finish_time_100ft_s": None,
                    "trap_speed_100ft_mph": None,
                    "distance_remaining_100ft_ft": 50.0,
                    "max_heading_error_deg": 180.0,
                    "max_lateral_drift_ft": 20.0,
                    "tilt_max_deg": 12.0,
                    "trunk_height_mean_m": 0.11,
                },
            },
        }
        score = policy_bench.score_evaluation(evaluation, "race5")
        self.assertGreater(score["performance"]["top_speed_mph"], 6.0)
        self.assertFalse(score["simulation_champion_eligible"])
        self.assertFalse(score["qualification_gates"]["long_run_heading"]["passed"])
        self.assertFalse(score["qualification_gates"]["long_run_drift"]["passed"])
        self.assertFalse(score["qualification_gates"]["a_to_b_100ft"]["passed"])

    def test_race5_reports_a_to_b_win_separately_from_control_qualification(self) -> None:
        candidate = {
            "phases": {
                "race": {
                    "steady_mean_world_forward_speed_mps": 0.7,
                    "acceleration_first_second_mps2": 0.5,
                },
                "max_speed": {
                    "finished_100ft": True,
                    "finish_time_100ft_s": 46.5,
                    "peak_horizontal_speed_mps": 0.82,
                    "max_heading_error_deg": 33.0,
                    "max_lateral_drift_ft": 22.0,
                },
            }
        }
        pollen = {
            "phases": {
                "race": {
                    "steady_mean_world_forward_speed_mps": 0.4,
                    "acceleration_first_second_mps2": 0.25,
                },
                "max_speed": {
                    "finished_100ft": False,
                    "finish_time_100ft_s": None,
                    "peak_horizontal_speed_mps": 0.68,
                    "max_heading_error_deg": 1700.0,
                    "max_lateral_drift_ft": 10.0,
                },
            }
        }

        comparison = policy_bench.race_baseline_comparison(
            candidate, pollen, self.root / "pollen.json"
        )

        self.assertTrue(comparison["a_to_b_improved"])
        self.assertFalse(comparison["improved"])
        self.assertEqual(
            comparison["verdict"],
            "Beat Pollen from A to B; control qualification pending",
        )

    def test_sprint_result_is_clear_in_dashboard_and_detail_report(self) -> None:
        sprint_dir = self.root / "logs" / "velocity_sprint" / "sprint-run"
        sprint_dir.mkdir(parents=True)
        (sprint_dir / "model_49.pt").write_bytes(b"sprint-checkpoint")
        (sprint_dir / "sprint.onnx").write_bytes(b"sprint-policy")
        manifest = self.bench.register(sprint_dir, task="sprint")
        candidate_path = self.root / "candidate-sprint.json"
        candidate_path.write_text(json.dumps(self.sprint_metrics()))
        baseline = self.sprint_metrics(-0.03)
        baseline_path = self.root / "pollen-sprint.json"
        baseline_path.write_text(json.dumps(baseline))
        with mock.patch.object(policy_bench, "SPRINT_BASELINE_REPORT", baseline_path):
            self.bench.attach_evaluation(manifest["run_id"], candidate_path, "sprint-v1")
        dashboard = self.bench.render_dashboard().read_text()
        report = (self.state / "runs" / manifest["run_id"] / "report.html").read_text()
        updated = self.bench.load_manifest(manifest["run_id"])
        self.assertIn("BEST OVERALL", dashboard)
        self.assertIn("sprint-run", dashboard)
        self.assertIn("1.14 mph", dashboard)
        self.assertIn("Best overall vs Pollen", dashboard)
        self.assertIn("DEFINITIVE RANKING", dashboard)
        self.assertIn("Pollen official roller", dashboard)
        self.assertIn("Try best model", dashboard)
        self.assertNotIn("podium-grid'>", dashboard)
        self.assertNotIn("All experiments and unscored runs", dashboard)
        self.assertEqual(updated["stage"], "sim-qualified")
        self.assertIn("YES — this run improved skating speed", report)
        self.assertIn("8/8 gates passed", report)
        self.assertIn("Speed across useful commands", report)
        self.assertIn("Sustained speed", report)
        self.assertIn("mph", report)
        self.assertIn("Agility score", report)

    def test_frontflip_score_requires_complete_clean_unassisted_rollouts(self) -> None:
        evaluation = {
            "frontflip": {
                "episodes": 256,
                "unassisted": True,
                "success_rate": 0.85,
                "takeoff_rate": 0.95,
                "landing_rate": 0.90,
                "settled_rate": 0.85,
                "body_contact_rate": 0.005,
                "median_peak_clearance_m": 0.08,
                "median_forward_rotation_deg": 360.0,
                "median_offaxis_rotation_deg": 10.0,
                "median_horizontal_drift_m": 0.04,
            }
        }
        score = policy_bench.score_evaluation(evaluation, "backflip")
        self.assertTrue(score["qualified"])
        self.assertGreater(score["overall"], 80.0)
        evaluation["frontflip"]["body_contact_rate"] = 0.25
        score = policy_bench.score_evaluation(evaluation, "backflip")
        self.assertFalse(score["qualified"])
        self.assertLessEqual(score["overall"], 49.0)
        self.assertLessEqual(score["overall"], 49.0)


if __name__ == "__main__":
    unittest.main()
