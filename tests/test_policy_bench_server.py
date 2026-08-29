from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlsplit


LAB_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if "policy_bench" not in sys.modules:
    load_module("policy_bench", LAB_ROOT / "tools" / "policy_bench.py")
server = load_module("policy_bench_server", LAB_ROOT / "tools" / "policy_bench_server.py")
policy_bench = sys.modules["policy_bench"]


class FakeProcess:
    next_pid = 5000

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


class PolicyBenchServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state = self.root / "state"
        run_dir = self.root / "logs" / "velocity_swizzle" / "run-a"
        run_dir.mkdir(parents=True)
        (run_dir / "model_250.pt").write_bytes(b"checkpoint")
        (run_dir / "run.onnx").write_bytes(b"policy")
        self.bench = policy_bench.Bench(self.state)
        self.bench.initialize()
        self.manifest = self.bench.register(run_dir)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_training_language_is_parsed_into_bounded_structure(self) -> None:
        request = server.parse_training_request(
            "Please train swizzle for 8,000 iterations with 2,048 environments"
        )
        self.assertEqual(
            request,
            {
                "task": "swizzle",
                "iterations": 8000,
                "environments": 2048,
                "resource_profile": "shared",
            },
        )
        self.assertIn("error", server.parse_training_request("train walking for 2 iterations"))

    def test_overnight_request_uses_training_priority(self) -> None:
        request = server.parse_training_request("train swizzle overnight for 8000 iterations")
        self.assertEqual(request["resource_profile"], "training-priority")

    def test_jump_request_maps_to_registered_hop_task(self) -> None:
        request = server.parse_training_request("train a roller jump overnight")
        self.assertEqual(request["task"], "hop")
        self.assertEqual(request["iterations"], 1500)
        self.assertEqual(request["resource_profile"], "training-priority")

    def test_unknown_training_skill_requests_clarification(self) -> None:
        request = server.parse_training_request("train something cool")
        self.assertIn("error", request)

    def test_shipped_capability_defaults_to_factory_playground(self) -> None:
        response = server.DashboardServer.chat(mock.Mock(), "train MicroDuck to skate backwards")
        self.assertEqual(response["kind"], "factory-play")
        self.assertEqual(response["url"], "http://localhost:8091/factory/?boot=1")

    @mock.patch.object(server, "running_training_processes", return_value=[])
    def test_explicit_custom_improvement_can_propose_training(self, _running) -> None:
        response = server.DashboardServer.chat(mock.Mock(), "train a custom swizzle policy for 8000 iterations")
        self.assertEqual(response["kind"], "confirm-training")

    def test_training_progress_includes_recent_reward_curve(self) -> None:
        reports = self.root / "reports"
        reports.mkdir()
        (reports / "train-test.log").write_text(
            "Learning iteration 9/20\nMean reward: 12.50\nETA: 00:01:00\n"
            "Learning iteration 10/20\nMean reward: 13.75\nETA: 00:00:57\n"
        )
        with mock.patch.object(server, "LAB_ROOT", self.root):
            progress = server.training_progress()
        assert progress is not None
        self.assertEqual(progress["iteration"], 10)
        self.assertEqual(progress["total"], 20)
        self.assertEqual(
            progress["reward_history"],
            [{"iteration": 9, "reward": 12.5}, {"iteration": 10, "reward": 13.75}],
        )
        self.assertEqual(progress["reward_history_full"], progress["reward_history"])
        self.assertEqual(progress["reward_history_count"], 2)
        self.assertEqual(progress["intelligence"]["current_reward"], 13.75)
        self.assertEqual(progress["intelligence"]["best_reward"], 13.75)
        self.assertEqual(progress["intelligence"]["trend"], "warming up")

    def test_training_intelligence_flags_hop_reward_without_takeoff(self) -> None:
        reports = self.root / "reports"
        reports.mkdir()
        blocks = []
        for iteration in range(1, 42):
            blocks.append(
                f"Learning iteration {iteration}/100\n"
                f"Mean reward: {iteration / 10:.2f}\n"
                "Steps per second: 30000\n"
                "Episode_Reward/hop_takeoff_velocity: 0.0001\n"
                "Episode_Reward/hop_clearance_progress: 2.5\n"
                "Episode_Reward/hop_landing: 0.02\n"
                "Episode_Reward/hop_landing_stillness: 0.0\n"
                "Episode_Termination/nan_state: 0.0\n"
                "ETA: 00:10:00\n"
            )
        (reports / "train-hop.log").write_text("".join(blocks))
        with mock.patch.object(server, "LAB_ROOT", self.root):
            progress = server.training_progress()
        assert progress is not None
        intelligence = progress["intelligence"]
        self.assertEqual(intelligence["trend"], "improving")
        self.assertEqual(intelligence["verdict_tone"], "watch")
        self.assertIn("takeoff signal is near zero", intelligence["verdict"])
        self.assertEqual(intelligence["steps_per_second"], 30000.0)

    def test_training_progress_full_curve_spans_entire_large_run(self) -> None:
        reports = self.root / "reports"
        reports.mkdir()
        lines = [
            f"Learning iteration {iteration}/1201\nMean reward: {iteration / 10:.2f}\n"
            for iteration in range(1, 1202)
        ]
        (reports / "train-large.log").write_text("".join(lines))
        with mock.patch.object(server, "LAB_ROOT", self.root):
            progress = server.training_progress()
        assert progress is not None
        self.assertEqual(progress["reward_history_count"], 1201)
        self.assertEqual(len(progress["reward_history"]), 80)
        self.assertEqual(len(progress["reward_history_full"]), 1000)
        self.assertEqual(progress["reward_history_full"][0]["iteration"], 1)
        self.assertEqual(progress["reward_history_full"][-1]["iteration"], 1201)

    def test_startup_cleanup_only_targets_policy_bench_viewers(self) -> None:
        proc = self.root / "proc"
        dashboard_process = proc / "123"
        manual_process = proc / "124"
        dashboard_process.mkdir(parents=True)
        manual_process.mkdir()
        snapshot = self.state / "runs" / self.manifest["run_id"] / "artifacts" / "model_250.pt"
        (dashboard_process / "cmdline").write_bytes(
            f"python\0tools/play_viser_compat.py\0--checkpoint-file\0{snapshot}\0".encode()
        )
        (manual_process / "cmdline").write_bytes(
            b"python\0tools/play_viser_compat.py\0--checkpoint-file\0/logs/model_250.pt\0"
        )
        with mock.patch.object(server.os, "getpid", return_value=999), \
             mock.patch.object(server.os, "getpgid", side_effect=lambda pid: pid), \
             mock.patch.object(server.os, "killpg") as killpg:
            cleaned = server.cleanup_orphaned_dashboard_viewers(self.state, proc)
        self.assertEqual(cleaned, 1)
        killpg.assert_called_once_with(123, server.signal.SIGTERM)

    @mock.patch.object(server, "port_available", return_value=True)
    @mock.patch.object(server.subprocess, "Popen", side_effect=FakeProcess)
    def test_play_uses_argument_vector_and_hashed_snapshot(self, popen, _port) -> None:
        manager = server.ProcessManager(self.bench)
        result = manager.launch_viewer(self.manifest["run_id"])
        process = manager.viewers[self.manifest["run_id"]].process
        self.assertEqual(result["run_id"], self.manifest["run_id"])
        self.assertEqual(result["label"], "run-a")
        self.assertEqual(result["task"], "swizzle")
        self.assertEqual(result["iteration"], 250)
        self.assertEqual(result["controller_url"], "http://localhost:8090/?arena_port=8080")
        self.assertEqual(result["open_url"], result["controller_url"])
        self.assertIn("Mjlab-Velocity-Swizzle-MicroDuck", process.command)
        self.assertNotIsInstance(process.command, str)
        self.assertEqual(process.kwargs["env"]["WANDB_MODE"], "disabled")
        self.assertEqual(process.kwargs["env"]["DUCKLAB_VISER_PORT"], "8080")
        self.assertEqual(process.kwargs["env"]["DUCKLAB_GAMEPAD_PORT"], "8090")
        process.returncode = 0
        manager.status()

    def test_product_play_opens_exact_policy_in_pollen_browser_arena(self) -> None:
        manager = server.ProcessManager(self.bench)
        result = manager.launch_simulator(self.manifest["run_id"])
        parsed = urlsplit(result["open_url"])
        params = parse_qs(parsed.query)

        self.assertEqual(result["renderer"], "pollen-browser-arena")
        self.assertEqual(result["iteration"], 250)
        self.assertEqual(parsed.path, "/factory/")
        self.assertEqual(params["preview_slot"], ["drive"])
        self.assertEqual(params["preview_loco"], ["rollers"])
        self.assertEqual(
            params["preview_policy"],
            [f"/runs/{self.manifest['run_id']}/artifacts/run.onnx"],
        )
        self.assertEqual(result["policy_sha256"], self.manifest["artifacts"]["policy"]["sha256"])

    def test_hop_preview_uses_full_three_second_phase(self) -> None:
        hop_dir = self.root / "logs" / "roller_hop" / "hop-a"
        hop_dir.mkdir(parents=True)
        (hop_dir / "model_300.pt").write_bytes(b"hop-checkpoint")
        (hop_dir / "hop.onnx").write_bytes(b"hop-policy")
        hop = self.bench.register(hop_dir)

        result = server.ProcessManager(self.bench).launch_simulator(hop["run_id"])
        params = parse_qs(urlsplit(result["open_url"]).query)
        self.assertEqual(params["preview_slot"], ["crouch"])
        self.assertEqual(params["preview_loco"], ["rollers"])
        self.assertEqual(params["preview_label"], ["Hop"])
        self.assertEqual(params["preview_period"], ["3.0"])
        self.assertEqual(params["preview_end"], ["1.0"])

    @mock.patch.object(server, "port_available", side_effect=lambda port: port != 8080)
    def test_play_skips_an_occupied_port_pair(self, _port) -> None:
        manager = server.ProcessManager(self.bench)
        with mock.patch.object(server.subprocess, "Popen", side_effect=FakeProcess):
            result = manager.launch_viewer(self.manifest["run_id"])
        self.assertFalse(result["reused"])
        self.assertEqual(result["viser_port"], 8081)
        self.assertEqual(result["controller_port"], 8092)
        manager.viewers[self.manifest["run_id"]].process.returncode = 0
        manager.status()

    @mock.patch.object(server, "port_available", return_value=True)
    @mock.patch.object(server.subprocess, "Popen", side_effect=FakeProcess)
    def test_live_training_view_samples_six_robots(self, _popen, _port) -> None:
        manager = server.ProcessManager(self.bench)
        result = manager.launch_training_viewer(self.manifest["run_id"])
        process = manager.viewers[self.manifest["run_id"]].process

        self.assertEqual(result["kind"], "training-preview")
        self.assertEqual(result["num_envs"], 6)
        env_index = process.command.index("--num-envs")
        self.assertEqual(process.command[env_index + 1], "6")
        self.assertEqual(result["open_url"], result["controller_url"])
        self.assertEqual(process.kwargs["env"]["DUCKLAB_VIEW_KIND"], "training-preview")
        self.assertEqual(process.kwargs["env"]["DUCKLAB_VIEW_NUM_ENVS"], "6")
        process.returncode = 0
        manager.status()

    @mock.patch.object(server, "port_available", return_value=True)
    def test_each_run_gets_an_isolated_viewer_and_reopen_reuses_it(self, _port) -> None:
        other_dir = self.root / "logs" / "velocity_swizzle" / "run-b"
        other_dir.mkdir(parents=True)
        (other_dir / "model_500.pt").write_bytes(b"other-checkpoint")
        other = self.bench.register(other_dir)
        manager = server.ProcessManager(self.bench)

        with mock.patch.object(server.subprocess, "Popen", side_effect=FakeProcess):
            first = manager.launch_viewer(self.manifest["run_id"])
            reopened = manager.launch_viewer(self.manifest["run_id"])
            second = manager.launch_viewer(other["run_id"])

        self.assertTrue(reopened["reused"])
        self.assertEqual(reopened["pid"], first["pid"])
        self.assertEqual((first["viser_port"], first["controller_port"]), (8080, 8090))
        self.assertEqual((second["viser_port"], second["controller_port"]), (8081, 8092))
        self.assertEqual(len(manager.status()["viewers"]), 2)
        for session in manager.viewers.values():
            session.process.returncode = 0
        manager.status()

    def test_deployment_check_returns_onnx_score(self) -> None:
        result_path = self.root / "deployment.json"
        result_path.write_text(json.dumps({"policy_bench_score": {"overall": 72.5}}))
        manager = server.ProcessManager(self.bench)
        with mock.patch.object(
            self.bench,
            "evaluate",
            return_value={"suite": "skating-v1", "path": str(result_path)},
        ):
            result = manager.run_deployment_check(self.manifest["run_id"])
        self.assertEqual(result["score"], 72.5)
        self.assertEqual(result["suite"], "skating-v1")
        self.assertEqual(result["report_url"], f"/runs/{self.manifest['run_id']}/report.html")

    @mock.patch.object(server, "running_training_processes", return_value=[])
    @mock.patch.object(server.subprocess, "Popen", side_effect=FakeProcess)
    def test_training_launch_is_structured_and_disables_wandb(self, popen, _running) -> None:
        manager = server.ProcessManager(self.bench)
        result = manager.start_training(
            {"task": "swizzle", "iterations": 8000, "environments": 4096}
        )
        process = manager.training
        assert process is not None
        self.assertTrue(result["started"])
        self.assertEqual(len(process.command), 1)
        self.assertEqual(process.kwargs["env"]["WANDB_MODE"], "disabled")
        self.assertEqual(process.kwargs["env"]["DUCKLAB_ITERATIONS"], "8000")
        self.assertEqual(process.kwargs["env"]["DUCKLAB_RESOURCE_PROFILE"], "shared")
        process.returncode = 0
        manager.status()

    @mock.patch.object(server, "running_training_processes", return_value=[])
    @mock.patch.object(server.subprocess, "Popen", side_effect=FakeProcess)
    def test_training_priority_is_forwarded_to_launcher(self, _popen, _running) -> None:
        manager = server.ProcessManager(self.bench)
        result = manager.start_training(
            {
                "task": "swizzle",
                "iterations": 8000,
                "environments": 4096,
                "resource_profile": "training-priority",
            }
        )
        assert manager.training is not None
        self.assertEqual(result["resource_profile"], "training-priority")
        self.assertEqual(
            manager.training.kwargs["env"]["DUCKLAB_RESOURCE_PROFILE"],
            "training-priority",
        )
        manager.training.returncode = 0
        manager.status()

    @mock.patch.object(server, "running_training_processes", return_value=[{"pid": 42}])
    def test_training_refuses_concurrent_run(self, _running) -> None:
        manager = server.ProcessManager(self.bench)
        with self.assertRaisesRegex(ValueError, "already running"):
            manager.start_training(
                {"task": "swizzle", "iterations": 8000, "environments": 4096}
            )


if __name__ == "__main__":
    unittest.main()
