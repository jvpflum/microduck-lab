from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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
            {"task": "swizzle", "iterations": 8000, "environments": 2048},
        )
        self.assertIn("error", server.parse_training_request("train walking for 2 iterations"))

    def test_unknown_training_skill_requests_clarification(self) -> None:
        request = server.parse_training_request("train something cool")
        self.assertIn("error", request)

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
        self.assertIn("Mjlab-Velocity-Swizzle-MicroDuck", process.command)
        self.assertNotIsInstance(process.command, str)
        self.assertEqual(process.kwargs["env"]["WANDB_MODE"], "disabled")
        self.assertEqual(process.kwargs["env"]["DUCKLAB_VISER_PORT"], "8080")
        self.assertEqual(process.kwargs["env"]["DUCKLAB_GAMEPAD_PORT"], "8090")
        process.returncode = 0
        manager.status()

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
        process.returncode = 0
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
