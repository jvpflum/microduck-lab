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

    @mock.patch.object(server, "port_available", return_value=True)
    @mock.patch.object(server.subprocess, "Popen", side_effect=FakeProcess)
    def test_play_uses_argument_vector_and_hashed_snapshot(self, popen, _port) -> None:
        manager = server.ProcessManager(self.bench)
        result = manager.launch_viewer(self.manifest["run_id"])
        process = manager.viewer
        assert process is not None
        self.assertEqual(result["run_id"], self.manifest["run_id"])
        self.assertIn("Mjlab-Velocity-Swizzle-MicroDuck", process.command)
        self.assertNotIsInstance(process.command, str)
        self.assertEqual(process.kwargs["env"]["WANDB_MODE"], "disabled")
        process.returncode = 0
        manager.status()

    @mock.patch.object(server, "port_available", side_effect=lambda port: port != 8080)
    def test_play_refuses_occupied_ports(self, _port) -> None:
        manager = server.ProcessManager(self.bench)
        with self.assertRaisesRegex(ValueError, "8080"):
            manager.launch_viewer(self.manifest["run_id"])

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
