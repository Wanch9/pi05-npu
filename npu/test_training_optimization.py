"""CPU unit tests for initialization ordering and baseline compatibility."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from openpi.models_pytorch import device as routing


class TrainingOptimizationTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"WORLD_SIZE": "4", "LOCAL_RANK": "2"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_determinism_precedes_device_and_collective_initialization(self):
        calls = []

        def device():
            self.assertEqual(os.environ["HCCL_DETERMINISTIC"], "true")
            calls.append("device")
            return SimpleNamespace(type="npu", index=2)

        def group(**kwargs):
            self.assertEqual(calls, ["device"])
            self.assertEqual(kwargs["backend"], "hccl")
            calls.append("group")

        with patch.object(routing, "local_device", side_effect=device), patch.object(
            torch, "npu", SimpleNamespace(is_initialized=lambda: False), create=True
        ), patch.object(torch.distributed, "is_initialized", return_value=False), patch.object(
            torch.distributed, "init_process_group", side_effect=group
        ):
            use_ddp, rank, selected = routing.setup_ddp(npu_training_optimization=True)
        self.assertEqual((use_ddp, rank, selected.type, calls), (True, 2, "npu", ["device", "group"]))

    def test_conflicting_environment_fails_before_device_initialization(self):
        os.environ["HCCL_DETERMINISTIC"] = "false"
        with patch.object(routing, "local_device") as device, patch.object(
            torch.distributed, "is_initialized", return_value=False
        ):
            with self.assertRaisesRegex(ValueError, "HCCL_DETERMINISTIC"):
                routing.setup_ddp(npu_training_optimization=True)
            device.assert_not_called()

    def test_existing_group_is_rejected(self):
        with patch.object(torch.distributed, "is_initialized", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "distributed initialization"):
                routing.setup_ddp(npu_training_optimization=True)

    def test_existing_npu_context_is_rejected(self):
        with patch.object(torch.distributed, "is_initialized", return_value=False), patch.object(
            torch, "npu", SimpleNamespace(is_initialized=lambda: True), create=True
        ):
            with self.assertRaisesRegex(RuntimeError, "NPU initialization"):
                routing.setup_ddp(npu_training_optimization=True)

    def test_opt_in_on_cpu_fails_before_collective(self):
        with patch.object(routing, "local_device", return_value=torch.device("cpu")), patch.object(
            torch.distributed, "is_initialized", return_value=False
        ), patch.object(torch.distributed, "init_process_group") as group:
            with self.assertRaisesRegex(ValueError, "requires an NPU"):
                routing.setup_ddp(npu_training_optimization=True)
            group.assert_not_called()

    def test_disabled_preserves_cpu_backend_and_environment(self):
        os.environ["HCCL_DETERMINISTIC"] = "false"
        with patch.object(routing, "local_device", return_value=torch.device("cpu")), patch.object(
            torch.distributed, "is_initialized", return_value=False
        ), patch.object(torch.distributed, "init_process_group") as group:
            self.assertEqual(routing.setup_ddp(), (True, 2, torch.device("cpu")))
            group.assert_called_once_with(backend="gloo", init_method="env://")
        self.assertEqual(os.environ["HCCL_DETERMINISTIC"], "false")


if __name__ == "__main__":
    unittest.main()
