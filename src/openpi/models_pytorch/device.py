"""Explicit accelerator routing; importing this module preserves CPU/CUDA support."""
import os

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def local_device():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested = os.environ.get("OPENPI_DEVICE", "auto")
    if requested not in {"auto", "npu", "cuda", "cpu"}:
        raise ValueError(f"Invalid OPENPI_DEVICE: {requested}")
    if requested == "npu" or (requested == "auto" and torch_npu is not None and torch.npu.is_available()):
        if torch_npu is None or not torch.npu.is_available():
            raise RuntimeError("NPU requested but torch_npu is unavailable")
        device = torch.device(f"npu:{rank}")
        torch.npu.set_device(device)
        return device
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def backend(device):
    return {"npu": "hccl", "cuda": "nccl", "cpu": "gloo"}[device.type]


def accelerator(device):
    return getattr(torch, device.type) if device.type in {"npu", "cuda"} else None


def setup_ddp(*, npu_training_optimization: bool = False):
    """Configure deterministic HCCL before device/process-group initialization."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if npu_training_optimization:
        if torch.distributed.is_initialized():
            raise RuntimeError("NPU training optimization must be configured before distributed initialization")
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_initialized():
            raise RuntimeError("NPU training optimization must be configured before NPU initialization")
        if os.environ.get("HCCL_DETERMINISTIC", "true").lower() != "true":
            raise ValueError("NPU training optimization requires HCCL_DETERMINISTIC=true; unset the conflicting value")
        os.environ["HCCL_DETERMINISTIC"] = "true"
    device = local_device()
    if npu_training_optimization and device.type != "npu":
        raise ValueError("--pytorch-npu-training-optimization requires an NPU device (OPENPI_DEVICE=npu)")
    if use_ddp and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend(device), init_method="env://")
        os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "INFO")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    return use_ddp, local_rank, device
