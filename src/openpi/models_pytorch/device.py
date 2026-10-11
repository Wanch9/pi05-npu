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
