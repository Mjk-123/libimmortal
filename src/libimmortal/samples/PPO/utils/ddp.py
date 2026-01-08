# ddp.py
# DDP utilities for torchrun.
# Comments are intentionally in English for consistency.

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import Optional, Tuple, Dict, Any


# -------------------------
# DDP helpers
# -------------------------
def ddp_is_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def ddp_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return ddp_rank() == 0


def ddp_setup(backend: str | None = None) -> None:
    """
    Initialize process group (torchrun env://).
    Default backend: NCCL if CUDA is available else GLOO.
    """
    if not ddp_is_enabled():
        return

    if dist.is_initialized():
        return

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # For NCCL, set per-process GPU.
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("DDP backend=nccl requested but CUDA is not available.")
        torch.cuda.set_device(ddp_local_rank())

    dist.init_process_group(backend=backend, init_method="env://")


def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def ddp_barrier():
    if dist.is_initialized():
        dist.barrier()


def seed_everything(seed: int):
    s = int(seed) + ddp_rank()
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_module(m):
    return m.module if hasattr(m, "module") else m


def ddp_wrap_model(m: torch.nn.Module) -> torch.nn.Module:
    if not ddp_is_enabled():
        return m
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requested (WORLD_SIZE>1) but CUDA is not available.")
    local_rank = ddp_local_rank()
    return DDP(
        m,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )