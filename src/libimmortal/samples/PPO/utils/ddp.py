# ddp.py
# DDP utilities for torchrun.
# Comments are intentionally in English for consistency.

from __future__ import annotations

import os, time
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

def _ddp_barrier(tag: str, ddp_barrier_timeout_s: float):
        if not ddp_is_enabled() or (not dist.is_initialized()):
            return
        try:
            # Version-compatible "timeout barrier"
            work = dist.barrier(async_op=True)

            # Fast path: Work.wait(timeout=...) exists in some builds
            try:
                work.wait(timeout=ddp_barrier_timeout_s)  # type: ignore[arg-type]
                return
            except TypeError:
                # No timeout support in wait()
                pass

            # Polling path: Work.is_completed() is widely available
            if hasattr(work, "is_completed"):
                t_deadline = time.time() + float(ddp_barrier_timeout_s)
                while time.time() < t_deadline:
                    if work.is_completed():  # type: ignore[attr-defined]
                        return
                    time.sleep(0.05)
                raise TimeoutError(f"DDP barrier timed out after {ddp_barrier_timeout_s}s (tag={tag})")

            # Fallback: no timeout possible (oldest builds)
            work.wait()
        except Exception as e:
             # Raise so we drop to finally and save on rank0.
             if is_main_process():
                 print(f"[DDP][barrier][{tag}] FAILED (timeout={ddp_barrier_timeout_s}s): {repr(e)}", flush=True)
             raise
        
def _ddp_barrier_soft(tag: str, timeout_s: float):
    if not ddp_is_enabled() or (not dist.is_initialized()):
        return False
    try:
        work = dist.barrier(async_op=True)

        # wait(timeout=) 지원이면 그걸 쓰고
        try:
            work.wait(timeout=timeout_s)  # type: ignore[arg-type]
            return True
        except TypeError:
            pass

        # polling
        if hasattr(work, "is_completed"):
            t_deadline = time.time() + float(timeout_s)
            while time.time() < t_deadline:
                if work.is_completed():  # type: ignore[attr-defined]
                    return True
                time.sleep(0.05)
            if is_main_process():
                print(f"[DDP][barrier-soft][{tag}] TIMEOUT after {timeout_s}s (continuing)", flush=True)
            return False

        # oldest fallback (timeout 불가) => 여기서는 아예 blocking barrier를 하면 안 됨
        if is_main_process():
            print(f"[DDP][barrier-soft][{tag}] NO TIMEOUT SUPPORT -> skip", flush=True)
        return False

    except Exception as e:
        if is_main_process():
            print(f"[DDP][barrier-soft][{tag}] FAILED: {repr(e)} (continuing)", flush=True)
        return False


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