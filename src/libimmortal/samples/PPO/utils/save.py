import os, glob
import torch
import numpy as np
import re
from typing import Optional, Tuple, Dict, Any

import libimmortal.samples.PPO.utils.ddp as ddp
from libimmortal.samples.PPO.reward import RewardConfig, RewardScaler


# -------------------------
# Checkpoint helpers
# -------------------------
def _checkpoint_dir() -> str:
    return r"/root/libimmortal/src/libimmortal/samples/PPO/checkpoints"


def _latest_checkpoint_path(ckpt_dir: str, prefix: str) -> Optional[str]:
    paths = glob.glob(os.path.join(ckpt_dir, f"{prefix}*.pth"))
    if not paths:
        return None
    paths.sort(key=os.path.getmtime, reverse=True)
    return paths[0]


def _infer_step_from_ckpt_path(ckpt_path: Optional[str], ckpt_prefix: str) -> int:
    if not ckpt_path:
        return 0
    base = os.path.basename(ckpt_path)
    m = re.match(re.escape(ckpt_prefix) + r"(\d+)\.pth$", base)
    return int(m.group(1)) if m else 0


def atomic_torch_save(make_obj_fn, path: str):
    tmp = path + ".tmp"
    obj = make_obj_fn()
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _optimizer_to_device(opt: torch.optim.Optimizer, device: torch.device):
    """
    Move optimizer state tensors onto `device`.
    This matters when you load optimizer state saved on CPU.
    """
    try:
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    except Exception:
        # Best-effort: do not crash training on optimizer migration
        pass


def _find_optimizers(ppo_agent) -> Dict[str, torch.optim.Optimizer]:
    """
    Try common attribute names used across PPO implementations.
    Also tries nested under policy module (best-effort).
    """
    cands = [
        "optimizer",
        "actor_optimizer",
        "critic_optimizer",
        "optimizer_actor",
        "optimizer_critic",
        "optim",
        "opt",
    ]

    out: Dict[str, torch.optim.Optimizer] = {}

    # 1) PPO object itself
    for name in cands:
        opt = getattr(ppo_agent, name, None)
        if opt is not None and hasattr(opt, "state_dict") and hasattr(opt, "load_state_dict"):
            out[f"ppo::{name}"] = opt

    # 2) policy module (some implementations keep optimizer there)
    try:
        pol = getattr(ppo_agent, "policy", None)
        pol = ddp.get_module(pol) if pol is not None else None
        if pol is not None:
            for name in cands:
                opt = getattr(pol, name, None)
                if opt is not None and hasattr(opt, "state_dict") and hasattr(opt, "load_state_dict"):
                    out[f"policy::{name}"] = opt
    except Exception:
        pass

    return out


def _make_checkpoint(
    ppo_agent,
    step: int,
    args,
    cfg: RewardConfig,
    reward_scaler: Optional[RewardScaler],
) -> Dict[str, Any]:
    """
    Full checkpoint: model + optimizer(s) + a bit of metadata.
    """
    ckpt: Dict[str, Any] = {
        "format": 2,
        "step": int(step),
        "policy_old": ddp.get_module(ppo_agent.policy_old).state_dict() if hasattr(ppo_agent, "policy_old") else None,
        "policy": ddp.get_module(ppo_agent.policy).state_dict() if hasattr(ppo_agent, "policy") else None,
        "args": vars(args),
        "reward_cfg": getattr(cfg, "__dict__", None),
    }

    # Optimizers (best-effort)
    opts = _find_optimizers(ppo_agent)
    for name, opt in opts.items():
        ckpt[f"opt::{name}"] = opt.state_dict()

    # Optional: reward scaler stats (only meaningful if you use it)
    if reward_scaler is not None:
        try:
            ckpt["reward_scaler"] = {
                "t": int(getattr(reward_scaler, "t", 0)),
                "ret": float(getattr(reward_scaler, "ret", 0.0)),
                "ret_rms_mean": float(np.asarray(reward_scaler.ret_rms.mean).reshape(())),
                "ret_rms_var": float(np.asarray(reward_scaler.ret_rms.var).reshape(())),
                "ret_rms_count": float(getattr(reward_scaler.ret_rms, "count", 1e-4)),
                "gamma": float(getattr(reward_scaler, "gamma", 0.99)),
            }
        except Exception:
            pass

    # Optional: RNG states (helps exact reproducibility)
    try:
        ckpt["rng"] = {
            "torch": torch.random.get_rng_state(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    except Exception:
        pass

    return ckpt


def _load_checkpoint_into(
    ppo_agent,
    obj: Any,
    model_device: torch.device,
    reward_scaler: Optional[RewardScaler],
) -> int:
    """
    Load either:
      - old format: a plain state_dict (weights only)
      - new format: dict with policy/policy_old + opt states
    Returns loaded step if available, else 0.
    """
    # Old format: state_dict directly (dict of tensors)
    if not isinstance(obj, dict) or ("policy_old" not in obj and "policy" not in obj and "format" not in obj):
        ddp.get_module(ppo_agent.policy).load_state_dict(obj)
        ddp.get_module(ppo_agent.policy_old).load_state_dict(obj)
        return 0

    # New-ish format
    if obj.get("policy_old") is not None and hasattr(ppo_agent, "policy_old"):
        ddp.get_module(ppo_agent.policy_old).load_state_dict(obj["policy_old"])
    if obj.get("policy") is not None and hasattr(ppo_agent, "policy"):
        ddp.get_module(ppo_agent.policy).load_state_dict(obj["policy"])

    # Optimizers
    opts = _find_optimizers(ppo_agent)
    for name, opt in opts.items():
        key = f"opt::{name}"
        if key in obj:
            try:
                opt.load_state_dict(obj[key])
                _optimizer_to_device(opt, model_device)
            except Exception:
                pass

    # Reward scaler stats (optional)
    if reward_scaler is not None and isinstance(obj.get("reward_scaler", None), dict):
        rs = obj["reward_scaler"]
        try:
            reward_scaler.t = int(rs.get("t", 0))
            reward_scaler.ret = float(rs.get("ret", 0.0))
            reward_scaler.ret_rms.mean = np.array(rs.get("ret_rms_mean", 0.0), dtype=np.float64)
            reward_scaler.ret_rms.var = np.array(rs.get("ret_rms_var", 1.0), dtype=np.float64)
            reward_scaler.ret_rms.count = float(rs.get("ret_rms_count", 1e-4))
        except Exception:
            pass

    # RNG restore (optional)
    try:
        rng = obj.get("rng", None)
        if isinstance(rng, dict):
            if rng.get("torch", None) is not None:
                torch.random.set_rng_state(rng["torch"])
            if rng.get("numpy", None) is not None:
                np.random.set_state(rng["numpy"])
            if torch.cuda.is_available() and rng.get("cuda", None) is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception:
        pass

    return int(obj.get("step", 0))