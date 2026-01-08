#!/usr/bin/env python3
import os
import glob
import time
import argparse
import re
import signal
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

from reward import RewardConfig, RewardShaper, RewardScaler

import gym
import numpy as np
import torch

# PPO.py should be in the same directory as this train.py
from PPO import PPO

# Optional: robust terminal detection if your env stuffs ML-Agents steps into info
try:
    from mlagents_envs.base_env import TerminalSteps
except Exception:
    TerminalSteps = None

try:
    from mlagents_envs.exception import UnityTimeOutException
except Exception:
    UnityTimeOutException = None

try:
    from mlagents_envs.exception import UnityEnvironmentException
except Exception:
    UnityEnvironmentException = None

from libimmortal.env import ImmortalSufferingEnv

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


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


def ddp_setup():
    if not ddp_is_enabled():
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(ddp_local_rank())
    dist.init_process_group(backend="nccl", init_method="env://")


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


# -------------------------
# Checkpoint helpers
# -------------------------
def _checkpoint_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "checkpoints")


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


def _infer_done(done_from_env, info) -> bool:
    done = bool(done_from_env)
    if TerminalSteps is None:
        return done
    if isinstance(info, dict):
        step_obj = info.get("step", None) or info.get("mlagents_step", None)
        if step_obj is not None and isinstance(step_obj, TerminalSteps):
            return True
    return done


def _get_action_space(env):
    if hasattr(env, "action_space"):
        return env.action_space
    if hasattr(env, "env") and hasattr(env.env, "action_space"):
        return env.env.action_space
    raise AttributeError("Cannot find action_space on env.")


class _SigintGuard:
    """
    During checkpoint save, ignore SIGINT so we don't corrupt / half-save.
    """
    def __enter__(self):
        self._old = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        return self

    def __exit__(self, exc_type, exc, tb):
        signal.signal(signal.SIGINT, self._old)
        return False


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
        pol = get_module(pol) if pol is not None else None
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
        "policy_old": get_module(ppo_agent.policy_old).state_dict() if hasattr(ppo_agent, "policy_old") else None,
        "policy": get_module(ppo_agent.policy).state_dict() if hasattr(ppo_agent, "policy") else None,
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
        get_module(ppo_agent.policy).load_state_dict(obj)
        get_module(ppo_agent.policy_old).load_state_dict(obj)
        return 0

    # New-ish format
    if obj.get("policy_old") is not None and hasattr(ppo_agent, "policy_old"):
        get_module(ppo_agent.policy_old).load_state_dict(obj["policy_old"])
    if obj.get("policy") is not None and hasattr(ppo_agent, "policy"):
        get_module(ppo_agent.policy).load_state_dict(obj["policy"])

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


# -------------------------
# Debug helpers (Player.log)
# -------------------------
def _find_latest_player_log() -> Optional[str]:
    # common default location on Linux
    pats = [
        "/root/.config/unity3d/DefaultCompany/**/Player.log",
        os.path.expanduser("~/.config/unity3d/DefaultCompany/**/Player.log"),
    ]
    cands = []
    for p in pats:
        cands.extend(glob.glob(p, recursive=True))
    if not cands:
        return None
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]


def _tail_file(path: str, n: int = 200) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read()
        lines = data.splitlines()[-n:]
        return "\n".join([ln.decode("utf-8", errors="replace") for ln in lines])
    except Exception as e:
        return f"[tail failed] {repr(e)}"


def _print_player_log_tail(prefix: str, n: int = 200):
    if not is_main_process():
        return
    pl = _find_latest_player_log()
    if pl is None:
        print(f"{prefix} no Player.log found under DefaultCompany/**/Player.log")
        return
    print(f"{prefix} Player.log: {pl}")
    print(f"{prefix} --- tail -n {n} ---")
    print(_tail_file(pl, n=n))
    print(f"{prefix} --- end tail ---")


# -------------------------
# Observation preprocessing
# -------------------------
def _make_map_and_vec(obs) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Return:
      id_map: (H,W) uint8 (IDs)
      vec_obs: (D,) float32
      K: number of IDs in palette (num_ids)
    """
    from libimmortal.utils import colormap_to_ids_and_onehot, parse_observation

    graphic_obs, vector_obs = parse_observation(obs)
    id_map, onehot = colormap_to_ids_and_onehot(graphic_obs)

    id_map = np.asarray(id_map, dtype=np.uint8)
    vec_obs = np.asarray(vector_obs, dtype=np.float32).reshape(-1)

    K = int(onehot.shape[0])
    return id_map, vec_obs, K


def _format_action_for_env(action_np, action_space):
    if isinstance(action_space, gym.spaces.Box):
        a = np.asarray(action_np, dtype=np.float32).reshape(action_space.shape)
        low = getattr(action_space, "low", None)
        high = getattr(action_space, "high", None)
        if low is not None and high is not None:
            a = np.clip(a, low, high)
        return a

    if isinstance(action_space, gym.spaces.Discrete):
        return int(np.asarray(action_np).item())

    if isinstance(action_space, gym.spaces.MultiDiscrete):
        a = np.asarray(action_np, dtype=np.int64).reshape(action_space.shape)
        a = np.minimum(np.maximum(a, 0), action_space.nvec - 1)
        return a

    raise TypeError(f"Unsupported action space: {type(action_space)}")


# -------------------------
# Training
# -------------------------
def train(args):
    print("============================================================================================")

    ddp_setup()
    rank = ddp_rank()
    local_rank = ddp_local_rank()
    world = ddp_world_size()

    if args.seed is not None:
        seed_everything(int(args.seed))

    # Per-rank device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_process():
        print(f"[Device] rank={rank} local_rank={local_rank} device={device}")

    # Unity env control
    port_stride = int(args.port_stride)
    max_env_restarts = int(args.max_env_restarts)
    unity_timeout_wait = int(args.unity_timeout_wait)
    no_graphics = bool(args.no_graphics)

    restart_box = {"id": 0}
    last_port_box = {"port": int(args.port)}

    def _compute_port() -> int:
        return (
            int(args.port)
            + int(local_rank) * port_stride
            + int(restart_box["id"]) * port_stride * int(world)
        )

    def _make_env():
        p = _compute_port()
        last_port_box["port"] = p

        kw = dict(
            game_path=args.game_path,
            port=p,
            time_scale=args.time_scale,
            seed=(int(args.seed) + rank) if args.seed is not None else None,
            width=args.width,
            height=args.height,
            verbose=args.verbose,
        )

        # If ImmortalSufferingEnv supports these kwargs, use them; else fallback.
        try:
            env = ImmortalSufferingEnv(
                **kw,
                no_graphics=no_graphics,
                timeout_wait=unity_timeout_wait,
            )
        except TypeError:
            env = ImmortalSufferingEnv(**kw)

        return env, p

    def _restart_env(reason: str):
        # rank-local restart counter
        restart_box["id"] += 1
        if restart_box["id"] > max_env_restarts:
            raise RuntimeError(f"[Env] exceeded max_env_restarts={max_env_restarts} (last reason={reason})")

        if is_main_process():
            print(f"[Env] restart #{restart_box['id']} reason={reason}")

        try:
            env.close()
        except Exception:
            pass

        # Helpful debug: tail player log when something bad happens
        _print_player_log_tail(prefix=f"[Env][{reason}]", n=int(args.player_log_tail))

        new_env, new_port = _make_env()
        return new_env, new_port

    def _is_timeout_exc(e: Exception) -> bool:
        return (UnityTimeOutException is not None) and isinstance(e, UnityTimeOutException)

    def _is_unity_env_unloaded_exc(e: Exception) -> bool:
        if UnityEnvironmentException is None:
            return False
        if not isinstance(e, UnityEnvironmentException):
            return False
        msg = str(e)
        # Be conservative: restart only for common transient/comm/unloaded cases
        if "No Unity environment is loaded" in msg:
            return True
        if "timed out" in msg.lower():
            return True
        if "communicator" in msg.lower():
            return True
        return False

    def _is_restartable_exc(e: Exception) -> bool:
        # In practice, ML-Agents comm errors can surface like these.
        restartable = (
            _is_timeout_exc(e)
            or _is_unity_env_unloaded_exc(e)
            or isinstance(e, (BrokenPipeError, ConnectionResetError, EOFError))
        )
        return bool(restartable)

    def _safe_reset(cur_env):
        while True:
            try:
                return cur_env.reset()
            except Exception as e:
                if _is_restartable_exc(e):
                    if is_main_process():
                        print(f"[Env][reset] exception={repr(e)} -> restart")
                    new_env, _ = _restart_env(reason="reset")
                    cur_env = new_env
                    continue
                raise

    # Build initial env
    env, port = _make_env()
    env_tag = "ImmortalSufferingEnv"

    # Runner-style: total interaction steps PER RANK
    MAX_STEPS = int(args.max_steps)

    # PPO hyperparameters
    max_ep_len = int(args.max_ep_len)
    update_timestep = int(args.update_timestep)
    K_epochs = int(args.k_epochs)
    eps_clip = float(args.eps_clip)
    gamma = float(args.gamma)
    lr_actor = float(args.lr_actor)
    lr_critic = float(args.lr_critic)
    save_model_freq = int(args.save_model_freq)

    action_std = float(args.action_std)
    action_std_decay_rate = float(args.action_std_decay_rate)
    min_action_std = float(args.min_action_std)
    action_std_decay_freq = int(args.action_std_decay_freq)
    mini_batch_size = int(args.mini_batch_size)

    # ---- step counters ----
    base_step = 0
    step = 0  # for finally

    # Infer dims from first reset (robust reset)
    obs = _safe_reset(env)
    id_map, vec_obs, K = _make_map_and_vec(obs)

    expected_id_shape = tuple(id_map.shape)
    expected_vec_dim = int(vec_obs.shape[0])

    # ----- For Debugging -----
    bfs_n = 0
    bfs_missing = 0
    bfs_sum = 0.0
    bfs_min = 1e9
    bfs_max = -1e9
    dd_sum = 0.0
    dd_n = 0
    closer_n = 0
    farther_n = 0

    action_space = _get_action_space(env)
    has_continuous_action_space = isinstance(action_space, gym.spaces.Box)

    if isinstance(action_space, gym.spaces.Box):
        action_dim = int(np.prod(action_space.shape))
        action_nvec = None
    elif isinstance(action_space, gym.spaces.Discrete):
        action_dim = int(action_space.n)
        action_nvec = None
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        action_nvec = action_space.nvec.astype(int).tolist()
        action_dim = len(action_nvec)
        has_continuous_action_space = False
    else:
        raise TypeError(f"Unsupported action space: {type(action_space)}")

    num_ids = int(K)

    if is_main_process():
        print(f"[DDP] world_size={world}, rank={rank}, local_rank={local_rank}")
        print(f"[Env] base_port={args.port}, this_rank_port={port}, port_stride={port_stride}, no_graphics={no_graphics}")
        print(f"[Obs] num_ids(K)={num_ids}, vec_dim={expected_vec_dim}, id_shape={expected_id_shape}")
        print(f"[Act] action_dim={action_dim}, continuous={has_continuous_action_space}, space={type(action_space)}")

    def _make_map_and_vec_checked(obs_):
        nonlocal expected_vec_dim, expected_id_shape, num_ids
        idm, vec, k = _make_map_and_vec(obs_)

        if tuple(idm.shape) != expected_id_shape:
            if is_main_process():
                print(f"[Obs][WARN] id_map shape changed: got={idm.shape}, expected={expected_id_shape}")
            expected_id_shape = tuple(idm.shape)

        if int(vec.shape[0]) != expected_vec_dim:
            if is_main_process():
                print(f"[Obs][WARN] vec_dim mismatch: got={vec.shape[0]}, expected={expected_vec_dim} (will pad/truncate)")
            if vec.shape[0] > expected_vec_dim:
                vec = vec[:expected_vec_dim]
            else:
                vec = np.pad(vec, (0, expected_vec_dim - vec.shape[0]), mode="constant")

        # K mismatch is rare; keep first K as authoritative
        if int(k) != int(num_ids):
            if is_main_process():
                print(f"[Obs][WARN] K changed: got={k}, expected={num_ids} (keeping expected)")
        return idm, vec, k

    # PPO agent
    ppo_agent = PPO(
        num_ids, expected_vec_dim, action_dim,
        lr_actor, lr_critic,
        gamma, K_epochs, eps_clip,
        has_continuous_action_space,
        action_std_init=action_std,
        mini_batch_size=mini_batch_size,
        action_nvec=action_nvec,
    )

    # Make sure both networks live on the correct per-rank device
    if hasattr(ppo_agent, "policy"):
        get_module(ppo_agent.policy).to(device)
    if hasattr(ppo_agent, "policy_old"):
        get_module(ppo_agent.policy_old).to(device)

    # DDP: wrap ppo_agent.policy (NOT policy_old)
    if hasattr(ppo_agent, "policy") and ddp_is_enabled():
        ddp_pol = ddp_wrap_model(ppo_agent.policy)
        base_pol = get_module(ddp_pol)

        # Make DDP forward run the same graph as evaluate
        if hasattr(base_pol, "evaluate"):
            if not hasattr(base_pol, "_orig_forward"):
                base_pol._orig_forward = base_pol.forward
            base_pol.forward = base_pol.evaluate

            def _ddp_evaluate(mb_map, mb_vec, mb_actions):
                return ddp_pol(mb_map, mb_vec, mb_actions)

            ddp_pol.evaluate = _ddp_evaluate  # type: ignore

        # Patch state_dict/load_state_dict so PPO can use them without "module." prefix trouble
        def _state_dict_no_prefix(*a, **kw):
            return base_pol.state_dict(*a, **kw)

        def _load_state_dict_no_prefix(sd, *a, **kw):
            return base_pol.load_state_dict(sd, *a, **kw)

        ddp_pol.state_dict = _state_dict_no_prefix  # type: ignore
        ddp_pol.load_state_dict = _load_state_dict_no_prefix  # type: ignore
        ppo_agent.policy = ddp_pol

    # Reward shaping
    cfg = RewardConfig(
        w_progress=args.w_progress,
        w_time=args.w_time,
        w_damage=args.w_damage,
        w_not_actionable=args.w_not_actionable,
        terminal_bonus=args.terminal_bonus,
        clip=args.reward_clip,
        success_if_raw_reward_ge=args.success_if_raw_reward_ge,
        time_limit=args.time_limit,
        success_speed_bonus=args.success_speed_bonus,
        use_bfs_progress=True,
        w_acceleration=1.0,
        bfs_update_every=10,
    )
    shaper = RewardShaper(cfg)
    shaper.reset(vec_obs, id_map)

    # IMPORTANT: reward_scaler is ALWAYS defined to avoid NameError in logs/config.
    reward_scaler = RewardScaler(
        gamma=gamma,
        clip=5.0,
        eps=1e-8,
        min_std=0.1,
        warmup_steps=10000,
    )

    # Checkpoints (only rank0 writes)
    ckpt_dir = _checkpoint_dir()
    if is_main_process():
        os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_prefix = f"PPO_{env_tag}_seed{int(args.seed) if args.seed is not None else 0}_"
    if is_main_process():
        print("checkpoint dir:", ckpt_dir)

    # Determine device from policy_old params (after moving to device)
    model_device = next(get_module(ppo_agent.policy_old).parameters()).device

    # Resume
    if args.resume:
        ckpt_to_load = args.checkpoint
        if ckpt_to_load is None and is_main_process():
            ckpt_to_load = _latest_checkpoint_path(ckpt_dir, prefix=ckpt_prefix)

        if ddp_is_enabled():
            obj_list = [ckpt_to_load]
            dist.broadcast_object_list(obj_list, src=0)
            ckpt_to_load = obj_list[0]

        if ckpt_to_load is None:
            raise FileNotFoundError(f"--resume set but no checkpoint found under {ckpt_dir} with prefix {ckpt_prefix}")

        inferred_step = _infer_step_from_ckpt_path(str(ckpt_to_load), ckpt_prefix)

        if is_main_process():
            print(f"[Resume] loading: {ckpt_to_load}")
            print(f"[Resume] inferred base_step(from filename)={inferred_step}")

        obj = torch.load(ckpt_to_load, map_location="cpu")
        loaded_step = _load_checkpoint_into(
            ppo_agent,
            obj,
            model_device=model_device,
            reward_scaler=reward_scaler,
        )

        base_step = loaded_step if loaded_step > 0 else inferred_step
        step = base_step

        if is_main_process():
            print(f"[Resume] base_step(used)={base_step} (loaded_step={loaded_step})")

        ddp_barrier()

    # Optional wandb (rank0 only)
    wandb = None
    if args.wandb and is_main_process():
        import wandb as _wandb
        wandb = _wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"{env_tag}-ddp{world}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=dict(
                env_tag=env_tag,
                max_steps=MAX_STEPS,
                max_ep_len=max_ep_len,
                update_timestep=update_timestep,
                K_epochs=K_epochs,
                eps_clip=eps_clip,
                gamma=gamma,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                action_std=action_std,
                action_std_decay_rate=action_std_decay_rate,
                min_action_std=min_action_std,
                action_std_decay_freq=action_std_decay_freq,
                save_model_freq=save_model_freq,
                mini_batch_size=mini_batch_size,
                seed=args.seed,
                ddp_world_size=world,
                base_step=base_step,
                w_progress=float(args.w_progress),
                w_time=float(args.w_time),
                w_damage=float(args.w_damage),
                w_not_actionable=float(args.w_not_actionable),
                terminal_bonus=float(args.terminal_bonus),
                reward_clip=float(args.reward_clip),
                scaler_clip=float(getattr(reward_scaler, "clip", 0.0)),
                w_acc=float(getattr(cfg, "w_acceleration", 0.0)),
                no_graphics=bool(no_graphics),
                unity_timeout_wait=int(unity_timeout_wait),
                max_env_restarts=int(max_env_restarts),
                port_stride=int(port_stride),
                reward_scaling=bool(args.reward_scaling),
            ),
        )

    # Runner-style main loop
    start_time = datetime.now().replace(microsecond=0)
    if is_main_process():
        print("Started training at:", start_time)
        print("============================================================================================")

    episode_reward = 0.0
    episode_raw_reward = 0.0
    episode_len = 0
    episode_idx = 0

    cur_id_map = id_map
    cur_vec_obs = vec_obs

    def _select_action(cur_map_np: np.ndarray, cur_vec_np: np.ndarray):
        map_t = torch.from_numpy(cur_map_np).to(model_device, dtype=torch.long).unsqueeze(0)
        vec_t = torch.from_numpy(cur_vec_np).to(model_device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_t, logp_t, value_t = get_module(ppo_agent.policy_old).act(map_t, vec_t)

        a_cpu = action_t.squeeze(0).detach().cpu()
        lp_cpu = logp_t.squeeze(0).detach().cpu()
        v_cpu = value_t.squeeze(0).detach().cpu()
        action_np = a_cpu.numpy()
        return a_cpu, lp_cpu, v_cpu, action_np

    # PPO update debug counters
    gae_lambda = 0.95

    upd_steps = 0
    upd_terminals = 0
    upd_goal_hits = 0
    upd_shaper_clip_hits = 0
    upd_scaler_clip_hits = 0

    def _reset_update_counters():
        nonlocal upd_steps, upd_terminals, upd_goal_hits, upd_shaper_clip_hits, upd_scaler_clip_hits
        upd_steps = 0
        upd_terminals = 0
        upd_goal_hits = 0
        upd_shaper_clip_hits = 0
        upd_scaler_clip_hits = 0

    def _update_counters(raw_reward_f: float, done_for_buffer: bool, shaped_reward_f: float, scaled_reward_f: float):
        nonlocal upd_steps, upd_terminals, upd_goal_hits, upd_shaper_clip_hits, upd_scaler_clip_hits

        upd_steps += 1
        if done_for_buffer:
            upd_terminals += 1
        if raw_reward_f >= 1.0:
            upd_goal_hits += 1

        if float(getattr(cfg, "clip", 0.0) or 0.0) > 0.0:
            c = float(cfg.clip)
            if abs(shaped_reward_f) >= (c - 1e-6):
                upd_shaper_clip_hits += 1

        if args.reward_scaling and float(getattr(reward_scaler, "clip", 0.0) or 0.0) > 0.0:
            c2 = float(reward_scaler.clip)
            if abs(scaled_reward_f) >= (c2 - 1e-6):
                upd_scaler_clip_hits += 1

    def _print_update_pre(step_i: int, buf_len_pre: int):
        if not is_main_process():
            return

        term_n = int(sum(1 for t in ppo_agent.buffer.is_terminals if t))
        term_frac = (term_n / buf_len_pre) if buf_len_pre > 0 else 0.0

        rw = np.asarray(ppo_agent.buffer.rewards, dtype=np.float32) if buf_len_pre > 0 else np.zeros((0,), dtype=np.float32)
        rw_mean = float(rw.mean()) if buf_len_pre > 0 else 0.0
        rw_std = float(rw.std()) if buf_len_pre > 0 else 0.0
        rw_min = float(rw.min()) if buf_len_pre > 0 else 0.0
        rw_max = float(rw.max()) if buf_len_pre > 0 else 0.0

        shaper_clip_frac = (upd_shaper_clip_hits / upd_steps) if upd_steps > 0 else 0.0
        scaler_clip_frac = (upd_scaler_clip_hits / upd_steps) if upd_steps > 0 else 0.0
        goal_hit_frac = (upd_goal_hits / upd_steps) if upd_steps > 0 else 0.0

        bfs_mean = (bfs_sum / bfs_n) if bfs_n > 0 else None
        dd_mean = (dd_sum / dd_n) if dd_n > 0 else None
        closer_pct = (100.0 * closer_n / dd_n) if dd_n > 0 else 0.0
        farther_pct = (100.0 * farther_n / dd_n) if dd_n > 0 else 0.0
        bfs_miss_pct = (100.0 * bfs_missing / max(1, (bfs_n + bfs_missing)))

        extra = ""
        if bfs_mean is not None:
            extra += f" bfs(mean/min/max)={bfs_mean:.4f}/{bfs_min:.4f}/{bfs_max:.4f}"
        extra += f" bfs_delta_mean={0.0 if dd_mean is None else dd_mean:.5f}"
        extra += f" closer/farther={closer_pct:.1f}%/{farther_pct:.1f}%"
        extra += f" bfs_missing={bfs_miss_pct:.1f}%"

        print(
            f"[PPO][begin] step={step_i} "
            f"T={buf_len_pre} terminals={term_n} ({term_frac:.2%}) "
            f"reward(mean/std/min/max)={rw_mean:+.3f}/{rw_std:.3f}/{rw_min:+.3f}/{rw_max:+.3f} "
            f"goal_hits={upd_goal_hits} ({goal_hit_frac:.2%}) "
            f"shaper_clip_hits={upd_shaper_clip_hits} ({shaper_clip_frac:.2%}) "
            f"scaler_clip_hits={upd_scaler_clip_hits} ({scaler_clip_frac:.2%})\n"
            f"| eps_clip={eps_clip:g} K_epochs={K_epochs} mb={mini_batch_size} "
            f"lr(a/c)={lr_actor:g}/{lr_critic:g} gamma={gamma:g} gae_lambda={gae_lambda:g}\n"
            f"| w_progress={float(args.w_progress):g} w_time={float(args.w_time):g} "
            f"w_damage={float(args.w_damage):g} w_not_actionable={float(args.w_not_actionable):g} "
            f"w_acc={float(getattr(cfg,'w_acceleration',0.0)):g} "
            f"reward_clip={float(getattr(cfg,'clip',0.0)):g} scaler_clip={float(getattr(reward_scaler,'clip',0.0)):g}\n"
            f"| reward_scaling={int(args.reward_scaling)}\n"
            f"|{extra}"
        )

    def _print_update_post(step_i: int, dt_s: float):
        if is_main_process():
            print(f"[PPO][end]   step={step_i} update_seconds={dt_s:.3f}")
        _reset_update_counters()

    timeouts = 0
    restarts = 0

    try:
        for local_step in range(1, MAX_STEPS + 1):
            step = base_step + local_step

            # Retry loop: restart env on this rank and retry this step if step/reset fails.
            while True:
                a_cpu, lp_cpu, v_cpu, action_np = _select_action(cur_id_map, cur_vec_obs)
                action_env = _format_action_for_env(action_np, action_space)

                try:
                    obs, raw_reward, done, info = env.step(action_env)

                    # Only after step succeeds, append to buffer
                    ppo_agent.buffer.add_state(cur_id_map, cur_vec_obs)
                    ppo_agent.buffer.actions.append(a_cpu)
                    ppo_agent.buffer.logprobs.append(lp_cpu)
                    ppo_agent.buffer.state_values.append(v_cpu)
                    break

                except Exception as e:
                    if _is_restartable_exc(e):
                        timeouts += int(_is_timeout_exc(e))
                        restarts += 1
                        if is_main_process():
                            print(f"[Env][step] exception={repr(e)} -> restart (timeouts={timeouts}, restarts={restarts})")
                        env, port = _restart_env(reason="step")
                        time.sleep(0.5)
                        obs = _safe_reset(env)
                        cur_id_map, cur_vec_obs, _ = _make_map_and_vec_checked(obs)
                        shaper.reset(cur_vec_obs, cur_id_map)
                        continue
                    raise

            done = _infer_done(done, info)
            done_env = _infer_done(done, info)
            next_id_map, next_vec_obs, _ = _make_map_and_vec_checked(obs)

            if info is None:
                info = {}
            elif not isinstance(info, dict):
                info = {"env_info": info}

            shaped_reward = float(shaper(raw_reward, next_vec_obs, next_id_map, done_env, info))   
            is_goal = (float(raw_reward) >= float(args.success_if_raw_reward_ge))
            done_for_buffer = (bool(done_env) or bool(shaper.virtual_done) or bool(is_goal))

            # Debug: BFS progress stats (cheap, no extra BFS compute)
            if getattr(shaper, "dbg_has_bfs", False) and (shaper.dbg_bfs_dist is not None):
                d = float(shaper.dbg_bfs_dist)
                bfs_n += 1
                bfs_sum += d
                bfs_min = min(bfs_min, d)
                bfs_max = max(bfs_max, d)
            else:
                bfs_missing += 1

            if (shaper.dbg_bfs_delta is not None):
                dd = float(shaper.dbg_bfs_delta)
                dd_sum += dd
                dd_n += 1
                if dd > 0.0:
                    closer_n += 1
                elif dd < 0.0:
                    farther_n += 1

            if args.reward_scaling:
                scaled_reward = float(reward_scaler(shaped_reward, done_for_buffer))
            else:
                scaled_reward = float(shaped_reward)

            done = done_for_buffer

            ppo_agent.buffer.rewards.append(float(scaled_reward))
            ppo_agent.buffer.is_terminals.append(bool(done_for_buffer))

            _update_counters(float(raw_reward), bool(done_for_buffer), float(shaped_reward), float(scaled_reward))

            episode_reward += float(scaled_reward)
            episode_raw_reward += float(raw_reward)
            episode_len += 1

            if is_goal and is_main_process():
                print(
                    f"[GOAL] step={step} raw={float(raw_reward):.10f} shaped={shaped_reward:.3f} scaled={scaled_reward:.3f} "
                    f"done_env={int(done_env)} done_buf={int(done_for_buffer)} virtual_done={int(bool(shaper.virtual_done))} "
                    f"ep_len={episode_len} ep_reward_running={episode_reward:.3f} "
                    f"clip={getattr(cfg,'clip',None)} w_time={args.w_time} w_prog={args.w_progress} termB={args.terminal_bonus}"
                )

            # PPO update (ALL ranks call update; DDP will sync gradients)
            if step % update_timestep == 0:
                ddp_barrier()
                t0 = time.time()

                # last_value for bootstrapping when rollout ends non-terminal
                with torch.no_grad():
                    if done_for_buffer:
                        ppo_agent.buffer.last_value = 0.0
                    else:
                        map_t = torch.from_numpy(next_id_map).to(model_device, dtype=torch.long).unsqueeze(0)
                        vec_t = torch.from_numpy(next_vec_obs).to(model_device, dtype=torch.float32).unsqueeze(0)
                        po = get_module(ppo_agent.policy_old)
                        feat = po.encode(map_t, vec_t)
                        v = po.critic_head(feat)
                        ppo_agent.buffer.last_value = float(v.item())

                if args.dump_id_map and is_main_process():
                    np.savetxt(args.dump_id_map, cur_id_map, delimiter=",", fmt="%d")

                buf_len_pre = int(len(ppo_agent.buffer.rewards))
                _print_update_pre(int(step), buf_len_pre)

                ppo_agent.update(gae_lambda)

                dt = time.time() - t0
                _print_update_post(int(step), float(dt))

                if wandb is not None:
                    wandb.log(
                        {
                            "train/updated": 1,
                            "train/update_seconds": float(dt),
                            "train/buffer_len": int(buf_len_pre),
                            "train/step": int(step),
                            "train/global_env_steps": int(step * world),
                            "train/update_goal_hits": int(upd_goal_hits),
                            "train/update_terminals": int(upd_terminals),
                            "train/update_shaper_clip_hits": int(upd_shaper_clip_hits),
                            "train/update_scaler_clip_hits": int(upd_scaler_clip_hits),
                            "env/timeouts": int(timeouts),
                            "env/restarts": int(restarts),
                            "env/port": int(last_port_box["port"]),
                        },
                        step=int(step),
                    )

                ddp_barrier()

            # Action std decay
            if has_continuous_action_space and (step % action_std_decay_freq == 0):
                if hasattr(ppo_agent, "decay_action_std"):
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
                    if wandb is not None:
                        wandb.log({"train/action_std": float(getattr(ppo_agent, "action_std", np.nan))}, step=int(step))

            # Save checkpoint (rank0 only) - atomic
            if (step % save_model_freq == 0) and is_main_process():
                ckpt_path = os.path.join(ckpt_dir, f"{ckpt_prefix}{step}.pth")
                print("--------------------------------------------------------------------------------------------")
                print("saving model at:", ckpt_path)
                with _SigintGuard():
                    atomic_torch_save(lambda: _make_checkpoint(ppo_agent, step, args, cfg, reward_scaler), ckpt_path)
                print("model saved")
                print("Elapsed Time:", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
                if wandb is not None:
                    wandb.log({"train/checkpoint_saved": 1}, step=int(step))
                    wandb.save(ckpt_path)

            # Lightweight wandb logs (rank0 only)
            if wandb is not None and ((step % args.wandb_log_freq == 0) or done_for_buffer or is_goal):
                wandb.log(
                    {
                        "train/reward": float(scaled_reward),
                        "train/raw_reward": float(raw_reward),
                        "train/done": int(done_for_buffer),
                        "train/done_env": int(done_env),
                        "train/is_goal": int(is_goal),
                        "train/virtual_done": int(bool(shaper.virtual_done)),
                        "train/episode_reward_running": float(episode_reward),
                        "train/episode_raw_reward_running": float(episode_raw_reward),
                        "train/episode_len_running": int(episode_len),
                        "env/timeouts": int(timeouts),
                        "env/restarts": int(restarts),
                        "env/port": int(last_port_box["port"]),
                    },
                    step=int(step),
                )

            # Episode boundary
            if done_for_buffer or (episode_len >= max_ep_len):
                if wandb is not None:
                    wandb.log(
                        {
                            "episode/reward": float(episode_reward),
                            "episode/raw_reward": float(episode_raw_reward),
                            "episode/len": int(episode_len),
                            "episode/index": int(episode_idx),
                        },
                        step=int(step),
                    )

                obs = _safe_reset(env)
                cur_id_map, cur_vec_obs, _ = _make_map_and_vec_checked(obs)
                shaper.reset(cur_vec_obs, cur_id_map)

                episode_reward = 0.0
                episode_raw_reward = 0.0
                episode_len = 0
                episode_idx += 1
            else:
                cur_id_map, cur_vec_obs = next_id_map, next_vec_obs

    except KeyboardInterrupt:
        if is_main_process():
            print("\n[Interrupted] saving checkpoint before exit...")

    except Exception as e:
        if is_main_process():
            print(f"\n[FATAL] exception={repr(e)}")
        _print_player_log_tail(prefix="[FATAL]", n=int(args.player_log_tail))
        raise

    finally:
        # Final save (rank0 only) - atomic + guard
        if is_main_process():
            with _SigintGuard():
                final_step = int(step) if int(step) > 0 else int(base_step)
                final_path = os.path.join(ckpt_dir, f"{ckpt_prefix}{final_step}.pth")
                atomic_torch_save(lambda: _make_checkpoint(ppo_agent, final_step, args, cfg, reward_scaler), final_path)
                print("Final model saved at:", final_path)

        # Don't hard-barrier in finally (can hang if ranks died)
        try:
            ddp_barrier()
        except Exception as e:
            if is_main_process():
                print(f"[DDP] barrier skipped in finally due to: {repr(e)}")

        try:
            env.close()
        except Exception:
            pass

        if wandb is not None:
            wandb.finish()

        end_time = datetime.now().replace(microsecond=0)
        if is_main_process():
            print("============================================================================================")
            print("Started training at:", start_time)
            print("Finished training at:", end_time)
            print("Total training time:", end_time - start_time)
            print(f"[Env][FINAL] timeouts={timeouts} restarts={restarts} last_port={last_port_box['port']}")
            print("============================================================================================")

        ddp_cleanup()


def build_argparser():
    p = argparse.ArgumentParser()

    # Immortal env args
    p.add_argument("--game_path", type=str, default=r"/root/immortal_suffering/immortal_suffering_linux_build.x86_64")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--port_stride", type=int, default=50)
    p.add_argument("--time_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--verbose", action="store_true")

    # Graphics flags: default is headless (no_graphics=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--no_graphics", dest="no_graphics", action="store_true", default=True)
    g.add_argument("--graphics", dest="no_graphics", action="store_false")

    p.add_argument("--unity_timeout_wait", type=int, default=120)
    p.add_argument("--max_env_restarts", type=int, default=20)

    # Debug
    p.add_argument("--player_log_tail", type=int, default=200)

    # Runner steps (PER RANK)
    p.add_argument("--max_steps", type=int, default=1000000)

    # PPO hyperparams
    p.add_argument("--max_ep_len", type=int, default=1000)
    p.add_argument("--update_timestep", type=int, default=4000)
    p.add_argument("--k_epochs", type=int, default=10)
    p.add_argument("--eps_clip", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr_actor", type=float, default=1e-4)
    p.add_argument("--lr_critic", type=float, default=2e-4)
    p.add_argument("--mini_batch_size", type=int, default=64)

    # Action std
    p.add_argument("--action_std", type=float, default=0.6)
    p.add_argument("--action_std_decay_rate", type=float, default=0.05)
    p.add_argument("--min_action_std", type=float, default=0.1)
    p.add_argument("--action_std_decay_freq", type=int, default=250000)

    # Saving (rank0 only)
    p.add_argument("--save_model_freq", type=int, default=35000)

    # Resume
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)

    # Reward shaping knobs
    p.add_argument("--w_progress", type=float, default=100.0)
    p.add_argument("--w_time", type=float, default=0.01)
    p.add_argument("--w_damage", type=float, default=0.1)
    p.add_argument("--w_not_actionable", type=float, default=0.01)
    p.add_argument("--terminal_bonus", type=float, default=4.0)
    p.add_argument("--reward_clip", type=float, default=20.0)
    p.add_argument("--success_if_raw_reward_ge", type=float, default=1.0)
    p.add_argument("--time_limit", type=float, default=300.0)
    p.add_argument("--success_speed_bonus", type=float, default=0.5)

    # Debug: dump id_map path (rank0 only, on update steps)
    p.add_argument("--dump_id_map", type=str, default=None)

    # Reward scaling toggle (default OFF)
    p.add_argument("--reward_scaling", action="store_true", default=False)

    # wandb (rank0 only)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="ppo-immortal")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_log_freq", type=int, default=200)

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)

"""
How to run

torchrun --standalone --nproc_per_node=4 ./src/libimmortal/samples/PPO/train.py \
  --port 5005 --port_stride 50 \
  --save_model_freq 20000 --wandb \
  --resume --checkpoint /root/libimmortal/src/libimmortal/samples/PPO/checkpoints/PPO_ImmortalSufferingEnv_seed42_116378.pth

If you want reward scaling:
  --reward_scaling

If you want graphics (NOT recommended on headless):
  --graphics
"""
