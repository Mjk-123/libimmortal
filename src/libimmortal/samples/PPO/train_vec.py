#!/usr/bin/env python3

"""
train_vec.py
- DDP (torchrun) + per-rank vectorized rollout via multiprocessing workers (each owns one Unity)
- PPOVec update on GPU (one GPU per rank)
- Robust env lifecycle: no env.reset() in main; each episode reset = close+recreate in worker
- Terminal condition: done OR raw_reward >= success_if_raw_reward_ge (goal)
- ppo.log logging format matches your previous train.py style
- wandb logging (rank0 only)

Run example:
torchrun --standalone --nproc_per_node=4 ./src/libimmortal/samples/PPO/train_vec.py \
  --port 5005 --port_stride 400 --envs_per_rank 2 --env_port_stride 50 \
  --update_timestep 4000 --max_steps 2000000 --save_model_freq 40000 --wandb \
  --seed 42 \
  --resume --checkpoint /root/libimmortal/src/libimmortal/samples/PPO/checkpoints/PPO_ImmortalSufferingEnv_seed42_950000.pth
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import random
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

# NOTE: your project imports (match your existing train.py layout)
from libimmortal.samples.PPO.reward import RewardConfig, RewardShaper, RewardScaler

import gym

import libimmortal.samples.PPO.utils.ddp as ddp
import libimmortal.samples.PPO.utils.save as save
import libimmortal.samples.PPO.utils.utilities as utilities
from libimmortal.samples.PPO.utils.ppolog import PPOFileLogger
from libimmortal.samples.PPO.utils.signaling import GracefulStop

from libimmortal.env import ImmortalSufferingEnv

# PPOVec / ActorCritic / RolloutBufferVec are expected to be in your project
from libimmortal.samples.PPO.ppo_vec import PPOVec, RolloutBufferVec, RolloutBatch, ActorCritic

# Optional ML-Agents exceptions (for nicer retry)
try:
    from mlagents_envs.exception import UnityWorkerInUseException
except Exception:
    UnityWorkerInUseException = None

try:
    from mlagents_envs.exception import UnityTimeOutException
except Exception:
    UnityTimeOutException = None

try:
    from mlagents_envs.exception import UnityEnvironmentException
except Exception:
    UnityEnvironmentException = None


# -------------------------
# Small helpers
# -------------------------

def _reward_scalar(r) -> float:
    try:
        if r is None:
            return 0.0
        if torch.is_tensor(r):
            rr = r.detach().view(-1)
            return float(rr[0].item()) if rr.numel() > 0 else 0.0
        if isinstance(r, np.ndarray):
            rr = r.reshape(-1)
            return float(rr[0]) if rr.size > 0 else 0.0
        if isinstance(r, (list, tuple)):
            return float(r[0]) if len(r) > 0 else 0.0
        return float(r)
    except Exception:
        return 0.0


def _get_action_space(env):
    if hasattr(env, "action_space"):
        return env.action_space
    if hasattr(env, "env") and hasattr(env.env, "action_space"):
        return env.env.action_space
    raise AttributeError("Cannot find action_space on env.")


def _noop_action_np(space: gym.Space) -> np.ndarray:
    # English comments for copy/paste friendliness.
    if isinstance(space, gym.spaces.Box):
        return np.zeros(space.shape, dtype=np.float32)
    if isinstance(space, gym.spaces.Discrete):
        return np.array(0, dtype=np.int64)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return np.zeros((len(space.nvec),), dtype=np.int64)
    raise TypeError(f"Unsupported action space: {type(space)}")


def _bootstrap_obs(_env) -> np.ndarray:
    # Do NOT call env.reset() here if reset is flaky; use one safe step to get an obs.
    space = _get_action_space(_env)
    a_np = _noop_action_np(space)
    a_env = utilities._format_action_for_env(a_np, space)
    obs_, _r, _d, _info = _env.step(a_env)
    return obs_


@dataclass
class ActionInfo:
    space_type: str
    action_dim: int
    action_nvec: Optional[List[int]]
    has_continuous_action_space: bool


def _infer_action_info(space: gym.Space) -> ActionInfo:
    if isinstance(space, gym.spaces.Box):
        return ActionInfo("box", int(np.prod(space.shape)), None, True)
    if isinstance(space, gym.spaces.Discrete):
        return ActionInfo("discrete", int(space.n), None, False)
    if isinstance(space, gym.spaces.MultiDiscrete):
        nvec = space.nvec.astype(int).tolist()
        return ActionInfo("multidiscrete", len(nvec), nvec, False)
    raise TypeError(f"Unsupported action space: {type(space)}")


def _set_global_rng(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# -------------------------
# Worker process
# -------------------------

def _worker_loop(conn, cfg: Dict[str, Any]):
    """
    Worker owns one Unity instance.

    Commands:
      - ("reset", {"seed": int, "bump_port": bool})
      - ("step",  {"action": np.ndarray})
      - ("close", {})

    Replies:
      - ("reset_ok", (id_map, vec_obs, K, env_info_dict))
      - ("step_ok",  (id_map, vec_obs, reward, done, info))
      - ("err", (where, repr(e), traceback_str))
    """
    env = None
    restart_id = 0
    env_info = None
    expected_id_shape = None
    expected_vec_dim = None
    num_ids = None

    def _close_env():
        nonlocal env
        if env is None:
            return
        try:
            env.close()
        except Exception:
            pass
        env = None

    def _make_env(seed: int):
        nonlocal env, env_info, restart_id, expected_id_shape, expected_vec_dim, num_ids

        # Port layout:
        # base_port + local_rank*port_stride + env_idx*env_port_stride + restart_id*port_stride*world
        port = (
            int(cfg["port"])
            + int(cfg["local_rank"]) * int(cfg["port_stride"])
            + int(cfg["env_idx"]) * int(cfg["env_port_stride"])
            + int(restart_id) * int(cfg["port_stride"]) * int(cfg["world_size"])
        )

        kw = dict(
            game_path=cfg["game_path"],
            port=int(port),
            time_scale=float(cfg["time_scale"]),
            seed=int(seed),
            width=int(cfg["width"]),
            height=int(cfg["height"]),
            verbose=bool(cfg["verbose"]),
        )

        # Retry loop on worker-in-use/timeout
        while True:
            try:
                try:
                    env = ImmortalSufferingEnv(
                        **kw,
                        no_graphics=bool(cfg.get("no_graphics", False)),
                        timeout_wait=int(cfg.get("unity_timeout_wait", 120)),
                    )
                except TypeError:
                    env = ImmortalSufferingEnv(**kw)

                obs0 = _bootstrap_obs(env)
                id_map, vec_obs, K = utilities._make_map_and_vec(obs0)

                if id_map is not None:
                    id_map = np.asarray(id_map, dtype=np.int16)
                    expected_id_shape = tuple(id_map.shape)
                if vec_obs is not None:
                    vec_obs = np.asarray(vec_obs, dtype=np.float32).reshape(-1)
                    expected_vec_dim = int(vec_obs.shape[0])

                num_ids = int(K) if K is not None else num_ids

                action_space = _get_action_space(env)
                env_info = _infer_action_info(action_space)
                return env, id_map, vec_obs, int(num_ids), env_info

            except Exception as e:
                # If port is in use or Unity is mid-shutdown, bump restart_id and try a new port.
                is_in_use = (UnityWorkerInUseException is not None) and isinstance(e, UnityWorkerInUseException)
                is_timeout = (UnityTimeOutException is not None) and isinstance(e, UnityTimeOutException)
                is_envexc = (UnityEnvironmentException is not None) and isinstance(e, UnityEnvironmentException)
                if is_in_use or is_timeout or is_envexc or ("worker number" in repr(e).lower()):
                    try:
                        _close_env()
                    except Exception:
                        pass
                    restart_id += 1
                    time.sleep(float(cfg.get("restart_sleep_s", 0.2)))
                    # Recompute port by continuing loop
                    port = (
                        int(cfg["port"])
                        + int(cfg["local_rank"]) * int(cfg["port_stride"])
                        + int(cfg["env_idx"]) * int(cfg["env_port_stride"])
                        + int(restart_id) * int(cfg["port_stride"]) * int(cfg["world_size"])
                    )
                    kw["port"] = int(port)
                    continue
                raise

    def _fix_obs(id_map, vec_obs):
        nonlocal expected_id_shape, expected_vec_dim

        # Stabilize vec
        if vec_obs is None:
            if expected_vec_dim is None:
                vec_obs = np.zeros((1,), dtype=np.float32)
                expected_vec_dim = 1
            else:
                vec_obs = np.zeros((expected_vec_dim,), dtype=np.float32)
        else:
            vec_obs = np.asarray(vec_obs, dtype=np.float32).reshape(-1)
            if expected_vec_dim is not None and vec_obs.shape[0] != expected_vec_dim:
                if vec_obs.shape[0] > expected_vec_dim:
                    vec_obs = vec_obs[:expected_vec_dim]
                else:
                    vec_obs = np.pad(vec_obs, (0, expected_vec_dim - vec_obs.shape[0]), mode="constant")

        # Stabilize map
        if id_map is None:
            if expected_id_shape is None:
                id_map = np.zeros((90, 160), dtype=np.int16)
                expected_id_shape = tuple(id_map.shape)
            else:
                id_map = np.zeros(expected_id_shape, dtype=np.int16)
        else:
            id_map = np.asarray(id_map, dtype=np.int16)
            if expected_id_shape is not None and tuple(id_map.shape) != expected_id_shape:
                H, W = expected_id_shape
                out = np.zeros((H, W), dtype=np.int16)
                hh = min(H, id_map.shape[0])
                ww = min(W, id_map.shape[1])
                out[:hh, :ww] = id_map[:hh, :ww]
                id_map = out

        return id_map, vec_obs

    try:
        while True:
            cmd, payload = conn.recv()

            if cmd == "close":
                _close_env()
                break

            if cmd == "reset":
                seed = int(payload.get("seed", 0))
                bump_port = bool(payload.get("bump_port", True))

                _close_env()
                if bump_port:
                    restart_id += 1

                env, id_map, vec_obs, K, env_info = _make_env(seed)
                id_map, vec_obs = _fix_obs(id_map, vec_obs)

                conn.send(("reset_ok", (id_map, vec_obs, int(K), env_info.__dict__)))
                continue

            if cmd == "step":
                if env is None:
                    raise RuntimeError("env is None, call reset first")

                action_np = payload["action"]
                action_space = _get_action_space(env)
                action_env = utilities._format_action_for_env(action_np, action_space)

                obs, reward, done, info = env.step(action_env)
                id_map, vec_obs, _K = utilities._make_map_and_vec(obs)
                id_map, vec_obs = _fix_obs(id_map, vec_obs)

                conn.send(("step_ok", (id_map, vec_obs, reward, bool(done), info)))
                continue

            raise ValueError(f"Unknown cmd: {cmd}")

    except Exception as e:
        conn.send(("err", ("worker_loop", repr(e), traceback.format_exc())))
        try:
            _close_env()
        except Exception:
            pass


# -------------------------
# Checkpoint helpers (for PPOVec)
# -------------------------

def _make_checkpoint_vec(ppo: PPOVec, step: int, args, cfg: RewardConfig, reward_scaler: Optional[RewardScaler]):
    # Keep it simple & robust: save state_dicts.
    pol = ppo.policy.module if hasattr(ppo.policy, "module") else ppo.policy
    obj = {
        "step": int(step),
        "args": vars(args),
        "reward_cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else dict(cfg=repr(cfg)),
        "policy": pol.state_dict(),
        "policy_old": ppo.policy_old.state_dict(),
        "optimizer": ppo.optimizer.state_dict() if hasattr(ppo, "optimizer") else None,
        "reward_scaler": getattr(reward_scaler, "__dict__", None) if reward_scaler is not None else None,
    }
    return obj


def _load_checkpoint_vec(ppo: PPOVec, ckpt_obj: dict, device: torch.device) -> int:
    pol = ppo.policy.module if hasattr(ppo.policy, "module") else ppo.policy

    sd = ckpt_obj.get("policy", None)
    if sd is None:
        raise KeyError("checkpoint missing 'policy'")

    msg = pol.load_state_dict(sd, strict=False)
    print(f"[CKPT] policy missing: {len(msg.missing_keys)} unexpected: {len(msg.unexpected_keys)}", flush=True)
    if len(msg.missing_keys) or len(msg.unexpected_keys):
        print("  missing sample:", msg.missing_keys[:8], flush=True)
        print("  unexpected sample:", msg.unexpected_keys[:8], flush=True)

    sd_old = ckpt_obj.get("policy_old", None)
    if sd_old is not None:
        ppo.policy_old.load_state_dict(sd_old, strict=False)

    opt = ckpt_obj.get("optimizer", None)
    if (opt is not None) and hasattr(ppo, "optimizer"):
        try:
            ppo.optimizer.load_state_dict(opt)
        except Exception as e:
            print("[CKPT][WARN] optimizer load failed:", repr(e), flush=True)

    return int(ckpt_obj.get("step", 0) or 0)


# -------------------------
# Training
# -------------------------

def train(args):
    stopper = GracefulStop()
    stopper.install()

    ddp.ddp_setup()
    rank = ddp.ddp_rank()
    local_rank = ddp.ddp_local_rank()
    world = ddp.ddp_world_size()

    if args.seed is not None:
        ddp.seed_everything(int(args.seed))

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if ddp.is_main_process():
        print(f"[Device] rank={rank} local_rank={local_rank} device={device}", flush=True)

    # (Important) Workers should not inherit CUDA context -> use spawn if possible.
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    import multiprocessing as mp

    # -------------------------
    # Spawn env workers (N per rank)
    # -------------------------
    N = int(args.envs_per_rank)
    assert N >= 1

    workers: List[mp.Process] = []
    conns = []

    base_cfg = dict(
        game_path=args.game_path,
        port=int(args.port),
        port_stride=int(args.port_stride),
        env_port_stride=int(args.env_port_stride),
        env_idx=0,  # filled per worker
        local_rank=int(local_rank),
        world_size=int(world),
        time_scale=float(args.time_scale),
        width=int(args.width),
        height=int(args.height),
        verbose=bool(args.verbose),
        no_graphics=bool(args.no_graphics),
        unity_timeout_wait=int(args.unity_timeout_wait),
        restart_sleep_s=float(args.restart_sleep_s),
    )

    for env_idx in range(N):
        parent_conn, child_conn = mp.Pipe()
        cfg = dict(base_cfg)
        cfg["env_idx"] = int(env_idx)

        p = mp.Process(target=_worker_loop, args=(child_conn, cfg), daemon=True)
        p.start()

        workers.append(p)
        conns.append(parent_conn)

    def _workers_close():
        for c in conns:
            try:
                c.send(("close", {}))
            except Exception:
                pass
        for p in workers:
            try:
                p.join(timeout=1.0)
            except Exception:
                pass
        for p in workers:
            try:
                if p.is_alive():
                    p.kill()
            except Exception:
                pass

    def _workers_reset(seeds: List[int], bump_port: bool = True):
        for i, c in enumerate(conns):
            c.send(("reset", {"seed": int(seeds[i]), "bump_port": bool(bump_port)}))
        out = []
        for c in conns:
            tag, payload = c.recv()
            if tag == "err":
                where, erepr, tb = payload
                raise RuntimeError(f"[WORKER_ERR] where={where} err={erepr}\n{tb}")
            assert tag == "reset_ok", tag
            out.append(payload)
        return out

    def _workers_step(actions_per_env: List[np.ndarray]):
        for i, c in enumerate(conns):
            c.send(("step", {"action": actions_per_env[i]}))
        out = []
        for c in conns:
            tag, payload = c.recv()
            if tag == "err":
                where, erepr, tb = payload
                raise RuntimeError(f"[WORKER_ERR] where={where} err={erepr}\n{tb}")
            assert tag == "step_ok", tag
            out.append(payload)
        return out

    # Initial reset
    base_seed = int(args.seed) if args.seed is not None else 0
    seeds0 = [base_seed + rank * 10000 + i for i in range(N)]
    reset_payloads = _workers_reset(seeds0, bump_port=False)

    id_maps = []
    vec_obs = []
    K0 = None
    env_info0 = None

    for (idm, vec, K, env_info_dict) in reset_payloads:
        id_maps.append(np.asarray(idm, dtype=np.int16))
        vec_obs.append(np.asarray(vec, dtype=np.float32).reshape(-1))
        K0 = int(K0 or K)
        env_info0 = env_info0 or env_info_dict

    id_maps = np.stack(id_maps, axis=0)  # (N,H,W)
    vec_obs = np.stack(vec_obs, axis=0)  # (N,D)
    vec_dim = int(vec_obs.shape[1])
    map_shape = tuple(id_maps.shape[1:])

    if env_info0 is None:
        raise RuntimeError("env_info missing from worker reset")

    action_info = ActionInfo(**env_info0)
    has_continuous_action_space = bool(action_info.has_continuous_action_space)
    action_dim = int(action_info.action_dim)
    action_nvec = action_info.action_nvec

    num_ids = int(K0)

    if ddp.is_main_process():
        print(f"[Vec] envs_per_rank={N} (total envs={N*world}) map_shape={map_shape} vec_dim={vec_dim} K={num_ids}", flush=True)
        print(f"[Act] space={action_info.space_type} action_dim={action_dim} action_nvec={action_nvec}", flush=True)
        print(f"[Ports] base_port={args.port} port_stride={args.port_stride} env_port_stride={args.env_port_stride}", flush=True)

    # -------------------------
    # PPOVec build
    # -------------------------
    ppo = PPOVec(
        ActorCritic,
        num_ids=num_ids,
        vec_dim=vec_dim,
        action_dim=action_dim,
        lr_actor=float(args.lr_actor),
        lr_critic=float(args.lr_critic),
        gamma=float(args.gamma),
        K_epochs=int(args.k_epochs),
        eps_clip=float(args.eps_clip),
        has_continuous_action_space=has_continuous_action_space,
        action_std_init=float(args.action_std),
        mini_batch_size=int(args.mini_batch_size),
        action_nvec=action_nvec,
        device=device,
    )

    # DDP wrap ppo.policy only
    if ddp.ddp_is_enabled():
        ppo.policy = ddp.ddp_wrap_model(ppo.policy)

    # -------------------------
    # Logging (rank0)
    # -------------------------
    ckpt_dir = save._checkpoint_dir()
    os.makedirs(ckpt_dir, exist_ok=True)

    ppo_logger = None
    if ddp.is_main_process():
        log_path = getattr(args, "ppo_log_path", None) or os.path.join(ckpt_dir, "ppo.log")
        ppo_logger = PPOFileLogger(log_path, also_stdout=False)
        ppo_logger.log(f"[Init] PPO log file: {log_path}")

    # wandb (rank0 only)
    wandb = None
    if args.wandb and ddp.is_main_process():
        import wandb as _wandb
        wandb = _wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"PPOVec-ddp{world}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(args),
        )

    # -------------------------
    # Resume
    # -------------------------
    base_step = 0
    step = 0

    ckpt_prefix = f"PPOVec_ImmortalSufferingEnv_seed{int(args.seed) if args.seed is not None else 0}_"

    if args.resume:
        ckpt_to_load = args.checkpoint
        if (ckpt_to_load is None) and ddp.is_main_process():
            ckpt_to_load = save._latest_checkpoint_path(ckpt_dir, prefix=ckpt_prefix)

        if ddp.ddp_is_enabled():
            obj_list = [ckpt_to_load]
            dist.broadcast_object_list(obj_list, src=0)
            ckpt_to_load = obj_list[0]

        if ckpt_to_load is None:
            raise FileNotFoundError(f"--resume set but no checkpoint found under {ckpt_dir} with prefix {ckpt_prefix}")

        if ddp.is_main_process():
            print(f"[Resume] loading: {ckpt_to_load}", flush=True)

        obj = torch.load(ckpt_to_load, map_location="cpu")
        loaded_step = _load_checkpoint_vec(ppo, obj, device=device)
        base_step = int(loaded_step)
        step = base_step

        ddp._ddp_barrier("resume", float(args.ddp_barrier_timeout_s))

    # -------------------------
    # Reward shaping (per-env shaper)
    # -------------------------
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

    shapers: List[RewardShaper] = []
    for i in range(N):
        s = RewardShaper(cfg)
        s.reset(vec_obs[i], id_maps[i])
        shapers.append(s)

    reward_scaler = None
    if args.reward_scaling:
        reward_scaler = RewardScaler(
            gamma=float(args.gamma),
            clip=5.0,
            eps=1e-8,
            min_std=0.1,
            warmup_steps=10000,
        )

    # -------------------------
    # Episode stats (per-env)
    # -------------------------
    ep_reward = np.zeros((N,), dtype=np.float64)
    ep_raw_reward = np.zeros((N,), dtype=np.float64)
    ep_len = np.zeros((N,), dtype=np.int64)
    ep_idx = np.zeros((N,), dtype=np.int64)

    # -------------------------
    # Update debug counters (rank-local)
    # -------------------------
    gae_lambda = float(args.gae_lambda)

    upd_steps = 0
    upd_terminals = 0
    upd_goal_hits = 0
    upd_shaper_clip_hits = 0
    upd_scaler_clip_hits = 0

    bfs_n = 0
    bfs_missing = 0
    bfs_sum = 0.0
    bfs_min = float("inf")
    bfs_max = float("-inf")
    bfs_best = float("inf")
    dd_sum = 0.0
    dd_n = 0
    closer_n = 0
    farther_n = 0

    def _reset_update_counters():
        nonlocal upd_steps, upd_terminals, upd_goal_hits, upd_shaper_clip_hits, upd_scaler_clip_hits
        nonlocal bfs_n, bfs_missing, bfs_sum, bfs_min, bfs_max, bfs_best
        nonlocal dd_sum, dd_n, closer_n, farther_n

        upd_steps = 0
        upd_terminals = 0
        upd_goal_hits = 0
        upd_shaper_clip_hits = 0
        upd_scaler_clip_hits = 0

        bfs_n = 0
        bfs_missing = 0
        bfs_sum = 0.0
        bfs_min = float("inf")
        bfs_max = float("-inf")

        dd_sum = 0.0
        dd_n = 0
        closer_n = 0
        farther_n = 0

    def _update_counters(raw_reward_f: float, done_for_buffer: bool, shaped_reward_f: float, scaled_reward_f: float):
        nonlocal upd_steps, upd_terminals, upd_goal_hits, upd_shaper_clip_hits, upd_scaler_clip_hits
        upd_steps += 1
        if done_for_buffer:
            upd_terminals += 1
        if raw_reward_f >= float(args.success_if_raw_reward_ge):
            upd_goal_hits += 1

        if float(getattr(cfg, "clip", 0.0) or 0.0) > 0.0:
            c = float(cfg.clip)
            if abs(shaped_reward_f) >= (c - 1e-6):
                upd_shaper_clip_hits += 1

        if args.reward_scaling and (reward_scaler is not None) and float(getattr(reward_scaler, "clip", 0.0) or 0.0) > 0.0:
            c2 = float(reward_scaler.clip)
            if abs(scaled_reward_f) >= (c2 - 1e-6):
                upd_scaler_clip_hits += 1

    def _print_update_pre(step_i: int, rewards_flat: np.ndarray, dones_flat: np.ndarray):
        if not ddp.is_main_process():
            return

        T = int(rewards_flat.size)
        term_n = int(dones_flat.sum())
        term_frac = (term_n / T) if T > 0 else 0.0

        rw = rewards_flat.astype(np.float32, copy=False)
        rw_mean = float(rw.mean()) if T > 0 else 0.0
        rw_std = float(rw.std()) if T > 0 else 0.0
        rw_min = float(rw.min()) if T > 0 else 0.0
        rw_max = float(rw.max()) if T > 0 else 0.0

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
            extra += f" bfs_best={bfs_best:.4f}"
        extra += f" bfs_delta_mean={0.0 if dd_mean is None else dd_mean:.5f}"
        extra += f" closer/farther={closer_pct:.1f}%/{farther_pct:.1f}%"
        extra += f" bfs_missing={bfs_miss_pct:.1f}%"

        msg = (
            f"[PPO][begin] step={step_i} "
            f"T={T} terminals={term_n} ({term_frac:.2%}) "
            f"reward(mean/std/min/max)={rw_mean:+.3f}/{rw_std:.3f}/{rw_min:+.3f}/{rw_max:+.3f} "
            f"goal_hits={upd_goal_hits} ({goal_hit_frac:.2%}) "
            f"shaper_clip_hits={upd_shaper_clip_hits} ({shaper_clip_frac:.2%}) "
            f"scaler_clip_hits={upd_scaler_clip_hits} ({scaler_clip_frac:.2%})\n"
            f"| eps_clip={float(args.eps_clip):g} K_epochs={int(args.k_epochs)} mb={int(args.mini_batch_size)} "
            f"lr(a/c)={float(args.lr_actor):g}/{float(args.lr_critic):g} gamma={float(args.gamma):g} gae_lambda={float(gae_lambda):g}\n"
            f"| w_progress={float(args.w_progress):g} w_time={float(args.w_time):g} "
            f"w_damage={float(args.w_damage):g} w_not_actionable={float(args.w_not_actionable):g} "
            f"w_acc={float(getattr(cfg,'w_acceleration',0.0)):g} "
            f"reward_clip={float(getattr(cfg,'clip',0.0)):g} scaler_clip={float(getattr(reward_scaler,'clip',0.0) if reward_scaler is not None else 0.0):g}\n"
            f"| reward_scaling={int(bool(args.reward_scaling))}\n"
            f"|{extra}"
        )

        if ppo_logger is not None:
            ppo_logger.log(msg)
        else:
            print(msg, flush=True)

    def _print_update_post(step_i: int, dt_s: float):
        if not ddp.is_main_process():
            return
        msg = f"[PPO][end]   step={step_i} update_seconds={dt_s:.3f}"
        if ppo_logger is not None:
            ppo_logger.log(msg)
        else:
            print(msg, flush=True)

    # -------------------------
    # Rollout horizon selection
    # -------------------------
    # Keep update size roughly similar to your old "update_timestep=4000 transitions per rank".
    # If horizon not explicitly set, infer horizon = update_timestep // N (at least 1).
    if args.rollout_horizon is None:
        horizon = max(1, int(args.update_timestep) // int(N))
    else:
        horizon = int(args.rollout_horizon)

    if ddp.is_main_process():
        print(f"[Rollout] horizon={horizon} envs_per_rank={N} => T_per_update={horizon*N}", flush=True)

    # IMPORTANT: construct RolloutBufferVec with POSITIONAL args (avoid T keyword mismatch)
    if action_nvec is None:
        num_branches = 1
    else:
        num_branches = int(len(action_nvec))

    buf = RolloutBufferVec(horizon, N, map_shape, vec_dim, num_branches)

    # -------------------------
    # Main loop
    # -------------------------
    MAX_STEPS = int(args.max_steps)  # "transitions per rank" budget (like old train.py)
    save_model_freq = int(args.save_model_freq)
    max_ep_len = int(args.max_ep_len)

    start_time = datetime.now().replace(microsecond=0)
    if ddp.is_main_process():
        print("============================================================================================", flush=True)
        print("Started training at:", start_time, flush=True)

    interrupted = False

    try:
        while True:
            if stopper.should_stop():
                interrupted = True
                break

            # Step budget: we count transitions per rank
            if step >= base_step + MAX_STEPS:
                break

            # -------------------------
            # Collect rollout: (horizon, N)
            # -------------------------
            buf.clear()

            # We'll store last done of the *last* timestep to mask last_values like your old code.
            last_done = np.zeros((N,), dtype=np.float32)

            for t in range(horizon):
                if stopper.should_stop():
                    interrupted = True
                    break

                # Batched act on GPU
                map_t = torch.from_numpy(id_maps.astype(np.int64)).to(device=device, dtype=torch.long)
                vec_t = torch.from_numpy(vec_obs.astype(np.float32)).to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    act_t, logp_t, val_t = ppo.act_batch(map_t, vec_t, deterministic=False)

                # Convert to CPU numpy for workers
                act_cpu = act_t.detach().cpu()
                logp_cpu = logp_t.detach().cpu().numpy().astype(np.float32)
                val_cpu = val_t.detach().cpu().numpy().astype(np.float32)

                if action_nvec is None:
                    # Discrete: (N,)
                    actions_np = act_cpu.numpy().astype(np.int64)
                    actions_list = [np.array(actions_np[i], dtype=np.int64) for i in range(N)]
                    act_store = actions_np.reshape(N, 1)  # store as (N,1) for buffer consistency
                else:
                    # MultiDiscrete: (N,branches)
                    actions_np = act_cpu.numpy().astype(np.int64)
                    actions_list = [actions_np[i].copy() for i in range(N)]
                    act_store = actions_np  # (N,B)

                # Step workers in parallel
                step_payloads = _workers_step(actions_list)

                # Build per-env reward/done for buffer
                rew_bn = np.zeros((N,), dtype=np.float32)
                done_bn = np.zeros((N,), dtype=np.float32)

                # Extra per-env signals for wandb (raw/goal/virtual_done)
                raw_bn = np.zeros((N,), dtype=np.float32)
                goal_bn = np.zeros((N,), dtype=np.float32)
                vdone_bn = np.zeros((N,), dtype=np.float32)
                done_env_bn = np.zeros((N,), dtype=np.float32)   # optional but useful

                next_id_maps = np.zeros_like(id_maps, dtype=np.int16)
                next_vec_obs = np.zeros_like(vec_obs, dtype=np.float32)

                for i, (idm2, vec2, raw_reward_any, done_flag, info) in enumerate(step_payloads):
                    raw_r = _reward_scalar(raw_reward_any)
                    is_goal = (raw_r >= float(args.success_if_raw_reward_ge))
                    done_env = bool(done_flag) or bool(is_goal)

                    # Make info dict-like
                    if info is None:
                        info = {}
                    elif not isinstance(info, dict):
                        info = {"env_info": info}
                    if is_goal:
                        info["is_goal"] = True

                    # Reward shaping
                    shaped = float(shapers[i](float(raw_r), vec2, idm2, bool(done_env), info))
                    done_for_buffer = bool(done_env) or bool(getattr(shapers[i], "virtual_done", False))

                    # Fill extra bn arrays (after shaper call so virtual_done is valid)
                    raw_bn[i] = float(raw_r)
                    goal_bn[i] = 1.0 if is_goal else 0.0
                    vdone_bn[i] = 1.0 if bool(getattr(shapers[i], "virtual_done", False)) else 0.0
                    done_env_bn[i] = 1.0 if done_env else 0.0

                    # BFS debug counters (best-effort)
                    try:
                        nonlocal_bfs = getattr(shapers[i], "dbg_has_bfs", False)
                        if nonlocal_bfs and (getattr(shapers[i], "dbg_bfs_dist", None) is not None):
                            d = float(shapers[i].dbg_bfs_dist)
                            bfs_n += 1
                            bfs_sum += d
                            bfs_min = min(bfs_min, d)
                            bfs_max = max(bfs_max, d)
                            bfs_best = min(bfs_best, d)
                        else:
                            bfs_missing += 1

                        if getattr(shapers[i], "dbg_bfs_delta", None) is not None:
                            ddv = float(shapers[i].dbg_bfs_delta)
                            dd_sum += ddv
                            dd_n += 1
                            if ddv > 0.0:
                                closer_n += 1
                            elif ddv < 0.0:
                                farther_n += 1
                    except Exception:
                        pass

                    # Optional scaling
                    if args.reward_scaling and (reward_scaler is not None):
                        scaled = float(reward_scaler(shaped, bool(done_for_buffer)))
                    else:
                        scaled = float(shaped)

                    rew_bn[i] = float(scaled)
                    done_bn[i] = 1.0 if done_for_buffer else 0.0

                    # Episode stats
                    ep_reward[i] += float(scaled)
                    ep_raw_reward[i] += float(raw_r)
                    ep_len[i] += 1

                    # Log ONCE per vector-step (after all envs are processed)
                    if wandb is not None and (
                        (step % args.wandb_log_freq == 0)
                        or bool(done_bn.any())
                        or bool(goal_bn.any())
                    ):
                        wandb.log(
                            {
                                "train/reward_mean": float(rew_bn.mean()),
                                "train/reward_min": float(rew_bn.min()),
                                "train/reward_max": float(rew_bn.max()),

                                "train/raw_reward_mean": float(raw_bn.mean()),
                                "train/raw_reward_min": float(raw_bn.min()),
                            "train/raw_reward_max": float(raw_bn.max()),

                                "train/done_frac": float(done_bn.mean()),        # terminal-for-buffer (includes virtual_done)
                                "train/done_env_frac": float(done_env_bn.mean()),# env-done-or-goal (excludes virtual_done)
                                "train/is_goal_frac": float(goal_bn.mean()),
                                "train/virtual_done_frac": float(vdone_bn.mean()),

                                "train/episode_reward_running_mean": float(ep_reward.mean()),
                                "train/episode_reward_running_max": float(ep_reward.max()),
                                "train/episode_len_running_mean": float(ep_len.mean()),
                                "train/episode_len_running_max": float(ep_len.max()),
                            },
                            step=int(step),
                        )

                    # If episode ended or max_ep_len reached: reset that env immediately
                    if done_for_buffer or (int(ep_len[i]) >= max_ep_len):
                        # Log per-episode (rank0 only, lightweight)
                        if ddp.is_main_process() and (is_goal or done_for_buffer):
                            if wandb is not None:
                                wandb.log(
                                    {
                                        "episode/reward": float(ep_reward[i]),
                                        "episode/raw_reward": float(ep_raw_reward[i]),
                                        "episode/len": int(ep_len[i]),
                                        "episode/is_goal": int(is_goal),
                                        "episode/env_i": int(i),
                                        "train/step": int(step),
                                    },
                                    step=int(step),
                                )

                        ep_idx[i] += 1
                        ep_reward[i] = 0.0
                        ep_raw_reward[i] = 0.0
                        ep_len[i] = 0

                        # Close+recreate reset in worker (bump_port=True)
                        seed_i = base_seed + rank * 10000 + i + int(step)
                        conns[i].send(("reset", {"seed": int(seed_i), "bump_port": True}))
                        tag, payload = conns[i].recv()
                        if tag == "err":
                            where, erepr, tb = payload
                            raise RuntimeError(f"[WORKER_ERR][reset] where={where} err={erepr}\n{tb}")
                        assert tag == "reset_ok", tag
                        idmR, vecR, _KR, _envinfo = payload
                        idm2 = np.asarray(idmR, dtype=np.int16)
                        vec2 = np.asarray(vecR, dtype=np.float32).reshape(-1)

                        # Reset shaper internal state too
                        shapers[i].reset(vec2, idm2)

                    # Save next state (already reset if terminal)
                    next_id_maps[i] = np.asarray(idm2, dtype=np.int16)
                    next_vec_obs[i] = np.asarray(vec2, dtype=np.float32).reshape(-1)

                    # Update counters for PPO log
                    _update_counters(float(raw_r), bool(done_bn[i] > 0.5), float(shaped), float(rew_bn[i]))

                # Add batch to buffer at time t (store current pre-step state)
                buf.add_batch(
                    map_bn=id_maps,         # (N,H,W)
                    vec_bn=vec_obs,         # (N,D)
                    act_bn=act_store,       # (N,B) or (N,1)
                    logp_bn=logp_cpu,       # (N,)
                    val_bn=val_cpu,         # (N,)
                    rew_bn=rew_bn,          # (N,)
                    done_bn=done_bn,        # (N,)
                )

                # Advance states
                id_maps[:] = next_id_maps
                vec_obs[:] = next_vec_obs
                last_done[:] = done_bn

                # Transition counter
                step += int(N)

            if interrupted:
                break

            # -------------------------
            # Bootstrap last values for GAE
            # -------------------------
            with torch.no_grad():
                map_t2 = torch.from_numpy(id_maps.astype(np.int64)).to(device=device, dtype=torch.long)
                vec_t2 = torch.from_numpy(vec_obs.astype(np.float32)).to(device=device, dtype=torch.float32)
                # Use policy_old for bootstrap
                feat = ppo.policy_old.encode(map_t2, vec_t2)
                v = ppo.policy_old.critic_head(feat).view(-1).detach().cpu().numpy().astype(np.float32)

            # Mask like your old code: if last step was terminal, last_value=0
            v = v * (1.0 - last_done.astype(np.float32))
            buf.set_last_values(v)

            # Build RolloutBatch (CPU tensors)
            batch = RolloutBatch(
                map_states=torch.from_numpy(buf.map_states),    # (H,N,H,W)
                vec_states=torch.from_numpy(buf.vec_states),    # (H,N,D)
                actions=torch.from_numpy(buf.actions),          # (H,N,B)
                logprobs=torch.from_numpy(buf.logprobs),        # (H,N)
                values=torch.from_numpy(buf.values),            # (H,N)
                rewards=torch.from_numpy(buf.rewards),          # (H,N)
                dones=torch.from_numpy(buf.dones),              # (H,N)
                last_values=torch.from_numpy(buf.last_values),  # (N,)
            )

            # PPO log pre
            if ddp.is_main_process():
                rewards_flat = buf.rewards.reshape(-1)
                dones_flat = buf.dones.reshape(-1)
                _print_update_pre(int(step), rewards_flat, dones_flat)

            # PPO update (all ranks)
            t0 = time.time()
            ppo.update(batch, gae_lambda=float(gae_lambda))
            dt = time.time() - t0

            if ddp.is_main_process():
                _print_update_post(int(step), float(dt))

            # wandb training logs (rank0 only)
            if wandb is not None:
                wandb.log(
                    {
                        "train/updated": 1,
                        "train/update_seconds": float(dt),
                        "train/step": int(step),
                        "train/global_env_steps": int(step * world),
                        "train/update_goal_hits": int(upd_goal_hits),
                        "train/update_terminals": int(upd_terminals),
                        "train/update_shaper_clip_hits": int(upd_shaper_clip_hits),
                        "train/update_scaler_clip_hits": int(upd_scaler_clip_hits),
                        "rollout/horizon": int(horizon),
                        "rollout/envs_per_rank": int(N),
                    },
                    step=int(step),
                )

            _reset_update_counters()

            # Save checkpoint (rank0 only)
            if (save_model_freq > 0) and (step % int(save_model_freq) == 0) and ddp.is_main_process():
                ckpt_path = os.path.join(ckpt_dir, f"{ckpt_prefix}{int(step)}.pth")
                if ppo_logger is not None:
                    ppo_logger.log("-" * 92)
                    ppo_logger.log(f"saving model at: {ckpt_path}")
                else:
                    print("saving model at:", ckpt_path, flush=True)

                with stopper.ignore_signals():
                    save.atomic_torch_save(
                        lambda: _make_checkpoint_vec(ppo, int(step), args, cfg, reward_scaler),
                        ckpt_path,
                    )

                if wandb is not None:
                    wandb.save(ckpt_path)

    except KeyboardInterrupt:
        interrupted = True
        if ddp.is_main_process():
            print("\n[KeyboardInterrupt] requested stop (will save in finally).", flush=True)

    finally:
        # Always close workers (best effort)
        try:
            _workers_close()
        except Exception:
            pass

        # Final save (rank0 only)
        force_exit = bool(interrupted) or bool(getattr(stopper, "stop_requested", False))

        if ddp.is_main_process():
            try:
                final_step = int(step)
                final_path = os.path.join(ckpt_dir, f"{ckpt_prefix}{final_step}.pth")
                if interrupted:
                    print("[Exit] stop requested -> saving final checkpoint:", final_path, flush=True)
                with stopper.ignore_signals():
                    save.atomic_torch_save(
                        lambda: _make_checkpoint_vec(ppo, final_step, args, cfg, reward_scaler),
                        final_path,
                    )
                print("Final model saved at:", final_path, flush=True)
            except Exception as e:
                print(f"[Exit][WARN] failed to save final checkpoint: {repr(e)}", flush=True)

        if (wandb is not None) and (not force_exit):
            try:
                wandb.finish()
            except Exception:
                pass

        end_time = datetime.now().replace(microsecond=0)
        if ddp.is_main_process():
            print("============================================================================================", flush=True)
            print("Started training at:", start_time, flush=True)
            print("Finished training at:", end_time, flush=True)
            print("Total training time:", end_time - start_time, flush=True)
            print("============================================================================================", flush=True)

        if ppo_logger is not None:
            try:
                ppo_logger.close()
            except Exception:
                pass

        if not force_exit:
            try:
                ddp.ddp_cleanup()
            except Exception:
                pass

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        if force_exit:
            os._exit(0)


# -------------------------
# Argparser
# -------------------------

def build_argparser():
    p = argparse.ArgumentParser()

    # Immortal env args
    p.add_argument("--game_path", type=str, default=r"/root/immortal_suffering/immortal_suffering_linux_build.x86_64")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--port_stride", type=int, default=200)          # per-rank stride (big!)
    p.add_argument("--env_port_stride", type=int, default=50)       # per-env stride within rank (big enough!)
    p.add_argument("--envs_per_rank", type=int, default=1)
    p.add_argument("--time_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--verbose", action="store_true")

    # Graphics flags
    g = p.add_mutually_exclusive_group()
    g.add_argument("--no_graphics", dest="no_graphics", action="store_true", default=False)
    g.add_argument("--graphics", dest="no_graphics", action="store_false")

    p.add_argument("--unity_timeout_wait", type=int, default=120)
    p.add_argument("--restart_sleep_s", type=float, default=0.2)

    # Runner steps (PER RANK, transitions)
    p.add_argument("--max_steps", type=int, default=2000000)

    # PPO hyperparams
    p.add_argument("--max_ep_len", type=int, default=1000)
    p.add_argument("--update_timestep", type=int, default=4000)     # target transitions per update (per rank)
    p.add_argument("--rollout_horizon", type=int, default=None)     # if None -> update_timestep // envs_per_rank
    p.add_argument("--k_epochs", type=int, default=10)
    p.add_argument("--eps_clip", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--lr_actor", type=float, default=1e-4)
    p.add_argument("--lr_critic", type=float, default=2e-4)
    p.add_argument("--mini_batch_size", type=int, default=64)

    # Action std (kept for compatibility; MultiDiscrete won't use)
    p.add_argument("--action_std", type=float, default=0.6)

    # Saving (rank0 only)
    p.add_argument("--save_model_freq", type=int, default=35000)

    # Resume
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)

    # Reward shaping knobs
    p.add_argument("--w_progress", type=float, default=100.0)
    p.add_argument("--w_time", type=float, default=0.02)
    p.add_argument("--w_damage", type=float, default=0.1)
    p.add_argument("--w_not_actionable", type=float, default=0.01)
    p.add_argument("--terminal_bonus", type=float, default=6.0)
    p.add_argument("--reward_clip", type=float, default=10.0)
    p.add_argument("--success_if_raw_reward_ge", type=float, default=1.0)
    p.add_argument("--time_limit", type=float, default=300.0)
    p.add_argument("--success_speed_bonus", type=float, default=3)

    # Reward scaling toggle (default OFF)
    p.add_argument("--reward_scaling", action="store_true", default=False)

    # DDP barrier timeout
    p.add_argument("--ddp_barrier_timeout_s", type=float, default=600.0)

    # ppo.log path (rank0 only)
    p.add_argument("--ppo_log_path", type=str, default=None)

    # wandb (rank0 only)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="ppo-immortal-vec")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_log_freq", type=int, default=2000) 

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
