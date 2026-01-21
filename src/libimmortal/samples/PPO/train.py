#!/usr/bin/env python3

"""
Read README.md

If you want reward scaling:
  --reward_scaling

If you want graphics (NOT recommended on headless):
  --graphics
"""

import os, sys
import random
import time
import json
import argparse
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from libimmortal.samples.PPO.reward import RewardConfig, RewardShaper, RewardScaler

import gym
import numpy as np
import torch
import torch.distributed as dist
import signal, faulthandler, traceback

from libimmortal.samples.PPO.PPO import PPO

import libimmortal.samples.PPO.utils.ddp as ddp
import libimmortal.samples.PPO.utils.save as save
import libimmortal.samples.PPO.utils.debug as dbg
import libimmortal.samples.PPO.utils.utilities as utilities
import libimmortal.samples.PPO.utils.excs as excs
from libimmortal.samples.PPO.utils.ppolog import PPOFileLogger

from libimmortal.samples.PPO.utils.signaling import GracefulStop

try:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except Exception:
    pass


# Optional: robust terminal detection if your env stuffs ML-Agents steps into info
try:
    from mlagents_envs.base_env import TerminalSteps
except Exception:
    TerminalSteps = None

try:
    # Raised when calling step() after done=True without reset()
    from mlagents_envs.envs.unity_gym_env import UnityGymException
except Exception:
    UnityGymException = None

try:
    from mlagents_envs.exception import UnityTimeOutException
except Exception:
    UnityTimeOutException = None

try:
    from mlagents_envs.exception import UnityEnvironmentException
except Exception:
    UnityEnvironmentException = None



from libimmortal.env import ImmortalSufferingEnv

faulthandler.enable(all_threads=True)

# =========================
# [ADD] Hard stop utilities
# =========================

_STOP_REQUESTED = False

def _mark_stop_requested(*_args):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True

def _stop_requested() -> bool:
    return bool(_STOP_REQUESTED)

class _StopTraining(SystemExit):
    """Used to unwind the Python stack into finally without os._exit()."""
    pass

def _hard_exit(code: int = 0):
    # Don't rely on Python cleanup in DDP+Unity situations.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(int(code))

# Register signals early (best effort)
try:
    signal.signal(signal.SIGTERM, _mark_stop_requested)
except Exception:
    pass
try:
    signal.signal(signal.SIGINT, _mark_stop_requested)
except Exception:
    pass

# =========================
# File writing utilities
# =========================

def _touch_file(path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"ok {time.time()}\n")
        return True
    except Exception:
        return False
    
def _write_text_atomic(path: str, text: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except Exception:
        return False

def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None

def _wait_for_file(path: str, timeout_s: float, poll_s: float = 0.1) -> bool:
    t0 = time.time()
    while (time.time() - t0) < float(timeout_s):
        if os.path.exists(path):
            return True
        time.sleep(float(poll_s))
    return False

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

import math

def p_random_by_episode(args, ep_idx: int) -> float:
    p0 = float(args.env_seed_mix_start)
    p1 = float(args.env_seed_mix_end)

    if args.env_seed_mix_schedule == "linear":
        warm = max(1, int(args.env_seed_mix_warmup_episodes))
        t = min(1.0, ep_idx / warm)
        return p0 + (p1 - p0) * t
    else:  # exp
        tau = max(1.0, float(args.env_seed_mix_tau))
        t = 1.0 - math.exp(-ep_idx / tau)
        return p0 + (p1 - p0) * t

def choose_env_seed(args, ep_idx: int, rank: int) -> Tuple[int, bool, float]:
    base_seed = int(args.seed)  # args.seed is not None 가정
    fixed = base_seed + rank
    mode = args.env_seed_mode
    if mode == "fixed":
        return fixed, False, 0.0
    if mode == "random":
        s = mix_seed(int(args.env_seed_base), ep_idx, rank)
        return s, True, 1.0
    # mix mode
    p = p_random_by_episode(args, ep_idx)
    u = random.Random(mix_seed(99991, ep_idx, rank)).random()
    use_random = (u < p)
    if use_random:
        s = mix_seed(int(args.env_seed_base), ep_idx, rank)
    else:
        s = fixed
    return s, use_random, p

def mix_seed(base: int, ep_idx: int, rank: int) -> int:
    x = (base ^ (ep_idx * 0x9E3779B1) ^ (rank * 0x85EBCA6B)) & 0xFFFFFFFF
    return int(x % (2**31 - 1))

import socket

def _port_is_free(p: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", p))
        return True
    except OSError:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass

# -------------------------
# Training
# -------------------------

def train(args):
    faulthandler.enable(all_threads=True)

    print("============================================================================================")

    stopper = GracefulStop()
    stopper.install()

    ddp.ddp_setup()
    rank = ddp.ddp_rank()
    local_rank = ddp.ddp_local_rank()
    world = ddp.ddp_world_size()

    if args.seed is not None:
        ddp.seed_everything(int(args.seed))

    # Per-rank device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if ddp.is_main_process():
        print(f"[Device] rank={rank} local_rank={local_rank} device={device}")

    # DDP barrier with timeout
    ddp_barrier_timeout_s = float(getattr(args, "ddp_barrier_timeout_s", 600.0))  # default: 10 min
    ddp_step_sync_every = int(getattr(args, "ddp_step_sync_every", 0))            # default: sync every step
    # IMPORTANT: pre/post-update barriers are optional; keeping them always-on can deadlock if ranks diverge.
    use_update_barriers = bool(getattr(args, "ddp_update_barriers", False))       # default: False


    # =========================
    # [CHANGE] stop condition
    # =========================

    # -------------------------
    # Stop / watchdog policy
    # -------------------------
    # Goal: (1) reduce NVML/NCCL issues by avoiding os._exit in normal Ctrl+C path,
    #       (2) still guarantee process termination even if env/DDP blocks.
    GRACE_SECONDS = float(getattr(args, "stop_grace_seconds", 5.0))   # after first Ctrl+C
    FORCE_SECONDS = float(getattr(args, "stop_force_seconds", 1.0))   # after second Ctrl+C
    _stop_t0: Optional[float] = None
    _force_t0: Optional[float] = None

    def _ddp_stop_any() -> bool:
        return bool(stopper.should_stop())

    def _ddp_force_any() -> bool:
        return bool(getattr(stopper, "should_force", lambda: False)())

    def _check_stop_or_raise(where: str) -> None:
        """Call frequently from Python-level code paths."""
        nonlocal _stop_t0, _force_t0
        if _ddp_force_any() and _force_t0 is None:
            _force_t0 = time.time()
        if _ddp_stop_any() and _stop_t0 is None:
            _stop_t0 = time.time()

        # Fast path: no stop requested
        if _stop_t0 is None:
            return

        # If second Ctrl+C happened, shorten the grace window drastically.
        if _force_t0 is not None:
            if (time.time() - _force_t0) >= FORCE_SECONDS:
                # Last resort hard-exit
                _hard_exit(130)
            # unwind ASAP
            raise _StopTraining(0)

        # First Ctrl+C: attempt graceful unwind; if it takes too long, hard-exit.
        if (time.time() - _stop_t0) >= GRACE_SECONDS:
            _hard_exit(0)
        raise _StopTraining(0)
    
    # NOTE: Avoid per-step collectives. They are a common deadlock source on interrupts.
    # -------------------------
    # Unity env control
    # -------------------------
    port_stride = int(args.port_stride)
    max_env_restarts = int(args.max_env_restarts)
    unity_timeout_wait = int(args.unity_timeout_wait)
    no_graphics = bool(args.no_graphics)

    restart_box = {"id": 0}
    last_port_box = {"port": int(args.port)}

    
    def _compute_port(args_) -> int:
        base = (
            int(args_.port)
            + int(local_rank) * port_stride
            + int(restart_box["id"]) * port_stride * int(world)
        )

        # ✅ 자기 stride 구간 안에서만 회피 (겹치지 않게!)
        for off in range(port_stride):
            p = base + off
            if _port_is_free(p):
                return p

        # 여기까지 왔으면, 이 restart_id 구간이 다 막힌 것
        return base

    def _make_env(seed_override: Optional[int] = None):
        nonlocal restart_box

        if _ddp_stop_any():
            raise RuntimeError("[Env] stop requested; abort env creation")

        if seed_override is None:
            seed_override = (int(args.seed) + rank) if args.seed is not None else None

        last_err = None
        backoff_s = 0.5

        for attempt in range(50):
            if _ddp_stop_any():
                raise RuntimeError("[Env] stop requested; abort env creation loop")

            p = _compute_port(args)
            last_port_box["port"] = p
            print(
                f"[Env] rank={rank} local_rank={local_rank} restart_id={restart_box['id']} "
                f"port={p} (try {attempt})",
                flush=True
            )

            kw = dict(
                game_path=args.game_path,
                port=p,
                time_scale=args.time_scale,
                seed=seed_override,
                width=args.width,
                height=args.height,
                verbose=args.verbose,
            )

            _env = None
            try:
                try:
                    _env = ImmortalSufferingEnv(
                        **kw, no_graphics=no_graphics, timeout_wait=unity_timeout_wait
                    )
                except TypeError:
                    _env = ImmortalSufferingEnv(**kw)

                return _env, p

            except Exception as e:
                last_err = e
                msg = repr(e)

                # ---- best-effort cleanup of partial env ----
                try:
                    if _env is not None:
                        _env.close()
                except Exception:
                    pass

                # stop이면 여기서도 즉시 중단
                if _ddp_stop_any():
                    raise RuntimeError("[Env] stop requested; abort after env create failure")

                # (A) 포트 충돌: 같은 restart_id에서 포트 찾기 실패 가능
                if ("Address already in use" in msg) or ("EADDRINUSE" in msg):
                    # 같은 restart_id 내에서 _compute_port가 off를 돌긴 하는데,
                    # 여긴 최소 backoff만 주고 계속
                    stopper.sleep_poll(backoff_s)
                    backoff_s = min(5.0, backoff_s * 1.3)
                    continue

                # (B) worker/유니티가 꼬여서 "still in use" 계열:
                # 포트만 바꾸면 해결 안 되는 경우가 많아서 restart_id를 올려서 대역을 바꾸고,
                # backoff를 더 강하게 준다.
                if ("UnityWorkerInUseException" in msg) or ("worker number 0 is still in use" in msg):
                    restart_box["id"] += 1
                    stopper.sleep_poll(max(1.0, backoff_s))
                    backoff_s = min(10.0, backoff_s * 1.7)
                    continue

                # 그 외는 즉시 throw (조용히 재시도하면 고아 프로세스만 늘어남)
                raise

        raise RuntimeError(f"[Env] failed to create env after retries. last_err={repr(last_err)}")


    def _restart_env(reason: str):
        nonlocal env
        if _ddp_stop_any():
            raise RuntimeError(f"[Env] stop requested; abort restart (reason={reason})")
        restart_box["id"] += 1
        if restart_box["id"] > max_env_restarts:
            raise RuntimeError(f"[Env] exceeded max_env_restarts={max_env_restarts} (last reason={reason})")

        if ddp.is_main_process():
            print(f"[Env] restart #{restart_box['id']} reason={reason}")

        try:
            env.close()
        except Exception:
            pass

        dbg._print_player_log_tail(prefix=f"[Env][{reason}]", n=int(args.player_log_tail))
        new_env, new_port = _make_env()
        return new_env, new_port

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
        # Do NOT call _env.reset() here (double-reset bug).
        space = _get_action_space(_env)
        a_np = _noop_action_np(space)
        a_env = utilities._format_action_for_env(a_np, space)
        obs_, _r, _d, _info = _env.step(a_env)
        return obs_

    # Robust reward scalarization (goal/terminal steps often change reward type/shape)
    def _reward_scalar(r) -> float:
        """Convert reward to a scalar float robustly across env wrappers."""
        try:
            if torch.is_tensor(r):
                return float(r.detach().view(-1)[0].item()) if r.numel() > 0 else 0.0
            if isinstance(r, np.ndarray):
                rr = r.reshape(-1)
                return float(rr[0]) if rr.size > 0 else 0.0
            if isinstance(r, (list, tuple)):
                return float(r[0]) if len(r) > 0 else 0.0
            return float(r)
        except Exception:
            return 0.0

    def _safe_reset():
        nonlocal env, port, step, episode_idx

        # reset 폭주 방지용
        max_reset_attempts = int(getattr(args, "max_reset_attempts", 30))
        backoff_s = 0.5

        def _stop_now() -> bool:
            return bool(_ddp_stop_any())

        for reset_attempt in range(max_reset_attempts):
            if _stop_now():
                # stop 요청 시 절대 유니티를 다시 띄우지 않는다
                raise RuntimeError("[Env][reset] stop requested")

            # (1) seed 선택
            if args.seed is not None:
                s, use_rand, p_mix = choose_env_seed(args, episode_idx, rank)
            else:
                s, use_rand, p_mix = (None, False, 0.0)

            if ddp.is_main_process():
                print(
                    f"[Env][reset] attempt={reset_attempt}/{max_reset_attempts} "
                    f"ep={episode_idx} p_rand={p_mix:.3f} use_rand={int(use_rand)} seed={s}",
                    flush=True
                )

            t0 = time.time()

            # (2) 기존 env 닫기 (best-effort)
            if env is not None:
                try:
                    if ddp.is_main_process():
                        print(f"[Env][reset] closing... step={step} port={last_port_box['port']}", flush=True)
                    env.close()
                except Exception:
                    pass
                env = None
                port = None

                # close 직후 즉시 재생성하면 worker가 덜 정리된 경우가 있어서 약간 쉼
                stopper.sleep_poll(min(1.0, backoff_s))

            if _stop_now():
                raise RuntimeError("[Env][reset] stop requested (after close)")

            # (3) 새 env 생성
            try:
                if ddp.is_main_process():
                    print(f"[Env][reset] creating new env...", flush=True)

                env, port = _make_env(seed_override=s)

                if ddp.is_main_process():
                    print(f"[Env][reset] created port={port} -> bootstrapping", flush=True)

                # (4) bootstrap (reset 대신 step 1번으로 obs 확보)
                obs0 = _bootstrap_obs(env)

                # (5) obs 유효성 체크 (policy가 필요로 하는 vec + id_map 둘 다)
                idm0, vec0, _k0 = utilities._make_map_and_vec(obs0)
                if vec0 is None or idm0 is None:
                    raise RuntimeError(f"bootstrap missing obs: id_map={idm0 is None} vec={vec0 is None}")

                if ddp.is_main_process():
                    dt = time.time() - t0
                    print(f"[Env][reset] bootstrap ok in {dt:.3f}s", flush=True)

                return obs0

            except Exception as e:
                # stop이면 여기서도 즉시 종료 (재시도 금지)
                if _stop_now():
                    raise RuntimeError(f"[Env][reset] stop requested while handling exception: {repr(e)}")

                # restartable이면 재생성 시도
                if excs._is_restartable_exc(e):
                    if ddp.is_main_process():
                        print(f"[Env][reset][WARN] restartable exception={repr(e)}", flush=True)

                    # 지금 env가 살아있으면 닫기 (best-effort)
                    try:
                        if env is not None:
                            env.close()
                    except Exception:
                        pass
                    env = None
                    port = None

                    # bootstrap invalid / worker in use 같은 케이스는 restart_id 올리는 게 유효
                    restart_box["id"] += 1
                    if restart_box["id"] > max_env_restarts:
                        raise RuntimeError(
                            f"[Env] exceeded max_env_restarts={max_env_restarts} (reset_fail)"
                        )

                    # Player.log tail 찍기(원하면)
                    try:
                        dbg._print_player_log_tail(prefix=f"[Env][reset_fail attempt={reset_attempt}]", n=int(args.player_log_tail))
                    except Exception:
                        pass

                    # 백오프 증가 (폭주 방지)
                    stopper.sleep_poll(backoff_s)
                    backoff_s = min(10.0, backoff_s * 1.7)

                    # 다음 루프에서 다시 시도
                    continue

                # restartable이 아니면 즉시 raise (조용한 재시도는 고아 유니티만 늘릴 수 있음)
                raise

        raise RuntimeError(f"[Env][reset] failed after {max_reset_attempts} attempts")


    # Build initial env (defer to _safe_reset)
    env = None
    port = None
    env_tag = "ImmortalSufferingEnv"
    episode_reward = 0.0
    episode_raw_reward = 0.0
    episode_len = 0
    episode_idx = 0

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
    step = 0  # for finally/logging

    # Infer dims from first reset
    _check_stop_or_raise("before_initial_reset")
    obs = _safe_reset()
    id_map, vec_obs, K = utilities._make_map_and_vec(obs)

    expected_id_shape = tuple(id_map.shape)
    expected_vec_dim = int(vec_obs.shape[0])

    # ----- Debug stats -----
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

    action_nvec = None

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

    if ddp.is_main_process():
        print(f"[DDP] world_size={world}, rank={rank}, local_rank={local_rank}")
        print(f"[Env] base_port={args.port}, this_rank_port={port}, port_stride={port_stride}, no_graphics={no_graphics}")
        print(f"[Obs] num_ids(K)={num_ids}, vec_dim={expected_vec_dim}, id_shape={expected_id_shape}")
        print(f"[Act] action_dim={action_dim}, continuous={has_continuous_action_space}, space={type(action_space)}")

    def _make_map_and_vec_checked(obs_):
        nonlocal expected_vec_dim, expected_id_shape, num_ids
        idm, vec, k = utilities._make_map_and_vec(obs_)

        # Make vec always a usable numpy array (policy input must be stable).
        if vec is None:
            if ddp.is_main_process():
                print(f"[Obs][WARN] vec_obs is None -> using zeros({expected_vec_dim})", flush=True)
            vec = np.zeros((expected_vec_dim,), dtype=np.float32)
        else:
            vec = np.asarray(vec, dtype=np.float32).reshape(-1)

        # If map is missing, do NOT touch idm.shape.
        # We still want vec to be well-formed (pad/truncate) so the policy can continue.
        if idm is None:
            if ddp.is_main_process():
                print(f"[Obs][WARN] id_map is None (map observation missing). vec_dim={getattr(vec,'shape',None)} k={k}", flush=True)
            # Keep vec dimension consistent
            if int(vec.shape[0]) != expected_vec_dim:
                if vec.shape[0] > expected_vec_dim:
                    vec = vec[:expected_vec_dim]
                else:
                    vec = np.pad(vec, (0, expected_vec_dim - vec.shape[0]), mode="constant")
            return None, vec, k

        if tuple(idm.shape) != expected_id_shape:
            if ddp.is_main_process():
                print(f"[Obs][WARN] id_map shape changed: got={idm.shape}, expected={expected_id_shape}", flush=True)
            expected_id_shape = tuple(idm.shape)

        if int(vec.shape[0]) != expected_vec_dim:
            if ddp.is_main_process():
                print(f"[Obs][WARN] vec_dim mismatch: got={vec.shape[0]}, expected={expected_vec_dim} (will pad/truncate)", flush=True)
            if vec.shape[0] > expected_vec_dim:
                vec = vec[:expected_vec_dim]
            else:
                vec = np.pad(vec, (0, expected_vec_dim - vec.shape[0]), mode="constant")

        if int(k) != int(num_ids):
            if ddp.is_main_process():
                print(f"[Obs][WARN] K changed: got={k}, expected={num_ids} (keeping expected)", flush=True)
        return idm, vec, k

    # PPO agent
    ppo_agent = PPO(
        num_ids=num_ids,
        action_nvec=action_nvec,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        mini_batch_size=mini_batch_size,
        device=device,
        model_dim=256,
        enemy_state_vocab=32,
        enemy_state_emb=16,
    )
    '''
    ppo_agent = PPO(
        num_ids, expected_vec_dim, action_dim,
        lr_actor, lr_critic,
        gamma, K_epochs, eps_clip,
        has_continuous_action_space,
        action_std_init=action_std,
        mini_batch_size=mini_batch_size,
        action_nvec=action_nvec,
    )
    '''

    if hasattr(ppo_agent, "policy"):
        ddp.get_module(ppo_agent.policy).to(device)
    if hasattr(ppo_agent, "policy_old"):
        ddp.get_module(ppo_agent.policy_old).to(device)

    # DDP: wrap ppo_agent.policy (NOT policy_old)
    if hasattr(ppo_agent, "policy") and ddp.ddp_is_enabled():
        ddp_pol = ddp.ddp_wrap_model(ppo_agent.policy)
        base_pol = ddp.get_module(ddp_pol)

        if hasattr(base_pol, "evaluate"):
            if not hasattr(base_pol, "_orig_forward"):
                base_pol._orig_forward = base_pol.forward
            base_pol.forward = base_pol.evaluate

            def _ddp_evaluate(mb_map, mb_vec, mb_actions):
                return ddp_pol(mb_map, mb_vec, mb_actions)

            ddp_pol.evaluate = _ddp_evaluate  # type: ignore

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
        reward_clip=args.reward_clip,
        success_if_raw_reward_ge=args.success_if_raw_reward_ge,
        time_limit=args.time_limit,
        success_speed_bonus=args.success_speed_bonus,
        use_bfs_progress=True,
        w_acceleration=1.0,
        bfs_update_every=10,
    )
    shaper = RewardShaper(cfg)
    shaper.reset(vec_obs, id_map)

    reward_scaler = RewardScaler(
        gamma=gamma,
        clip=5.0,
        eps=1e-8,
        min_std=0.1,
        warmup_steps=10000,
    )

    ckpt_dir = save._checkpoint_dir()
    os.makedirs(ckpt_dir, exist_ok=True)

    # Per-rank debug traces
    dbg_dir = os.path.join(ckpt_dir, "debug_traces")
    os.makedirs(dbg_dir, exist_ok=True)

    # Ring buffer for last N steps (per rank)
    DBG_N = int(getattr(args, "debug_trace_n", 64))
    dbg_buf = []

    def _dbg_push(step_i, cur_map, cur_vec, act_np, raw_reward_any, done_any, info_any):
        item = dict(
            step=int(step_i),
            action=np.asarray(act_np),
            raw_reward=raw_reward_any,
            done=done_any,
            info_repr=repr(info_any),
            id_map=np.asarray(cur_map, dtype=np.int16),
            vec_obs=np.asarray(cur_vec, dtype=np.float32),
            raw_type=str(type(raw_reward_any)),
            raw_shape=getattr(raw_reward_any, "shape", None),
        )
        dbg_buf.append(item)
        if len(dbg_buf) > DBG_N:
            dbg_buf.pop(0)

    def _dbg_dump(tag: str):
        path = os.path.join(dbg_dir, f"{tag}_rank{rank}_step{int(step)}.npz")
        try:
            np.savez_compressed(
                path,
                steps=np.array([x["step"] for x in dbg_buf], dtype=np.int64),
                actions=np.stack([x["action"] for x in dbg_buf], axis=0) if len(dbg_buf) else np.zeros((0,)),
                raw_reward=np.array([_reward_scalar(x["raw_reward"]) for x in dbg_buf], dtype=np.float32),
                done=np.array([int(bool(x["done"])) for x in dbg_buf], dtype=np.int8),
                raw_type=np.array([x["raw_type"] for x in dbg_buf], dtype=object),
                raw_shape=np.array([repr(x["raw_shape"]) for x in dbg_buf], dtype=object),
                info_repr=np.array([x["info_repr"] for x in dbg_buf], dtype=object),
                id_map=np.stack([x["id_map"] for x in dbg_buf], axis=0) if len(dbg_buf) else np.zeros((0,)),
                vec_obs=np.stack([x["vec_obs"] for x in dbg_buf], axis=0) if len(dbg_buf) else np.zeros((0,)),
            )
            print(f"[DBG] dumped {tag} to {path}", flush=True)
        except Exception as e:
            print(f"[DBG][WARN] dump failed: {repr(e)}", flush=True)

    # Rank0 PPO log
    ppo_logger = None
    if ddp.is_main_process():
        log_path = getattr(args, "ppo_log_path", None) or os.path.join(ckpt_dir, "ppo.log")
        ppo_logger = PPOFileLogger(log_path, also_stdout=False)
        ppo_logger.log(f"[Init] PPO log file: {log_path}")

    ckpt_prefix = f"Necto2_{env_tag}_seed{int(args.seed) if args.seed is not None else 0}_"
    if ddp.is_main_process():
        print("checkpoint dir:", ckpt_dir)

    model_device = next(ddp.get_module(ppo_agent.policy_old).parameters()).device

    # Resume
    if args.resume:
        # --------- File-based rendezvous (NO collectives) ----------
        # Using dist.broadcast_object_list() can deadlock if any rank diverges/crashes
        # during Unity setup/reset. Use a simple file in ckpt_dir instead.
        ckpt_to_load = args.checkpoint
        rendezvous_dir = ckpt_dir if (ckpt_dir is not None) else "/tmp"
        ckpt_path_file = os.path.join(rendezvous_dir, f".resume_ckpt_path.rank0")

        if ddp.is_main_process():
            if ckpt_to_load is None:
                ckpt_to_load = save._latest_checkpoint_path(ckpt_dir, prefix=ckpt_prefix)
            _write_text_atomic(ckpt_path_file, "" if ckpt_to_load is None else str(ckpt_to_load))
        else:
            _wait_for_file(ckpt_path_file, timeout_s=120.0)
            txt = _read_text(ckpt_path_file)
            ckpt_to_load = (txt.strip() if txt is not None else "") or None

        if ckpt_to_load is None:
            raise FileNotFoundError(f"--resume set but no checkpoint found under {ckpt_dir} with prefix {ckpt_prefix}")

        inferred_step = save._infer_step_from_ckpt_path(str(ckpt_to_load), ckpt_prefix)

        if ddp.is_main_process():
            print(f"[Resume] loading: {ckpt_to_load}")
            print(f"[Resume] inferred base_step(from filename)={inferred_step}")

        obj = torch.load(ckpt_to_load, map_location="cpu")
        loaded_step = save._load_checkpoint_into(
            ppo_agent,
            obj,
            model_device=model_device,
            reward_scaler=reward_scaler,
        )

        base_step = loaded_step if loaded_step > 0 else inferred_step
        step = base_step

        if ddp.is_main_process():
            print(f"[Resume] base_step(used)={base_step} (loaded_step={loaded_step})")

        # Avoid hard barriers here; resume should not deadlock even if a rank is slow.
        try:
            ddp._ddp_barrier_soft("resume_soft", timeout_s=10.0)
        except Exception:
            pass

    # Optional wandb (rank0 only)
    wandb = None
    if args.wandb and ddp.is_main_process():
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
    if ddp.is_main_process():
        if ppo_logger is not None:
            ppo_logger.log(f"Started training at: {start_time}")
            ppo_logger.log("=" * 92)
        else:
            print("Started training at:", start_time)
            print("============================================================================================")

    cur_id_map = id_map
    cur_vec_obs = vec_obs

    def _select_action(cur_map_np: np.ndarray, cur_vec_np: np.ndarray):
        map_t = torch.from_numpy(cur_map_np).to(model_device, dtype=torch.long).unsqueeze(0)
        vec_t = torch.from_numpy(cur_vec_np).to(model_device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_t, logp_t, value_t = ddp.get_module(ppo_agent.policy_old).act(map_t, vec_t)

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
        nonlocal bfs_n, bfs_sum, bfs_min, bfs_max, bfs_missing
        nonlocal dd_n, dd_sum, closer_n, farther_n
        upd_steps = 0
        upd_terminals = 0
        upd_goal_hits = 0
        upd_shaper_clip_hits = 0
        upd_scaler_clip_hits = 0

        bfs_n = 0
        bfs_sum = 0.0
        bfs_min = float("inf")
        bfs_max = float("-inf")
        bfs_missing = 0

        dd_n = 0
        dd_sum = 0.0
        closer_n = 0
        farther_n = 0

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
        if not ddp.is_main_process():
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
            extra += f" bfs_best={bfs_best:.4f}"
        extra += f" bfs_delta_mean={0.0 if dd_mean is None else dd_mean:.5f}"
        extra += f" closer/farther={closer_pct:.1f}%/{farther_pct:.1f}%"
        extra += f" bfs_missing={bfs_miss_pct:.1f}%"

        msg = (
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
        if ppo_logger is not None:
            ppo_logger.log(msg)
        else:
            print(msg)

    def _print_update_post(step_i: int, dt_s: float):
        if not ddp.is_main_process():
            return
        msg = f"[PPO][end]   step={step_i} update_seconds={dt_s:.3f}"
        if ppo_logger is not None:
            ppo_logger.log(msg)
        else:
            print(msg)

    timeouts = 0
    restarts = 0
    bfs_best = float("inf")
    bfs_min_episode = float("inf")

    interrupted = False

    try:
        for local_step in range(1, MAX_STEPS + 1):
            _check_stop_or_raise("top_of_step")

            step = base_step + local_step

            # -------------------------
            # 1) Collect one transition (retry on env failures)
            # -------------------------
            while True:
                _check_stop_or_raise("collect_transition_loop")

                a_cpu, lp_cpu, v_cpu, action_np = _select_action(cur_id_map, cur_vec_obs)
                action_env = utilities._format_action_for_env(action_np, action_space)

                try:
                    obs, raw_reward_any, done, info = env.step(action_env)
                    _dbg_push(step, cur_id_map, cur_vec_obs, action_np, raw_reward_any, done, info)
                    break
                except Exception as e:
                    if excs._is_step_after_done_exc(e):
                        if ddp.is_main_process():
                            print(f"[Env][step] step-after-done -> reset and retry: {repr(e)}", flush=True)
                        obs = _safe_reset()
                        cur_id_map, cur_vec_obs, _ = _make_map_and_vec_checked(obs)
                        shaper.reset(cur_vec_obs, cur_id_map)
                        episode_reward = 0.0
                        episode_raw_reward = 0.0
                        episode_len = 0
                        episode_idx += 1
                        continue

                    if excs._is_restartable_exc(e):
                        timeouts += int(excs._is_timeout_exc(e))
                        restarts += 1
                        if ddp.is_main_process():
                            print(f"[Env][step] exception={repr(e)} -> restart (timeouts={timeouts}, restarts={restarts})", flush=True)
                        env, port = _restart_env(reason="step")
                        stopper.sleep_poll(0.5)
                        _check_stop_or_raise("post_restart_before_reset")
                        obs = _safe_reset()
                        cur_id_map, cur_vec_obs, _ = _make_map_and_vec_checked(obs)
                        shaper.reset(cur_vec_obs, cur_id_map)
                        continue

                    raise

            if interrupted:
                break

            # -------------------------
            # 2) Post-step processing (must NOT leave buffer half-written)
            # -------------------------
            post_fail_local = 0
            try:
                # Normalize reward robustly (some wrappers return array/list/tensor).
                raw_r = _reward_scalar(raw_reward_any)

                # Ensure info is always a dict BEFORE using it anywhere.
                if info is None:
                    info = {}
                elif not isinstance(info, dict):
                    info = {"env_info": info}

                done_env = bool(_infer_done(done, info))

                # ------------------------------------------------------------
                # Guard 1) obs is None/malformed -> never parse, treat as terminal
                # ------------------------------------------------------------
                if obs is None:
                    info["obs_none"] = True
                    next_id_map = None
                    next_vec_obs = np.array(cur_vec_obs, copy=True)  # keep shapes stable
                    done_env = True
                    _dbg_dump("obs_none")
                else:
                    next_id_map, next_vec_obs, _ = _make_map_and_vec_checked(obs)

                    # Guard 2) vec missing -> treat as terminal (keep shapes stable)
                    if next_vec_obs is None:
                        info["vec_none"] = True
                        next_vec_obs = np.array(cur_vec_obs, copy=True)
                        done_env = True

                    # Guard 3) map missing -> treat as terminal (prevents later BFS/parse crashes)
                    if next_id_map is None:
                        info["map_missing"] = True
                        done_env = True
                        _dbg_dump("map_missing")

                # Goal -> fold into done_env to keep terminal semantics consistent.
                is_goal = (float(raw_r) >= float(args.success_if_raw_reward_ge))
                if is_goal:
                    info["is_goal"] = True
                done_env = bool(done_env) or bool(is_goal)

                # Call shaper safely: allow next_id_map=None and rely on RewardShaper fallback.
                shaped_reward = float(shaper(float(raw_r), next_vec_obs, next_id_map, bool(done_env), info))
                done_for_buffer = (bool(done_env) or bool(shaper.virtual_done))

                # Debug: BFS progress stats
                if getattr(shaper, "dbg_has_bfs", False) and (shaper.dbg_bfs_dist is not None):
                    d = float(shaper.dbg_bfs_dist)
                    bfs_n += 1
                    bfs_sum += d
                    bfs_min = min(bfs_min, d)
                    bfs_max = max(bfs_max, d)
                    bfs_best = min(bfs_best, d)

                    bfs_min_episode = min(bfs_min_episode, d)
                else:
                    bfs_missing += 1

                if shaper.dbg_bfs_delta is not None:
                    ddv = float(shaper.dbg_bfs_delta)
                    dd_sum += ddv
                    dd_n += 1
                    if ddv > 0.0:
                        closer_n += 1
                    elif ddv < 0.0:
                        farther_n += 1

                if args.reward_scaling:
                    scaled_reward = float(reward_scaler(shaped_reward, bool(done_for_buffer)))
                else:
                    scaled_reward = float(shaped_reward)

                # Now append to buffer (atomic w.r.t. exceptions)
                ppo_agent.buffer.add_state(cur_id_map, cur_vec_obs)
                ppo_agent.buffer.actions.append(a_cpu)
                ppo_agent.buffer.logprobs.append(lp_cpu)
                ppo_agent.buffer.state_values.append(v_cpu)
                ppo_agent.buffer.rewards.append(float(scaled_reward))
                ppo_agent.buffer.is_terminals.append(bool(done_for_buffer))

                _update_counters(float(raw_r), bool(done_for_buffer), float(shaped_reward), float(scaled_reward))

                episode_reward += float(scaled_reward)
                episode_raw_reward += float(raw_r)
                episode_len += 1

                # Dump trace around goal on ALL ranks (rare event)
                if is_goal:
                    _dbg_dump("goal")

                if is_goal and ddp.is_main_process():
                    msg = (
                        f"[GOAL] step={step} raw={float(raw_r):.10f} shaped={shaped_reward:.3f} scaled={scaled_reward:.3f} "
                        f"done_env={int(done_env)} done_buf={int(done_for_buffer)} virtual_done={int(bool(shaper.virtual_done))} "
                        f"ep_len={episode_len} ep_reward_running={episode_reward:.3f} "
                        f"clip={getattr(cfg,'clip',None)} w_time={args.w_time} w_prog={args.w_progress} termB={args.terminal_bonus}"
                    )
                    if ppo_logger is not None:
                        ppo_logger.log(msg)
                    else:
                        print(msg)

            except Exception as e:
                post_fail_local = 1
                # Per-rank fatal log + dump trace (DO NOT try per-rank recovery; it can desync DDP).
                try:
                    fatal_path = os.path.join(dbg_dir, f"fatal_rank{rank}_step{int(step)}.log")
                    with open(fatal_path, "w", encoding="utf-8") as f:
                        f.write(f"rank={rank} local_rank={local_rank} step={int(step)}\n")
                        f.write(f"exception={repr(e)}\n\n")
                        f.write(traceback.format_exc())
                    print(f"[FATAL] rank={rank} wrote {fatal_path}", flush=True)
                except Exception:
                    pass
                try:
                    _dbg_dump("post_step_fatal")
                except Exception:
                    pass

                # Sync post-step fatal across ranks; if any rank failed, stop all ranks together.
            if post_fail_local:
                # If any rank hits a fatal error, exit that rank immediately.
                # torchrun will tear down the other ranks.
                interrupted = True 
                if ddp.is_main_process():
                    print(f"[FATAL] post-step failed on rank={rank} step={int(step)} -> exiting", flush=True)
                os._exit(1)

            # -------------------------
            # PPO update (ALL ranks)
            # -------------------------

            # If stopping, skip updates entirely (updates are long and prone to divergence on interrupt).
            if step % update_timestep == 0:
                _check_stop_or_raise("pre_update")

                # Hard barriers around updates are dangerous in this env (ranks can diverge on reset/restart).
                # Keep them off by default; even when enabled, use only soft barrier.
                if use_update_barriers and (not stopper.should_stop()):
                    ddp._ddp_barrier_soft(f"pre_update@{int(step)}", timeout_s=5.0)
                t0 = time.time()

                # Computing last_value must be robust even if next_id_map is None.
                with torch.no_grad():
                    try:
                        if done_for_buffer or (next_id_map is None) or (next_vec_obs is None):
                            ppo_agent.buffer.last_value = 0.0
                        else:
                            map_t = torch.from_numpy(next_id_map).to(model_device, dtype=torch.long).unsqueeze(0)
                            vec_t = torch.from_numpy(next_vec_obs).to(model_device, dtype=torch.float32).unsqueeze(0)
                            po = ddp.get_module(ppo_agent.policy_old)
                            feat = po.encode(map_t, vec_t)
                            v = po.critic_head(feat)
                            ppo_agent.buffer.last_value = float(v.item())
                    except Exception:
                        # Never crash only one rank here; fall back to 0.
                        ppo_agent.buffer.last_value = 0.0

                if args.dump_id_map and ddp.is_main_process():
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
                _reset_update_counters()
                if use_update_barriers and (not stopper.should_stop()):
                    ddp._ddp_barrier_soft(f"post_update@{int(step)}", timeout_s=5.0)

            # Action std decay
            if has_continuous_action_space and (step % action_std_decay_freq == 0):
                if hasattr(ppo_agent, "decay_action_std"):
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
                    if wandb is not None:
                        wandb.log({"train/action_std": float(getattr(ppo_agent, "action_std", np.nan))}, step=int(step))

            # Save checkpoint (rank0 only)
            if (step % save_model_freq == 0) and ddp.is_main_process():
                _check_stop_or_raise("pre_checkpoint_save")
                ckpt_path = os.path.join(ckpt_dir, f"{ckpt_prefix}{step}.pth")

                if ppo_logger is not None:
                    ppo_logger.log("-" * 92)
                    ppo_logger.log(f"saving model at: {ckpt_path}")
                else:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at:", ckpt_path)

                # IMPORTANT: do NOT ignore signals here. If user Ctrl+C during save,
                # we prefer to abort promptly rather than "hang and feel unkillable".
                save.atomic_torch_save(
                    lambda: save._make_checkpoint(ppo_agent, step, args, cfg, reward_scaler),
                    ckpt_path,
                )

                if ppo_logger is not None:
                    ppo_logger.log("-" * 92)
                    ppo_logger.log(f"saving model at: {ckpt_path}")
                else:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at:", ckpt_path)

                if wandb is not None:
                    wandb.log({"train/checkpoint_saved": 1}, step=int(step))
                    wandb.save(ckpt_path)

            # Lightweight wandb logs (rank0 only)
            if wandb is not None and ((step % args.wandb_log_freq == 0) or done_for_buffer or is_goal):
                wandb.log(
                    {
                        "train/reward": float(scaled_reward),
                        "train/raw_reward": float(raw_r),
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
                bfs_min_episode_log = None if not np.isfinite(bfs_min_episode) else float(bfs_min_episode)
                if wandb is not None:
                    wandb.log(
                        {
                            "episode/reward": float(episode_reward),
                            "episode/raw_reward": float(episode_raw_reward),
                            "episode/len": int(episode_len),
                            "episode/index": int(episode_idx),
                            "episode/bfs_min": float(bfs_min_episode_log),
                        },
                        step=int(step),
                    )

                episode_idx += 1
                _check_stop_or_raise("pre_episode_reset")
                obs = _safe_reset()
                cur_id_map, cur_vec_obs, _ = _make_map_and_vec_checked(obs)
                shaper.reset(cur_vec_obs, cur_id_map)

                episode_reward = 0.0
                episode_raw_reward = 0.0
                episode_len = 0
                bfs_min_episode = float("inf")
            else:
                cur_id_map, cur_vec_obs = next_id_map, next_vec_obs

            # Step-level DDP synchronization
            if ddp_step_sync_every > 0 and (local_step % ddp_step_sync_every == 0) and (not stopper.should_stop()):
                ddp._ddp_barrier(f"step_sync@{int(step)}", ddp_barrier_timeout_s)

    except _StopTraining:
        interrupted = True
        if ddp.is_main_process():
            print("\n[STOP] requested stop -> unwinding to finally", flush=True)

    except KeyboardInterrupt:
        interrupted = True
        if ddp.is_main_process():
            print("\n[KeyboardInterrupt] requested stop (will save in finally).")

    except Exception as e:
        # Always write per-rank fatal log + dump traces.
        try:
            fatal_path = os.path.join(dbg_dir, f"fatal_rank{rank}_step{int(step)}.log")
            with open(fatal_path, "w", encoding="utf-8") as f:
                f.write(f"rank={rank} local_rank={local_rank} step={int(step)}\n")
                f.write(f"exception={repr(e)}\n\n")
                f.write(traceback.format_exc())
            print(f"[FATAL] rank={rank} wrote {fatal_path}", flush=True)
        except Exception:
            pass
        try:
            _dbg_dump("fatal_outer")
        except Exception:
            pass

        if ddp.is_main_process():
            print(f"\n[FATAL] exception={repr(e)}")
        dbg._print_player_log_tail(prefix="[FATAL]", n=int(args.player_log_tail))
        raise

    finally:
        # Final save intent (we prioritize exiting cleanly; saving is best-effort)
        force_exit = bool(interrupted) or bool(getattr(stopper, "stop_requested", False))

        # Decide final step/path early (used by flags + logging)
        try:
            final_step = int(step) if int(step) > 0 else int(base_step)
        except Exception:
            final_step = int(base_step) if "base_step" in locals() else 0

        try:
            final_path = os.path.join(ckpt_dir, f"{ckpt_prefix}{final_step}.pth")
        except Exception:
            # if ckpt_dir/ckpt_prefix not defined somehow
            final_path = None

        # ---- rendezvous flags (file-based, no deadlock) ----
        # Keep them in ckpt_dir so you can inspect after crashes.
        save_flag = None
        start_flag = None
        if "ckpt_dir" in locals() and (ckpt_dir is not None) and (final_step is not None):
            save_flag = os.path.join(ckpt_dir, f".final_save_done_step{final_step}.rank0")
            start_flag = os.path.join(ckpt_dir, f".final_save_start_step{final_step}.rank0")

        # (A) short soft barrier: align ranks entering finally (optional, never blocks forever)
        if ddp.ddp_is_enabled():
            try:
                ddp._ddp_barrier_soft("enter_finally", timeout_s=5.0)
            except Exception:
                pass

        # (B) rank0 touches "start" flag so other ranks can decide to wait
        if ddp.is_main_process() and start_flag is not None:
            _touch_file(start_flag)

        # (C) no barrier required; file flags are enough (barriers can still deadlock on divergent ranks).
        # (D) rank0 save (BEST-EFFORT)
        if ddp.is_main_process():
            try:
                if final_path is not None:
                    if interrupted:
                        print("[Exit] stop requested -> saving final checkpoint:", final_path, flush=True)
                    # Do NOT ignore signals: user wants reliable exit over guaranteed save.
                    save.atomic_torch_save(
                        lambda: save._make_checkpoint(ppo_agent, final_step, args, cfg, reward_scaler),
                        final_path,
                    )
                    print("Final model saved at:", final_path, flush=True)
                else:
                    print("[Exit][WARN] final_path is None; skipping final checkpoint save", flush=True)

                # touch done flag only if saving did not throw
                if save_flag is not None:
                    _touch_file(save_flag)

            except Exception as e:
                print(f"[Exit][WARN] failed to save final checkpoint: {repr(e)}", flush=True)

        # Rank0 wandb finish AFTER checkpoint (still in shutdown path). Keep best-effort.
        if ddp.is_main_process():
            if ("wandb" in locals()) and (wandb is not None):
                try:
                    with stopper.ignore_signals():
                        wandb.finish()
                except Exception:
                    pass

        # (E) non-rank0 waits (SHORT). Prefer exiting fast over perfect rendezvous.
        if (not ddp.is_main_process()):
            try:
                if start_flag is not None:
                    _wait_for_file(start_flag, timeout_s=2.0)
                if save_flag is not None:
                    _wait_for_file(save_flag, timeout_s=10.0)
            except Exception:
                pass

        # (F) final soft barrier (optional) - if it works, great; if not, never deadlocks
        if ddp.ddp_is_enabled():
            try:
                ddp._ddp_barrier_soft("final_save_done", timeout_s=30.0)
            except Exception:
                pass

        # ---- best-effort cleanup ----
        # On interrupt, Unity can hang. Close env only best-effort, and NEVER restart it.
        try:
            if "env" in locals() and env is not None:
                env.close()
        except Exception:
            pass

        # ---- end-of-run logging (preserve original behavior) ----
        end_time = datetime.now().replace(microsecond=0)
        if ddp.is_main_process():
            try:
                if "ppo_logger" in locals() and (ppo_logger is not None):
                    ppo_logger.log("=" * 92)
                    ppo_logger.log(f"Started training at: {start_time}")
                    ppo_logger.log(f"Finished training at: {end_time}")
                    ppo_logger.log(f"Total training time: {end_time - start_time}")
                    ppo_logger.log(f"[Env][FINAL] timeouts={timeouts} restarts={restarts} last_port={last_port_box['port']}")
                    ppo_logger.log("=" * 92)
                else:
                    print("============================================================================================")
                    print("Started training at:", start_time)
                    print("Finished training at:", end_time)
                    print("Total training time:", end_time - start_time)
                    try:
                        print(f"[Env][FINAL] timeouts={timeouts} restarts={restarts} last_port={last_port_box['port']}")
                    except Exception:
                        pass
                    print("============================================================================================")
            except Exception:
                # never let logging crash shutdown
                pass

        if "ppo_logger" in locals() and (ppo_logger is not None):
            try:
                ppo_logger.close()
            except Exception:
                pass

        # cleanup도 best-effort (force_exit여도 시도는 해보되 실패하면 무시)
        if ddp.ddp_is_enabled():
            try:
                ddp.ddp_cleanup()
            except Exception:
                pass

        # Always ensure buffers flushed
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        # If stop was requested: attempt to exit normally (let python unwind).
        # As last resort, if we're still here and force_requested was set, hard-exit.
        if force_exit and bool(getattr(stopper, "force_requested", False)):
            _hard_exit(130)

import libimmortal.samples.PPO.config as config

if __name__ == "__main__":
    args = config.build_argparser().parse_args()
    train(args)
