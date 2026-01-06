#!/usr/bin/env python3
import os
import glob
import time
import argparse
from datetime import datetime
from typing import Optional, Tuple
import re
import signal

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
    """
    Initialize torch.distributed process group.
    Launch with:
      torchrun --standalone --nproc_per_node=4 train_ddp.py ...
    """
    if not ddp_is_enabled():
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(ddp_local_rank())

    # NCCL for single-node multi-GPU
    dist.init_process_group(backend="nccl", init_method="env://")


def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def ddp_barrier():
    if dist.is_initialized():
        dist.barrier()


def seed_everything(seed: int):
    """
    Offset seed by global rank so each GPU explores different trajectories.
    """
    s = int(seed) + ddp_rank()
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_module(m):
    """
    If wrapped by DDP, return underlying module, else return itself.
    """
    return m.module if hasattr(m, "module") else m


def ddp_wrap_model(m: torch.nn.Module) -> torch.nn.Module:
    """
    Wrap module with DDP if enabled.
    """
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
    """
    Save checkpoints under:
      <this train.py directory>/checkpoints
    """
    return os.path.join(os.path.dirname(__file__), "checkpoints")


def _latest_checkpoint_path(ckpt_dir: str, prefix: str) -> Optional[str]:
    """Return latest checkpoint matching prefix (by mtime), or None."""
    paths = glob.glob(os.path.join(ckpt_dir, f"{prefix}*.pth"))
    if not paths:
        return None
    paths.sort(key=os.path.getmtime, reverse=True)
    return paths[0]

def _infer_step_from_ckpt_path(ckpt_path: Optional[str], ckpt_prefix: str) -> int:
    """
    Expect: <ckpt_prefix><step>.pth
    Example: PPO_ImmortalSufferingEnv_seed42_4000.pth -> 4000
    Return 0 if cannot parse.
    """
    if not ckpt_path:
        return 0
    base = os.path.basename(ckpt_path)
    m = re.match(re.escape(ckpt_prefix) + r"(\d+)\.pth$", base)
    return int(m.group(1)) if m else 0

def _infer_done(done_from_env, info) -> bool:
    """Combine env's done with ML-Agents TerminalSteps signal (if present)."""
    done = bool(done_from_env)
    if TerminalSteps is None:
        return done

    if isinstance(info, dict):
        step_obj = info.get("step", None) or info.get("mlagents_step", None)
        if step_obj is not None and isinstance(step_obj, TerminalSteps):
            return True
    return done


def _get_action_space(env):
    """
    ImmortalSufferingEnv sometimes wraps an inner gym env.
    Try common attributes.
    """
    if hasattr(env, "action_space"):
        return env.action_space
    if hasattr(env, "env") and hasattr(env.env, "action_space"):
        return env.env.action_space
    raise AttributeError("Cannot find action_space on env.")

# Helper function

'''
def save_checkpoint_safely(state_dict, path: str):
    """
    Save checkpoint while temporarily ignoring SIGINT so Ctrl+C doesn't corrupt the save.
    """
    old = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        torch.save(state_dict, path)
    finally:
        signal.signal(signal.SIGINT, old)
'''

def _rollback_last_rollout_entry(buf):
    """
    Undo the last (state, action, logprob, value) appended BEFORE env.step().
    Keeps buffer lengths consistent when env.step() fails.
    """
    if getattr(buf, "map_states", None) and len(buf.map_states) > 0:
        buf.map_states.pop()
    if getattr(buf, "vec_states", None) and len(buf.vec_states) > 0:
        buf.vec_states.pop()
    if getattr(buf, "actions", None) and len(buf.actions) > 0:
        buf.actions.pop()
    if getattr(buf, "logprobs", None) and len(buf.logprobs) > 0:
        buf.logprobs.pop()
    if getattr(buf, "state_values", None) and len(buf.state_values) > 0:
        buf.state_values.pop()

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

def atomic_torch_save_state_dict(make_sd_fn, path: str):
    """
    Save to temp then os.replace for atomic commit.
    Prevents broken checkpoints if the job dies mid-write.
    """
    tmp = path + ".tmp"
    sd = make_sd_fn()
    torch.save(sd, tmp)
    os.replace(tmp, path)

# -------------------------
# Observation preprocessing
# -------------------------
def _make_map_and_vec(obs) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Return:
      id_map: (H,W) uint8 (IDs)
      vec_obs: (103,) float32
      K: number of IDs in palette (num_ids)
    """
    from libimmortal.utils import colormap_to_ids_and_onehot, parse_observation

    graphic_obs, vector_obs = parse_observation(obs)

    # id_map: (H,W) uint8, onehot: (K,H,W) uint8
    id_map, onehot = colormap_to_ids_and_onehot(graphic_obs)

    id_map = np.asarray(id_map, dtype=np.uint8)
    vec_obs = np.asarray(vector_obs, dtype=np.float32).reshape(-1)

    K = int(onehot.shape[0])
    return id_map, vec_obs, K


def _format_action_for_env(action_np, action_space):
    # Make action compatible with gym spaces before env.step()
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
    from libimmortal.env import ImmortalSufferingEnv

    # Optional: Unity timeout exception type
    try:
        from mlagents_envs.exception import UnityTimeOutException  # type: ignore
    except Exception:
        UnityTimeOutException = None  # type: ignore

    import os
    import time
    import numpy as np
    import gym
    import torch
    import torch.distributed as dist
    from datetime import datetime

    # IMPORTANT: DDP must be setup BEFORE creating PPO / any CUDA tensors
    ddp_setup()

    rank = ddp_rank()
    local_rank = ddp_local_rank()
    world = ddp_world_size()

    # Each rank must use a unique *PORT RANGE* (Unity uses multiple ports internally).
    # Use a stride to avoid collisions across ranks and restarts.
    port_stride = int(getattr(args, "port_stride", 10))
    max_env_restarts = int(getattr(args, "max_env_restarts", 3))
    no_graphics = bool(getattr(args, "no_graphics", True))
    unity_timeout_wait = int(getattr(args, "unity_timeout_wait", 60))

    # Use a mutable box so nested fns can read/write without Python scoping issues.
    restart_box = {"id": 0}

    def _infer_step_from_ckpt_path(ckpt_path: str) -> int:
        """
        Infer step from checkpoint filename like:
          .../PPO_ImmortalSufferingEnv_seed42_69198.pth  -> 69198
        Returns 0 if not parsable.
        """
        import re
        m = re.search(r"_(\d+)\.pth$", str(ckpt_path))
        return int(m.group(1)) if m else 0

    def _compute_port() -> int:
        return (
            int(args.port)
            + int(local_rank) * port_stride
            + int(restart_box["id"]) * port_stride * int(world)
        )

    def _make_env():
        p = _compute_port()
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
            return ImmortalSufferingEnv(**kw, no_graphics=no_graphics, timeout_wait=unity_timeout_wait), p
        except TypeError:
            return ImmortalSufferingEnv(**kw), p

    class _SigintGuard:
        """
        Prevent repeated SIGINT from interrupting the critical save section.
        """
        def __enter__(self):
            import signal
            self._old = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            return self

        def __exit__(self, exc_type, exc, tb):
            import signal
            signal.signal(signal.SIGINT, self._old)
            return False

    # Seeds (offset by rank)
    if args.seed is not None:
        seed_everything(int(args.seed))

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
    # base_step: already-trained steps (per-rank) inferred from checkpoint filename
    # step:      running step counter (per-rank) used for logging/saving/scheduling
    base_step = 0
    step = 0  # always defined for finally

    # Infer dims from first reset
    obs = env.reset()
    id_map, vec_obs, K = _make_map_and_vec(obs)

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

    vec_dim = int(vec_obs.shape[0])
    num_ids = int(K)

    if is_main_process():
        print(f"[DDP] world_size={world}, rank={rank}, local_rank={local_rank}")
        print(f"[Env] base_port={args.port}, this_rank_port={port}")
        print(f"[Env] num_ids(K)={num_ids}, vec_dim={vec_dim}, action_dim={action_dim}, continuous={has_continuous_action_space}")

    # PPO agent
    ppo_agent = PPO(
        num_ids, vec_dim, action_dim,
        lr_actor, lr_critic,
        gamma, K_epochs, eps_clip,
        has_continuous_action_space,
        action_std_init=action_std,
        mini_batch_size=mini_batch_size,
        action_nvec=action_nvec,
    )

    # DDP: wrap ppo_agent.policy (NOT policy_old) and patch the interface so PPO.update() can keep using
    #   - self.policy.evaluate(...)
    #   - self.policy.state_dict() / load_state_dict(...)
    # without breaking due to "DistributedDataParallel has no attribute evaluate" and "module." prefix keys.
    if hasattr(ppo_agent, "policy") and ddp_is_enabled():
        ddp_pol = ddp_wrap_model(ppo_agent.policy)
        base_pol = get_module(ddp_pol)

        # Make DDP forward run the same graph as evaluate (so DDP reducer is prepared correctly).
        # Then expose .evaluate() on the DDP wrapper to route through ddp_pol(...).
        if hasattr(base_pol, "evaluate"):
            if not hasattr(base_pol, "_orig_forward"):
                base_pol._orig_forward = base_pol.forward
            base_pol.forward = base_pol.evaluate

            def _ddp_evaluate(mb_map, mb_vec, mb_actions):
                return ddp_pol(mb_map, mb_vec, mb_actions)

            ddp_pol.evaluate = _ddp_evaluate  # type: ignore

        # Patch state_dict/load_state_dict to return/load *module* keys (no "module." prefix),
        # so PPO.update() internal "policy_old.load_state_dict(policy.state_dict())" won't break.
        def _state_dict_no_prefix(*args, **kwargs):
            return base_pol.state_dict(*args, **kwargs)

        def _load_state_dict_no_prefix(state_dict, *args, **kwargs):
            return base_pol.load_state_dict(state_dict, *args, **kwargs)

        ddp_pol.state_dict = _state_dict_no_prefix  # type: ignore
        ddp_pol.load_state_dict = _load_state_dict_no_prefix  # type: ignore

        ppo_agent.policy = ddp_pol
    
    from libimmortal.utils.aux_func import DEFAULT_ENCODER

    ENC = DEFAULT_ENCODER

    WALL_ID = ENC.name2id["WALL"]
    GOAL_ID = ENC.name2id["GOAL"]

    # Player marker on minimap
    PLAYER_IDS = [ENC.name2id["KNIGHT"], ENC.name2id["KNIGHT_ATTACK"]]

    # By default, only WALL blocks movement in BFS.
    DEFAULT_BLOCKED_IDS = [WALL_ID]

    cfg = RewardConfig(
        w_progress=args.w_progress,
        w_time=args.w_time,
        w_damage=args.w_damage,
        w_not_actionable=args.w_not_actionable,
        terminal_failure_penalty=args.terminal_failure_penalty,
        terminal_bonus=args.terminal_bonus,
        clip=args.reward_clip,
        success_if_raw_reward_ge=args.success_if_raw_reward_ge,
        time_limit=args.time_limit,
        success_speed_bonus=args.success_speed_bonus,
        use_bfs_progress=True,  
        w_acceleration=1.0,     # 자석 효과 강도
        bfs_update_every=10     # 갱신 주기
    )
    shaper = RewardShaper(cfg)
    shaper.reset(vec_obs, id_map)

    # Reward scaler (do NOT create inside loop)
    reward_scaler = RewardScaler(gamma=gamma, clip=5.0, eps=1e-8, min_std=0.1, warmup_steps=10000)

    # Checkpoints (only rank0 writes)
    ckpt_dir = _checkpoint_dir()
    if is_main_process():
        os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_prefix = f"PPO_{env_tag}_seed{int(args.seed) if args.seed is not None else 0}_"
    if is_main_process():
        print("checkpoint dir:", ckpt_dir)

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

        base_step = _infer_step_from_ckpt_path(str(ckpt_to_load))
        step = base_step

        if is_main_process():
            print(f"[Resume] loading: {ckpt_to_load}")
            print(f"[Resume] inferred base_step={base_step}")

        state = torch.load(ckpt_to_load, map_location="cpu")
        get_module(ppo_agent.policy).load_state_dict(state)
        get_module(ppo_agent.policy_old).load_state_dict(state)
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
            ),
        )

    # -------------------------
    # Runner-style main loop
    # -------------------------
    start_time = datetime.now().replace(microsecond=0)
    if is_main_process():
        print("Started training at (GMT):", start_time)
        print("============================================================================================")

    episode_reward = 0.0
    episode_raw_reward = 0.0
    episode_len = 0
    episode_idx = 0

    cur_id_map = id_map
    cur_vec_obs = vec_obs

    # Determine device from policy_old params
    model_device = next(ppo_agent.policy_old.parameters()).device

    def _select_action(cur_map_np: np.ndarray, cur_vec_np: np.ndarray):
        """
        Runs policy_old.act (no grad), returns:
          - a_cpu: action tensor on CPU
          - lp_cpu: logprob tensor on CPU
          - v_cpu: value tensor on CPU
          - action_np: numpy action for env
        NOTE: We do NOT write into buffer here. We only write after env.step() succeeds.
        """
        map_t = torch.from_numpy(cur_map_np).to(model_device, dtype=torch.long).unsqueeze(0)
        vec_t = torch.from_numpy(cur_vec_np).to(model_device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_t, logp_t, value_t = get_module(ppo_agent.policy_old).act(map_t, vec_t)

        a_cpu = action_t.squeeze(0).detach().cpu()
        lp_cpu = logp_t.squeeze(0).detach().cpu()
        v_cpu = value_t.squeeze(0).detach().cpu()
        action_np = a_cpu.numpy()
        return a_cpu, lp_cpu, v_cpu, action_np

    try:
        for local_step in range(1, MAX_STEPS + 1):
            step = base_step + local_step  # per-rank cumulative step

            # Retry loop: Unity can time out; restart env on this rank and retry this step.
            while True:
                a_cpu, lp_cpu, v_cpu, action_np = _select_action(cur_id_map, cur_vec_obs)
                action_env = _format_action_for_env(action_np, action_space)

                try:
                    obs, raw_reward, done, info = env.step(action_env)

                    # ✅ Only after step succeeds, we append to buffer (prevents DDP buffer-length mismatch)
                    ppo_agent.buffer.add_state(cur_id_map, cur_vec_obs)
                    ppo_agent.buffer.actions.append(a_cpu)
                    ppo_agent.buffer.logprobs.append(lp_cpu)
                    ppo_agent.buffer.state_values.append(v_cpu)

                    break
                except Exception as e:
                    is_timeout = (UnityTimeOutException is not None) and isinstance(e, UnityTimeOutException)
                    if is_timeout:
                        try:
                            env.close()
                        except Exception:
                            pass

                        restart_box["id"] += 1
                        if restart_box["id"] > max_env_restarts:
                            raise

                        env, port = _make_env()
                        obs = env.reset()
                        cur_id_map, cur_vec_obs, _ = _make_map_and_vec(obs)
                        shaper.reset(cur_vec_obs, cur_id_map)
                        continue

                    raise

            done = _infer_done(done, info)
            next_id_map, next_vec_obs, _ = _make_map_and_vec(obs)

            if info is None:
                info = {}
            elif not isinstance(info, dict):
                info = {"env_info": info}

            reward = shaper(raw_reward, next_vec_obs, next_id_map, done, info)
            reward = reward_scaler(reward, done)
            done = done or shaper.virtual_done

            ppo_agent.buffer.rewards.append(float(reward))
            ppo_agent.buffer.is_terminals.append(bool(done))

            episode_reward += float(reward)
            episode_raw_reward += float(raw_reward)
            episode_len += 1

            # PPO update (ALL ranks call update; DDP will sync gradients)
            if step % update_timestep == 0:
                ddp_barrier()
                t0 = time.time()

                with torch.no_grad():
                    if done:
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

                ppo_agent.update(0.95)  # GAE lambda
                dt = time.time() - t0

                if wandb is not None:
                    wandb.log(
                        {
                            "train/updated": 1,
                            "train/update_seconds": float(dt),
                            "train/buffer_len": int(update_timestep),
                            "train/step": int(step),
                            "train/global_env_steps": int(step * world),
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

            # Save checkpoint (rank0 only)
            if (step % save_model_freq == 0) and is_main_process():
                ckpt_path = os.path.join(ckpt_dir, f"{ckpt_prefix}{step}.pth")
                print("--------------------------------------------------------------------------------------------")
                print("saving model at:", ckpt_path)
                torch.save(get_module(ppo_agent.policy_old).state_dict(), ckpt_path)
                print("model saved")
                print("Elapsed Time:", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
                if wandb is not None:
                    wandb.log({"train/checkpoint_saved": 1}, step=int(step))
                    wandb.save(ckpt_path)

            # Lightweight wandb logs (rank0 only)
            if wandb is not None and (step % args.wandb_log_freq == 0):
                wandb.log(
                    {
                        "train/reward": float(reward),
                        "train/raw_reward": float(raw_reward),
                        "train/done": int(done),
                        "train/episode_reward_running": float(episode_reward),
                        "train/episode_raw_reward_running": float(episode_raw_reward),
                        "train/episode_len_running": int(episode_len),
                    },
                    step=int(step),
                )

            # Episode boundary
            if done or (episode_len >= max_ep_len):
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

                obs = env.reset()
                cur_id_map, cur_vec_obs, _ = _make_map_and_vec(obs)
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

    finally:
        # Final save (rank0 only) - atomic + guard against repeated SIGINT
        if is_main_process():
            with _SigintGuard():
                final_step = int(step) if int(step) > 0 else int(base_step)
                final_path = os.path.join(ckpt_dir, f"{ckpt_prefix}{final_step}.pth")
                tmp = final_path + ".tmp"
                torch.save(get_module(ppo_agent.policy_old).state_dict(), tmp)
                os.replace(tmp, final_path)
                print("Final model saved at:", final_path)

        # Don't hard-barrier in finally: if ranks die, barrier can hang.
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
            print("Started training at (GMT):", start_time)
            print("Finished training at (GMT):", end_time)
            print("Total training time:", end_time - start_time)
            print("============================================================================================")

        ddp_cleanup()


def build_argparser():
    p = argparse.ArgumentParser()

    # Immortal env args
    p.add_argument("--game_path", type=str, default=r"/root/immortal_suffering/immortal_suffering_linux_build.x86_64")
    # IMPORTANT: this is the BASE port; each rank uses (port + LOCAL_RANK)
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--port_stride", type=int, default=50)
    p.add_argument("--time_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--verbose", action="store_true")

    p.add_argument("--no_graphics", action="store_true")   # if env wrapper supports it
    p.add_argument("--unity_timeout_wait", type=int, default=120)  # if env wrapper supports it
    p.add_argument("--max_env_restarts", type=int, default=20)

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
    p.add_argument("--w_progress", type=float, default=40.0) # 20.0, 40.0, 60.0, 80.0
    p.add_argument("--w_time", type=float, default=0.01)
    p.add_argument("--w_damage", type=float, default=0.05)
    p.add_argument("--w_not_actionable", type=float, default=0.01)
    p.add_argument("--terminal_bonus", type=float, default=1.0)

    p.add_argument("--terminal_failure_penalty", type=float, default=3.0)
    p.add_argument("--reward_clip", type=float, default=8.0)
    p.add_argument("--success_if_raw_reward_ge", type=float, default=1.0)
    p.add_argument("--time_limit", type=float, default=300.0)
    p.add_argument("--success_speed_bonus", type=float, default=10.0)

    # Debug: dump id_map path (rank0 only, on update steps)
    p.add_argument("--dump_id_map", type=str, default=None)

    # wandb (rank0 only)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="ppo-immortal")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_log_freq", type=int, default=200)

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)

'''
How to run

# torchrun --standalone --nproc_per_node=4 ./src/libimmortal/samples/PPO/train.py \
  --port 5005 --port_stride 10 \
  --save_model_freq 20000 --wandb \
  --resume --checkpoint ./src/libimmortal/samples/PPO/checkpoints/PPO_ImmortalSufferingEnv_seed42_4000.pth

'''

'''
# torchrun --standalone --nproc_per_node=4  --log_dir /tmp/torchrun_logs  \
 ./src/libimmortal/samples/PPO/train.py   --port 5205   --wandb
'''