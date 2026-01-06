#!/usr/bin/env python3
import os
import glob
import time
import argparse
from datetime import datetime
from typing import Optional, Tuple

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

    # IMPORTANT: DDP must be setup BEFORE creating PPO / any CUDA tensors
    ddp_setup()

    rank = ddp_rank()
    local_rank = ddp_local_rank()
    world = ddp_world_size()

    # Each rank must use a unique port (Unity/ML-Agents style env will conflict otherwise)
    port = int(args.port) + int(local_rank) * int(args.port_stride)

    # Seeds (offset by rank)
    if args.seed is not None:
        seed_everything(int(args.seed))

    env = ImmortalSufferingEnv(
        game_path=args.game_path,
        port=port,
        time_scale=args.time_scale,
        seed=(int(args.seed) + rank) if args.seed is not None else None,
        width=args.width,
        height=args.height,
        verbose=args.verbose,
    )

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
    # NOTE: PPO.py must NOT hardcode cuda:0. It should use torch.device("cuda") or current device.
    ppo_agent = PPO(
        num_ids, vec_dim, action_dim,
        lr_actor, lr_critic,
        gamma, K_epochs, eps_clip,
        has_continuous_action_space,
        action_std_init=action_std,
        mini_batch_size=mini_batch_size,
        action_nvec=action_nvec,
    )

    # Wrap BOTH policy and policy_old with DDP so PPO.update() internal weight copying won't break.
    if hasattr(ppo_agent, "policy") and hasattr(ppo_agent, "policy_old") and ddp_is_enabled():
        ppo_agent.policy = ddp_wrap_model(ppo_agent.policy)

    # Reward shaping
    cfg = RewardConfig(
        w_progress=args.w_progress,
        w_time=args.w_time,
        w_damage=args.w_damage,
        w_not_actionable=args.w_not_actionable,
        terminal_failure_penalty=args.terminal_failure_penalty,
        clip=args.reward_clip,
        success_if_raw_reward_ge=args.success_if_raw_reward_ge,
        time_limit=args.time_limit,
        success_speed_bonus=args.success_speed_bonus,
    )
    shaper = RewardShaper(cfg)
    shaper.reset(vec_obs, id_map)

    # Reward scaler (FIX: do NOT create this inside the loop)
    reward_scaler = RewardScaler(gamma=gamma, clip=5.0, eps=1e-8, min_std=0.1, warmup_steps=10000)

    # Checkpoints (only rank0 writes)
    ckpt_dir = _checkpoint_dir()
    if is_main_process():
        os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_prefix = f"PPO_{env_tag}_seed{int(args.seed) if args.seed is not None else 0}_"
    if is_main_process():
        print("checkpoint dir:", ckpt_dir)

    # Resume (every rank loads the same weights, then barrier)
    if args.resume:
        ckpt_to_load = args.checkpoint
        if ckpt_to_load is None and is_main_process():
            ckpt_to_load = _latest_checkpoint_path(ckpt_dir, prefix=ckpt_prefix)
        if ddp_is_enabled():
            # Broadcast chosen path from rank0 to all ranks
            obj_list = [ckpt_to_load]
            dist.broadcast_object_list(obj_list, src=0)
            ckpt_to_load = obj_list[0]

        if ckpt_to_load is None:
            raise FileNotFoundError(f"--resume set but no checkpoint found under {ckpt_dir} with prefix {ckpt_prefix}")

        if is_main_process():
            print(f"[Resume] loading: {ckpt_to_load}")

        state = torch.load(ckpt_to_load, map_location="cpu")
        get_module(ppo_agent.policy).load_state_dict(state)
        get_module(ppo_agent.policy_old).load_state_dict(state)
        ddp_barrier()

    # Optional wandb (only rank0)
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

    def _select_action_and_store(map_np: np.ndarray, vec_np: np.ndarray) -> np.ndarray:
        """
        Runs policy_old.act on GPU (or CPU), stores (state, action, logprob, value) in buffer,
        returns action as numpy.
        """
        map_t = torch.from_numpy(map_np).to(model_device, dtype=torch.long).unsqueeze(0)
        vec_t = torch.from_numpy(vec_np).to(model_device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_t, logp_t, value_t = ppo_agent.policy_old.act(map_t, vec_t)

        # Store states as CPU numpy (buffer expects numpy)
        ppo_agent.buffer.add_state(map_np, vec_np)

        # Store these as CPU tensors to avoid growing GPU memory during rollout
        ppo_agent.buffer.actions.append(action_t.squeeze(0).detach().cpu())
        ppo_agent.buffer.logprobs.append(logp_t.squeeze(0).detach().cpu())
        ppo_agent.buffer.state_values.append(value_t.squeeze(0).detach().cpu())

        action_np = action_t.squeeze(0).detach().cpu().numpy()
        return action_np

    step = 0
    try:
        for step in range(1, MAX_STEPS + 1):
            action = _select_action_and_store(cur_id_map, cur_vec_obs)
            action = _format_action_for_env(action, action_space)

            obs, raw_reward, done, info = env.step(action)
            done = _infer_done(done, info)

            next_id_map, next_vec_obs, _ = _make_map_and_vec(obs)

            if info is None:
                info = {}
            elif not isinstance(info, dict):
                info = {"env_info": info}

            reward = shaper(raw_reward, next_vec_obs, next_id_map, done, info)
            reward = reward_scaler(reward, done)

            ppo_agent.buffer.rewards.append(float(reward))
            ppo_agent.buffer.is_terminals.append(bool(done))

            episode_reward += float(reward)
            episode_raw_reward += float(raw_reward)
            episode_len += 1

            # PPO update (ALL ranks call update; DDP will sync gradients)
            if step % update_timestep == 0:
                t0 = time.time()
                with torch.no_grad():
                    if done:
                        ppo_agent.buffer.last_value = 0.0
                    else:
                        map_t = torch.from_numpy(next_id_map).to(model_device, dtype=torch.long).unsqueeze(0)
                        vec_t = torch.from_numpy(next_vec_obs).to(model_device, dtype=torch.float32).unsqueeze(0)
                        feat = ppo_agent.policy_old.encode(map_t, vec_t)
                        v = ppo_agent.policy_old.critic_head(feat)  # (1,1)
                        ppo_agent.buffer.last_value = float(v.item())

                # NOTE: Avoid heavy I/O from all ranks
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
                        },
                        step=step,
                    )

                # Keep all ranks aligned after update
                # ddp_barrier()

            # Action std decay
            if has_continuous_action_space and (step % action_std_decay_freq == 0):
                if hasattr(ppo_agent, "decay_action_std"):
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
                    if wandb is not None:
                        wandb.log({"train/action_std": float(getattr(ppo_agent, "action_std", np.nan))}, step=step)

            # Save checkpoint (rank0 only; save underlying module for portability)
            if step % save_model_freq == 0 and is_main_process():
                ckpt_path = os.path.join(ckpt_dir, f"{ckpt_prefix}{step}.pth")
                print("--------------------------------------------------------------------------------------------")
                print("saving model at:", ckpt_path)
                torch.save(get_module(ppo_agent.policy_old).state_dict(), ckpt_path)
                print("model saved")
                print("Elapsed Time:", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
                if wandb is not None:
                    wandb.log({"train/checkpoint_saved": 1}, step=step)
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
                    step=step,
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
                        step=step,
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
        # Final save (rank0 only)
        if is_main_process():
            final_path = os.path.join(ckpt_dir, f"{ckpt_prefix}{step}.pth")
            torch.save(get_module(ppo_agent.policy_old).state_dict(), final_path)
            print("Final model saved at:", final_path)

        # Make sure rank0 finishes saving before others exit
        ddp_barrier()

        env.close()
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
    p.add_argument("--port_stride", type=int, default=10)
    p.add_argument("--time_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--verbose", action="store_true")

    # Runner steps (PER RANK)
    p.add_argument("--max_steps", type=int, default=100000)

    # PPO hyperparams
    p.add_argument("--max_ep_len", type=int, default=1000)
    p.add_argument("--update_timestep", type=int, default=4000)
    p.add_argument("--k_epochs", type=int, default=10)
    p.add_argument("--eps_clip", type=float, default=0.15)
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
    p.add_argument("--save_model_freq", type=int, default=30000)

    # Resume
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)

    # Reward shaping knobs
    p.add_argument("--w_progress", type=float, default=0.2)
    p.add_argument("--w_time", type=float, default=0.001)
    p.add_argument("--w_damage", type=float, default=0.05)
    p.add_argument("--w_not_actionable", type=float, default=0.01)
    p.add_argument("--terminal_bonus", type=float, default=1.0)

    p.add_argument("--terminal_failure_penalty", type=float, default=5.0)
    p.add_argument("--reward_clip", type=float, default=3.0)
    p.add_argument("--success_if_raw_reward_ge", type=float, default=1.0)
    p.add_argument("--time_limit", type=float, default=300.0)
    p.add_argument("--success_speed_bonus", type=float, default=1.0)

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

# torchrun --standalone --nproc_per_node=4 train.py --port 5005 --time_scale 1.0 --max_steps 100000