import os
import glob
import argparse
from datetime import datetime
from typing import Optional

import numpy as np
import torch

from PPO import PPO
from reward import RewardShaper, RewardConfig
from libimmortal.utils import parse_observation, colormap_to_ids_and_onehot

def _make_state_and_vector(obs):
    graphic_obs, vector_obs = parse_observation(obs)

    # id_map: (H,W) uint8, onehot: (K,H,W) uint8
    id_map, onehot = colormap_to_ids_and_onehot(graphic_obs)

    # Flatten id_map and (optionally) normalize to [0,1]
    g = np.asarray(id_map, dtype=np.float32).reshape(-1)
    K = int(onehot.shape[0])
    if K > 1:
        g = g / float(K - 1)

    v = np.asarray(vector_obs, dtype=np.float32).reshape(-1)
    state = np.concatenate([g, v], axis=0)
    return state, v

# Optional: robust terminal detection if your env stuffs ML-Agents steps into info
try:
    from mlagents_envs.base_env import TerminalSteps
except Exception:
    TerminalSteps = None


def _checkpoint_dir() -> str:
    """
    Save checkpoints under:
    <this train.py directory>/checkpoints

    i.e., ./src/libimmortal/samples/PPO/checkpoints
    regardless of where you run the script from.
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
    Your runner used env.env.action_space; keep that behavior.
    """
    if hasattr(env, "env") and hasattr(env.env, "action_space"):
        return env.env.action_space
    return env.action_space


def train(args):
    print("============================================================================================")

    # Import here so PYTHONPATH / module mode works cleanly
    from libimmortal.env import ImmortalSufferingEnv

    env = ImmortalSufferingEnv(
        game_path=args.game_path,
        port=args.port,
        time_scale=args.time_scale,  # NOTE: assessment uses 1.0
        seed=args.seed,              # NOTE: assessment uses random seed
        width=args.width,
        height=args.height,
        verbose=args.verbose,
    )

    env_tag = "ImmortalSufferingEnv"  # only for filenames/logging

    # -------------------- hyperparameters --------------------
    # Total interaction steps (runner-style)
    MAX_STEPS = args.max_steps

    # PPO defaults (keep your previous values)
    max_ep_len = args.max_ep_len               # safety cap per episode
    update_timestep = args.update_timestep
    K_epochs = args.k_epochs
    eps_clip = args.eps_clip
    gamma = args.gamma
    lr_actor = args.lr_actor
    lr_critic = args.lr_critic

    save_model_freq = args.save_model_freq
    action_std = args.action_std
    action_std_decay_rate = args.action_std_decay_rate
    min_action_std = args.min_action_std
    action_std_decay_freq = args.action_std_decay_freq
    # ----------------------------------------------------------

    random_seed = int(args.seed) if args.seed is not None else 0
    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        try:
            env.seed(random_seed)
        except Exception:
            pass

    # -------------------- infer state_dim from first reset --------------------
    cfg = RewardConfig(
        w_progress=1.0,
        w_time=0.001,
        w_damage=0.05,
        w_not_actionable=0.01,
        terminal_failure_penalty=1.0,
        clip=1.0,
        time_limit=300.0,              # TIME_ELAPSED 단위가 "초"라면 300.0, "스텝"이면 18000
        success_speed_bonus=1.0,
    )
    shaper = RewardShaper(cfg)

    obs = env.reset()
    state, vector_obs = _make_state_and_vector(obs)
    shaper.reset(vector_obs)
    state_dim = int(state.shape[0])

    action_space = _get_action_space(env)
    has_continuous_action_space = hasattr(action_space, "shape") and action_space.shape is not None

    if has_continuous_action_space:
        action_dim = int(np.prod(action_space.shape))
    else:
        action_dim = int(action_space.n)

    print(f"[Env] state_dim={state_dim}, action_dim={action_dim}, continuous={has_continuous_action_space}")

    # -------------------- checkpoints --------------------
    ckpt_dir = _checkpoint_dir()
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_prefix = f"PPO_{env_tag}_{random_seed}_"
    default_ckpt_path = os.path.join(ckpt_dir, f"{ckpt_prefix}0.pth")  # same naming style as before

    print("checkpoint dir:", ckpt_dir)
    print("default checkpoint path:", default_ckpt_path)

    # -------------------- PPO agent --------------------
    ppo_agent = PPO(
        state_dim, action_dim,
        lr_actor, lr_critic,
        gamma, K_epochs, eps_clip,
        has_continuous_action_space, action_std
    )

    # -------------------- resume --------------------
    if args.resume:
        ckpt_to_load = args.checkpoint
        if ckpt_to_load is None:
            ckpt_to_load = _latest_checkpoint_path(ckpt_dir, prefix=ckpt_prefix)

        if ckpt_to_load is None:
            raise FileNotFoundError(
                f"--resume set but no checkpoint found under {ckpt_dir} with prefix {ckpt_prefix}"
            )

        if not hasattr(ppo_agent, "load"):
            raise AttributeError("PPO class has no .load(path). Implement it or adjust the call.")

        print(f"Resuming from: {ckpt_to_load}")
        ppo_agent.load(ckpt_to_load)

    # -------------------- wandb (optional) --------------------
    wandb = None
    if args.wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except Exception as e:
            raise RuntimeError(
                "You passed --wandb but wandb is not available. Install wandb or run without --wandb."
            ) from e

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"{env_tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=dict(
                env_tag=env_tag,
                MAX_STEPS=MAX_STEPS,
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
                seed=random_seed,
                resume=args.resume,
                checkpoint=args.checkpoint,
            ),
        )

    # -------------------- runner-style training loop --------------------
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT):", start_time)
    print("============================================================================================")

    # episode bookkeeping (runner resets on done)
    episode_reward = 0.0
    episode_raw_reward = 0.0
    episode_len = 0
    episode_idx = 0

    # Optional: prevent runaway episodes if env fails to set done
    # (Still compatible with runner, just a safety cap)
    MAX_EP_LEN = max_ep_len

    for step in range(1, MAX_STEPS + 1):

        # action from PPO
        action = ppo_agent.select_action(state)
        if has_continuous_action_space:
            action = np.asarray(action, dtype=np.float32).reshape(action_space.shape)

        obs, raw_reward, done, info = env.step(action)
        done = _infer_done(done, info)

        # preprocess next observation (+ vector_obs for reward shaping)
        next_state, next_vector_obs = _make_state_and_vector(obs)
        reward = shaper(raw_reward, next_vector_obs, done, info)

        # store reward/terminal for PPO update
        ppo_agent.buffer.rewards.append(float(reward))
        ppo_agent.buffer.is_terminals.append(bool(done))

        episode_reward += float(reward)
        episode_raw_reward += float(raw_reward)
        episode_len += 1

        # PPO update
        if step % update_timestep == 0:
            ppo_agent.update()
            if wandb is not None:
                wandb.log({"train/updated": 1}, step=step)

        # std decay
        if has_continuous_action_space and (step % action_std_decay_freq == 0):
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
            if wandb is not None:
                wandb.log({"train/action_std": getattr(ppo_agent, "action_std", np.nan)}, step=step)

        # checkpoint save
        if step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at:", default_ckpt_path)
            ppo_agent.save(default_ckpt_path)
            print("model saved")
            print("Elapsed Time:", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
            if wandb is not None:
                wandb.log({"train/checkpoint_saved": 1}, step=step)
                wandb.save(default_ckpt_path)

        # wandb step logging (lightweight)
        if wandb is not None and (step % args.wandb_log_freq == 0):
            wandb.log(
                {
                    "train/reward": float(reward),
                    "train/done": int(done),
                    "train/episode_reward_running": float(episode_reward),
                    "train/episode_reward_raw": float(episode_raw_reward),
                    "train/episode_len_running": int(episode_len),
                },
                step=step,
            )

        # episode termination (runner behavior)
        if done or (episode_len >= MAX_EP_LEN):
            if done and args.verbose:
                print("[DONE] reward=", reward, "info=", info)

            if wandb is not None:
                wandb.log(
                    {
                        "episode/reward": float(episode_reward),
                        "episode/reward_raw":float(episode_raw_reward),
                        "episode/len": int(episode_len),
                        "episode/index": int(episode_idx),
                    },
                    step=step,
                )

            # reset episode
            obs = env.reset()
            state, vector_obs = _make_state_and_vector(obs)
            shaper.reset(vector_obs)

            episode_reward = 0.0
            episode_raw_reward = 0.0
            episode_len = 0
            episode_idx += 1
        else:
            state = next_state

    # final save
    print("--------------------------------------------------------------------------------------------")
    print("saving final model at:", default_ckpt_path)
    ppo_agent.save(default_ckpt_path)
    print("final model saved")
    print("--------------------------------------------------------------------------------------------")

    env.close()
    if wandb is not None:
        wandb.finish()

    end_time = datetime.now().replace(microsecond=0)
    print("============================================================================================")
    print("Started training at (GMT):", start_time)
    print("Finished training at (GMT):", end_time)
    print("Total training time:", end_time - start_time)
    print("============================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ImmortalSufferingEnv ctor args (runner-compatible)
    parser.add_argument(
        "--game_path",
        type=str,
        default=r"/root/immortal_suffering/immortal_suffering_linux_build.x86_64",
        help="Path to the Unity executable",
    )
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--time_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--verbose", action="store_true")

    # runner-style step budget (assessment uses 18000)
    parser.add_argument("--max_steps", type=int, default=18000)

    # safety cap per episode (doesn't change runner semantics unless done never triggers)
    parser.add_argument("--max_ep_len", type=int, default=1000)

    # PPO hyperparams (exposed so you can tune without editing code)
    parser.add_argument("--update_timestep", type=int, default=4000)
    parser.add_argument("--k_epochs", type=int, default=80)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-3)

    parser.add_argument("--action_std", type=float, default=0.6)
    parser.add_argument("--action_std_decay_rate", type=float, default=0.05)
    parser.add_argument("--min_action_std", type=float, default=0.1)
    parser.add_argument("--action_std_decay_freq", type=int, default=int(2.5e5))

    parser.add_argument("--save_model_freq", type=int, default=int(1e5))

    # resume
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)

    # wandb (optional)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="ppo-immortal")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb_log_freq", type=int, default=200)

    args = parser.parse_args()
    train(args)
