'''
How to run

xvfb-run -a -s "-screen 0 1024x768x24" \
python3 submission_runner.py \
  --ckpt_path /root/libimmortal/src/libimmortal/samples/PPO/checkpoints/Necto2_ImmortalSufferingEnv_randseed_2400000.pth \
  --seed 42 \
  --max_ep_len 4500
  --max_steps 45000 \
  --debug_dump_every 100 \
  --verbose

xvfb-run -a -s "-screen 0 1024x768x24" \
python3 submission_runner.py \
  --ckpt_path /root/libimmortal/src/libimmortal/samples/PPO/checkpoints/Necto2_ImmortalSufferingEnv_seed_mix_42_2400000.pth \
  --seed 42 \
  --max_ep_len 1500
  --max_steps 15000 \

'''

import time
import random
import numpy as np

from libimmortal.env import ImmortalSufferingEnv

# If you still want these debug utilities, keep them:
from libimmortal.utils import colormap_to_ids_and_onehot, parse_observation

import torch
import tqdm

# PPO utilities (your project path 기준)
import src.libimmortal.samples.PPO.utils.utilities as utilities
from src.libimmortal.samples.PPO import PPO


def _reward_scalar(reward_any) -> float:
    """Robustly convert reward (float/np/torch/list) -> float."""
    try:
        if reward_any is None:
            return 0.0
        if torch.is_tensor(reward_any):
            r = reward_any.detach().view(-1)
            return float(r[0].item()) if r.numel() > 0 else 0.0
        if isinstance(reward_any, np.ndarray):
            r = reward_any.reshape(-1)
            return float(r[0]) if r.size > 0 else 0.0
        if isinstance(reward_any, (list, tuple)):
            return float(reward_any[0]) if len(reward_any) > 0 else 0.0
        return float(reward_any)
    except Exception:
        return 0.0


def _close_env_silent(env):
    if env is None:
        return
    try:
        env.close()
    except Exception:
        pass


def _make_env(args, seed: int) -> ImmortalSufferingEnv:
    return ImmortalSufferingEnv(
        game_path=args.game_path,
        port=args.port,
        time_scale=args.time_scale,
        seed=int(seed),
        width=args.width,
        height=args.height,
        verbose=args.verbose,
    )


def _set_global_rng(seed: int):
    # English comments for copy/paste friendliness.
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _noop_action_np(space):
    """
    Return a "do nothing" action in numpy form, matching the gym space.
    NOTE: we intentionally avoid importing gym here; instead we check attributes.
    """
    # Box: has shape + dtype float
    if hasattr(space, "shape") and space.__class__.__name__ == "Box":
        return np.zeros(space.shape, dtype=np.float32)

    # Discrete: has n
    if hasattr(space, "n") and space.__class__.__name__ == "Discrete":
        return np.array(0, dtype=np.int64)

    # MultiDiscrete: has nvec
    if hasattr(space, "nvec") and space.__class__.__name__ == "MultiDiscrete":
        return np.zeros((len(space.nvec),), dtype=np.int64)

    raise TypeError(f"Unsupported action space: {type(space)}")


def _bootstrap_obs(env):
    """
    Requirement (1): env.reset() alone is unreliable.
    So we get a valid first observation by doing one noop step.
    This matches the workaround used in your training loop.
    """
    action_space = env.env.action_space
    a_np = _noop_action_np(action_space)
    a_env = utilities._format_action_for_env(a_np, action_space)
    obs, _r, _done, _info = env.step(a_env)
    return obs


def _start_new_episode(args, env, restart_sleep_s: float, episode_idx: int):
    """
    Requirement (1): reset alone is unreliable -> always close and recreate env for each episode.
    Also avoid env.reset() and bootstrap with one noop step.
    """
    _close_env_silent(env)
    if restart_sleep_s > 0:
        time.sleep(float(restart_sleep_s))

    # --- Seed override logic ---
    # If --fixed_seed is provided, it overrides --seed (and any external random seed).
    base_seed = int(args.fixed_seed) if (getattr(args, "fixed_seed", None) is not None) else int(args.seed)
    offset = int(getattr(args, "seed_offset_per_episode", 0) or 0)
    env_seed = int(base_seed + episode_idx * offset)

    if bool(getattr(args, "set_global_rng", False)):
        _set_global_rng(env_seed)

    env = _make_env(args, seed=env_seed)

    # Requirement (1): do NOT rely on env.reset(); bootstrap via noop step.
    obs = _bootstrap_obs(env)

    # PPO pipeline: id_map + vec_obs
    id_map, vec_obs, K = utilities._make_map_and_vec(obs)
    return env, obs, id_map, vec_obs, int(K)


def _select_action(agent, id_map_np: np.ndarray, vec_obs_np: np.ndarray, action_space, device, deterministic: bool):
    """
    Handles ActorCritic.act returning:
      - action only
      - or (action, logp, value)
    Then formats action into env-compatible format.
    """
    id_t = torch.from_numpy(np.asarray(id_map_np)).to(device=device, dtype=torch.long).unsqueeze(0)        # (1,H,W)
    vec_t = torch.from_numpy(np.asarray(vec_obs_np)).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,D)

    with torch.no_grad():
        try:
            out = agent.act(id_t, vec_t, deterministic=deterministic)
        except TypeError:
            out = agent.act(id_t, vec_t)

    if isinstance(out, (tuple, list)):
        action_t = out[0]
    else:
        action_t = out

    # MultiDiscrete: (1,branches) or (branches,)
    action_cpu = action_t.detach().cpu()
    if action_cpu.dim() >= 2 and action_cpu.size(0) == 1:
        action_cpu = action_cpu.squeeze(0)
    action_np = action_cpu.numpy()

    action_env = utilities._format_action_for_env(action_np, action_space)
    return action_env


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PPO Submission Runner (episode = close+recreate env)")

    # Env args
    parser.add_argument("--game_path", type=str, default=r"/root/immortal_suffering/immortal_suffering_linux_build.x86_64")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--time_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max_steps", type=int, default=15000)

    # PPO args (your shared args)
    PPO.add_ppo_args(parser)

    # Runner/agent extras (your request)
    g = parser.add_argument_group("PPO Runner Extras")
    g.add_argument("--deterministic", action="store_true", help="Use argmax actions instead of sampling")
    g.add_argument("--goal_reward", type=float, default=1.0, help="Terminal if reward >= goal_reward")
    g.add_argument("--restart_sleep_s", type=float, default=0.2, help="Sleep after env.close() before recreating")
    g.add_argument("--debug_dump_every", type=int, default=0, help="Dump debug files every N steps (0 disables)")

    # Repro / seed control
    s = parser.add_argument_group("Repro")
    s.add_argument(
        "--fixed_seed",
        type=int,
        default=None,
        help="If set, override env seed with this value for every episode (ignores --seed).",
    )
    s.add_argument(
        "--seed_offset_per_episode",
        type=int,
        default=0,
        help="env_seed = fixed_seed + episode_idx * offset (default 0 keeps identical seed).",
    )
    s.add_argument(
        "--set_global_rng",
        action="store_true",
        help="Also set python/numpy/torch RNGs to env_seed at each episode start.",
    )

    args = parser.parse_args()

    # Prefer-old default unless prefer_new set
    prefer_old = True
    if getattr(args, "prefer_new", False):
        prefer_old = False
    if getattr(args, "prefer_old", False):
        prefer_old = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # Episode stats (your request #3)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    cur_ep_return = 0.0
    cur_ep_len = 0

    # IMPORTANT: define episode_idx BEFORE first _start_new_episode()
    episode_idx = 0

    # --- initial episode (close+recreate pattern) ---
    env = None
    env, obs, id_map, vec_obs, K = _start_new_episode(args, env, restart_sleep_s=0.0, episode_idx=episode_idx)

    # Action space comes from wrapped gym env
    action_space = env.env.action_space

    num_ids = int(K)
    vec_dim = int(np.asarray(vec_obs).reshape(-1).shape[0])

    # Build agent
    agent, info = PPO.getPPOAgent(
        args,
        num_ids=num_ids,
        vec_dim=vec_dim,
        action_space=action_space,
        device=device,
        ckpt_path=getattr(args, "ckpt_path", None),
        prefer_old=prefer_old,
    )
    print("[Agent] loaded:", info)

    MAX_STEPS = int(args.max_steps)
    goal_reward = float(args.goal_reward)

    # -------------------------
    # Debug helper (kept from your original)
    # -------------------------
    def save_graphic_obs_to_txt(graphic_obs, file_path="/root/libimmortal/graphic_obs.txt"):
        """
        [데이터 구조 정상화 버전]
        graphic_obs: 현재 (3, 90, 160)으로 들어오지만 내부 데이터는 섞여 있는 상태를 가정
        """
        corrected_hwc = graphic_obs.reshape(90, 160, 3)
        obs_chw = np.transpose(corrected_hwc, (2, 0, 1))

        channels, height, width = obs_chw.shape
        channel_names = ["CHANNEL 0 (RED)", "CHANNEL 1 (GREEN)", "CHANNEL 2 (BLUE)"]

        with open(file_path, "w") as f:
            f.write(f"Original Input Shape: {graphic_obs.shape}\n")
            f.write(f"Corrected CHW Shape: {obs_chw.shape}\n")
            f.write("Note: Reshaped to (90, 160, 3) then transposed to (3, 90, 160)\n")
            f.write("=" * 50 + "\n\n")

            for c in range(channels):
                f.write(f"[[ {channel_names[c]} ]]\n")
                np.savetxt(f, obs_chw[c], fmt="%3d", delimiter=" ")
                f.write("\n" + "=" * 50 + "\n\n")

        print(f"graphic_obs saved with correction: {file_path}")

    # For step-level debug
    num_steps = 1

    # -------------------------
    # Main loop
    # -------------------------
    for _ in tqdm.tqdm(range(MAX_STEPS), desc="Stepping through environment"):
        # Action from PPO agent
        action_env = _select_action(
            agent,
            id_map_np=id_map,
            vec_obs_np=vec_obs,
            action_space=action_space,
            device=device,
            deterministic=bool(args.deterministic),
        )

        obs, reward_any, done, info = env.step(action_env)
        r = _reward_scalar(reward_any)

        # accumulate episode stats
        cur_ep_return += float(r)
        cur_ep_len += 1

        # parse next obs for PPO
        id_map, vec_obs, _K2 = utilities._make_map_and_vec(obs)

        # optional debug dump (kept similar to your original)
        if args.debug_dump_every and (num_steps % int(args.debug_dump_every) == 0):
            try:
                graphic_obs, vector_obs = parse_observation(obs)
                id_map_dbg, _onehot = colormap_to_ids_and_onehot(graphic_obs)

                DEFAULT_BLOCKED_IDS = 1
                passable = ~np.isin(id_map_dbg, np.asarray(DEFAULT_BLOCKED_IDS, dtype=id_map_dbg.dtype))
                np.savetxt("/root/libimmortal/id_map.txt", id_map_dbg, delimiter=",", fmt="%d")
                np.savetxt("/root/libimmortal/passable.txt", passable.astype(np.int8), delimiter=",", fmt="%d")
                print("id_map/passable saved at num_step:", num_steps)
                save_graphic_obs_to_txt(graphic_obs)
            except Exception as e:
                print("[DEBUG][WARN] dump failed:", repr(e))

        # Requirement (2): terminal if done OR reward >= goal_reward
        # terminal = bool(done) or (float(r) >= goal_reward) 
        terminal = bool(done) or (float(r) >= goal_reward) or (cur_ep_len >= int(args.max_ep_len)) # By episode

        if terminal:
            episode_idx += 2
            episode_returns.append(float(cur_ep_return))
            episode_lengths.append(int(cur_ep_len))

            print(
                f"[EP_END] ep={episode_idx} terminal_by={'reward' if float(r) >= goal_reward else 'done'} "
                f"last_r={float(r):.6f} ep_return={float(cur_ep_return):.6f} ep_len={int(cur_ep_len)} info={info}"
            )

            # Requirement (1): close + recreate env every episode
            env, obs, id_map, vec_obs, _K = _start_new_episode(
                args, env, restart_sleep_s=float(args.restart_sleep_s), episode_idx=episode_idx
            )

            # reset episode accumulators
            cur_ep_return = 0.0
            cur_ep_len = 0

        num_steps += 1

    # If the last episode didn't terminate, still report it as partial
    if cur_ep_len > 0:
        episode_returns.append(float(cur_ep_return))
        episode_lengths.append(int(cur_ep_len))
        print(f"[EP_PARTIAL] ep={len(episode_returns)} ep_return={float(cur_ep_return):.6f} ep_len={int(cur_ep_len)}")

    _close_env_silent(env)

    # -------------------------
    # Final print (your request #3)
    # -------------------------
    total_return = float(np.sum(np.asarray(episode_returns, dtype=np.float64))) if episode_returns else 0.0
    total_steps = int(np.sum(np.asarray(episode_lengths, dtype=np.int64))) if episode_lengths else 0

    print("=" * 90)
    print(f"[SUMMARY] episodes={len(episode_returns)} total_steps={total_steps} total_return={total_return:.6f}")
    if episode_returns:
        print(
            f"[SUMMARY] mean_return={float(np.mean(episode_returns)):.6f} "
            f"max_return={float(np.max(episode_returns)):.6f} "
            f"mean_len={float(np.mean(episode_lengths)):.2f} "
            f"max_len={int(np.max(episode_lengths))}"
        )
        print("-" * 90)
        print("ep_idx, ep_len, ep_return")
        for i, (L, R) in enumerate(zip(episode_lengths, episode_returns), start=1):
            print(f"{i:04d}, {int(L):5d}, {float(R): .6f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
