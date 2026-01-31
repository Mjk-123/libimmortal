import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from libimmortal.env import ImmortalSufferingEnv
from libimmortal.utils import colormap_to_ids_and_onehot, parse_observation

from collections import Counter, deque


class VecDimTracker:
    """
    Tracks vector observation dimension statistics without spamming logs.
    """
    def __init__(self, expected_dim: int | None = None, recent_k: int = 32):
        self.expected_dim = expected_dim
        self.recent = deque(maxlen=int(recent_k))
        self.counter = Counter()
        self.total = 0
        self.mismatch = 0
        self.first_mismatch_events = []  # list of (step, got, expected, where), capped

    def _flatten_dim(self, vec_obs) -> int:
        if vec_obs is None:
            return 0
        v = np.asarray(vec_obs)
        return int(v.reshape(-1).shape[0])

    def update(self, vec_obs, step: int, where: str):
        dim = self._flatten_dim(vec_obs)
        self.total += 1
        self.counter[dim] += 1
        self.recent.append(dim)

        if self.expected_dim is not None and dim != int(self.expected_dim):
            self.mismatch += 1
            if len(self.first_mismatch_events) < 20:
                self.first_mismatch_events.append((int(step), int(dim), int(self.expected_dim), str(where)))

        return dim

    def summary_str(self, prefix: str = "[VecDim]") -> str:
        dist = ", ".join([f"{k}:{v}" for k, v in sorted(self.counter.items())])
        return (
            f"{prefix} expected={self.expected_dim} total={self.total} mismatch={self.mismatch} "
            f"dist=({dist}) recent={list(self.recent)}"
        )


def main():
    import tqdm
    import argparse
    import time

    # Optional: Unity timeout exception type
    try:
        from mlagents_envs.exception import UnityTimeOutException  # type: ignore
    except Exception:
        UnityTimeOutException = None  # type: ignore

    parser = argparse.ArgumentParser(description="Test Immortal Suffering Environment")
    parser.add_argument(
        "--game_path",
        type=str,
        required=False,
        default=r"/root/immortal_suffering/immortal_suffering_linux_build.x86_64",
        help="Path to the Unity executable",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=False,
        default=5005,
        help="Port number for the Unity environment and python api to communicate",
    )
    parser.add_argument(
        "--time_scale",
        type=float,
        required=False,
        default=1.0,  # !NOTE: This will be set as 1.0 in assessment
        help="Speed of the simulation, maximum 2.0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Seed that controls enemy spawn",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=False,
        default=720,
        help="Visualized game screen width",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=False,
        default=480,
        help="Visualized game screen height",
    )
    parser.add_argument("--verbose", action="store_true", help="Whether to print logs")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,  # !NOTE: This will be set as 18000 (5 minutes in real-time) in assessment
        help="Number of steps to run the environment",
    )

    # -------------------------
    # Added args for vec_dim probing
    # -------------------------
    parser.add_argument(
        "--vecdim_report_every",
        type=int,
        default=1000,
        help="Report vec_dim stats every N steps (0 disables periodic reports).",
    )
    parser.add_argument(
        "--expected_vecdim",
        type=int,
        default=-1,
        help="If >=0, treat as expected vec_dim and count mismatches. If <0, use the first reset vec_dim.",
    )
    parser.add_argument(
        "--vecdim_recent_k",
        type=int,
        default=32,
        help="How many recent vec_dim values to keep in memory for reporting.",
    )

    # -------------------------
    # Optional env restart on timeout
    # -------------------------
    parser.add_argument(
        "--enable_env_restart",
        action="store_true",
        help="If set, restart env on UnityTimeOutException by closing and recreating with a different port.",
    )
    parser.add_argument(
        "--port_stride",
        type=int,
        default=50,
        help="Port stride used when restarting env (to avoid collision).",
    )
    parser.add_argument(
        "--max_env_restarts",
        type=int,
        default=5,
        help="Maximum env restarts on timeout before giving up.",
    )
    parser.add_argument(
        "--restart_sleep",
        type=float,
        default=2.0,
        help="Sleep seconds before recreating env after timeout.",
    )

    args = parser.parse_args()

    # -------------------------
    # Env creation helper (to support restarts)
    # -------------------------
    restart_id = 0

    def make_env():
        nonlocal restart_id
        port = int(args.port) + int(restart_id) * int(args.port_stride)
        env = ImmortalSufferingEnv(
            game_path=args.game_path,
            port=port,
            time_scale=args.time_scale,  # !NOTE: This will be set as 1.0 in assessment
            seed=args.seed,  # !NOTE: This will be set as random number in assessment
            width=args.width,
            height=args.height,
            verbose=args.verbose,
        )
        return env, port

    env, port = make_env()

    MAX_STEPS = args.max_steps
    obs = env.reset()
    graphic_obs, vector_obs = parse_observation(obs)
    id_map, graphic_obs_onehot = colormap_to_ids_and_onehot(
        graphic_obs
    )  # one-hot encoded graphic observation

    # -------------------------
    # Import your AI agent here
    # -------------------------
    from src.libimmortal.samples.randomAgent import RandomAgent
    agent = RandomAgent(env.env.action_space)

    num_steps = 1

    # -------------------------
    # VecDim tracking init
    # -------------------------
    first_dim = int(np.asarray(vector_obs).reshape(-1).shape[0])
    expected = first_dim if args.expected_vecdim < 0 else int(args.expected_vecdim)
    vecdim = VecDimTracker(expected_dim=expected, recent_k=int(args.vecdim_recent_k))
    vecdim.update(vector_obs, step=num_steps, where="reset/initial")

    # For debugging graphic obs dump
    def save_graphic_obs_to_txt(graphic_obs, file_path="/root/libimmortal/graphic_obs.txt"):
        """
        [Normalization attempt]
        NOTE: This assumes the buffer is flattened RGBRGB... and reshapes to (90,160,3).
        Keep as-is from your original code.
        """
        corrected_hwc = graphic_obs.reshape(90, 160, 3)
        obs_chw = np.transpose(corrected_hwc, (2, 0, 1))

        channels, height, width = obs_chw.shape
        channel_names = ["CHANNEL 0 (RED)", "CHANNEL 1 (GREEN)", "CHANNEL 2 (BLUE)"]

        with open(file_path, 'w') as f:
            f.write(f"Original Input Shape: {graphic_obs.shape}\n")
            f.write(f"Corrected CHW Shape: {obs_chw.shape}\n")
            f.write("Note: Reshaped to (90, 160, 3) then transposed to (3, 90, 160)\n")
            f.write("=" * 50 + "\n\n")

            for c in range(channels):
                f.write(f"[[ {channel_names[c]} ]]\n")
                np.savetxt(f, obs_chw[c], fmt='%3d', delimiter=' ')
                f.write("\n" + "=" * 50 + "\n\n")

        print(f"graphic_obs saved with correction: {file_path}")

    # Restart counters
    timeout_count = 0
    restart_count = 0

    pbar = tqdm.tqdm(range(MAX_STEPS), desc=f"Stepping through environment (port={port})")

    for _ in pbar:
        # Agent action (keep your original input signature)
        action = agent.act((graphic_obs, vector_obs))

        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            is_timeout = (UnityTimeOutException is not None) and isinstance(e, UnityTimeOutException)

            if is_timeout:
                timeout_count += 1
                tqdm.tqdm.write(f"[Timeout] UnityTimeOutException at num_steps={num_steps} (port={port}).")

                if not args.enable_env_restart:
                    tqdm.tqdm.write("[Timeout] enable_env_restart is False -> exiting loop.")
                    break

                restart_count += 1
                if restart_count > int(args.max_env_restarts):
                    tqdm.tqdm.write(f"[Timeout] exceeded max_env_restarts={args.max_env_restarts} -> exiting loop.")
                    break

                # Restart env with new port
                try:
                    env.close()
                except Exception:
                    pass

                restart_id += 1
                time.sleep(float(args.restart_sleep))
                env, port = make_env()
                pbar.set_description_str(f"Stepping through environment (port={port})")

                # Reset after restart
                obs = env.reset()
                graphic_obs, vector_obs = parse_observation(obs)
                id_map, graphic_obs_onehot = colormap_to_ids_and_onehot(graphic_obs)

                # Update vecdim stats on reset
                vecdim.update(vector_obs, step=num_steps, where="reset/after_timeout")
                continue

            # Non-timeout exception: re-raise
            raise

        # Parse obs
        graphic_obs, vector_obs = parse_observation(obs)
        id_map, graphic_onehot = colormap_to_ids_and_onehot(graphic_obs)

        # Update vecdim stats
        got_dim = vecdim.update(vector_obs, step=num_steps, where="step")

        # Periodic report (no spam)
        if args.vecdim_report_every > 0 and (num_steps % int(args.vecdim_report_every) == 0):
            tqdm.tqdm.write(vecdim.summary_str(prefix=f"[VecDim][step={num_steps}]"))
            tqdm.tqdm.write(f"[Env] done={bool(done)} reward={float(reward):.4f} timeouts={timeout_count} restarts={restart_count} port={port}")
        '''
        # Your existing debug dumps
        if num_steps % 200 == 0:
            np.savetxt("/root/libimmortal/id_map.txt", id_map, delimiter=',', fmt='%d')
            tqdm.tqdm.write(f"id_map saved at num_step: {num_steps}")
            try:
                save_graphic_obs_to_txt(graphic_obs)
            except Exception as e:
                tqdm.tqdm.write(f"[Warn] save_graphic_obs_to_txt failed: {repr(e)}")
        '''
        # Episode reset
        if done:
            tqdm.tqdm.write(f"[DONE] reward={reward} info={info}")
            obs = env.reset()
            graphic_obs, vector_obs = parse_observation(obs)
            id_map, graphic_obs_onehot = colormap_to_ids_and_onehot(graphic_obs)
            vecdim.update(vector_obs, step=num_steps, where="reset/done")

        num_steps += 1

    # Final summary
    print("============================================================")
    print(vecdim.summary_str(prefix="[VecDim][FINAL]"))
    if vecdim.first_mismatch_events:
        print("[VecDim][FINAL] first mismatch events (up to 20):")
        for (st, got, exp, where) in vecdim.first_mismatch_events:
            print(f"  step={st} got={got} expected={exp} where={where}")
    print(f"[Env][FINAL] timeouts={timeout_count} restarts={restart_count} last_port={port}")
    print(f"[Finished] done={done} reward={reward} info={info}")
    print("============================================================")

    env.close()


if __name__ == "__main__":
    main()
