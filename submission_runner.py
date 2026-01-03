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



def main():
    import tqdm
    import argparse

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
        default=18000,  # !NOTE: This will be set as 18000 (5 minutes in real-time) in assessment
        help="Number of steps to run the environment",
    )
    
    ###################################
    """
    You can add more arguments here for your AI agent if needed.
    """
    ###################################
    
    
    args = parser.parse_args()

    env = ImmortalSufferingEnv(
        game_path=args.game_path,
        port=args.port,
        time_scale=args.time_scale,  # !NOTE: This will be set as 1.0 in assessment
        seed=args.seed,  # !NOTE: This will be set as random number in assessment
        width=args.width,
        height=args.height,
        verbose=args.verbose,
    )

    MAX_STEPS = args.max_steps
    obs = env.reset()
    graphic_obs, vector_obs = parse_observation(obs)
    id_map, graphic_obs = colormap_to_ids_and_onehot(
        graphic_obs
    )  # one-hot encoded graphic observation

    ###################################
    """
    Import your AI agent here and replace the random action below with your agent's action.
    """

    from src.libimmortal.samples.randomAgent import RandomAgent

    agent = RandomAgent(env.env.action_space)
    # agent = 

    ###################################

    for _ in tqdm.tqdm(range(MAX_STEPS), desc="Stepping through environment"):
        
        ###################################
        """
        Do whatever you want with the observation here and get action from your AI agent.
        Replace the random action below with your agent's action.
        """
        ###################################
        
        action = agent.act((graphic_obs, vector_obs))  # REPLACE this with your AI agent's action
        obs, reward, done, info = env.step(action)
        graphic_obs, vector_obs = parse_observation(obs)
        id_map, graphic_obs = colormap_to_ids_and_onehot(
            graphic_obs
        )  # one-hot encoded graphic observation

        if done:
            print("[DONE] reward=", reward, "info=", info)
            obs = env.reset()
            graphic_obs, vector_obs = parse_observation(obs)
            id_map, graphic_obs = colormap_to_ids_and_onehot(graphic_obs)
    
    print(f"[Finished] done = {done}, reward={reward}, info={info}")

    env.close()


if __name__ == "__main__":
    main()

