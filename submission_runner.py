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
    num_steps = 1
    ###################################

    import numpy as np
    
    ###################################
    # For debugging
    ###################################

    def save_graphic_obs_to_txt(graphic_obs, file_path="/root/libimmortal/graphic_obs.txt"):
        """
        [데이터 구조 정상화 버전]
        graphic_obs: 현재 (3, 90, 160)으로 들어오지만 내부 데이터는 섞여 있는 상태를 가정
        """
        # 1. 꼬여있는 데이터를 원래의 H, W, C 구조로 복원 (90, 160, 3)
        # 현재 텍스트 파일의 정황상 (3, 90, 160)의 전체 크기인 43,200개의 숫자가 
        # 사실은 [R, G, B, R, G, B...] 순서로 14,400픽셀만큼 나열되어 있음
        corrected_hwc = graphic_obs.reshape(90, 160, 3)
        
        # 2. 저장과 분석을 위해 (3, 90, 160)으로 차원 순서 변경
        # Axis 0: Channel, Axis 1: Height, Axis 2: Width
        obs_chw = np.transpose(corrected_hwc, (2, 0, 1))
        
        channels, height, width = obs_chw.shape
        channel_names = ["CHANNEL 0 (RED)", "CHANNEL 1 (GREEN)", "CHANNEL 2 (BLUE)"]
        
        with open(file_path, 'w') as f:
            # 데이터가 어떻게 변환되었는지 정보 기록
            f.write(f"Original Input Shape: {graphic_obs.shape}\n")
            f.write(f"Corrected CHW Shape: {obs_chw.shape}\n")
            f.write("Note: Reshaped to (90, 160, 3) then transposed to (3, 90, 160)\n")
            f.write("=" * 50 + "\n\n")
            
            for c in range(channels):
                f.write(f"[[ {channel_names[c]} ]]\n")
                # 이제 각 채널에는 R이면 R, G면 G 한 가지 색상 강도만 들어있게 됩니다.
                np.savetxt(f, obs_chw[c], fmt='%3d', delimiter=' ')
                f.write("\n" + "="*50 + "\n\n")
                
        print(f"graphic_obs saved with correction: {file_path}")

    ###################################
    ###################################
    '''
    def fix_obs_structure(obs):
        """
        (3, 90, 160) 형태이지만 RGB가 섞여있는 데이터를 
        제대로 된 (3, 90, 160) CHW 구조로 바꿈
        """
        # 1. 일렬로 나열된 RGB를 (H, W, 3)으로 제대로 쌓기
        # 이 과정에서 [8, 19, 49]가 하나의 픽셀 묶음이 됨
        h, w = 90, 160
        corrected_hwc = obs.reshape(h, w, 3)
        
        # 2. 인코더가 기대하는 (3, H, W) 순서로 축 변경
        corrected_chw = np.transpose(corrected_hwc, (2, 0, 1))
        
        return corrected_chw
    '''

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
        # graphic_obs = fix_obs_structure(graphic_obs)
        id_map, graphic_onehot = colormap_to_ids_and_onehot(
            graphic_obs
        )  # one-hot encoded graphic observation
        
        if num_steps % 200 == 0:
            DEFAULT_BLOCKED_IDS = 1
            passable = ~np.isin(id_map, np.asarray(DEFAULT_BLOCKED_IDS, dtype=id_map.dtype))
            np.savetxt("/root/libimmortal/id_map.txt", id_map, delimiter=',', fmt='%.2f')
            np.savetxt("/root/libimmortal/passable.txt", passable, delimiter=',', fmt='%.2f')
            print("id_map saved at num_step:", num_steps)
            save_graphic_obs_to_txt(graphic_obs)
            #print("vector_obs:", vector_obs)
        
        if done:
            print("[DONE] reward=", reward, "info=", info)
            obs = env.reset()
            graphic_obs, vector_obs = parse_observation(obs)
            # graphic_obs = fix_obs_structure(graphic_obs)
            id_map, graphic_onehot = colormap_to_ids_and_onehot(graphic_obs)
        
        num_steps += 1
    
    print(f"[Finished] done = {done}, reward={reward}, info={info}")

    env.close()


if __name__ == "__main__":
    main()

