# libimmortal: 2025 IST-TECH AI Video Game Project

## Introduction
This is the Python Gym-like API for video game **"Immortal suffering"**
Make an AI Agent that take out various enemies and reach the goalpoint as fast as possible.

## Installation

### For windows

1. Download window build of [immortal suffering](https://github.com/ist-tech-AI-games/immortal_suffering/releases/download/v1.0.1/immortal_suffering_windows_x86_64.zip)  
**KNOWN ISSUE**: Windows 11 Defender misdianoses the build file as a trojan.  
It is a false positive, so it is **safe to use**. 

2. Unzip the immortal_suffering_windows_x86_64.zip

3. Import conda virtual environment (this might take a while...)
```
conda env create -f libimmortal.yaml -n <env_name_here>
```

4. Install libimmortal
```
pip install -e .
```

5. Install pytorch that are compatible with you local gpu

### For Linux
1. Build docker image
```sh
docker build -t libimmortal:1.0.1 .
```

2. create docker container
```sh
docker compose up -d
```

3. Access docker container
## Patch Note
- v1.0  
Known issue: Sprite rendering error while on animation playing  
Known issue: Unity Major security issue(CVE-2025-59489)  

- v1.0.1  
Upgraded MLAgent from 3.0.0 to 4.0.0  
Migrated Sentis to Inference engine  
Updated Unity Version to correspond security issue  
Handled sprite rendering error while on animation playing  
Added Obs parsing function ```libimmortal.utils.parse_observation```  

## Content
```sh
.
├── docker
│   └── start_xvfb.sh
├── docker-compose.yml
├── Dockerfile
├── env.py  # This is the environment file
├── __init__.py
├── libimmortal.yaml
├── README.md
├── requirements.txt
├── samples
│   ├── agents.py  # Sample agent will be located here (WIP)
│   └── __init__.py
└── utils
    ├── aux_func.py  # auxilary functions such as feature extraction or finding free ports are located here
    ├── enums.py  # enums that are useful for feature extraction are located here
    ├── __init__.py
    └── obs_limits.py  # limit values for normailization are located here
```
## How to run
```python
from libimmortal import ImmortalSufferingEnv
from libimmortal.utils import colormap_to_ids_and_onehot, parse_observation

env = ImmortalSufferingEnv(
    game_path=args.game_path,  # Put you game path here. (For windows, <path -for-Immortal Suffering.exe>. For linux, <path-for immortal_suffering_linux_build.x86_64>)
    port=args.port,  # you can use immortal_suffering.utils.aux_func.find_free_tcp_port() to find free usable port 
    time_scale=args.time_scale,  # 1.0~2.0 is recommended
    seed=args.seed,  # integer seed that determines enemy spawn position and type
    width=args.width,  # Game play screen width (only for visualization)
    height=args.height,  # Game play screen height (only for visualization)
    verbose=args.verbose,  # Whether to print logs or not
)

MAX_STEPS = args.max_steps
obs = env.reset()
graphic_obs, vector_obs = parse_observation(obs)
    id_map, graphic_obs = colormap_to_ids_and_onehot(
        graphic_obs
    )  # one-hot encoded graphic observation

for _ in tqdm.tqdm(range(MAX_STEPS), desc="Stepping through environment"):
    action = env.env.action_space.sample()  # Change here with your AI agent
    obs, reward, done, info = env.step(action)  # Receive reward 1 if player Goal, else 0
    graphic_obs, vector_obs = parse_observation(obs)
    id_map, graphic_obs = colormap_to_ids_and_onehot(
        graphic_obs
    )  # one-hot encoded graphic observation

env.close()  # !NECESSARY TO CLOSE AFTER 1 ENV RUN
```

## Observation
there are two observations that are provided from the environment.
**Graphic observation**, **Vector observation**

### Graphic Observation
Graphic Observation is ```3 (RGB channel) x 90 (Height) x 160 (Width)-dimensional tensor```.  
This is the downscaled image of the game screen, and the game entities are drawn with specific color, which can be mapped as id.  
There is a utility function(```libimmortal.utils.colormap_to_ids_and_onehot```) that maps raw graphic observation to one-hot encoded id tensor map.

### Vector Observation
Vector Observation is a ```103-dimensional vector```.  
This contains player-related informations and enemy-related infomations.  

```python
obs, reward, done, info = env.step(action)
graphic_obs, vector_obs = obs["graphic"], obs["vector"]
player_obs = vector_obs[0:13]
enemy_obs = vector_obs[13:103]

player_obs = [
    PLAYER_POSITION_X, 
    PLAYER_POSITION_Y, 
    PLAYER_VELOCITY_X, 
    PLAYER_VELOCITY_Y,
    PLAYER_CULULATED_DAMAGE,
    PLAYER_IS_ACTIONABLE,
    PLAYER_IS_HITTING,
    PLAYER_IS_DOBBLE_JUMP_AVAILABLE,
    PLAYER_IS_ATTACKABLE,
    GOAL_POSITION_X,
    GOAL_POSITION_Y,
    GOAL_PLAYER_DISTANCE,
    TIME_ELAPSED
    ]

enemy_obs = [
    ENEMY_TYPE_SKELETON,
    ENEMY_TYPE_BOMBKID,
    ENEMY_TYPE_TURRET,
    ENEMY_POSITION_X,
    ENEMY_POSITION_Y,
    ENEMY_VELOCITY_X,
    ENEMY_VELOCITY_Y,
    ENEMY_HEALTH,
    ENEMY_STATE
] * 10
```

Observation of 10 enemies is seriealized in ```vector_obs[13:103]```, if there is less then 10 enemies, the back of the ```enemy_obs``` vector is zero-padded at the tail.


## Tips for Reinforcement Learning
1. **Parallel episode collection**  
Included parallel processing library **"ray"**.  
Immortal suffering supports parallel running.  
Also there is a utility function(```libimmortal.utils.find_free_tcp_port```, ```libimmortal.utils.find_n_free_tcp_ports```) to find free ports for connecting Immortal Suffering and libimmortal Python API.
2. **Reward shaping**  
The default reward only gives 1 when goal is reached, else 0.  
Modify reward fucntion using given observations.
3. **Feature extraction**  
Both graphic observation and vector observation are provided in obs.
4. **Monitor training process with visualization**  
Included training visualizing library "tensorflow" and "wandb"

## Run PPO
1. **How to run**
Use torchrun for DDP. We used 4 nodes for one run.

```
torchrun --standalone --nproc_per_node=[number_of_nodes] ./src/libimmortal/samples/PPO/train.py \
  --port 5005 --port_stride 50 \
  --save_model_freq [save_frequency] --wandb [Optional]
```

2. **Use checkpoint and resume training.**
Use ```--resume``` and ```--checkpoint``` flag to resume your learning continuously.

```
torchrun --standalone --nproc_per_node=[number_of_nodes] ./src/libimmortal/samples/PPO/train.py \
  --port 5005 --port_stride 50 \
  --save_model_freq [save_frequency] --wandb [Optional] \
  --resume --checkpoint [checkpoint_path]
```

3. **use ```KeyboardInterrupt``` to put ```SIGINT``` into process so that pause learning.**
Then, your checkpoint will be saved in ```./src/libimmortal/samples/PPO/checkpioints```. Do not put ```KeyboardInterrupt``` more than once despite your shell is not interrupted. Just wait. Two or more interruption can disturb saving checkpoint.
