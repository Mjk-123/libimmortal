# samples/random_agent.pyimport argparse

import tqdm

from libimmortal import ImmortalSufferingEnv
from libimmortal.utils import parse_observation, colormap_to_ids_and_onehot, find_free_tcp_port

'''
==========================================================================================
Random Agent
==========================================================================================
action_space: MultiDiscrete([2 2 2 2 2 2 2 2])
observation_space: Tuple(Box(0, 255, (3, 90, 160), uint8), Box(-inf, inf, (103,), float32))
graphic shape/dtype: (3, 90, 160) uint8
vector  shape/dtype: (103,) float32
sample action: [0 0 1 1 0 1 0 1]
==========================================================================================
'''

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        # Ignore obs; just sample a valid action from the environment space.
        return self.action_space.sample()