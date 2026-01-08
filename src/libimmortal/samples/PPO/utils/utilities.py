import numpy as np
import gym
from typing import Optional, Tuple, Dict, Any

# -------------------------
# Observation preprocessing
# -------------------------
def _make_map_and_vec(obs) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Return:
      id_map: (H,W) uint8 (IDs)
      vec_obs: (D,) float32
      K: number of IDs in palette (num_ids)
    """
    from libimmortal.utils import colormap_to_ids_and_onehot, parse_observation

    graphic_obs, vector_obs = parse_observation(obs)
    id_map, onehot = colormap_to_ids_and_onehot(graphic_obs)

    id_map = np.asarray(id_map, dtype=np.uint8)
    vec_obs = np.asarray(vector_obs, dtype=np.float32).reshape(-1)

    K = int(onehot.shape[0])
    return id_map, vec_obs, K


def _format_action_for_env(action_np, action_space):
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

