import numpy as np
import gym
from typing import Optional, Tuple, Dict, List, Any

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

# -----------------------------------------------------------------------------
# BFS utilities
# -----------------------------------------------------------------------------

from collections import deque

def _find_centroid_xy(id_map: np.ndarray, target_ids: List[int]) -> Optional[Tuple[int, int]]:
    """Return (x,y) centroid of all pixels whose id is in target_ids, else None."""
    if target_ids is None or len(target_ids) == 0:
        return None
    mask = np.isin(id_map, np.asarray(target_ids, dtype=id_map.dtype))
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    cx = int(np.round(xs.mean()))
    cy = int(np.round(ys.mean()))
    return cx, cy


def _find_left_center_xy(id_map: np.ndarray, target_ids: List[int]) -> Optional[Tuple[int, int]]:
    """Return (x_left, y_centroid) for pixels in target_ids, else None."""
    if target_ids is None or len(target_ids) == 0:
        return None
    mask = np.isin(id_map, np.asarray(target_ids, dtype=id_map.dtype))
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x_left = int(xs.min())
    y_centroid = int(np.round(ys.mean()))
    return x_left, y_centroid


def _find_right_center_xy(id_map: np.ndarray, target_ids: List[int]) -> Optional[Tuple[int, int]]:
    """Return (x_right, y_centroid) for pixels in target_ids, else None."""
    if target_ids is None or len(target_ids) == 0:
        return None
    mask = np.isin(id_map, np.asarray(target_ids, dtype=id_map.dtype))
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x_right = int(xs.max())
    y_centroid = int(np.round(ys.mean()))
    return x_right, y_centroid


def _bfs_distance_map(passable: np.ndarray, goal_xy: Tuple[int, int]) -> np.ndarray:
    """4-neighbor BFS distance-to-goal. Returns inf for unreachable cells."""
    H, W = passable.shape
    gx, gy = goal_xy
    dist = np.full((H, W), np.inf, dtype=np.float32)

    if not (0 <= gx < W and 0 <= gy < H):
        return dist
    if not passable[gy, gx]:
        return dist

    q = deque()
    dist[gy, gx] = 0.0
    q.append((gx, gy))

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        x, y = q.popleft()
        d = dist[y, x] + 1.0
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and passable[ny, nx]:
                if d < dist[ny, nx]:
                    dist[ny, nx] = d
                    q.append((nx, ny))

    return dist
