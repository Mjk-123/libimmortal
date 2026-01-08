# reward.py
# Reward shaping with BFS shortest-path distance on id_map.
# Vector obs indices are integrated:
#   IDX_CUM_DAMAGE = 4
#   IDX_IS_ACTIONABLE = 5
#   IDX_GOAL_DIST = 11
#   IDX_TIME = 12

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

# -------------------------
# Vector observation indices
# -------------------------
IDX_CUM_DAMAGE = 4
IDX_IS_ACTIONABLE = 5
IDX_GOAL_DIST = 11
IDX_TIME = 12


# -------------------------
# ColorMap IDs (from your encoder)
# -------------------------
# Adjust import path to wherever DEFAULT_ENCODER lives.

from libimmortal.utils.aux_func import DEFAULT_ENCODER

ENC = DEFAULT_ENCODER

WALL_ID = ENC.name2id["WALL"]
GOAL_ID = ENC.name2id["GOAL"]

# Player marker on minimap
PLAYER_IDS = [ENC.name2id["KNIGHT"], ENC.name2id["KNIGHT_ATTACK"]]

# By default, only WALL blocks movement in BFS.
DEFAULT_BLOCKED_IDS = [WALL_ID]
# If PLATFORM is also solid in the minimap, uncomment:
# DEFAULT_BLOCKED_IDS = [WALL_ID, ENC.name2id["PLATFORM"]]


# -------------------------
# BFS utilities
# -------------------------
'''
def _find_first_xy(id_map: np.ndarray, target_ids: List[int]) -> Optional[Tuple[int, int]]:
    """Return (x,y) of the first occurrence of any target id, else None."""
    for tid in target_ids:
        ys, xs = np.where(id_map == tid)
        if xs.size > 0:
            return int(xs[0]), int(ys[0])
    return None
'''

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
'''
def _find_left_center_xy(id_map: np.ndarray, target_ids: List[int]) -> Optional[Tuple[int, int]]:
    """Return (x_left, y_centroid) for the object pixels in target_ids, else None."""
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
    """Return (x_right, y_centroid) for the object pixels in target_ids, else None."""
    if target_ids is None or len(target_ids) == 0:
        return None
    mask = np.isin(id_map, np.asarray(target_ids, dtype=id_map.dtype))
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x_right = int(xs.max())
    y_centroid = int(np.round(ys.mean()))
    return x_right, y_centroid
'''

def _bfs_distance_map(passable: np.ndarray, goal_xy: Tuple[int, int]) -> np.ndarray:
    """4-neighbor BFS distance to goal. Returns inf for unreachable."""
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


# -------------------------
# Reward config
# -------------------------
@dataclass
class RewardConfig:
    w_progress: float = 1.0
    w_time: float = 0.0
    w_damage: float = 0.0
    w_not_actionable: float = 0.0
    terminal_bonus: float = 5.0
    clip: Optional[float] = None

    # Success heuristics / bonus
    success_if_raw_reward_ge: Optional[float] = None
    time_limit: Optional[int] = None
    success_speed_bonus: float = 0.0

    # BFS progress shaping
    use_bfs_progress: bool = True
    w_acceleration: float = 1.0,
    bfs_update_every: int = 10  # recompute BFS map every N steps (goal fixed => can be large)


class RewardShaper:
    def __init__(self, cfg: Any, gamma: float = 0.99):
        self.map_h, self.map_w = 90, 160
        self.max_dist = float(self.map_h + self.map_w)

        self.cfg = cfg
        self.gamma = float(gamma)

        # Config fallback
        self.update_every = int(getattr(self.cfg, "bfs_update_every", 10))

        self._step = 0
        self._prev_cum_damage = None
        self._prev_time = None
        self._prev_bfs_dist = None

        # Bounded goal potential (delta-based magnet shaping)
        self._prev_goal_phi = None

        self._cached_goal_xy = None
        self._cached_dist_map = None

        self.virtual_done = False

        self.dbg_has_bfs = False
        self.dbg_bfs_dist = None          # normalized [0,1]
        self.dbg_bfs_delta = None         # prev - curr (normalized)
        self.dbg_closer = False
        self.dbg_farther = False
        self.dbg_is_success = False

    def reset(self, vec_obs: np.ndarray, id_map: Optional[np.ndarray] = None):
        self._step = 0
        self.virtual_done = False
        self._prev_cum_damage = float(vec_obs[IDX_CUM_DAMAGE]) if vec_obs is not None else None
        self._prev_time = float(vec_obs[IDX_TIME]) if vec_obs is not None else None
        self._prev_bfs_dist = None
        self._prev_goal_phi = None
        self._cached_goal_xy, self._cached_dist_map = None, None

        if id_map is not None and self.cfg.use_bfs_progress:
            d0 = self._get_bfs_dist(id_map)
            self._prev_bfs_dist = float(d0) if d0 is not None else None

            # Initialize bounded potential to avoid a large first-step delta
            if self._prev_bfs_dist is not None:
                alpha = 30.0  # larger => more emphasis near the goal, still bounded in (0,1]
                self._prev_goal_phi = float(np.exp(-alpha * float(self._prev_bfs_dist)))

    def _get_bfs_dist(self, id_map: np.ndarray) -> Optional[float]:
        player_xy = _find_centroid_xy(id_map, PLAYER_IDS)
        goal_xy = _find_centroid_xy(id_map, [GOAL_ID])
        if player_xy is None or goal_xy is None:
            return None

        if (
            self._cached_dist_map is None
            or self._cached_goal_xy != goal_xy
            or (self._step % self.update_every) == 0
        ):
            passable = ~np.isin(id_map, np.asarray(DEFAULT_BLOCKED_IDS, dtype=id_map.dtype))
            gx, gy = goal_xy
            passable[gy, gx] = True
            self._cached_dist_map = _bfs_distance_map(passable, goal_xy)
            self._cached_goal_xy = goal_xy

        px, py = player_xy
        raw_d = float(self._cached_dist_map[py, px])
        return raw_d / self.max_dist if np.isfinite(raw_d) else None

    def __call__(
        self,
        raw_reward: float,
        vec_obs: np.ndarray,
        id_map: Optional[np.ndarray],
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        info = info or {}
        self._step += 1
        r = float(raw_reward)

        # 1) Distance signals
        prev_d_bfs = self._prev_bfs_dist
        d_bfs = self._get_bfs_dist(id_map) if (self.cfg.use_bfs_progress and id_map is not None) else None
        raw_goal_dist = float(vec_obs[IDX_GOAL_DIST]) if vec_obs is not None else 999.0

        # 2) Unified success condition
        # BFS-based (within ~1.5 tiles) OR vector-based (< 0.6) OR raw_reward>=1.0
        is_success = (float(raw_reward) >= 1.0) or (
            (d_bfs is not None and d_bfs <= (1.5 / self.max_dist)) or (raw_goal_dist < 0.6)
        )
        self.dbg_has_bfs = (d_bfs is not None)
        self.dbg_bfs_dist = float(d_bfs) if d_bfs is not None else None
        if (prev_d_bfs is not None) and (d_bfs is not None):
            dd = float(prev_d_bfs - d_bfs)
            self.dbg_bfs_delta = dd
            self.dbg_closer = (dd > 0.0)
            self.dbg_farther = (dd < 0.0)
        else:
            self.dbg_bfs_delta = None
            self.dbg_closer = False
            self.dbg_farther = False
        self.dbg_is_success = bool(is_success)

        # -----------------------------------------------------------
        # [Progress & Magnet Reward]
        # -----------------------------------------------------------
        if d_bfs is not None:
            # A) Linear progress (only reward if getting closer)
            if self._prev_bfs_dist is not None:
                r += float(self.cfg.w_progress) * (self._prev_bfs_dist - d_bfs)

            # B) Bounded "magnet" shaping: delta of exp(-alpha*d)
            # - phi(d) in (0,1], bounded => no explosion
            # - derivative increases as d->0 => stronger pull near goal
            alpha = 30.0
            w_acc = float(getattr(self.cfg, "w_acceleration", 1.0))

            cur_phi = float(np.exp(-alpha * float(d_bfs)))
            if self._prev_goal_phi is not None:
                r += w_acc * (cur_phi - self._prev_goal_phi)
            self._prev_goal_phi = cur_phi

            self._prev_bfs_dist = d_bfs

        # -----------------------------------------------------------
        # [Terminal Logic]
        # -----------------------------------------------------------
        did_just_succeed = False

        if is_success and (not self.virtual_done):
            # Pay terminal reward only once per episode
            self.virtual_done = True
            did_just_succeed = True

        # -----------------------------------------------------------
        # [Penalty Terms]
        # -----------------------------------------------------------
        if vec_obs is not None:
            # Time penalty (constant per step)
            r += float(self.cfg.w_time) * -1.0

            # Damage penalty (delta of cumulative damage)
            dmg = float(vec_obs[IDX_CUM_DAMAGE])
            if self._prev_cum_damage is not None:
                r += float(self.cfg.w_damage) * -max(0.0, dmg - self._prev_cum_damage)
            self._prev_cum_damage = dmg

            # Not-actionable penalty
            if float(vec_obs[IDX_IS_ACTIONABLE]) < 0.5:
                r += float(self.cfg.w_not_actionable) * -1.0

        # -----------------------------------------------------------
        # [Guarantee: goal entry is the largest per-step reward]
        # -----------------------------------------------------------
        if self.cfg.clip:
            clip = float(self.cfg.clip)

            # Terminal target can be > clip (terminal is "outside" the per-step clip).
            terminal_bonus = float(getattr(self.cfg, "terminal_bonus", 1.0))

            if did_just_succeed:
                # Success step becomes exactly terminal_bonus (not clipped).
                return float(terminal_bonus)

            # Non-success steps: clip normally, but if terminal_bonus <= clip, force strict separation.
            hi = clip if terminal_bonus > clip else (terminal_bonus - 1e-3)
            return float(np.clip(r, -clip, hi))

        return float(r)

class RunningMeanStd:
    """Track running mean/variance with Welford-style parallel update."""
    def __init__(self, epsilon: float = 1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta * delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class RewardScaler:
    def __init__(self, gamma: float, clip: float = 5.0, eps: float = 1e-8, min_std: float = 0.1, warmup_steps: int = 2000):
        self.gamma = float(gamma)
        self.clip = float(clip)
        self.eps = float(eps)
        self.min_std = float(min_std)
        self.warmup_steps = int(warmup_steps)

        self.ret_rms = RunningMeanStd(shape=())
        self.ret = 0.0
        self.t = 0

    def __call__(self, reward: float, done: bool) -> float:
        self.t += 1
        self.ret = self.ret * self.gamma + float(reward)
        self.ret_rms.update(np.array([self.ret], dtype=np.float64))

        if done:
            self.ret = 0.0

        # During warmup, return raw reward (or lightly clipped raw reward)
        if self.t < self.warmup_steps:
            return float(np.clip(reward, -self.clip, self.clip)) if self.clip is not None else float(reward)

        std = float(np.sqrt(self.ret_rms.var + self.eps))
        std = max(std, self.min_std)

        r = float(reward) / std
        if self.clip is not None:
            r = float(np.clip(r, -self.clip, self.clip))
        return r