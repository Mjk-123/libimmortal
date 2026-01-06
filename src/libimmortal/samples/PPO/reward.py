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
    mask = np.isin(id_map, np.asarray(target_ids, dtype=id_map.dtype))
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    cx = int(np.round(xs.mean()))
    cy = int(np.round(ys.mean()))
    return cx, cy

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
    terminal_failure_penalty: float = 0.0
    clip: Optional[float] = None

    # Success heuristics / bonus
    success_if_raw_reward_ge: Optional[float] = None
    time_limit: Optional[int] = None
    success_speed_bonus: float = 0.0

    # BFS progress shaping
    use_bfs_progress: bool = True
    bfs_update_every: int = 10  # recompute BFS map every N steps (goal fixed => can be large)
    bfs_blocked_ids: Optional[List[int]] = None
    bfs_player_ids: Optional[List[int]] = None
    bfs_goal_ids: Optional[List[int]] = None


class RewardShaper:
    """
    Call signature (recommended):
        reward = shaper(raw_reward, next_vec_obs, next_id_map, done, info)

    Reset signature (recommended):
        shaper.reset(initial_vec_obs, initial_id_map)

    Notes:
    - Progress term uses BFS shortest-path distance on id_map by default.
    - Falls back to vec_obs[IDX_GOAL_DIST] if player/goal not detectable on id_map.
    - Damage term uses delta of cumulative damage.
    - Time term uses delta of time.
    - Not-actionable term uses vec_obs[IDX_IS_ACTIONABLE] (0/1).
    """

    def __init__(self, cfg: RewardConfig, gamma: float = 0.99):
        # Distance normalizer
        self.map_h = 90 
        self.map_w = 160
        self.max_dist = float(self.map_h + self.map_w)

        self.cfg = cfg
        self.gamma = float(gamma)

        if self.cfg.bfs_blocked_ids is None:
            self.cfg.bfs_blocked_ids = list(DEFAULT_BLOCKED_IDS)
        if self.cfg.bfs_player_ids is None:
            self.cfg.bfs_player_ids = list(PLAYER_IDS)
        if self.cfg.bfs_goal_ids is None:
            self.cfg.bfs_goal_ids = [GOAL_ID]

        self._step = 0

        # previous vec values
        self._prev_cum_damage: Optional[float] = None
        self._prev_time: Optional[float] = None
        self._prev_goal_dist_fallback: Optional[float] = None

        # previous BFS distance (player->goal)
        self._prev_bfs_dist: Optional[float] = None

        # BFS cache
        self._cached_goal_xy: Optional[Tuple[int, int]] = None
        self._cached_dist_map: Optional[np.ndarray] = None

    def reset(self, vec_obs: np.ndarray, id_map: Optional[np.ndarray] = None):
        self._step = 0
        self._prev_cum_damage = float(vec_obs[IDX_CUM_DAMAGE]) if vec_obs is not None else None
        self._prev_time = float(vec_obs[IDX_TIME]) if vec_obs is not None else None
        self._prev_goal_dist_fallback = (float(vec_obs[IDX_GOAL_DIST]) / self.max_dist) if vec_obs is not None else None

        self._prev_bfs_dist = None
        self._cached_goal_xy = None
        self._cached_dist_map = None

        if id_map is not None and self.cfg.use_bfs_progress:
            d0 = self._get_bfs_dist(id_map)
            if d0 is not None:
                self._prev_bfs_dist = float(d0)

    def _passable_mask(self, id_map: np.ndarray) -> np.ndarray:
        blocked = np.isin(id_map, np.asarray(self.cfg.bfs_blocked_ids, dtype=id_map.dtype))
        return ~blocked

    def _get_bfs_dist(self, id_map: np.ndarray) -> Optional[float]:
        player_xy = _find_centroid_xy(id_map, self.cfg.bfs_player_ids)
        goal_xy   = _find_centroid_xy(id_map, self.cfg.bfs_goal_ids)
        if player_xy is None or goal_xy is None:
            return None

        need_recompute = (
            self._cached_dist_map is None
            or self._cached_goal_xy != goal_xy
            or (self._step % int(self.cfg.bfs_update_every)) == 0
        )

        if need_recompute:
            passable = self._passable_mask(id_map)
            gx, gy = goal_xy
            passable[gy, gx] = True
            self._cached_dist_map = _bfs_distance_map(passable, goal_xy)
            self._cached_goal_xy = goal_xy

        px, py = player_xy
        raw_d = float(self._cached_dist_map[py, px])
        
        if np.isfinite(raw_d):
            # --- Distance Normalization (0.0 ~ 1.0) ---
            return raw_d / self.max_dist
        return None

    def _is_success(self, raw_reward: float, done: bool, info: Dict[str, Any]) -> bool:
        if not done:
            return False
        if info is not None and bool(info.get("success", False)):
            return True
        if self.cfg.success_if_raw_reward_ge is not None:
            return float(raw_reward) >= float(self.cfg.success_if_raw_reward_ge)
        return False

    def __call__(
        self,
        raw_reward: float,
        vec_obs: np.ndarray,
        id_map: Optional[np.ndarray],
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        if info is None:
            info = {}

         # Step counter: increment exactly once per env step.
        self._step += 1
        r = 0.0

        # (0) Keep raw env reward (if you want purely shaped reward, set raw_reward weight outside)
        r += float(raw_reward)

        # (1) Progress term (BFS distance preferred, fallback to vec goal distance)
        progress = 0.0
        used_bfs = False

        if self.cfg.use_bfs_progress and id_map is not None:
            d_bfs = self._get_bfs_dist(id_map)
            if d_bfs is not None:
                used_bfs = True
                if self._prev_bfs_dist is not None:
                    progress = float(self._prev_bfs_dist - d_bfs)
                self._prev_bfs_dist = float(d_bfs)

        if (not used_bfs) and vec_obs is not None:
            d_fallback = float(vec_obs[IDX_GOAL_DIST]) / self.max_dist
            if self._prev_goal_dist_fallback is not None:
                progress = float(self._prev_goal_dist_fallback - d_fallback)
            self._prev_goal_dist_fallback = float(d_fallback)

        r += float(self.cfg.w_progress) * float(progress)

        # (2) Time penalty (delta time)
        if vec_obs is not None and self.cfg.w_time != 0.0:
            t = float(vec_obs[IDX_TIME])
            if self._prev_time is not None:
                dt = max(0.0, t - float(self._prev_time))
                r += float(self.cfg.w_time) * (-dt)  # w_time > 0 means penalize time passing
            self._prev_time = t

        # (3) Damage penalty (delta of cumulative damage)
        if vec_obs is not None and self.cfg.w_damage != 0.0:
            dmg = float(vec_obs[IDX_CUM_DAMAGE])
            if self._prev_cum_damage is not None:
                ddmg = max(0.0, dmg - float(self._prev_cum_damage))
                r += float(self.cfg.w_damage) * (-ddmg)  # w_damage > 0 means penalize taking damage
            self._prev_cum_damage = dmg

        # (4) Not actionable penalty
        if vec_obs is not None and self.cfg.w_not_actionable != 0.0:
            actionable = float(vec_obs[IDX_IS_ACTIONABLE])
            if actionable < 0.5:
                r += float(self.cfg.w_not_actionable) * (-1.0)

        # (5) Terminal shaping: failure penalty / success speed bonus
        if done:
            success = self._is_success(raw_reward=float(raw_reward), done=done, info=info)

            if not success:
                r += float(self.cfg.terminal_failure_penalty)

            # Speed bonus: more bonus if success with smaller time
            if success and self.cfg.success_speed_bonus != 0.0:
                if (self.cfg.time_limit is not None) and (vec_obs is not None):
                    t = float(vec_obs[IDX_TIME])
                    tl = float(self.cfg.time_limit)
                    frac = 1.0 - np.clip(t / max(tl, 1.0), 0.0, 1.0)
                    r += float(self.cfg.success_speed_bonus) * float(frac)
                else:
                    r += float(self.cfg.success_speed_bonus)

        # (6) Clip
        if self.cfg.clip is not None:
            r = float(np.clip(r, -float(self.cfg.clip), float(self.cfg.clip)))

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