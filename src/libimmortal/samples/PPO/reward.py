# reward.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import numpy as np

from libimmortal.utils.aux_func import DEFAULT_ENCODER

import libimmortal.samples.PPO.utils.utilities as utilities

# -----------------------------------------------------------------------------
# Vector observation indices
# -----------------------------------------------------------------------------
IDX_CUM_DAMAGE = 4
IDX_IS_ACTIONABLE = 5
IDX_GOAL_DIST = 11
IDX_TIME = 12


# -----------------------------------------------------------------------------
# Encoder / IDs
# -----------------------------------------------------------------------------
ENC = DEFAULT_ENCODER

WALL_ID = ENC.name2id["WALL"]
GOAL_ID = ENC.name2id["GOAL"]

# Player marker on minimap
PLAYER_IDS = [ENC.name2id["KNIGHT"], ENC.name2id["KNIGHT_ATTACK"]]

# By default, only WALL blocks movement in BFS.
DEFAULT_BLOCKED_IDS = [WALL_ID]
# If PLATFORM is also solid in the minimap, uncomment:
# DEFAULT_BLOCKED_IDS = [WALL_ID, ENC.name2id["PLATFORM"]]


# -----------------------------------------------------------------------------
# Reward config
# -----------------------------------------------------------------------------
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
    w_acceleration: float = 1.0  # IMPORTANT: no trailing comma (tuple bug)
    bfs_update_every: int = 10  # kept for compatibility; with caching, can be ignored

    # Optional: map shape if you want to override defaults
    map_h: int = 90
    map_w: int = 160


# -----------------------------------------------------------------------------
# Reward shaper
# -----------------------------------------------------------------------------
class RewardShaper:
    """
    Call signature (recommended):
        reward = shaper(raw_reward, vec_obs, id_map, done, info)

    Reset signature (recommended):
        shaper.reset(initial_vec_obs, initial_id_map)

    Notes:
    - With fixed goal_xy + fixed passable, dist_map is fixed and should be computed once at reset.
    - Per-step BFS distance becomes O(1): dist_map[py, px].
    - The remaining per-step cost is finding player_xy from id_map.
    """

    def __init__(self, cfg: Any, gamma: float = 0.99):
        self.cfg = cfg
        self.gamma = float(gamma)

        self.map_h = int(getattr(cfg, "map_h", 90))
        self.map_w = int(getattr(cfg, "map_w", 160))
        self.max_dist = float(self.map_h + self.map_w)

        # Config fallback
        self.update_every = int(getattr(self.cfg, "bfs_update_every", 10))

        self._step = 0
        self._prev_cum_damage: Optional[float] = None
        self._prev_time: Optional[float] = None
        self._prev_bfs_dist: Optional[float] = None

        # Bounded goal potential (delta-based magnet shaping)
        self._prev_goal_phi: Optional[float] = None

        # Cached BFS artifacts (goal/passable/dist map)
        self._cached_goal_xy: Optional[Tuple[int, int]] = None
        self._cached_dist_map: Optional[np.ndarray] = None
        self._cached_passable: Optional[np.ndarray] = None  # bool mask

        self.virtual_done = False

        # Debug
        self.dbg_has_bfs = False
        self.dbg_bfs_dist: Optional[float] = None          # normalized [0,1]
        self.dbg_bfs_delta: Optional[float] = None         # prev - curr (normalized)
        self.dbg_closer = False
        self.dbg_farther = False
        self.dbg_is_success = False

    def _ensure_bfs_cache(self, id_map: np.ndarray) -> None:
        """
        Ensure passable, goal_xy, and dist_map are cached.
        Assumes map/passable/goal are fixed for the episode (or for the whole run).
        """
        H, W = id_map.shape[:2]

        # (1) Cache passable mask once per shape.
        if (self._cached_passable is None) or (self._cached_passable.shape != (H, W)):
            self._cached_passable = ~np.isin(
                id_map, np.asarray(DEFAULT_BLOCKED_IDS, dtype=id_map.dtype)
            )

        # (2) Cache goal_xy (try to set it if missing).
        if self._cached_goal_xy is None:
            self._cached_goal_xy = utilities._find_centroid_xy(id_map, [GOAL_ID])

        # (3) Cache dist_map once per shape (depends only on passable + goal_xy).
        if self._cached_goal_xy is None:
            self._cached_dist_map = None
            return

        if (self._cached_dist_map is None) or (self._cached_dist_map.shape != (H, W)):
            gx, gy = self._cached_goal_xy
            # Copy passable so we can force-goal cell passable without mutating base mask.
            passable = self._cached_passable.copy()
            if 0 <= gx < W and 0 <= gy < H:
                passable[gy, gx] = True
            self._cached_dist_map = utilities._bfs_distance_map(passable, self._cached_goal_xy)

    def reset(self, vec_obs: np.ndarray, id_map: Optional[np.ndarray] = None):
        self._step = 0
        self.virtual_done = False

        self._prev_cum_damage = float(vec_obs[IDX_CUM_DAMAGE]) if vec_obs is not None else None
        self._prev_time = float(vec_obs[IDX_TIME]) if vec_obs is not None else None
        self._prev_bfs_dist = None
        self._prev_goal_phi = None

        # Do NOT blindly clear BFS caches; map/goal are assumed fixed.
        # We'll rebuild only when missing or shape changed.
        if id_map is not None and bool(getattr(self.cfg, "use_bfs_progress", True)):
            self._ensure_bfs_cache(id_map)

            d0 = self._get_bfs_dist(id_map)
            self._prev_bfs_dist = float(d0) if d0 is not None else None

            # Initialize bounded potential to avoid a large first-step delta
            if self._prev_bfs_dist is not None:
                alpha = 30.0  # larger => more emphasis near the goal, still bounded in (0,1]
                self._prev_goal_phi = float(np.exp(-alpha * float(self._prev_bfs_dist)))

    def _get_bfs_dist(self, id_map: np.ndarray) -> Optional[float]:
        # You can swap centroid -> right-center later if you want:
        # player_xy = _find_right_center_xy(id_map, PLAYER_IDS)
        player_xy = utilities._find_centroid_xy(id_map, PLAYER_IDS)
        if player_xy is None:
            return None

        if self._cached_dist_map is None:
            # Lazy build (in case reset() didn't receive id_map)
            self._ensure_bfs_cache(id_map)

        if self._cached_dist_map is None:
            return None

        px, py = player_xy
        H, W = self._cached_dist_map.shape
        if not (0 <= px < W and 0 <= py < H):
            return None

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
        # If map observation is missing, skip BFS-based terms.
        # Keep training stable by falling back to vec-based distance or no progress term.
        if id_map is None:
            self.dbg_has_bfs = True
            self.dbg_bfs_dist = None
            self.dbg_bfs_delta = None
            # IMPORTANT: do not raise; just compute reward without BFS.
            return self._reward_without_bfs(raw_reward, vec_obs, done, info)
        
        info = info or {}
        self._step += 1
        r = float(raw_reward)

        # 1) Distance signals
        prev_d_bfs = self._prev_bfs_dist
        d_bfs = self._get_bfs_dist(id_map) if (bool(getattr(self.cfg, "use_bfs_progress", True)) and id_map is not None) else None
        raw_goal_dist = float(vec_obs[IDX_GOAL_DIST]) if vec_obs is not None else 999.0

        # 2) Unified success condition
        # BFS-based (within ~1.5 tiles) OR vector-based (< 0.6) OR raw_reward>=1.0
        is_success = (float(raw_reward) >= 1.0) or (
            (d_bfs is not None and d_bfs <= (1.5 / self.max_dist)) or (raw_goal_dist < 0.6)
        )

        # Debug stats for trainer-side logging
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
            # A) Linear progress (SIGNED; no max(0, ...) farming)
            if self._prev_bfs_dist is not None:
                r += float(getattr(self.cfg, "w_progress", 0.0)) * (self._prev_bfs_dist - d_bfs)

            # B) Bounded "magnet" shaping: delta of exp(-alpha*d)
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
            r += float(getattr(self.cfg, "w_time", 0.0)) * -1.0

            # Damage penalty (delta of cumulative damage)
            dmg = float(vec_obs[IDX_CUM_DAMAGE])
            if self._prev_cum_damage is not None:
                r += float(getattr(self.cfg, "w_damage", 0.0)) * -max(0.0, dmg - self._prev_cum_damage)
            self._prev_cum_damage = dmg

            # Not-actionable penalty
            if float(vec_obs[IDX_IS_ACTIONABLE]) < 0.5:
                r += float(getattr(self.cfg, "w_not_actionable", 0.0)) * -1.0

        # -----------------------------------------------------------
        # [Guarantee: goal entry is the largest per-step reward]
        # -----------------------------------------------------------
        if getattr(self.cfg, "clip", None):
            clip = float(getattr(self.cfg, "clip", 0.0))
            terminal_bonus = float(getattr(self.cfg, "terminal_bonus", 1.0))

            if did_just_succeed:
                # Success step becomes exactly terminal_bonus (not clipped).
                return float(terminal_bonus)

            # Non-success steps: clip normally, but if terminal_bonus <= clip, force strict separation.
            hi = clip if terminal_bonus > clip else (terminal_bonus - 1e-3)
            return float(np.clip(r, -clip, hi))

        return float(r)
    
    def _reward_without_bfs(
        self,
        raw_reward: float,
        vec_obs: Optional[np.ndarray],
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Fallback reward when id_map is missing (id_map=None).
        Mirrors __call__'s structure but skips BFS-based progress/magnet terms.
        """
        info = info or {}
        self._step += 1

        r = float(raw_reward)

        # Vector-based goal distance fallback (if available)
        raw_goal_dist = float(vec_obs[IDX_GOAL_DIST]) if vec_obs is not None else 999.0

        # Same success condition as __call__ but without BFS term
        is_success = (r >= 1.0) or (raw_goal_dist < 0.6)

        # Debug flags (no BFS available)
        self.dbg_has_bfs = False
        self.dbg_bfs_dist = None
        self.dbg_bfs_delta = None
        self.dbg_closer = False
        self.dbg_farther = False
        self.dbg_is_success = bool(is_success)

        # Break BFS continuity so we don't compute deltas using stale values later.
        self._prev_bfs_dist = None
        self._prev_goal_phi = None

        # Terminal-on-missing-map often happens; ensure we don't carry damage deltas across episode boundary.
        if done:
            self._prev_cum_damage = None

        # -----------------------------------------------------------
        # [Terminal Logic]
        # -----------------------------------------------------------
        did_just_succeed = False
        if is_success and (not self.virtual_done):
            self.virtual_done = True
            did_just_succeed = True

        # -----------------------------------------------------------
        # [Penalty Terms]
        # -----------------------------------------------------------
        if vec_obs is not None:
            r += float(getattr(self.cfg, "w_time", 0.0)) * -1.0

            dmg = float(vec_obs[IDX_CUM_DAMAGE])
            if self._prev_cum_damage is not None:
                r += float(getattr(self.cfg, "w_damage", 0.0)) * -max(0.0, dmg - self._prev_cum_damage)
            self._prev_cum_damage = dmg

            if float(vec_obs[IDX_IS_ACTIONABLE]) < 0.5:
                r += float(getattr(self.cfg, "w_not_actionable", 0.0)) * -1.0

        # -----------------------------------------------------------
        # [Guarantee: goal entry is the largest per-step reward]
        # -----------------------------------------------------------
        if getattr(self.cfg, "clip", None):
            clip = float(getattr(self.cfg, "clip", 0.0))
            terminal_bonus = float(getattr(self.cfg, "terminal_bonus", 1.0))

            if did_just_succeed:
                return float(terminal_bonus)

            hi = clip if terminal_bonus > clip else (terminal_bonus - 1e-3)
            return float(np.clip(r, -clip, hi))

        if did_just_succeed:
            return float(getattr(self.cfg, "terminal_bonus", 1.0))

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