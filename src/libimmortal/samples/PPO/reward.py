# reward.py (baseline episode-relative version)

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

IDX_CUM_DAMAGE = 4
IDX_IS_ACTIONABLE = 5
IDX_GOAL_DIST = 11
IDX_TIME = 12


@dataclass
class RewardConfig:
    w_progress: float = 1.0
    w_time: float = 0.001
    w_damage: float = 0.05
    w_not_actionable: float = 0.01

    terminal_failure_penalty: float = 1.0
    clip: float = 5.0

    success_if_raw_reward_ge: float = 1.0

    # speed bonus on success (episode-relative)
    time_limit: float = 300.0
    success_speed_bonus: float = 1.0


class RewardShaper:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

        # episode baselines
        self.t0: Optional[float] = None
        self.dmg0: Optional[float] = None

        # previous values within episode (episode-relative)
        self.prev_dist: Optional[float] = None
        self.prev_t_ep: Optional[float] = None
        self.prev_dmg_ep: Optional[float] = None

    def reset(self, vector_obs: np.ndarray) -> None:
        v = np.asarray(vector_obs, dtype=np.float32).reshape(-1)
        t = float(v[IDX_TIME])
        dmg = float(v[IDX_CUM_DAMAGE])
        dist = float(v[IDX_GOAL_DIST])

        self.t0 = t
        self.dmg0 = dmg

        self.prev_dist = dist
        self.prev_t_ep = 0.0
        self.prev_dmg_ep = 0.0

    def __call__(self, raw_reward: float, vector_obs: np.ndarray, done: bool, info: Optional[Dict[str, Any]] = None) -> float:
        if self.t0 is None:
            self.reset(vector_obs)

        v = np.asarray(vector_obs, dtype=np.float32).reshape(-1)

        dist = float(v[IDX_GOAL_DIST])
        t = float(v[IDX_TIME])
        dmg = float(v[IDX_CUM_DAMAGE])
        actionable = float(v[IDX_IS_ACTIONABLE])

        # episode-relative scalars
        t_ep = t - self.t0
        dmg_ep = dmg - self.dmg0

        # deltas
        d_progress = (self.prev_dist - dist)
        d_time = max(0.0, t_ep - self.prev_t_ep)
        d_dmg = max(0.0, dmg_ep - self.prev_dmg_ep)

        shaped = float(raw_reward)
        shaped += self.cfg.w_progress * d_progress
        shaped -= self.cfg.w_time * d_time
        shaped -= self.cfg.w_damage * d_dmg
        shaped -= self.cfg.w_not_actionable * (1.0 - actionable)

        success = float(raw_reward) >= self.cfg.success_if_raw_reward_ge

        if done and success:
            remaining_frac = max(0.0, 1.0 - (t_ep / max(1e-6, self.cfg.time_limit)))
            shaped += self.cfg.success_speed_bonus * remaining_frac

        if done and not success:
            shaped -= self.cfg.terminal_failure_penalty

        # update prevs
        self.prev_dist = dist
        self.prev_t_ep = t_ep
        self.prev_dmg_ep = dmg_ep

        if self.cfg.clip is not None:
            shaped = float(np.clip(shaped, -self.cfg.clip, self.cfg.clip))

        if isinstance(info, dict):
            info["raw_reward"] = float(raw_reward)
            info["shaped_reward"] = float(shaped)
            info["time_ep"] = float(t_ep)
            info["dmg_ep"] = float(dmg_ep)
            info["delta_progress"] = float(d_progress)

        return shaped
