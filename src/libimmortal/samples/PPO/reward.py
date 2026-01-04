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
    
import numpy as np

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
