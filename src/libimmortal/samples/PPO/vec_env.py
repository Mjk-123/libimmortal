# src/libimmortal/samples/PPO/vec_env.py
from __future__ import annotations

import time
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from libimmortal.env import ImmortalSufferingEnv
import libimmortal.samples.PPO.utils.utilities as utilities


def _get_action_space(env):
    if hasattr(env, "action_space"):
        return env.action_space
    if hasattr(env, "env") and hasattr(env.env, "action_space"):
        return env.env.action_space
    raise AttributeError("Cannot find action_space on env.")


def _noop_action_np(space) -> np.ndarray:
    # English comments for copy/paste friendliness.
    import gym
    if isinstance(space, gym.spaces.Box):
        return np.zeros(space.shape, dtype=np.float32)
    if isinstance(space, gym.spaces.Discrete):
        return np.array(0, dtype=np.int64)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return np.zeros((len(space.nvec),), dtype=np.int64)
    raise TypeError(f"Unsupported action space: {type(space)}")


def _bootstrap_obs(env):
    # Do NOT call env.reset() (reset is unreliable / can double-reset).
    space = _get_action_space(env)
    a_np = _noop_action_np(space)
    a_env = utilities._format_action_for_env(a_np, space)
    obs, r, d, info = env.step(a_env)
    return obs


@dataclass
class EnvInfo:
    has_continuous: bool
    action_dim: int
    action_nvec: Optional[List[int]]


def _infer_action_info(action_space) -> EnvInfo:
    import gym
    if isinstance(action_space, gym.spaces.Box):
        return EnvInfo(True, int(np.prod(action_space.shape)), None)
    if isinstance(action_space, gym.spaces.Discrete):
        return EnvInfo(False, int(action_space.n), None)
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        nvec = action_space.nvec.astype(int).tolist()
        # In your codebase, action_dim for MultiDiscrete is number of branches
        return EnvInfo(False, int(len(nvec)), nvec)
    raise TypeError(f"Unsupported action space: {type(action_space)}")


def _worker_loop(conn, cfg: Dict[str, Any]):
    """
    Worker owns one Unity instance.
    Commands:
      - ("reset", {"seed": int, "bump_port": bool})
      - ("step",  {"action": np.ndarray})
      - ("close", {})
    Replies:
      - ("reset_ok", (id_map, vec_obs, K, env_info_dict))
      - ("step_ok",  (id_map, vec_obs, reward, done, info))
      - ("err", (where, repr(e), traceback_str))
    """
    env = None
    restart_id = 0
    env_info = None
    expected_id_shape = None
    expected_vec_dim = None
    num_ids = None

    def _close_env():
        nonlocal env
        if env is None:
            return
        try:
            env.close()
        except Exception:
            pass
        env = None

    def _make_env(seed: int):
        nonlocal env, env_info, restart_id, expected_id_shape, expected_vec_dim, num_ids

        # Port layout:
        # base_port + local_rank*port_stride + env_idx*env_port_stride + restart_id*port_stride*world
        port = (
            int(cfg["port"])
            + int(cfg["local_rank"]) * int(cfg["port_stride"])
            + int(cfg["env_idx"]) * int(cfg["env_port_stride"])
            + int(restart_id) * int(cfg["port_stride"]) * int(cfg["world_size"])
        )

        kw = dict(
            game_path=cfg["game_path"],
            port=int(port),
            time_scale=float(cfg["time_scale"]),
            seed=int(seed),
            width=int(cfg["width"]),
            height=int(cfg["height"]),
            verbose=bool(cfg["verbose"]),
        )

        try:
            env = ImmortalSufferingEnv(
                **kw,
                no_graphics=bool(cfg.get("no_graphics", False)),
                timeout_wait=int(cfg.get("unity_timeout_wait", 120)),
            )
        except TypeError:
            env = ImmortalSufferingEnv(**kw)

        # Bootstrap instead of reset
        obs0 = _bootstrap_obs(env)
        id_map, vec_obs, K = utilities._make_map_and_vec(obs0)

        # Stabilize shapes
        if id_map is not None:
            id_map = np.asarray(id_map, dtype=np.int16)
            expected_id_shape = tuple(id_map.shape)
        if vec_obs is not None:
            vec_obs = np.asarray(vec_obs, dtype=np.float32).reshape(-1)
            expected_vec_dim = int(vec_obs.shape[0])

        num_ids = int(K) if K is not None else num_ids

        # Action info
        action_space = _get_action_space(env)
        env_info = _infer_action_info(action_space)

        return env, id_map, vec_obs, int(num_ids), env_info

    def _fix_obs(id_map, vec_obs):
        nonlocal expected_id_shape, expected_vec_dim

        if vec_obs is None:
            if expected_vec_dim is None:
                vec_obs = np.zeros((1,), dtype=np.float32)
                expected_vec_dim = 1
            else:
                vec_obs = np.zeros((expected_vec_dim,), dtype=np.float32)
        else:
            vec_obs = np.asarray(vec_obs, dtype=np.float32).reshape(-1)
            if expected_vec_dim is not None and vec_obs.shape[0] != expected_vec_dim:
                if vec_obs.shape[0] > expected_vec_dim:
                    vec_obs = vec_obs[:expected_vec_dim]
                else:
                    vec_obs = np.pad(vec_obs, (0, expected_vec_dim - vec_obs.shape[0]), mode="constant")

        if id_map is None:
            if expected_id_shape is None:
                id_map = np.zeros((90, 160), dtype=np.int16)
                expected_id_shape = tuple(id_map.shape)
            else:
                id_map = np.zeros(expected_id_shape, dtype=np.int16)
        else:
            id_map = np.asarray(id_map, dtype=np.int16)
            if expected_id_shape is not None and tuple(id_map.shape) != expected_id_shape:
                # If shape changes, force to expected by padding/cropping (rare).
                H, W = expected_id_shape
                out = np.zeros((H, W), dtype=np.int16)
                hh = min(H, id_map.shape[0])
                ww = min(W, id_map.shape[1])
                out[:hh, :ww] = id_map[:hh, :ww]
                id_map = out

        return id_map, vec_obs

    try:
        while True:
            cmd, payload = conn.recv()

            if cmd == "close":
                _close_env()
                break

            if cmd == "reset":
                seed = int(payload.get("seed", 0))
                bump_port = bool(payload.get("bump_port", True))

                _close_env()
                if bump_port:
                    restart_id += 1

                env, id_map, vec_obs, K, env_info = _make_env(seed)
                id_map, vec_obs = _fix_obs(id_map, vec_obs)

                conn.send(("reset_ok", (id_map, vec_obs, int(K), env_info.__dict__)))
                continue

            if cmd == "step":
                if env is None:
                    raise RuntimeError("env is None, call reset first")

                action_np = payload["action"]
                action_space = _get_action_space(env)
                action_env = utilities._format_action_for_env(action_np, action_space)

                obs, reward, done, info = env.step(action_env)
                id_map, vec_obs, _K = utilities._make_map_and_vec(obs)
                id_map, vec_obs = _fix_obs(id_map, vec_obs)

                conn.send(("step_ok", (id_map, vec_obs, reward, bool(done), info)))
                continue

            raise ValueError(f"Unknown cmd: {cmd}")

    except Exception as e:
        conn.send(("err", ("worker_loop", repr(e), traceback.format_exc())))
        try:
            _close_env()
        except Exception:
            pass


class VecImmortalEnv:
    """
    Vectorized env using N subprocess workers.

    Parent API:
      - reset_all(seeds: List[int]) -> (id_maps(N,H,W), vec_obs(N,D), K, env_info)
      - step(actions: np.ndarray (N,branches) or (N,)) -> (id_maps, vec_obs, rewards, dones, infos)
      - reset_one(i, seed)
      - close()
    """

    def __init__(self, *, num_envs: int, cfg_base: Dict[str, Any]):
        self.num_envs = int(num_envs)
        self.cfg_base = dict(cfg_base)
        self.ctx = mp.get_context("spawn")
        self.conns = []
        self.procs = []
        self.env_info = None
        self.K = None

        for env_idx in range(self.num_envs):
            parent_conn, child_conn = self.ctx.Pipe()
            cfg = dict(self.cfg_base)
            cfg["env_idx"] = int(env_idx)

            p = self.ctx.Process(target=_worker_loop, args=(child_conn, cfg), daemon=True)
            p.start()

            self.conns.append(parent_conn)
            self.procs.append(p)

    def _rpc(self, i: int, cmd: str, payload: Dict[str, Any]):
        self.conns[i].send((cmd, payload))

    def _recv(self, i: int):
        tag, data = self.conns[i].recv()
        if tag == "err":
            where, err_repr, tb = data
            raise RuntimeError(f"[VecImmortalEnv][{where}] {err_repr}\n{tb}")
        return tag, data

    def reset_all(self, seeds: List[int], bump_port: bool = True):
        assert len(seeds) == self.num_envs
        for i in range(self.num_envs):
            self._rpc(i, "reset", {"seed": int(seeds[i]), "bump_port": bool(bump_port)})

        id_maps = []
        vec_obs = []
        env_info = None
        K = None

        for i in range(self.num_envs):
            tag, data = self._recv(i)
            assert tag == "reset_ok"
            idm, vec, k, info_dict = data
            id_maps.append(idm)
            vec_obs.append(vec)
            K = int(k)
            if env_info is None:
                env_info = dict(info_dict)

        self.env_info = env_info
        self.K = int(K)

        id_maps = np.stack(id_maps, axis=0)  # (N,H,W)
        vec_obs = np.stack(vec_obs, axis=0)  # (N,D)
        return id_maps, vec_obs, int(self.K), dict(self.env_info)

    def reset_one(self, i: int, seed: int, bump_port: bool = True):
        self._rpc(i, "reset", {"seed": int(seed), "bump_port": bool(bump_port)})
        tag, data = self._recv(i)
        assert tag == "reset_ok"
        idm, vec, k, info_dict = data
        self.K = int(k)
        if self.env_info is None:
            self.env_info = dict(info_dict)
        return idm, vec

    def step(self, actions: np.ndarray):
        # actions: (N,branches) or (N,)
        actions = np.asarray(actions)

        for i in range(self.num_envs):
            self._rpc(i, "step", {"action": actions[i]})

        id_maps, vec_obs, rewards, dones, infos = [], [], [], [], []
        for i in range(self.num_envs):
            tag, data = self._recv(i)
            assert tag == "step_ok"
            idm, vec, r, d, info = data
            id_maps.append(idm)
            vec_obs.append(vec)
            rewards.append(r)
            dones.append(bool(d))
            infos.append(info)

        id_maps = np.stack(id_maps, axis=0)
        vec_obs = np.stack(vec_obs, axis=0)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(self.num_envs)
        dones = np.asarray(dones, dtype=np.bool_).reshape(self.num_envs)
        return id_maps, vec_obs, rewards, dones, infos

    def close(self, force: bool = False):
        for i in range(self.num_envs):
            try:
                self._rpc(i, "close", {})
            except Exception:
                pass
        for p in self.procs:
            try:
                p.join(timeout=1.0)
            except Exception:
                pass
        if force:
            for p in self.procs:
                try:
                    if p.is_alive():
                        p.terminate()
                except Exception:
                    pass
