#!/usr/bin/env bash
set -euo pipefail

GAME_PATH="${GAME_PATH:-/root/immortal_suffering/immortal_suffering_linux_build.x86_64}"

chmod +x "$GAME_PATH" 2>/dev/null || true
pkill -f immortal_suffering_linux_build.x86_64 2>/dev/null || true

python3 - <<'PY'
from libimmortal import ImmortalSufferingEnv
from libimmortal.utils import parse_observation
import socket

GAME_PATH = "/root/immortal_suffering/immortal_suffering_linux_build.x86_64"

def find_free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

port = find_free_port()
env = ImmortalSufferingEnv(
    game_path=GAME_PATH,
    port=port,
    time_scale=1.0,
    seed=0,
    width=160,
    height=90,
    verbose=False,
)

try:
    print("=== INSPECT ===")
    print("action_space:", env.env.action_space)
    print("observation_space:", env.env.observation_space)

    obs = env.reset()
    graphic_obs, vector_obs = parse_observation(obs)
    print("graphic shape/dtype:", graphic_obs.shape, graphic_obs.dtype)
    print("vector  shape/dtype:", vector_obs.shape, vector_obs.dtype)
    print("sample action:", env.env.action_space.sample())
finally:
    env.close()
    print("closed ok")
PY

python3 - <<'PY'
from libimmortal import ImmortalSufferingEnv
from libimmortal.utils import parse_observation
import socket

GAME_PATH = "/root/immortal_suffering/immortal_suffering_linux_build.x86_64"

def find_free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    def act(self, obs):
        return self.action_space.sample()

port = find_free_port()
env = ImmortalSufferingEnv(
    game_path=GAME_PATH,
    port=port,
    time_scale=3.0,
    seed=0,
    width=160,
    height=90,
    verbose=False,
)

try:
    agent = RandomAgent(env.env.action_space)
    obs = env.reset()
    total_reward = 0.0

    print("=== RANDOM RUN (200 steps) ===")
    for t in range(200):
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        if done:
            obs = env.reset()

    print("done. total_reward:", total_reward)
finally:
    env.close()
    print("closed ok")
PY

