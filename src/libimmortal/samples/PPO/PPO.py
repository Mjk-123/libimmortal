# PPO.py
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:
    """
    Stores multimodal observations:
      - id_map: (H,W) int/uint8 class ids
      - vec_obs: (V,) float
    We keep them on CPU to avoid GPU memory blowup during rollout collection.
    """

    def __init__(self):
        self.id_maps: List[torch.Tensor] = []
        self.vec_obs: List[torch.Tensor] = []

        self.actions: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.state_values: List[torch.Tensor] = []
        self.is_terminals: List[bool] = []

    def add_state(self, id_map_np: np.ndarray, vec_np: np.ndarray):
        # Store on CPU
        id_map = torch.as_tensor(id_map_np, dtype=torch.uint8, device="cpu")
        vec = torch.as_tensor(vec_np, dtype=torch.float32, device="cpu")
        self.id_maps.append(id_map)
        self.vec_obs.append(vec)

    def clear(self):
        self.id_maps.clear()
        self.vec_obs.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.act(x + identity)
        return x


class CNNEncoder(nn.Module):
    """
    Encodes id_map (B,H,W) -> feature (B, C)
    We first embed ids -> (B,H,W,E), then permute -> (B,E,H,W), then CNN.
    """

    def __init__(self, num_ids: int, embed_dim: int = 16, base_channels: int = 32, out_dim: int = 256):
        super().__init__()
        self.id_embed = nn.Embedding(num_ids, embed_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(embed_dim, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.block1 = ResidualBlock(base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)

        self.block2 = ResidualBlock(base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.block3 = ResidualBlock(base_channels * 4)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels * 4, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, id_map_long: torch.Tensor) -> torch.Tensor:
        # id_map_long: (B,H,W) long
        x = self.id_embed(id_map_long)          # (B,H,W,E)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B,E,H,W)

        x = self.stem(x)
        x = self.block1(x)
        x = F.relu(self.down1(x), inplace=True)

        x = self.block2(x)
        x = F.relu(self.down2(x), inplace=True)

        x = self.block3(x)
        x = self.head(x)  # (B,out_dim)
        return x


class VecEncoder(nn.Module):
    def __init__(self, vec_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vec_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.net(v)


class Fusion(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    """
    Multimodal:
      - id_map: (B,H,W) long
      - vec_obs: (B,V) float
    """

    def __init__(
        self,
        num_ids: int,
        vec_dim: int,
        action_dim: int,
        action_std_init: float = 0.6,
        embed_dim: int = 16,
        cnn_base: int = 32,
        feat_dim: int = 256,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        self.cnn = CNNEncoder(num_ids=num_ids, embed_dim=embed_dim, base_channels=cnn_base, out_dim=feat_dim)
        self.vec = VecEncoder(vec_dim=vec_dim, out_dim=feat_dim)
        self.fuse = Fusion(in_dim=feat_dim * 2, out_dim=feat_dim)

        # Actor head (mean of Gaussian)
        self.actor = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        # Critic head (state value)
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def set_action_std(self, new_action_std: float):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def encode(self, id_map: torch.Tensor, vec_obs: torch.Tensor) -> torch.Tensor:
        # id_map: (B,H,W) long ; vec_obs: (B,V) float
        img_feat = self.cnn(id_map)
        vec_feat = self.vec(vec_obs)
        feat = self.fuse(torch.cat([img_feat, vec_feat], dim=-1))
        return feat

    def act(self, id_map: torch.Tensor, vec_obs: torch.Tensor):
        feat = self.encode(id_map, vec_obs)
        action_mean = self.actor(feat)

        cov_mat = torch.diag(self.action_var).unsqueeze(0).expand(action_mean.shape[0], -1, -1)
        dist = MultivariateNormal(action_mean, covariance_matrix=cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(feat).squeeze(-1)

        return action, action_logprob, state_val

    def evaluate(self, id_map: torch.Tensor, vec_obs: torch.Tensor, action: torch.Tensor):
        feat = self.encode(id_map, vec_obs)
        action_mean = self.actor(feat)

        cov_mat = torch.diag(self.action_var).unsqueeze(0).expand(action_mean.shape[0], -1, -1)
        dist = MultivariateNormal(action_mean, covariance_matrix=cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(feat).squeeze(-1)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(
        self,
        num_ids: int,
        vec_dim: int,
        action_dim: int,
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        K_epochs: int,
        eps_clip: float,
        action_std_init: float = 0.6,
        minibatch_size: int = 64,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.minibatch_size = minibatch_size

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(num_ids, vec_dim, action_dim, action_std_init).to(device)
        self.policy_old = ActorCritic(num_ids, vec_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
                {"params": self.policy.cnn.parameters(), "lr": lr_actor},
                {"params": self.policy.vec.parameters(), "lr": lr_actor},
                {"params": self.policy.fuse.parameters(), "lr": lr_actor},
            ]
        )

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std: float):
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate: float, min_action_std: float):
        # You can track current std outside if needed; here we read from policy_old.action_var
        cur_std = float(torch.sqrt(self.policy_old.action_var[0]).detach().cpu().item())
        new_std = max(min_action_std, round(cur_std - action_std_decay_rate, 4))
        self.set_action_std(new_std)

    @torch.no_grad()
    def select_action(self, id_map_np: np.ndarray, vec_np: np.ndarray) -> np.ndarray:
        # Store state first
        self.buffer.add_state(id_map_np, vec_np)

        id_map = torch.as_tensor(id_map_np, dtype=torch.long, device=device).unsqueeze(0)  # (1,H,W)
        vec = torch.as_tensor(vec_np, dtype=torch.float32, device=device).unsqueeze(0)     # (1,V)

        action, action_logprob, state_val = self.policy_old.act(id_map, vec)

        self.buffer.actions.append(action.squeeze(0).detach().cpu())
        self.buffer.logprobs.append(action_logprob.squeeze(0).detach().cpu())
        self.buffer.state_values.append(state_val.squeeze(0).detach().cpu())

        return action.squeeze(0).detach().cpu().numpy()

    def update(self):
        # ---------- compute returns ----------
        returns = []
        discounted = 0.0
        for r, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted = 0.0
            discounted = float(r) + self.gamma * discounted
            returns.insert(0, discounted)

        returns = torch.tensor(returns, dtype=torch.float32)  # CPU
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # ---------- stack rollout (CPU) ----------
        old_maps = torch.stack(self.buffer.id_maps, dim=0)          # (T,H,W) uint8 CPU
        old_vec = torch.stack(self.buffer.vec_obs, dim=0)           # (T,V) float32 CPU
        old_actions = torch.stack(self.buffer.actions, dim=0)       # (T,A) float32 CPU
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0)     # (T,) CPU
        old_state_values = torch.stack(self.buffer.state_values, dim=0)  # (T,) CPU

        advantages = returns - old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        T = old_actions.shape[0]
        idx_all = torch.arange(T)

        # ---------- PPO epochs w/ minibatches ----------
        for _ in range(self.K_epochs):
            perm = idx_all[torch.randperm(T)]

            for start in range(0, T, self.minibatch_size):
                mb_idx = perm[start : start + self.minibatch_size]

                mb_map = old_maps[mb_idx].to(device=device, dtype=torch.long)   # (B,H,W)
                mb_vec = old_vec[mb_idx].to(device=device, dtype=torch.float32) # (B,V)
                mb_actions = old_actions[mb_idx].to(device=device, dtype=torch.float32)  # (B,A)
                mb_old_logprobs = old_logprobs[mb_idx].to(device=device, dtype=torch.float32)  # (B,)
                mb_returns = returns[mb_idx].to(device=device, dtype=torch.float32)  # (B,)
                mb_adv = advantages[mb_idx].to(device=device, dtype=torch.float32)  # (B,)

                logprobs, state_values, dist_entropy = self.policy.evaluate(mb_map, mb_vec, mb_actions)

                ratios = torch.exp(logprobs - mb_old_logprobs)

                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv

                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = 0.5 * self.MseLoss(state_values, mb_returns)
                loss_entropy = -0.01 * dist_entropy.mean()

                loss = loss_actor + loss_critic + loss_entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path: str):
        payload = {
            "policy_old": self.policy_old.state_dict(),
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(payload, checkpoint_path)

    def load(self, checkpoint_path: str):
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.policy_old.load_state_dict(payload["policy_old"])
        self.policy.load_state_dict(payload["policy"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        # move to device
        self.policy.to(device)
        self.policy_old.to(device)
