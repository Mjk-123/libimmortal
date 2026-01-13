# src/libimmortal/samples/PPO/ppo_vec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal

@dataclass
class RolloutBatch:
    """
    Vectorized rollout batch.

    Shapes:
      H = horizon steps per env
      N = number of envs
      T = H*N

    map_states: (H, N, Hmap, Wmap) int64 (CPU)
    vec_states: (H, N, D) float32 (CPU)
    actions:    (H, N, branches) int64 for MultiDiscrete OR (H, N) for Discrete (CPU)
    logprobs:   (H, N) float32 (CPU)
    values:     (H, N) float32 (CPU)
    rewards:    (H, N) float32 (CPU)
    dones:      (H, N) float32 (CPU)   # 1.0 if terminal for buffer
    last_values:(N,)   float32 (CPU)   # bootstrap V(s_{H}) (0 for terminal envs)
    """
    map_states: torch.Tensor
    vec_states: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    last_values: torch.Tensor

class RolloutBufferVec:
    def __init__(self, H_rollout: int, N: int, map_shape, vec_dim: int, num_branches: int):
        # H_rollout = horizon steps per env
        self.H = int(H_rollout)
        self.N = int(N)
        H, W = map_shape

        # Pre-allocate on CPU (numpy) for speed + simplicity
        self.map_states = np.zeros((self.H, self.N, H, W), dtype=np.int16)     # ids fit in int16 usually
        self.vec_states = np.zeros((self.H, self.N, vec_dim), dtype=np.float32)

        self.actions = np.zeros((self.H, self.N, num_branches), dtype=np.int64)   # MultiDiscrete
        self.logprobs = np.zeros((self.H, self.N), dtype=np.float32)
        self.values = np.zeros((self.H, self.N), dtype=np.float32)
        self.rewards = np.zeros((self.H, self.N), dtype=np.float32)
        self.dones = np.zeros((self.H, self.N), dtype=np.float32)

        self.last_values = np.zeros((N,), dtype=np.float32)
        self.ptr = 0

    def add_batch(self, map_bn, vec_bn, act_bn, logp_bn, val_bn, rew_bn, done_bn):
        # map_bn: (N,H,W), vec_bn: (N,D), act_bn:(N,B), logp/val/rew/done:(N,)
        t = self.ptr
        assert t < self.H

        self.map_states[t] = map_bn
        self.vec_states[t] = vec_bn
        self.actions[t] = act_bn
        self.logprobs[t] = logp_bn
        self.values[t] = val_bn
        self.rewards[t] = rew_bn
        self.dones[t] = done_bn

        self.ptr += 1

    def set_last_values(self, last_values_n):
        # last_values_n: (N,)
        self.last_values[:] = last_values_n

    def to_batch(self, *, assert_full: bool = True) -> RolloutBatch:
        """
        Convert internal numpy buffers -> torch RolloutBatch on CPU.
        If assert_full=True, require ptr == H (typical PPO rollout).
        """
        if assert_full:
            assert self.ptr == self.H, f"rollout not full: ptr={self.ptr}, H={self.H}"
        H_used = self.ptr

        map_states = torch.from_numpy(self.map_states[:H_used])      # int16 CPU
        vec_states = torch.from_numpy(self.vec_states[:H_used])      # float32 CPU
        actions    = torch.from_numpy(self.actions[:H_used])         # int64 CPU
        logprobs   = torch.from_numpy(self.logprobs[:H_used])        # float32 CPU
        values     = torch.from_numpy(self.values[:H_used])          # float32 CPU
        rewards    = torch.from_numpy(self.rewards[:H_used])         # float32 CPU
        dones      = torch.from_numpy(self.dones[:H_used])           # float32 CPU
        last_values= torch.from_numpy(self.last_values.copy())       # float32 CPU

        return RolloutBatch(
            map_states=map_states,
            vec_states=vec_states,
            actions=actions,
            logprobs=logprobs,
            values=values,
            rewards=rewards,
            dones=dones,
            last_values=last_values,
        )

    def clear(self):
        self.ptr = 0

class ResBlock2D(nn.Module):
    """Basic 2D residual block with optional downsampling."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU()

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(y + self.skip(x))


class ResMLPBlock(nn.Module):
    """Residual MLP block: x -> Linear(d,d) -> ReLU -> Linear(d,d) + x -> ReLU."""
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.act(self.fc1(x))
        y = self.fc2(y)
        return self.act(x + y)

class SpatialAttnPool2d(nn.Module):
    """Attention pooling: (B,C,H,W) -> (B,C)."""
    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        a = self.attn(x).view(B, 1, H * W)          # (B,1,HW)
        a = torch.softmax(a, dim=-1)                # (B,1,HW)
        x_flat = x.view(B, C, H * W)                # (B,C,HW)
        pooled = torch.bmm(x_flat, a.transpose(1,2))# (B,C,1)
        return pooled.squeeze(-1)                   # (B,C)

class ModalityGate(nn.Module):
    """
    Compute 2-way soft gate for (map_feat, vec_feat).
    Returns weights (B,2): [w_map, w_vec], sum to 1.
    """
    def __init__(self, map_dim: int, vec_dim: int, hidden: int = 128, use_layernorm: bool = True):
        super().__init__()
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln_map = nn.LayerNorm(map_dim)
            self.ln_vec = nn.LayerNorm(vec_dim)

        self.net = nn.Sequential(
            nn.Linear(map_dim + vec_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),  # logits for 2 modalities
        )

        # Start near equal weighting (optional but often stabilizes early PPO)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, map_feat: torch.Tensor, vec_feat: torch.Tensor) -> torch.Tensor:
        if self.use_layernorm:
            map_feat = self.ln_map(map_feat)
            vec_feat = self.ln_vec(vec_feat)

        logits = self.net(torch.cat([map_feat, vec_feat], dim=-1))  # (B,2)
        w = torch.softmax(logits, dim=-1)                           # (B,2)
        return w


class ActorCritic(nn.Module):
    """
    Actor-Critic backbone for (id_map, vector_obs) input.

    Supports:
    - Discrete: gym.spaces.Discrete(n)          -> action_nvec = None
    - MultiDiscrete: gym.spaces.MultiDiscrete  -> action_nvec = [n1, n2, ...]
    - Continuous: (kept for compatibility)      -> has_continuous_action_space = True
    """
    def __init__(
        self,
        num_ids: int,
        vec_dim: int,
        action_dim: int,
        has_continuous_action_space: bool,
        action_std_init: float,
        *,
        emb_dim: int = 32,
        map_feat_dim: int = 256,
        vec_feat_dim: int = 128,
        fused_dim: int = 256,
        fusion_blocks: int = 2,
        action_nvec=None,
    ):
        super().__init__()

        self.has_continuous_action_space = has_continuous_action_space

        # Determine action type
        if action_nvec is None:
            self.is_multidiscrete = False
            self.action_nvec = None
            self.action_dim = int(action_dim)  # Discrete(n)
        else:
            self.is_multidiscrete = True
            if isinstance(action_nvec, np.ndarray):
                action_nvec = action_nvec.tolist()
            elif torch.is_tensor(action_nvec):
                action_nvec = action_nvec.detach().cpu().tolist()
            self.action_nvec = [int(x) for x in action_nvec]  # e.g. [2,2,2,2,2,2,2,2]
            self.num_branches = len(self.action_nvec)
            self.sum_action_dims = int(sum(self.action_nvec))  # here: 16 for 8 branches * 2
            # Keep action_dim for compatibility, but MultiDiscrete uses branches
            self.action_dim = self.num_branches

        # --- ID embedding ---
        self.id_embed = nn.Embedding(num_embeddings=num_ids, embedding_dim=emb_dim)

        # --- Map encoder ---
        self.map_stem = nn.Sequential(
            nn.Conv2d(emb_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.map_backbone = nn.Sequential(
            ResBlock2D(64, 64, stride=1),
            ResBlock2D(64, 64, stride=1),

            ResBlock2D(64, 128, stride=2),
            ResBlock2D(128, 128, stride=1),

            ResBlock2D(128, 256, stride=2),
            ResBlock2D(256, 256, stride=1),
        )

        self.map_pool = SpatialAttnPool2d(256)
        self.map_proj = nn.Sequential(
            nn.Linear(256, map_feat_dim),
            nn.ReLU(),
        )

        # --- Vector encoder ---
        self.vec_proj = nn.Sequential(
            nn.Linear(vec_dim, vec_feat_dim),
            nn.ReLU(),
        )
        self.vec_res = nn.Sequential(
            ResMLPBlock(vec_feat_dim),
            ResMLPBlock(vec_feat_dim),
        )

        # --- Fusion trunk ---
        fusion_in = map_feat_dim + vec_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fused_dim),
            nn.ReLU(),
            *[ResMLPBlock(fused_dim) for _ in range(fusion_blocks)],
        )

        self.mod_gate = ModalityGate(map_feat_dim, vec_feat_dim, hidden=128, use_layernorm=True)

        # --- Actor / Critic heads ---
        if self.has_continuous_action_space:
            # NOTE: kept only for API compatibility; your env is MultiDiscrete so this won't be used.
            self.actor_head = nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.ReLU(),
                nn.Linear(fused_dim, action_dim),
            )
            self.register_buffer(
                "action_var",
                torch.full((action_dim,), float(action_std_init) * float(action_std_init)),
            )
        else:
            if self.is_multidiscrete:
                # Flat logits for all branches, later split by action_nvec
                self.actor_head = nn.Sequential(
                    nn.Linear(fused_dim, fused_dim),
                    nn.ReLU(),
                    nn.Linear(fused_dim, self.sum_action_dims),
                )
            else:
                # Single Discrete(n): logits of size action_dim
                self.actor_head = nn.Sequential(
                    nn.Linear(fused_dim, fused_dim),
                    nn.ReLU(),
                    nn.Linear(fused_dim, self.action_dim),
                )

        self.critic_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, 1),
        )

    def encode(self, id_map, vec_obs):
        if id_map.dim() == 2:
            id_map = id_map.unsqueeze(0)
        if vec_obs.dim() == 1:
            vec_obs = vec_obs.unsqueeze(0)

        id_map = id_map.long()
        vec_obs = vec_obs.float()

        x = self.id_embed(id_map).permute(0, 3, 1, 2).contiguous()
        x = self.map_stem(x)
        x = self.map_backbone(x)
        x = self.map_pool(x)               # (B,256)
        map_feat = self.map_proj(x)        # (B,map_feat_dim)

        vec_feat = self.vec_res(self.vec_proj(vec_obs))  # (B,vec_feat_dim)

        w = self.mod_gate(map_feat, vec_feat)            # (B,2)
        map_feat = map_feat * w[:, 0:1]
        vec_feat = vec_feat * w[:, 1:2]

        fused = torch.cat([map_feat, vec_feat], dim=1)
        fused = self.fusion(fused)         # (B,fused_dim)
        return fused

    def _split_branch_logits(self, flat_logits: torch.Tensor):
        # flat_logits: (B, sum(action_nvec)) -> list of tensors [(B,n1), (B,n2), ...]
        return list(torch.split(flat_logits, self.action_nvec, dim=-1))

    @torch.no_grad()
    def act(self, id_map, vec_obs, deterministic: bool = False):
        feat = self.encode(id_map, vec_obs)
        state_value = self.critic_head(feat)  # (B,1)

        if self.has_continuous_action_space:
            action_mean = self.actor_head(feat)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
            # NOTE: deterministic uses the mean action (no sampling)
            action = action_mean if deterministic else dist.sample()
            action_logprob = dist.log_prob(action)
            return action, action_logprob, state_value

        logits = self.actor_head(feat)

        if not self.is_multidiscrete:
            dist = Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)          # (B,)
            else:
                action = dist.sample()                         # (B,)
            action_logprob = dist.log_prob(action)             # (B,)
            return action, action_logprob, state_value

        logits_list = self._split_branch_logits(logits)
        actions, logps = [], []
        for lg in logits_list:
            dist = Categorical(logits=lg)
            if deterministic:
                a = torch.argmax(lg, dim=-1)                   # (B,)
            else:
                a = dist.sample()                              # (B,)
            actions.append(a)
            logps.append(dist.log_prob(a))

        action = torch.stack(actions, dim=-1)                 # (B,num_branches)
        action_logprob = torch.stack(logps, dim=-1).sum(-1)   # (B,)
        return action, action_logprob, state_value

    def evaluate(self, id_map, vec_obs, action):
        feat = self.encode(id_map, vec_obs)
        state_values = self.critic_head(feat)  # (B,1)

        if self.has_continuous_action_space:
            action_mean = self.actor_head(feat)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            return action_logprobs, state_values, dist_entropy

        logits = self.actor_head(feat)

        if not self.is_multidiscrete:
            dist = Categorical(logits=logits)
            action = action.long()
            if action.dim() > 1:
                action = action.squeeze(-1)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            return action_logprobs, state_values, dist_entropy

        # MultiDiscrete: action must be (B, num_branches) or (num_branches,)
        action = action.long()

        if action.dim() == 1:
            if action.numel() == self.num_branches:
                action = action.unsqueeze(0)  # (1,num_branches)
            else:
                raise ValueError(
                    f"[ActorCritic.evaluate] Expected action shape (B,{self.num_branches}) "
                    f"but got 1D tensor of length {action.numel()}. "
                    f"This usually means actions were flattened in PPO.update()."
                )

        # Now action must be (B, num_branches)
        if action.dim() != 2 or action.size(1) != self.num_branches:
            raise ValueError(
                f"[ActorCritic.evaluate] Bad action shape: {tuple(action.shape)} expected (B,{self.num_branches})"
            )

        logits_list = self._split_branch_logits(logits)
        logps, ents = [], []
        for i, lg in enumerate(logits_list):
            dist = Categorical(logits=lg)
            a_i = action[:, i]
            logps.append(dist.log_prob(a_i))
            ents.append(dist.entropy())

        action_logprobs = torch.stack(logps, dim=-1).sum(-1)  # (B,)
        dist_entropy = torch.stack(ents, dim=-1).sum(-1)      # (B,)
        return action_logprobs, state_values, dist_entropy
    
    def forward(self, id_map, vec_obs, action):
        """
        DDP 호환을 위해 forward를 evaluate로 연결.
        PPO.update()에서 self.policy(mb_map, mb_vec, mb_actions)로 호출하게 됨.
        """
        return self.evaluate(id_map, vec_obs, action)


'''
PPO implementation
'''

class PPOVec:
    """
    PPO that expects vectorized rollouts (H, N, ...).
    ActorCritic is assumed to be your existing backbone (with MultiDiscrete support).
    """

    def __init__(
        self,
        actor_critic_ctor,
        *,
        num_ids: int,
        vec_dim: int,
        action_dim: int,
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        K_epochs: int,
        eps_clip: float,
        has_continuous_action_space: bool,
        action_std_init: float = 0.6,
        mini_batch_size: int = 128,
        action_nvec=None,
        device: Optional[torch.device] = None,
    ):
        self.has_continuous_action_space = bool(has_continuous_action_space)
        self.gamma = float(gamma)
        self.eps_clip = float(eps_clip)
        self.K_epochs = int(K_epochs)
        self.mini_batch_size = int(mini_batch_size)
        self.action_nvec = action_nvec

        self.device = device if device is not None else torch.device("cpu")

        if self.has_continuous_action_space:
            self.action_std = float(action_std_init)

        # Build models
        self.policy = actor_critic_ctor(
            num_ids=num_ids,
            vec_dim=vec_dim,
            action_dim=action_dim,
            has_continuous_action_space=self.has_continuous_action_space,
            action_std_init=action_std_init,
            action_nvec=action_nvec,
        ).to(self.device)

        self.policy_old = actor_critic_ctor(
            num_ids=num_ids,
            vec_dim=vec_dim,
            action_dim=action_dim,
            has_continuous_action_space=self.has_continuous_action_space,
            action_std_init=action_std_init,
            action_nvec=action_nvec,
        ).to(self.device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        # Optimizer groups (match your previous grouping style)
        shared_and_actor = []
        critic_only = []

        shared_and_actor += list(self.policy.id_embed.parameters())
        shared_and_actor += list(self.policy.map_stem.parameters())
        shared_and_actor += list(self.policy.map_backbone.parameters())
        shared_and_actor += list(self.policy.map_proj.parameters())
        shared_and_actor += list(self.policy.vec_proj.parameters())
        shared_and_actor += list(self.policy.map_pool.parameters())
        shared_and_actor += list(self.policy.mod_gate.parameters())
        shared_and_actor += list(self.policy.vec_res.parameters())
        shared_and_actor += list(self.policy.fusion.parameters())
        shared_and_actor += list(self.policy.actor_head.parameters())
        critic_only += list(self.policy.critic_head.parameters())

        self.optimizer = torch.optim.Adam(
            [{"params": shared_and_actor, "lr": float(lr_actor)},
             {"params": critic_only, "lr": float(lr_critic)}]
        )

        self.MseLoss = nn.MSELoss()

    @torch.no_grad()
    def act_batch(self, map_t: torch.Tensor, vec_t: torch.Tensor, deterministic: bool = False):
        """
        Batched action selection using policy_old.
        map_t: (N,H,W) long on device
        vec_t: (N,D)   float on device
        Returns:
          action: (N,branches) or (N,) on device
          logp:   (N,) on device
          value:  (N,) on device
        """
        out = self.policy_old.act(map_t, vec_t, deterministic=deterministic)
        if isinstance(out, (tuple, list)):
            action_t, logp_t, v_t = out[0], out[1], out[2]
        else:
            raise RuntimeError("ActorCritic.act must return (action, logp, value).")

        return action_t, logp_t.view(-1), v_t.view(-1)

    def _compute_gae_vec(self, rewards, dones, values, last_values, gae_lambda: float):
        """
        rewards,dones,values: (H,N)
        last_values: (N,)
        returns advantages, returns: both (H,N)
        """
        H, N = rewards.shape
        next_values = torch.zeros_like(values)
        if H > 1:
            next_values[:-1] = values[1:]
        # IMPORTANT: last_values must be masked to 0 where the last step was terminal.
        # We also defensively mask here using dones[-1].
        last_values = last_values.to(device=rewards.device, dtype=rewards.dtype)
        last_values = last_values * (1.0 - dones[-1])
        next_values[-1] = last_values

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros((N,), dtype=rewards.dtype, device=rewards.device)

        for t in reversed(range(H)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * float(gae_lambda) * mask * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, batch: RolloutBatch, gae_lambda: float = 0.95):
        """
        PPO update from a vectorized rollout batch.
        All batch tensors are expected on CPU; we move minibatches to self.device.
        """
        # CPU tensors
        old_map = batch.map_states.long()     # (H,N,Hmap,Wmap)
        old_vec = batch.vec_states.float()    # (H,N,D)
        old_actions = batch.actions           # (H,N,...) torch CPU
        old_logprobs = batch.logprobs.float() # (H,N)
        values = batch.values.float()         # (H,N)
        rewards = batch.rewards.float()       # (H,N)
        dones = batch.dones.float()           # (H,N)
        last_values = batch.last_values.float()  # (N,)

        H, N = rewards.shape
        T = int(H * N)

        # VecGAE on CPU
        advantages, returns = self._compute_gae_vec(rewards, dones, values, last_values, gae_lambda=gae_lambda)

        # Flatten to (T, ...)
        advantages = advantages.view(-1)
        returns = returns.view(-1)
        old_values = values.view(-1)

        # Normalize advantages (per-rank is fine)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        old_map = old_map.view(T, *old_map.shape[2:])   # (T,Hmap,Wmap)
        old_vec = old_vec.view(T, old_vec.shape[2])     # (T,D)
        old_logprobs = old_logprobs.view(T)             # (T,)

        # Actions flatten
        if self.has_continuous_action_space:
            old_actions = old_actions.view(T, -1).float()
        else:
            # For MultiDiscrete we expect (H,N,branches) -> (T,branches)
            if old_actions.dim() == 3:
                old_actions = old_actions.view(T, old_actions.size(-1)).long()
            else:
                old_actions = old_actions.view(T).long()

        mb = int(self.mini_batch_size)
        vf_clip = float(self.eps_clip)
        vf_coef = 0.5
        ent_coef = 0.01

        for _ in range(int(self.K_epochs)):
            idxs = torch.randperm(T)

            for start in range(0, T, mb):
                mb_idx = idxs[start:start + mb]

                mb_map = old_map[mb_idx].to(self.device, non_blocking=True)
                mb_vec = old_vec[mb_idx].to(self.device, non_blocking=True)
                mb_actions = old_actions[mb_idx].to(self.device, non_blocking=True)
                mb_old_logp = old_logprobs[mb_idx].to(self.device, non_blocking=True)

                mb_returns = returns[mb_idx].to(self.device, non_blocking=True)
                mb_adv = advantages[mb_idx].to(self.device, non_blocking=True)
                mb_old_values = old_values[mb_idx].to(self.device, non_blocking=True)

                logprobs, state_values, dist_entropy = self.policy(mb_map, mb_vec, mb_actions)
                state_values = state_values.view(-1)

                ratios = torch.exp(logprobs - mb_old_logp.detach())
                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv

                v_pred = state_values
                v_pred_clipped = mb_old_values + torch.clamp(v_pred - mb_old_values, -vf_clip, vf_clip)

                loss_v1 = (v_pred - mb_returns).pow(2)
                loss_v2 = (v_pred_clipped - mb_returns).pow(2)
                loss_critic = torch.max(loss_v1, loss_v2)

                loss = -torch.min(surr1, surr2) + vf_coef * loss_critic - ent_coef * dist_entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        # Sync policy_old
        src = self.policy.module if hasattr(self.policy, "module") else self.policy
        self.policy_old.load_state_dict(src.state_dict())
