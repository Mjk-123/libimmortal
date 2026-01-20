import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical


class ResBlock2D(nn.Module):
    """Light 2D residual block with normalization."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 1):
        super().__init__()
        # GroupNorm(1, C) ~= LayerNorm over channels for conv features
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=out_ch)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.ReLU()

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(y + self.skip(x))


class ResMLPBlock(nn.Module):
    """Residual MLP block with LayerNorm."""
    def __init__(self, d: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.act(self.fc1(self.ln1(x)))
        y = self.fc2(self.ln2(y))
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

        # --- Map encoder (LIGHT) ---
        self.map_stem = nn.Sequential(
            nn.Conv2d(emb_dim, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
        )

        # Downsample only once (stride=2) and keep channels modest
        self.map_backbone = nn.Sequential(
            ResBlock2D(32, 32, stride=1, groups=1),
            ResBlock2D(32, 64, stride=2, groups=1),   # single downsample
            ResBlock2D(64, 64, stride=1, groups=1),
            ResBlock2D(64, 64, stride=1, groups=1),
        )

        self.map_pool = SpatialAttnPool2d(64)
        self.map_proj = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, map_feat_dim),
            nn.ReLU(),
        )

        # --- Vector encoder ---
        self.vec_proj = nn.Sequential(
            nn.LayerNorm(vec_dim),
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
            nn.LayerNorm(fusion_in),
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
            actor_out = self.sum_action_dims if self.is_multidiscrete else self.action_dim
            self.actor_head = nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.ReLU(),
                nn.Linear(fused_dim, actor_out),
            )

        self.critic_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Linear(fused_dim // 2, 1),
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