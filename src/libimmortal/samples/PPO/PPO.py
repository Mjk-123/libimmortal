import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RolloutBuffer:
    def __init__(self):
        self.actions = []

        # CHANGED: split state into 2 streams
        # map_states: (H,W) id_map (uint8)  -- keep uint8 to save memory
        self.map_states = []
        # vec_states: (103,) float32
        self.vec_states = []

        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.map_states[:]
        del self.vec_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add_state(self, id_map, vec_obs):
        """
        id_map: np.ndarray of shape (H,W) from colormap encoder
        vec_obs: np.ndarray of shape (103,)
        """
        self.map_states.append(np.asarray(id_map, dtype=np.uint8))
        self.vec_states.append(np.asarray(vec_obs, dtype=np.float32).reshape(-1))


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
        emb_dim: int = 16,
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
    def act(self, id_map, vec_obs):
        feat = self.encode(id_map, vec_obs)
        state_value = self.critic_head(feat)  # (B,1)

        if self.has_continuous_action_space:
            action_mean = self.actor_head(feat)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            return action, action_logprob, state_value

        logits = self.actor_head(feat)

        if not self.is_multidiscrete:
            dist = Categorical(logits=logits)
            action = dist.sample()                 # (B,)
            action_logprob = dist.log_prob(action) # (B,)
            return action, action_logprob, state_value

        logits_list = self._split_branch_logits(logits)
        actions, logps = [], []
        for lg in logits_list:
            dist = Categorical(logits=lg)
            a = dist.sample()          # (B,)
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

        # MultiDiscrete: action should be (B,num_branches) or (num_branches,)
        action = action.long()
        if action.dim() == 1:
            if action.numel() == self.num_branches:
                action = action.unsqueeze(0)  # (1,num_branches)
            else:
                action = action.view(-1, self.num_branches)

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


class PPO:
    def __init__(self, 
                 num_ids, vec_dim, action_dim,
                 lr_actor, lr_critic,
                 gamma, K_epochs, eps_clip,
                 has_continuous_action_space,
                 action_std_init=0.6,
                 mini_batch_size=128,
                 action_nvec=None):   # <-- NEW
        self.has_continuous_action_space = has_continuous_action_space
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.mini_batch_size = mini_batch_size  # <-- NEW

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            num_ids=num_ids, vec_dim=vec_dim, action_dim=action_dim,
            has_continuous_action_space=has_continuous_action_space,
            action_std_init=action_std_init,
            action_nvec=action_nvec
        ).to(device)

        self.policy_old = ActorCritic(
            num_ids=num_ids, vec_dim=vec_dim, action_dim=action_dim,
            has_continuous_action_space=has_continuous_action_space,
            action_std_init=action_std_init,
            action_nvec=action_nvec
        ).to(device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        # Optimizer (원하는대로 유지/수정 가능)
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
            [{"params": shared_and_actor, "lr": lr_actor},
             {"params": critic_only, "lr": lr_critic}]
        )

        self.MseLoss = nn.MSELoss()

    def update(self):
        # 1) Compute discounted returns (CPU)
        rewards = []
        discounted_reward = 0.0
        for r, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0.0
            discounted_reward = float(r) + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)  # CPU
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 2) Rollout tensors (keep big map on CPU)
        old_map = torch.from_numpy(np.stack(self.buffer.map_states, axis=0)).long()    # (T,H,W) CPU
        old_vec = torch.from_numpy(np.stack(self.buffer.vec_states, axis=0)).float()  # (T,103) CPU

        old_actions = torch.stack(self.buffer.actions, dim=0).detach().cpu()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().cpu().view(-1)       # (T,)
        old_state_values = torch.stack(self.buffer.state_values, dim=0).detach().cpu().view(-1)  # (T,)

        # ---- FIX: handle Single Discrete vs MultiDiscrete correctly ----
        action_nvec = getattr(self, "action_nvec", None)
        is_multidiscrete = (not self.has_continuous_action_space) and (action_nvec is not None)
        num_branches = len(action_nvec) if is_multidiscrete else None

        if self.has_continuous_action_space:
            # (T, action_dim)
            old_actions = old_actions.view(old_actions.size(0), -1).float()
        else:
            if not is_multidiscrete:
                # Single Discrete: (T,)
                old_actions = old_actions.view(-1).long()
            else:
                # MultiDiscrete: keep (T, num_branches)
                old_actions = old_actions.long()

                # Common shapes:
                # (T,1,B) -> (T,B)
                if old_actions.dim() == 3 and old_actions.size(1) == 1:
                    old_actions = old_actions.squeeze(1)

                # (T*B,) -> (T,B) (safety for legacy buffer formats)
                if old_actions.dim() == 1:
                    assert old_actions.numel() % num_branches == 0, \
                        f"Cannot reshape old_actions of size {old_actions.numel()} into (-1,{num_branches})"
                    old_actions = old_actions.view(-1, num_branches)

                # Sanity check
                assert old_actions.dim() == 2 and old_actions.size(1) == num_branches, \
                    f"old_actions shape wrong for MultiDiscrete: {old_actions.shape}, expected (T,{num_branches})"

        # 3) Advantages (CPU)
        advantages = rewards - old_state_values

        T = rewards.shape[0]
        mb = int(self.mini_batch_size)

        # Optional sanity checks
        assert old_logprobs.shape[0] == T and old_state_values.shape[0] == T, \
            f"Rollout length mismatch: T={T}, logp={old_logprobs.shape}, v={old_state_values.shape}"
        if is_multidiscrete:
            assert old_actions.shape[0] == T, f"Actions length mismatch: {old_actions.shape} vs T={T}"

        for _ in range(self.K_epochs):
            idxs = torch.randperm(T)  # CPU indices

            for start in range(0, T, mb):
                mb_idx = idxs[start:start + mb]

                # Move minibatch to GPU
                mb_map = old_map[mb_idx].to(device, non_blocking=True)
                mb_vec = old_vec[mb_idx].to(device, non_blocking=True)
                mb_actions = old_actions[mb_idx].to(device, non_blocking=True)
                mb_old_logp = old_logprobs[mb_idx].to(device, non_blocking=True)
                mb_rewards = rewards[mb_idx].to(device, non_blocking=True)
                mb_adv = advantages[mb_idx].to(device, non_blocking=True)

                # Ensure correct dtype for discrete actions
                if not self.has_continuous_action_space:
                    mb_actions = mb_actions.long()

                logprobs, state_values, dist_entropy = self.policy.evaluate(mb_map, mb_vec, mb_actions)
                state_values = state_values.view(-1)

                ratios = torch.exp(logprobs - mb_old_logp.detach())

                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv

                # NOTE: For MultiDiscrete, dist_entropy is sum over branches -> can be larger.
                # You may consider scaling entropy coef by 1/num_branches if exploration is too strong.
                loss = -torch.min(surr1, surr2) \
                    + 0.5 * self.MseLoss(state_values, mb_rewards) \
                    - 0.01 * dist_entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

