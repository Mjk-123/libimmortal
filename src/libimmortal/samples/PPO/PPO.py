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


class ActorCritic(nn.Module):
    """
    Actor-Critic backbone for (id_map, vector_obs) input.

    - id_map: (H,W) or (B,H,W), integer IDs in [0, num_ids-1]
    - vector_obs: (103,) or (B,103), float
    """
    def __init__(
        self,
        num_ids: int,                 # K (palette size)
        vec_dim: int,                 # 103
        action_dim: int,
        has_continuous_action_space: bool,
        action_std_init: float,
        *,
        emb_dim: int = 16,            # id embedding dim
        map_feat_dim: int = 256,      # map encoder output dim
        vec_feat_dim: int = 128,      # vector encoder output dim
        fused_dim: int = 256,         # fusion trunk width
        fusion_blocks: int = 2,       # residual blocks in fusion MLP
    ):
        super().__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim

        # --- ID embedding (id_map -> embedding channels) ---
        self.id_embed = nn.Embedding(num_embeddings=num_ids, embedding_dim=emb_dim)

        # --- Map encoder (CNN + residual) ---
        # Input after embedding: (B, emb_dim, H, W)
        self.map_stem = nn.Sequential(
            nn.Conv2d(emb_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Deeper CNN with downsampling
        self.map_backbone = nn.Sequential(
            ResBlock2D(64, 64, stride=1),
            ResBlock2D(64, 64, stride=1),

            ResBlock2D(64, 128, stride=2),  # downsample
            ResBlock2D(128, 128, stride=1),

            ResBlock2D(128, 256, stride=2), # downsample
            ResBlock2D(256, 256, stride=1),
        )

        # Make output size-independent of (90,160)
        self.map_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.map_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, map_feat_dim),
            nn.ReLU(),
        )

        # --- Vector encoder (FFN + residual) ---
        self.vec_proj = nn.Sequential(
            nn.Linear(vec_dim, vec_feat_dim),
            nn.ReLU(),
        )
        self.vec_res = nn.Sequential(
            ResMLPBlock(vec_feat_dim),
            ResMLPBlock(vec_feat_dim),
        )

        # --- Fusion trunk (concat -> residual FFN) ---
        fusion_in = map_feat_dim + vec_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fused_dim),
            nn.ReLU(),
            *[ResMLPBlock(fused_dim) for _ in range(fusion_blocks)],
        )

        # --- Actor head / Critic head ---
        if self.has_continuous_action_space:
            self.actor_head = nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.ReLU(),
                nn.Linear(fused_dim, action_dim),  # NOTE: no Tanh (unbounded mean)
            )
            # Keep as buffer so it follows .to(device)
            self.register_buffer(
                "action_var",
                torch.full((action_dim,), action_std_init * action_std_init),
            )
        else:
            self.actor_head = nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.ReLU(),
                nn.Linear(fused_dim, action_dim),
                nn.Softmax(dim=-1),
            )

        self.critic_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, 1),
        )

    def encode(self, id_map, vec_obs):
        # id_map: (H,W) or (B,H,W) -> (B,H,W)
        if id_map.dim() == 2:
            id_map = id_map.unsqueeze(0)
        if vec_obs.dim() == 1:
            vec_obs = vec_obs.unsqueeze(0)

        # Ensure dtypes
        id_map = id_map.long()
        vec_obs = vec_obs.float()

        # Embedding: (B,H,W) -> (B,H,W,emb_dim) -> (B,emb_dim,H,W)
        x = self.id_embed(id_map).permute(0, 3, 1, 2).contiguous()

        x = self.map_stem(x)
        x = self.map_backbone(x)
        x = self.map_pool(x)
        map_feat = self.map_proj(x)  # (B, map_feat_dim)

        vec_feat = self.vec_proj(vec_obs)
        vec_feat = self.vec_res(vec_feat)  # (B, vec_feat_dim)

        fused = torch.cat([map_feat, vec_feat], dim=1)
        fused = self.fusion(fused)  # (B, fused_dim)
        return fused

    def act(self, id_map, vec_obs):
        """Sample action + logprob + state value (for rollout collection)."""
        feat = self.encode(id_map, vec_obs)
        state_value = self.critic_head(feat)

        if self.has_continuous_action_space:
            action_mean = self.actor_head(feat)
            action_var = self.action_var.expand_as(action_mean)               # (B, action_dim)
            cov_mat = torch.diag_embed(action_var)                             # (B, action_dim, action_dim)
            dist = MultivariateNormal(action_mean, cov_mat)

            action = dist.sample()
            action_logprob = dist.log_prob(action)
            return action, action_logprob, state_value
        else:
            action_probs = self.actor_head(feat)
            dist = Categorical(action_probs)

            action = dist.sample()
            action_logprob = dist.log_prob(action)
            return action, action_logprob, state_value

    def evaluate(self, id_map, vec_obs, action):
        """Compute logprobs, values, entropy for PPO update."""
        feat = self.encode(id_map, vec_obs)
        state_values = self.critic_head(feat)

        if self.has_continuous_action_space:
            action_mean = self.actor_head(feat)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            return action_logprobs, state_values, dist_entropy
        else:
            action_probs = self.actor_head(feat)
            dist = Categorical(action_probs)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, 
                 num_ids, vec_dim, action_dim,
                 lr_actor, lr_critic,
                 gamma, K_epochs, eps_clip,
                 has_continuous_action_space,
                 action_std_init=0.6,
                 mini_batch_size=128):   # <-- NEW
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
            action_std_init=action_std_init
        ).to(device)

        self.policy_old = ActorCritic(
            num_ids=num_ids, vec_dim=vec_dim, action_dim=action_dim,
            has_continuous_action_space=has_continuous_action_space,
            action_std_init=action_std_init
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
        # 1) returns 계산 (CPU에서 해도 OK)
        rewards = []
        discounted_reward = 0.0
        for r, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0.0
            discounted_reward = float(r) + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)  # CPU
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 2) rollout tensors (큰 map은 CPU 유지!)
        # map: (T,H,W) long (CPU)
        old_map = torch.from_numpy(np.stack(self.buffer.map_states, axis=0)).long()     # CPU
        old_vec = torch.from_numpy(np.stack(self.buffer.vec_states, axis=0)).float()   # CPU

        # actions/logprobs/values는 크기 작으니 CPU로 통일해도 충분
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().cpu()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().cpu().view(-1)
        old_state_values = torch.stack(self.buffer.state_values, dim=0).detach().cpu().view(-1)

        # continuous면 action shape 정리
        if self.has_continuous_action_space:
            old_actions = old_actions.view(old_actions.size(0), -1)  # (T, action_dim)
        else:
            old_actions = old_actions.view(-1)

        advantages = rewards - old_state_values

        T = rewards.shape[0]
        mb = int(self.mini_batch_size)

        for _ in range(self.K_epochs):
            idxs = torch.randperm(T)  # CPU indices

            for start in range(0, T, mb):
                mb_idx = idxs[start:start + mb]

                # 미니배치만 GPU로 이동
                mb_map = old_map[mb_idx].to(device, non_blocking=True)          # (B,H,W)
                mb_vec = old_vec[mb_idx].to(device, non_blocking=True)          # (B,103)
                mb_actions = old_actions[mb_idx].to(device, non_blocking=True)
                mb_old_logp = old_logprobs[mb_idx].to(device, non_blocking=True)
                mb_rewards = rewards[mb_idx].to(device, non_blocking=True)
                mb_adv = advantages[mb_idx].to(device, non_blocking=True)

                logprobs, state_values, dist_entropy = self.policy.evaluate(mb_map, mb_vec, mb_actions)
                state_values = state_values.view(-1)

                ratios = torch.exp(logprobs - mb_old_logp.detach())

                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv

                loss = -torch.min(surr1, surr2) \
                       + 0.5 * self.MseLoss(state_values, mb_rewards) \
                       - 0.01 * dist_entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.mean().backward()

                # (선택) 안정성용 grad clip 추천
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
