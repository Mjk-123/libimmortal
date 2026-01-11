import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical

# --- REMOVE THIS (DDP import-time cuda:0 trap) ---
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _default_device() -> torch.device:
    """
    DDP-safe default device.
    Assumes torch.cuda.set_device(local_rank) has already been called (in train.py).
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.map_states = []
        self.vec_states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.last_value = 0.0

    def clear(self):
        del self.actions[:]
        del self.map_states[:]
        del self.vec_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add_state(self, id_map, vec_obs):
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

class PPO:
    def __init__(
        self,
        num_ids, vec_dim, action_dim,
        lr_actor, lr_critic,
        gamma, K_epochs, eps_clip,
        has_continuous_action_space,
        action_std_init=0.6,
        mini_batch_size=128,
        action_nvec=None,
        device: torch.device | None = None,   # <-- NEW
    ):
        self.has_continuous_action_space = has_continuous_action_space
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.mini_batch_size = mini_batch_size

        # Keep action_nvec for shape handling in update()
        self.action_nvec = action_nvec

        # DDP-safe device (depends on torch.cuda.set_device called in train.py)
        self.device = device if device is not None else _default_device()

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            num_ids=num_ids, vec_dim=vec_dim, action_dim=action_dim,
            has_continuous_action_space=has_continuous_action_space,
            action_std_init=action_std_init,
            action_nvec=action_nvec
        ).to(self.device)

        self.policy_old = ActorCritic(
            num_ids=num_ids, vec_dim=vec_dim, action_dim=action_dim,
            has_continuous_action_space=has_continuous_action_space,
            action_std_init=action_std_init,
            action_nvec=action_nvec
        ).to(self.device)

        self.policy_old.load_state_dict(self.policy.state_dict())

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

    def update(self, gae_lambda: float = 0.95):
        # 1) Build tensors on CPU
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32)          # (T,)
        dones = torch.tensor(self.buffer.is_terminals, dtype=torch.float32)       # (T,)

        values = torch.stack(self.buffer.state_values, dim=0).detach().cpu().view(-1)  # (T,)
        last_value = torch.tensor([float(getattr(self.buffer, "last_value", 0.0))], dtype=torch.float32)  # (1,)
        next_values = torch.cat([values[1:], last_value], dim=0)  # (T,)

        # 2) GAE
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(rewards.size(0))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * gae_lambda * mask * gae
            advantages[t] = gae

        # 3) Returns
        returns = advantages + values

        # 4) Normalize advantages (per-rank; OK for DDP, though global norm is also possible)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # 5) Rollout tensors
        old_map = torch.from_numpy(np.stack(self.buffer.map_states, axis=0)).long()     # (T,H,W) CPU
        old_vec = torch.from_numpy(np.stack(self.buffer.vec_states, axis=0)).float()   # (T,103) CPU

        old_actions = torch.stack(self.buffer.actions, dim=0).detach().cpu()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().cpu().view(-1)  # (T,)

        # Keep MultiDiscrete shape
        if self.has_continuous_action_space:
            old_actions = old_actions.view(old_actions.size(0), -1).float()
        else:
            action_nvec = self.action_nvec if self.action_nvec is not None else getattr(self.policy, "action_nvec", None)

            if action_nvec is None:
                old_actions = old_actions.view(-1).long()
            else:
                num_branches = len(action_nvec)
                old_actions = old_actions.long()
                if old_actions.dim() == 3 and old_actions.size(1) == 1:
                    old_actions = old_actions.squeeze(1)
                if old_actions.dim() == 1:
                    assert old_actions.numel() % num_branches == 0
                    old_actions = old_actions.view(-1, num_branches)
                assert old_actions.dim() == 2 and old_actions.size(1) == num_branches

        T = rewards.shape[0]
        mb = int(self.mini_batch_size)

        old_values = values  # (T,)
        vf_clip = self.eps_clip
        vf_coef = 0.5
        ent_coef = 0.01

        # --- FIX: remove accidental double K_epochs loop ---
        for _ in range(self.K_epochs):
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
                state_values = state_values.view(-1)  # (B,)

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

        src = self.policy.module if hasattr(self.policy, "module") else self.policy
        self.policy_old.load_state_dict(src.state_dict())
        self.buffer.clear()


def add_ppo_args(p):
    """
    Shared argparse definitions used by both train.py and submission_runner.py.
    Keep defaults in one place to avoid mismatches.
    """
    # PPO hyperparams
    g = p.add_argument_group("PPO")
    g.add_argument("--max_ep_len", type=int, default=1000)
    g.add_argument("--update_timestep", type=int, default=4000)
    g.add_argument("--k_epochs", type=int, default=10)
    g.add_argument("--eps_clip", type=float, default=0.2)
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--lr_actor", type=float, default=1e-4)
    g.add_argument("--lr_critic", type=float, default=2e-4)
    g.add_argument("--mini_batch_size", type=int, default=64)

    # Checkpoint loading (for eval/inference scripts)
    g.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint to load for evaluation/inference.",
    )
    g.add_argument(
        "--prefer_old",
        action="store_true",
        default=True,
        help="Prefer policy_old when checkpoint contains both policy and policy_old. (default: True)",
    )
    g.add_argument(
        "--prefer_new",
        action="store_true",
        default=False,
        help="Prefer policy (new) instead of policy_old when checkpoint contains both.",
    )
    return p

def getPPOAgent(
    args,
    num_ids: int,
    vec_dim: int,
    action_space,
    device,
    ckpt_path: str | None = None,
    prefer_old: bool = True,  # True면 checkpoint에서 policy_old 우선 로드
):
    """
    Build and (optionally) load an ActorCritic agent for inference/eval.

    Returns:
        agent: ActorCritic (nn.Module) on `device`, set to eval()
        info: dict(action_dim, action_nvec, has_continuous_action_space)
    """
    import numpy as np
    import torch

    # gym vs gymnasium compatibility
    try:
        import gym
        spaces = gym.spaces
    except Exception:
        import gymnasium as gym
        spaces = gym.spaces

    def _strip_module_prefix(sd: dict) -> dict:
        # If keys start with "module.", strip it for non-DDP models.
        if not sd:
            return sd
        if all(isinstance(k, str) and k.startswith("module.") for k in sd.keys()):
            return {k[len("module."):]: v for k, v in sd.items()}
        return sd

    def _extract_policy_state(ckpt_obj: object) -> dict:
        """
        Accepts:
          - raw state_dict
          - checkpoint dict containing policy/policy_old (state_dict or nested)
        """
        if isinstance(ckpt_obj, dict):
            # Common patterns
            # 1) ckpt["policy_old"] is a state_dict
            # 2) ckpt["policy"] is a state_dict
            # 3) ckpt["state_dict"] is a state_dict
            cand_keys = []
            if prefer_old:
                cand_keys += ["policy_old", "actor_critic_old", "model_old"]
                cand_keys += ["policy", "actor_critic", "model"]
            else:
                cand_keys += ["policy", "actor_critic", "model"]
                cand_keys += ["policy_old", "actor_critic_old", "model_old"]
            cand_keys += ["state_dict", "model_state_dict"]

            for k in cand_keys:
                if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                    sd = ckpt_obj[k]
                    # Sometimes nested: ckpt["policy"]["state_dict"]
                    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
                        sd = sd["state_dict"]
                    return sd

            # Heuristic: looks like a state_dict already (param tensors)
            if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
                return ckpt_obj

        raise ValueError("Unsupported checkpoint format: cannot find a policy state_dict.")

    # Infer action spec
    if isinstance(action_space, spaces.Box):
        has_continuous_action_space = True
        action_dim = int(np.prod(action_space.shape))
        action_nvec = None

    elif isinstance(action_space, spaces.Discrete):
        has_continuous_action_space = False
        action_dim = int(action_space.n)  # logits size n
        action_nvec = None

    elif isinstance(action_space, spaces.MultiDiscrete):
        has_continuous_action_space = False
        action_nvec = action_space.nvec.astype(int).tolist()
        action_dim = int(len(action_nvec))  # branches count (ActorCritic uses action_nvec anyway)

    else:
        raise TypeError(f"Unsupported action space: {type(action_space)}")

    # Build ActorCritic agent
    agent = ActorCritic(
        num_ids=int(num_ids),
        vec_dim=int(vec_dim),
        action_dim=int(action_dim),
        has_continuous_action_space=bool(has_continuous_action_space),
        action_std_init=float(getattr(args, "action_std", 0.6)),
        action_nvec=action_nvec,
    ).to(device)

    # Optional: load weights
    if ckpt_path is not None:
        ckpt_obj = torch.load(ckpt_path, map_location="cpu")
        sd = _extract_policy_state(ckpt_obj)
        sd = _strip_module_prefix(sd)
        missing, unexpected = agent.load_state_dict(sd, strict=False)
        # If you want strict=True, flip it, but strict=False is safer across small refactors.
        if getattr(args, "verbose", False):
            print(f"[Load] ckpt={ckpt_path}")
            print(f"[Load] missing={len(missing)} unexpected={len(unexpected)}")

    agent.eval()

    info = {
        "action_dim": action_dim,
        "action_nvec": action_nvec,
        "has_continuous_action_space": has_continuous_action_space,
    }
    return agent, info
