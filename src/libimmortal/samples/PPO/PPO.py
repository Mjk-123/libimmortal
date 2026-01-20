import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical

import libimmortal.samples.PPO.models.simple as simple
import libimmortal.samples.PPO.models.necto as necto

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


'''
PPO implementation
'''

ActorCriticCls = simple.ActorCritic
# ActorCriticCls = necto.ActorCritic

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

        self.policy = ActorCriticCls(
            num_ids=num_ids, vec_dim=vec_dim, action_dim=action_dim,
            has_continuous_action_space=has_continuous_action_space,
            action_std_init=action_std_init,
            action_nvec=action_nvec
        ).to(self.device)

        self.policy_old = ActorCriticCls(
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
        '''
        Freezing test
        '''
        T = len(self.buffer.rewards)
        print(f"PPO enters T = {T}", flush=True)

        assert T == len(self.buffer.is_terminals)
        assert T == len(self.buffer.map_states)
        assert T == len(self.buffer.vec_states)
        assert T == len(self.buffer.actions)
        assert T == len(self.buffer.logprobs)
        assert T == len(self.buffer.state_values)
        assert T <= 7000, f"T too large: {T} (buffer not cleared / rollout not bounded?)"

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
    import gym
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
    agent = ActorCriticCls(
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
        print("[CKPT] missing:", len(missing), "unexpected:", len(unexpected))
        print("  missing sample:", missing[:5])
        print("  unexpected sample:", unexpected[:5])
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
