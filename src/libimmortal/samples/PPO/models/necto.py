import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from libimmortal.samples.PPO.models.backbone import DualStreamBidirBackbone
from libimmortal.samples.PPO.models.head import MultiDiscreteActorHead, CriticHead


class ActorCritic(nn.Module):
    """
    - backbone: id_map + vec -> state_vec [B,D]
    - actor: flat logits -> per-dim Categorical
    - critic: value
    """
    def __init__(
        self,
        num_ids: int,
        action_nvec: list[int],
        model_dim: int = 256,
        enemy_state_vocab: int = 32,
        enemy_state_emb: int = 16,
    ):
        super().__init__()
        self.action_nvec = list(action_nvec)

        self.backbone = DualStreamBidirBackbone(
            num_ids=num_ids,
            model_dim=model_dim,
            id_emb_dim=64,
            patch_hw=(9,10),
            map_layers=3,
            ent_layers=2,
            heads=8,
            dropout=0.0,
            enemy_state_vocab=enemy_state_vocab,
            enemy_state_emb=enemy_state_emb,
        )

        self.actor_head = MultiDiscreteActorHead(model_dim, self.action_nvec, hidden=256)
        self.critic_head = CriticHead(model_dim, hidden=256)

    @torch.no_grad()
    def act(self, id_map: torch.Tensor, vec: torch.Tensor):
        """
        id_map: [B,90,160] long
        vec:    [B,103] float

        returns:
          action: [B, len(action_nvec)] long  (각 차원별 sampled action index)
          logp:   [B] float                  (합산 logprob)
          value:  [B] float                  (critic value)
        """
        state_vec = self.backbone(id_map, vec)          # [B,D]
        flat_logits = self.actor_head(state_vec)        # [B,total]
        logits_list = self.actor_head.split_logits(flat_logits)

        actions = []
        logps = []
        entropies = []

        for logits in logits_list:
            dist = Categorical(logits=logits)           # logits: [B, n_i]
            a = dist.sample()                           # [B]
            actions.append(a)
            logps.append(dist.log_prob(a))              # [B]
            entropies.append(dist.entropy())            # [B]

        action = torch.stack(actions, dim=-1)           # [B, dims]
        logp = torch.stack(logps, dim=-1).sum(dim=-1)   # [B]
        value = self.critic_head(state_vec).squeeze(-1) # [B]

        return action, logp, value

    def evaluate(self, id_map: torch.Tensor, vec: torch.Tensor, action: torch.Tensor):
        """
        PPO update에서 사용:
        action: [B, dims] long (저장해둔 action)
        returns:
          logp: [B]
          value: [B]
          entropy: [B]
        """
        state_vec = self.backbone(id_map, vec)          # [B,D]
        flat_logits = self.actor_head(state_vec)        # [B,total]
        logits_list = self.actor_head.split_logits(flat_logits)

        # action split
        assert action.dim() == 2 and action.size(1) == len(self.action_nvec)
        logps = []
        entropies = []

        for i, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            a_i = action[:, i]                          # [B]
            logps.append(dist.log_prob(a_i))
            entropies.append(dist.entropy())

        logp = torch.stack(logps, dim=-1).sum(dim=-1)   # [B]
        entropy = torch.stack(entropies, dim=-1).sum(dim=-1)  # [B]
        value = self.critic_head(state_vec).squeeze(-1) # [B]

        return logp, value, entropy
