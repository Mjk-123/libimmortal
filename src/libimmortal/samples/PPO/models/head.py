import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MultiDiscreteActorHead(nn.Module):
    """
    action_nvec: 예) [5, 3, 2, ...]  (각 액션 차원의 category 개수)
    출력 logits은 총합 sum(action_nvec)로 뽑고, 필요할 때 split해서 Categorical로 만듦.
    """
    def __init__(self, in_dim: int, action_nvec: list[int], hidden: int = 256):
        super().__init__()
        self.action_nvec = list(action_nvec)
        self.total = int(sum(self.action_nvec))

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.total),
        )

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        # [B, total_logits]
        return self.net(state_vec)

    def split_logits(self, flat_logits: torch.Tensor) -> list[torch.Tensor]:
        # flat_logits: [B, total]
        return list(torch.split(flat_logits, self.action_nvec, dim=-1))


class CriticHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.net(state_vec)  # [B,1]

