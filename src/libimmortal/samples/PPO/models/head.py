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

        '''
        # Old self.net: simple MLP
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.total),
        )
        '''

        self.block1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.total),
        )

        self._init_all_weights()

    def _init_all_weights(self):
        # self.modules()는 클래스 내 모든 서브모듈을 재귀적으로 탐색합니다.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He 초기화: 깊은 모델에서 Gradient Flow를 돕습니다.
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm은 초기 상태에서 입력을 그대로 유지하도록 설정합니다.
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        # [B, total_logits]
        x = self.block1(state_vec)
        logits = self.block2(x)
        return logits

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


class DeepCriticHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.414)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.net(state_vec)