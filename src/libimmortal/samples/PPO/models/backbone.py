import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical

import libimmortal.samples.PPO.models.simple as simple
import libimmortal.samples.PPO.models.backbone as backbone

# ----------------------------
# Basic blocks
# ----------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x

class CrossAttn(nn.Module):
    """Cross attention: Q attends to KV. Shapes: Q=[B,Tq,D], KV=[B,Tk,D] -> [B,Tq,D]"""
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        return q + out

# ----------------------------
# Tokenizers
# ----------------------------

class MapTokenizer(nn.Module):
    """
    id_map: [B, 90, 160] long (values in [0..num_ids-1])
    -> map_tokens: [B, 160, D]  (patch=(9,10) -> 10x16=160 tokens)
    """
    def __init__(
        self,
        num_ids: int = 11,
        id_emb_dim: int = 64,
        model_dim: int = 256,
        patch_hw=(9, 10),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_ids = num_ids
        self.id_embed = nn.Embedding(num_ids, id_emb_dim)

        pH, pW = patch_hw
        assert 90 % pH == 0 and 160 % pW == 0, "Patch size must divide map size (H=90,W=160)."
        self.grid_h = 90 // pH
        self.grid_w = 160 // pW
        self.num_tokens = self.grid_h * self.grid_w  # 160 for (9,10)

        # patchify conv
        self.patch = nn.Conv2d(
            id_emb_dim, model_dim,
            kernel_size=(pH, pW),
            stride=(pH, pW),
            bias=True
        )
        self.pos = nn.Parameter(torch.zeros(1, self.num_tokens, model_dim))
        self.drop = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, id_map: torch.Tensor) -> torch.Tensor:
        # id_map: [B, 90, 160] long
        x = self.id_embed(id_map)                      # [B, 90, 160, E]
        x = x.permute(0, 3, 1, 2).contiguous()         # [B, E, 90, 160]
        x = self.patch(x)                              # [B, D, 10, 16]
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, 160, D]
        x = self.drop(x + self.pos)
        return x

class EntityTokenizer(nn.Module):
    """
    vec: [B, 103] float
    player: [0:13] -> 1 token
    enemy : [13:103] -> reshape [B,10,9]
       indices:
         ZERO=0, TYPE_SKELETON=1, TYPE_BOMKID=2, POS_X=3, POS_Y=4, VEL_X=5, VEL_Y=6, HEALTH=7, STATE=8
    -> ent_tokens: [B, 11, D]
    """
    def __init__(
        self,
        model_dim: int = 256,
        enemy_state_vocab: int = 32,  # 넉넉하게. 실제 최대값+1로 맞추면 더 좋음
        enemy_state_emb: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.enemy_state_vocab = enemy_state_vocab

        self.player_proj = nn.Sequential(
            nn.LayerNorm(13),
            nn.Linear(13, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.state_emb = nn.Embedding(enemy_state_vocab, enemy_state_emb)

        # enemy float dims = type(2) + pos(2) + vel(2) + health(1) = 7
        enemy_in = 7 + enemy_state_emb
        self.enemy_proj = nn.Sequential(
            nn.LayerNorm(enemy_in),
            nn.Linear(enemy_in, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        # vec: [B, 103]
        B = vec.size(0)

        player = vec[:, 0:13]                          # [B, 13]
        player_tok = self.player_proj(player).unsqueeze(1)  # [B,1,D]

        enemy_flat = vec[:, 13:103]                    # [B, 90]
        enemy = enemy_flat.view(B, 10, 9)              # [B,10,9]

        # select fields (drop ZERO)
        # type flags
        t_skel = enemy[:, :, 1:2]                      # [B,10,1]
        t_bomb = enemy[:, :, 2:3]                      # [B,10,1]
        # pos/vel/health
        pos = enemy[:, :, 3:5]                         # [B,10,2]
        vel = enemy[:, :, 5:7]                         # [B,10,2]
        health = enemy[:, :, 7:8]                      # [B,10,1]
        enemy_float = torch.cat([t_skel, t_bomb, pos, vel, health], dim=-1)  # [B,10,7]

        # state categorical
        state_raw = enemy[:, :, 8]  # [B,10] float
        state_id = torch.round(state_raw).to(torch.long)  # ★ 반올림 후 정수화
        state_id = state_id.clamp(min=0, max=self.enemy_state_vocab - 1)
        state_vec = self.state_emb(state_id)

        enemy_in = torch.cat([enemy_float, state_vec], dim=-1)  # [B,10,7+S]
        enemy_tok = self.enemy_proj(enemy_in)           # [B,10,D]

        ent_tokens = torch.cat([player_tok, enemy_tok], dim=1)  # [B,11,D]
        return ent_tokens

# ----------------------------
# Full backbone
# ----------------------------

class DualStreamBidirBackbone(nn.Module):
    """
    Shared backbone that outputs a single state vector [B,D] for actor/critic heads.
    """
    def __init__(
        self,
        num_ids: int = 11,
        model_dim: int = 256,
        id_emb_dim: int = 64,
        patch_hw=(9, 10),
        map_layers: int = 3,
        ent_layers: int = 2,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        enemy_state_vocab: int = 32,
        enemy_state_emb: int = 16,
        cross_layers: int = 1,
    ):
        super().__init__()
        self.map_tok = MapTokenizer(
            num_ids=num_ids,
            id_emb_dim=id_emb_dim,
            model_dim=model_dim,
            patch_hw=patch_hw,
            dropout=dropout,
        )
        self.ent_tok = EntityTokenizer(
            model_dim=model_dim,
            enemy_state_vocab=enemy_state_vocab,
            enemy_state_emb=enemy_state_emb,
            dropout=dropout,
        )

        self.map_enc = nn.ModuleList([
            TransformerBlock(model_dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(map_layers)
        ])
        self.ent_enc = nn.ModuleList([
            TransformerBlock(model_dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(ent_layers)
        ])

        # bidirectional cross attention (1 round)
        assert cross_layers >= 1
        self.cross_layers = int(cross_layers)
        self.ent_to_map = nn.ModuleList([
            CrossAttn(model_dim, heads=heads, dropout=dropout)
            for _ in range(self.cross_layers)
        ])
        self.map_to_ent = nn.ModuleList([
            CrossAttn(model_dim, heads=heads, dropout=dropout)
            for _ in range(self.cross_layers)
        ])

        # readout: state token (learnable query)
        self.state_q = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.state_q, std=0.02)
        self.readout = CrossAttn(model_dim, heads=heads, dropout=dropout)

        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, id_map: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        id_map: [B,90,160] long
        vec:    [B,103] float
        returns: state_vec [B, D]
        """
        map_tokens = self.map_tok(id_map)   # [B,160,D]
        ent_tokens = self.ent_tok(vec)      # [B,11,D]

        for blk in self.map_enc:
            map_tokens = blk(map_tokens)
        for blk in self.ent_enc:
            ent_tokens = blk(ent_tokens)

        # fusion (bidir cross-attn) -- multi rounds
        for i in range(self.cross_layers):
            map_tokens = self.ent_to_map[i](map_tokens, ent_tokens) # q = ent, kv = map, [B,160,D]
            ent_tokens = self.map_to_ent[i](ent_tokens, map_tokens) # q = map, kv = ent, [B,11,D]

        # readout
        kv = torch.cat([ent_tokens, map_tokens], dim=1)       # [B,171,D]
        q = self.state_q.expand(kv.size(0), -1, -1)           # [B,1,D]
        state = self.readout(q, kv)                           # [B,1,D]
        state = self.final_norm(state).squeeze(1)             # [B,D]
        return state

# ----------------------------
# Demo: run with your tensors
# ----------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Example shapes (your real tensors will come from numpy like you showed)

    backbone = DualStreamBidirBackbone(
        num_ids=11,
        model_dim=256,
        id_emb_dim=64,
        patch_hw=(9,10),
        map_layers=3,
        ent_layers=2,
        heads=8,
        dropout=0.0,
        enemy_state_vocab=32,  # adjust if you know exact
        enemy_state_emb=16,
    ).to(device)

    id_map_np = np.zeros((90, 160), dtype=np.int32)
    vec_obs_np = np.zeros((103,), dtype=np.float32)

    id_t = torch.from_numpy(id_map_np).to(device=device, dtype=torch.long).unsqueeze(0)   # (1,90,160)
    vec_t = torch.from_numpy(vec_obs_np).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,103)

    with torch.no_grad():
        state_vec = backbone(id_t, vec_t)  # (1, D)

    print("state_vec shape:", state_vec.shape)
    print("state_vec dtype:", state_vec.dtype)
    print("state_vec min/max:", state_vec.min().item(), state_vec.max().item())
    print("state_vec has nan:", torch.isnan(state_vec).any().item())
    print("state_vec has inf:", torch.isinf(state_vec).any().item())

    assert state_vec.shape == (1, 256)
    assert not torch.isnan(state_vec).any()
    assert not torch.isinf(state_vec).any()
