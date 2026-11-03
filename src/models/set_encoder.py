"""Joint T-day Zone/Type Encoder + Shared Day PE (SPEC 4.2, 4.3)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.patch_embedding import DayPatchEmbedding, SinusoidalDayPE
from src.models.temporal_decoder import feed_forward

ROLE_TARGET, ROLE_PEER = 0, 1


def slot_index(t: int, g: int, device=None) -> torch.Tensor:
    """day-major Flatten된 `T×G` Token의 그룹 내 건물 위치. 0 = 대상 건물. 같은 slot = 같은 건물 (SPEC 3.2)."""
    return torch.arange(g, device=device).repeat(t)


class SameBuildingSelfAttention(nn.Module):
    """Multi-Head Self-Attention + 동일 건물 Bias `β_h·1[slot_i = slot_j]` (SPEC 4.2.1).

    `[L, L]` Bias 행렬을 만들지 않고 Q/K 뒤에 slot one-hot을 이어 붙인다:
    `[q, β_h·√d_h·s_i]·[k, s_j] / √d_h = q·k/√d_h + β_h·1[s_i = s_j]`. 메모리는 `O(L·G)`.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, bias_init: float):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.beta = nn.Parameter(torch.full((n_heads,), float(bias_init)))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, valid: torch.Tensor, slots: torch.Tensor, n_slots: int) -> torch.Tensor:
        """x: `[B, L, D]`, valid: `[B, L]` (True = 유효 Key), slots: `[L]` -> `[B, L, D]`."""
        b, l, d = x.shape
        h, dh = self.n_heads, self.head_dim
        q, k, v = self.qkv(x).view(b, l, 3, h, dh).permute(2, 0, 3, 1, 4)  # 각 [B, H, L, dh]
        onehot = F.one_hot(slots, n_slots).to(x.dtype)[None, None].expand(b, h, l, n_slots)
        q = torch.cat([q, onehot * (self.beta * dh**0.5)[None, :, None, None]], dim=-1)
        k = torch.cat([k, onehot], dim=-1)
        v = torch.cat([v, torch.zeros_like(onehot)], dim=-1)
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=valid[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0, scale=dh**-0.5,
        )
        return self.out(o[..., :dh].transpose(1, 2).reshape(b, l, d))


class SameBuildingEncoderLayer(nn.Module):
    """Self-Attention(동일 건물 Bias) -> FFN. 각 Sublayer는 Residual + LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, norm_first: bool, bias_init: float):
        super().__init__()
        self.norm_first = norm_first
        self.attn = SameBuildingSelfAttention(d_model, n_heads, dropout, bias_init)
        self.ffn = feed_forward(d_model, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, valid, slots, n_slots):
        if self.norm_first:
            h = h + self.dropout(self.attn(self.norm1(h), valid, slots, n_slots))
            return h + self.dropout(self.ffn(self.norm2(h)))
        h = self.norm1(h + self.dropout(self.attn(h, valid, slots, n_slots)))
        return self.norm2(h + self.dropout(self.ffn(h)))


class JointDayEncoder(nn.Module):
    """그룹 건물들의 T일 Day Token을 day-major로 Flatten해 하나의 Self-Attention 시퀀스로 처리한다.

    건물 나열 순서용 PE는 없고, 같은 날짜 Token에는 같은 PE(d)를 더한다.
    날짜 Causal Mask 없이 Window 전체를 양방향으로 본다 (오프라인/사후 탐지).
    v1.0: 대상/Peer Role Embedding, 동일 건물 Attention Bias (SPEC 4.2.1). 둘 다 끄면 v0.9 Encoder.
    """

    def __init__(
        self, patch: DayPatchEmbedding, day_pe: SinusoidalDayPE | None,
        d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float, norm_first: bool,
        role_embedding: bool = False, same_building_bias: bool = False, bias_init: float = 0.0,
    ):
        super().__init__()
        self.patch, self.day_pe = patch, day_pe
        # 0으로 초기화: 학습 시작 시점에는 Role 정보가 없는 모델과 같다.
        self.role = nn.Embedding(2, d_model) if role_embedding else None
        if self.role is not None:
            nn.init.zeros_(self.role.weight)
        self.same_building_bias = same_building_bias
        if same_building_bias:
            self.layers = nn.ModuleList(
                SameBuildingEncoderLayer(d_model, n_heads, d_ff, dropout, norm_first, bias_init)
                for _ in range(n_layers)
            )
        else:
            self.layers = nn.ModuleList(
                nn.TransformerEncoderLayer(
                    d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True, norm_first=norm_first
                )
                for _ in range(n_layers)
            )
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def embed(self, daily: torch.Tensor) -> torch.Tensor:
        """daily: `[B, T, G, 24]` -> Flatten된 입력 Token `[B, T×G, D]`."""
        b, t, g, _ = daily.shape
        tokens = self.patch(daily)
        if self.day_pe is not None:
            tokens = tokens + self.day_pe(t)[None, :, None, :]
        if self.role is not None:
            role = torch.full((g,), ROLE_PEER, dtype=torch.long, device=daily.device)
            role[0] = ROLE_TARGET
            tokens = tokens + self.role(role)[None, None]
        return tokens.reshape(b, t * g, -1)

    def forward(self, daily: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """mask: `[B, T×G]`, True = 유효 Token(그룹 패딩이 아님) -> 메모리 `[B, T×G, D]`."""
        _, t, g, _ = daily.shape
        h = self.embed(daily)
        slots = slot_index(t, g, daily.device)
        for layer in self.layers:
            if self.same_building_bias:
                h = layer(h, mask, slots, g)
            else:
                h = layer(h, src_key_padding_mask=~mask)
        return self.norm(h)
