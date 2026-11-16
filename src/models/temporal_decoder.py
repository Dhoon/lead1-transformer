"""Temporal Decoder (SPEC 4.4, 4.5).

sequential (기본): Masked Self-Attn+FFN -> Zone Cross+FFN -> Type Cross+FFN
parallel_concat_mlp: Masked Self-Attn+FFN -> [Zone Cross ∥ Type Cross] -> Concat MLP -> FFN
  (docs/LEAD_v3pm_Parallel_Site_Type_CrossAttention_Change.md)
"""
from __future__ import annotations

import torch
from torch import nn


def causal_mask(t: int, device=None) -> torch.Tensor:
    """True = 참조 금지. 날짜 i는 j > i인 날짜를 볼 수 없다."""
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)


def feed_forward(d_model: int, d_ff: int, dropout: float) -> nn.Module:
    return nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))


class TemporalDecoderLayer(nn.Module):
    """`block_names` 순서대로 Sublayer를 적용한다. 각 Sublayer는 Residual + LayerNorm.

    기본: ["self", "ffn", "zone", "ffn", "type", "ffn"]. Cross-Attention은 Encoder의 T일 전체 메모리를 본다.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, norm_first: bool,
        cross_order: tuple[str, ...], ffn_after_each_attention: bool,
    ):
        super().__init__()
        self.norm_first = norm_first
        attentions = ["self", *cross_order]
        if ffn_after_each_attention:
            self.block_names = [n for a in attentions for n in (a, "ffn")]
        else:
            self.block_names = [*attentions, "ffn"]
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for name in self.block_names:
            if name == "ffn":
                self.blocks.append(feed_forward(d_model, d_ff, dropout))
            else:
                self.blocks.append(nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True))
            self.norms.append(nn.LayerNorm(d_model))
        self.dropout = nn.Dropout(dropout)

    def _sublayer(self, name, block, x, memories, attn_mask):
        if name == "ffn":
            return block(x)
        if name == "self":
            return block(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        memory, mask = memories[name]
        return block(x, memory, memory, key_padding_mask=~mask, need_weights=False)[0]

    def forward(self, h: torch.Tensor, memories: dict, attn_mask: torch.Tensor | None) -> torch.Tensor:
        for name, block, norm in zip(self.block_names, self.blocks, self.norms):
            if self.norm_first:
                h = h + self.dropout(self._sublayer(name, block, norm(h), memories, attn_mask))
            else:
                h = norm(h + self.dropout(self._sublayer(name, block, h, memories, attn_mask)))
        return h


class ParallelFusionDecoderLayer(nn.Module):
    """Site(Zone)/Type Cross-Attention을 같은 Query에서 병렬로 하고 MLP로 합친다 (Pre-LN 전용).

        H_self = h + Drop(SelfAttn(LN(h)));  H_self = H_self + Drop(FFN(LN(H_self)))
        Z = LN(H_self)
        O_c = CrossAttn_c(Q=Z, K/V=C_c)              c ∈ cross_order, Branch마다 독립 파라미터
        F = MLP([Z; O_zone; O_type])                  (1+n)D -> D -> D
        H_fused = H_self + Drop(F);  out = H_fused + Drop(FFN(LN(H_fused)))

    availability=True (docs/LEAD_v3pm_PeerOnly_AllPeer_Parallel_Change.md 5장):
        m_c = 샘플 Window 안에 유효 Memory Token이 하나라도 있는가 `[B]`
        F = MLP([Z; m_zone·O_zone; m_type·O_type; m_zone; m_type])   (1+n)D+n -> D -> D
        m_c = 0인 샘플은 전부 가린 Softmax(NaN)를 피하려고 Key 하나를 풀어 계산한 뒤 출력을 0으로 바꾼다.

    same_day=True (같은 문서 7장 B안): d일 Query는 d일 Peer Key만 본다. Memory는 day-major `[B, T×G, D]`.
        Score Mask M[d, (d', j)] = 0 if d' = d and 유효, else -inf.
        m_c는 날짜별 `[B, T]`: 그날 유효 Peer가 하나라도 있는가. 0인 (샘플, 날짜)는 출력 0, 플래그 0.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, cross_order: tuple[str, ...],
        availability: bool = False, same_day: bool = False,
    ):
        super().__init__()
        if same_day and not availability:
            raise ValueError("same_day Cross-Attention은 날짜별 Availability 플래그와 함께 쓴다.")
        self.cross_order = tuple(cross_order)
        self.availability = availability
        self.same_day = same_day
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        self.ffn1 = feed_forward(d_model, d_ff, dropout)
        self.ffn1_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)  # Z = LN(H_self): 두 Branch가 같은 Query 입력을 쓴다
        # nn.Module에는 type() 메서드가 있어 "type" 이름으로 등록할 수 없으므로 cross_order 순서의 ModuleList로 둔다.
        self.cross = nn.ModuleList(
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True) for _ in self.cross_order
        )
        self.context_fusion = nn.Sequential(
            nn.Linear((1 + len(self.cross_order)) * d_model + (len(self.cross_order) if availability else 0), d_model),
            nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.ffn2 = feed_forward(d_model, d_ff, dropout)
        self.ffn2_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, memories: dict, attn_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.self_norm(h)
        h = h + self.dropout(self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0])
        h = h + self.dropout(self.ffn1(self.ffn1_norm(h)))
        z = self.cross_norm(h)
        outs, flags = [], []
        for name, attn in zip(self.cross_order, self.cross):
            memory, mask = memories[name]
            if not self.availability:
                outs.append(attn(z, memory, memory, key_padding_mask=~mask, need_weights=False)[0])  # Cross는 Causal 없음
                continue
            if self.same_day:
                o, m = self._same_day_attention(attn, z, memory, mask)   # m: [B, T, 1]
            else:
                avail = mask.any(dim=1)                                  # [B]
                safe = mask.clone()
                safe[~avail, 0] = True                                   # 빈 샘플: 계산만 하고 아래에서 버린다
                o = attn(z, memory, memory, key_padding_mask=~safe, need_weights=False)[0]
                m = avail[:, None, None].expand(-1, z.shape[1], 1)
            outs.append(torch.where(m, o, torch.zeros_like(o)))
            flags.append(m.to(z.dtype))
        h = h + self.dropout(self.context_fusion(torch.cat([z, *outs, *flags], dim=-1)))
        return h + self.dropout(self.ffn2(self.ffn2_norm(h)))

    @staticmethod
    def _same_day_attention(attn: nn.MultiheadAttention, z, memory, mask):
        """z: `[B, T, D]`, memory: `[B, T×G, D]`, mask: `[B, T×G]` -> (출력 `[B, T, D]`, 날짜별 존재 `[B, T, 1]`)."""
        b, t, _ = z.shape
        g = memory.shape[1] // t
        if g * t != memory.shape[1]:
            raise ValueError("same_day는 day-major T×G Memory가 필요하다.")
        key_day = torch.arange(t * g, device=z.device) // g                       # [T×G]
        allowed = (key_day[None, None, :] == torch.arange(t, device=z.device)[None, :, None]) & mask[:, None, :]
        avail = allowed.any(dim=-1)                                                 # [B, T]
        safe = allowed.clone()
        safe[..., 0] |= ~avail                                                      # 빈 (샘플, 날짜): 계산만 하고 버린다
        block = (~safe).repeat_interleave(attn.num_heads, dim=0)                    # [B×H, T, T×G], True = 차단
        o = attn(z, memory, memory, attn_mask=block, need_weights=False)[0]
        return o, avail[..., None]


class TemporalDecoder(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float, norm_first: bool,
        cross_order: tuple[str, ...], ffn_after_each_attention: bool, causal: bool, fusion: str = "sequential",
        fusion_availability: bool = False, same_day: bool = False,
    ):
        super().__init__()
        self.causal = causal
        if fusion == "parallel_concat_mlp":
            if not norm_first:
                raise ValueError("parallel_concat_mlp는 Pre-LN(layer_norm: pre)에서만 구현했다.")
            self.layers = nn.ModuleList(
                ParallelFusionDecoderLayer(d_model, n_heads, d_ff, dropout, cross_order, fusion_availability, same_day)
                for _ in range(n_layers)
            )
        else:
            self.layers = nn.ModuleList(
                TemporalDecoderLayer(d_model, n_heads, d_ff, dropout, norm_first, cross_order, ffn_after_each_attention)
                for _ in range(n_layers)
            )
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, h: torch.Tensor, memories: dict) -> torch.Tensor:
        """h: `[B, T, D]`, memories: {context: (`[B, M, D]`, `[B, M]`)} -> `[B, T, D]`."""
        attn_mask = causal_mask(h.shape[1], h.device) if self.causal else None
        for layer in self.layers:
            h = layer(h, memories, attn_mask)
        return self.norm(h)
