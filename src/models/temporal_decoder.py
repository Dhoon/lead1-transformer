"""Temporal Decoder: Masked Self-Attn+FFN -> Zone Cross+FFN -> Type Cross+FFN (SPEC 4.4, 4.5)."""
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


class TemporalDecoder(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float, norm_first: bool,
        cross_order: tuple[str, ...], ffn_after_each_attention: bool, causal: bool,
    ):
        super().__init__()
        self.causal = causal
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
