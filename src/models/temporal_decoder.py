"""Temporal Decoder: Causal Self-Attention + 날짜별 Zone/Type Cross-Attention (SPEC 4.4, 4.5)."""
from __future__ import annotations

import torch
from torch import nn

from src.models.set_encoder import safe_key_mask

FUSIONS = ("parallel", "sequential", "gate")


def causal_mask(t: int, device=None) -> torch.Tensor:
    """True = 참조 금지. 날짜 i는 j > i인 날짜를 볼 수 없다."""
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)


class DayCrossAttention(nn.Module):
    """날짜 d의 Decoder 상태가 같은 날짜 d의 Peer Encoder 출력만 참조한다.

    Peer 그룹이 비면 학습 가능한 `NULL_CONTEXT`를 반환한다 (Building ID와 무관한 공용 벡터).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.null_context = nn.Parameter(torch.zeros(d_model))

    def forward(self, query: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """query: `[B, T, D]`, memory: `[B, T, S, D]`, mask: `[B, T, S]` -> `[B, T, D]`."""
        b, t, s, d = memory.shape
        key_padding, empty = safe_key_mask(mask.reshape(b * t, s))
        out, _ = self.attn(
            query.reshape(b * t, 1, d), memory.reshape(b * t, s, d), memory.reshape(b * t, s, d),
            key_padding_mask=key_padding, need_weights=False,
        )
        out = torch.where(empty[:, None, None], self.null_context.expand_as(out), out)
        return out.reshape(b, t, d)


class TemporalDecoderLayer(nn.Module):
    """Causal Self-Attention -> Zone/Type Cross-Attention(융합 방식 선택) -> FFN, Residual + LayerNorm."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, norm_first: bool,
        contexts: tuple[str, ...], fusion: str,
    ):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"cross_attention_fusion must be one of {FUSIONS}")
        self.norm_first, self.contexts, self.fusion = norm_first, contexts, fusion
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(d_model)
        # nn.ModuleDict는 "type" 키가 내장 메서드와 충돌하므로 `{context}_cross` 속성으로 등록한다.
        for c in contexts:
            setattr(self, f"{c}_cross", DayCrossAttention(d_model, n_heads, dropout))
            setattr(self, f"{c}_norm", nn.LayerNorm(d_model))
        if contexts and fusion == "parallel":
            self.fuse = nn.Linear(len(contexts) * d_model, d_model)
        if contexts and fusion == "gate":
            self.gate = nn.Linear((len(contexts) + 1) * d_model, len(contexts))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _residual(self, h, norm, fn):
        if self.norm_first:
            return h + self.dropout(fn(norm(h)))
        return norm(h + self.dropout(fn(h)))

    def _cross_block(self, q, memories):
        outs = [getattr(self, f"{c}_cross")(q, *memories[c]) for c in self.contexts]
        if self.fusion == "parallel":
            return self.fuse(torch.cat(outs, dim=-1))
        g = torch.softmax(self.gate(torch.cat([q, *outs], dim=-1)), dim=-1)
        return sum(g[..., i : i + 1] * o for i, o in enumerate(outs))

    def forward(self, h: torch.Tensor, memories: dict, attn_mask: torch.Tensor | None) -> torch.Tensor:
        h = self._residual(
            h, self.norm_self, lambda x: self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        )
        if self.contexts:
            if self.fusion == "sequential":
                for c in self.contexts:
                    cross = getattr(self, f"{c}_cross")
                    h = self._residual(h, getattr(self, f"{c}_norm"), lambda x: cross(x, *memories[c]))
            else:
                # 병렬/게이트 융합은 하나의 Residual Branch로 묶고 첫 Context의 LayerNorm을 Query 정규화에 쓴다.
                h = self._residual(h, getattr(self, f"{self.contexts[0]}_norm"), lambda x: self._cross_block(x, memories))
        return self._residual(h, self.norm_ffn, self.ffn)


class TemporalDecoder(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float, norm_first: bool,
        contexts: tuple[str, ...], fusion: str, causal: bool,
    ):
        super().__init__()
        self.causal = causal
        self.layers = nn.ModuleList(
            TemporalDecoderLayer(d_model, n_heads, d_ff, dropout, norm_first, contexts, fusion)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, h: torch.Tensor, memories: dict) -> torch.Tensor:
        """h: `[B, T, D]`, memories: {context: (`[B, T, S, D]`, `[B, T, S]`)} -> `[B, T, D]`."""
        attn_mask = causal_mask(h.shape[1], h.device) if self.causal else None
        for layer in self.layers:
            h = layer(h, memories, attn_mask)
        return self.norm(h)
