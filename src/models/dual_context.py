"""Dual-Context Transformer Backbone: Zone Encoder + Type Encoder + Temporal Decoder (SPEC 4)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
from torch import nn

from src.models.patch_embedding import DayPatchEmbedding, SinusoidalDayPE
from src.models.set_encoder import SetEncoder
from src.models.temporal_decoder import TemporalDecoder


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    head_dim: int | None = 32
    d_ff: int = 512
    zone_encoder_layers: int = 2
    type_encoder_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.1
    layer_norm: str = "pre"
    patch_size_hours: int = 24
    decoder_window_days: int = 7
    id_lookup_embedding: bool = False
    encoder_positional_encoding: bool = False
    decoder_positional_encoding: bool = True
    decoder_causal_mask: bool = True
    cross_attention_fusion: str = "parallel"
    exclude_target_from_peers: bool = True
    # Ablation 스위치 (SPEC 5.3, 9: E1–E4, Context-only)
    use_zone_encoder: bool = True
    use_type_encoder: bool = True
    share_patch_embedding: bool = True
    target_input: bool = True

    def __post_init__(self):
        if self.id_lookup_embedding:
            raise ValueError("SPEC 2.1 [확정]: Input-side Building ID Lookup Embedding은 사용하지 않는다.")
        if self.encoder_positional_encoding:
            raise ValueError("SPEC 4.2/4.3 [확정]: Zone/Type Encoder에는 Positional Encoding을 넣지 않는다.")
        if self.patch_size_hours != 24:
            raise ValueError("SPEC 3.1 [확정]: 하루 24시간 = 토큰 하나.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model은 n_heads로 나누어떨어져야 한다.")
        if self.head_dim is not None and self.head_dim != self.d_model // self.n_heads:
            raise ValueError("head_dim은 d_model / n_heads와 같아야 한다.")
        if self.layer_norm not in ("pre", "post"):
            raise ValueError("layer_norm은 pre 또는 post.")

    @classmethod
    def from_dict(cls, cfg: dict) -> "ModelConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def contexts(self) -> tuple[str, ...]:
        return tuple(c for c, on in (("zone", self.use_zone_encoder), ("type", self.use_type_encoder)) if on)


class DualContextTransformer(nn.Module):
    """입력은 전력 소비량뿐이다. forward는 Building ID를 받지 않는다."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, norm_first = cfg.d_model, cfg.layer_norm == "pre"
        self.patch = DayPatchEmbedding(d, cfg.patch_size_hours)
        # Context별 모듈은 `zone_encoder`, `type_encoder`, `{context}_patch` 속성으로 등록한다.
        layers = {"zone": cfg.zone_encoder_layers, "type": cfg.type_encoder_layers}
        for c in cfg.contexts:
            setattr(self, f"{c}_patch", self.patch if cfg.share_patch_embedding else DayPatchEmbedding(d))
            setattr(self, f"{c}_encoder", SetEncoder(d, cfg.n_heads, cfg.d_ff, layers[c], cfg.dropout, norm_first))
        self.day_pe = SinusoidalDayPE(d) if cfg.decoder_positional_encoding else nn.Identity()
        self.decoder = TemporalDecoder(
            d, cfg.n_heads, cfg.d_ff, cfg.decoder_layers, cfg.dropout, norm_first,
            cfg.contexts, cfg.cross_attention_fusion, cfg.decoder_causal_mask,
        )

    def encode_context(self, name: str, daily: torch.Tensor, mask: torch.Tensor):
        """daily: `[B, T, S, 24]`, mask: `[B, T, S]`. 날짜마다 독립적으로 건물 집합을 인코딩한다."""
        b, t, s, _ = daily.shape
        tokens = getattr(self, f"{name}_patch")(daily).reshape(b * t, s, -1)
        enc = getattr(self, f"{name}_encoder")(tokens, mask.reshape(b * t, s))
        return enc.reshape(b, t, s, -1), mask

    def forward(
        self,
        target_daily: torch.Tensor,
        zone_daily: torch.Tensor | None = None,
        zone_mask: torch.Tensor | None = None,
        type_daily: torch.Tensor | None = None,
        type_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """target_daily: `[B, T, 24]` -> 날짜별 Decoder 상태 `[B, T, D]`."""
        if not self.cfg.target_input:
            target_daily = torch.zeros_like(target_daily)
        h = self.day_pe(self.patch(target_daily))
        given = {"zone": (zone_daily, zone_mask), "type": (type_daily, type_mask)}
        memories = {}
        for c in self.cfg.contexts:
            daily, mask = given[c]
            if daily is None:
                raise ValueError(f"{c} Encoder가 켜져 있으면 {c}_daily/{c}_mask가 필요하다.")
            memories[c] = self.encode_context(c, daily, mask)
        return self.decoder(h, memories)


def backbone_inputs(batch: dict) -> dict:
    """Batch dict에서 모델 입력만 골라낸다. building_target 등 정답은 전달하지 않는다."""
    keys = ("target_daily", "zone_daily", "zone_mask", "type_daily", "type_mask")
    return {k: batch[k] for k in keys if k in batch}
