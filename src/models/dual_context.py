"""Dual-Context Transformer: Zone/Type Joint Encoder + Temporal Decoder + Hourly Head (SPEC 4)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import torch
from torch import nn

from src.models.hourly_head import HourlyAnomalyHead
from src.models.patch_embedding import DayPatchEmbedding, SinusoidalDayPE
from src.models.set_encoder import JointDayEncoder
from src.models.temporal_decoder import TemporalDecoder

CONTEXTS = ("zone", "type")

# SPEC 8.1에서 값이 하나로 확정된 키. 다른 값이 들어오면 구현과 SPEC이 어긋난 것이므로 거부한다.
FIXED = {
    "patch_size_hours": 24,
    "id_lookup_embedding": False,
    "decoder_window_days": "variable",
    "encoder_joint_days": True,
    "encoder_tokens_layout": "day_major",
    "target_first_per_day": True,
    "encoder_self_attention_temporal_mask": "none",
    "cross_attention_temporal_mask": "none",
    "cross_attention_fusion": "sequential",
    "exclude_target_from_peers": False,
}


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
    encoder_positional_encoding: str = "shared_sinusoidal_day"  # none = E8 Ablation
    decoder_positional_encoding: str = "sinusoidal_day"
    decoder_causal_mask: bool = True
    cross_attention_order: list[str] = field(default_factory=lambda: ["zone", "type"])  # E1–E3, E5
    ffn_after_each_attention: bool = True
    share_patch_embedding: bool = True  # [제안] Linear(24,D) 공유부터 시작
    # v1.0 (SPEC 4.2.1). 기본값 False = v0.9 Encoder라서 이전 Checkpoint도 그대로 불러온다. configs/train.yaml에서 켠다.
    target_role_embedding: bool = False  # E14
    same_building_attention_bias: bool = False  # E13
    same_building_bias_init: float = 0.0
    encoder_peers: str = "all"  # all = E4 / target_only = E1b
    max_window_days: int = 64

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model은 n_heads로 나누어떨어져야 한다.")
        if self.head_dim is not None and self.head_dim != self.d_model // self.n_heads:
            raise ValueError("head_dim은 d_model / n_heads와 같아야 한다.")
        if self.layer_norm not in ("pre", "post"):
            raise ValueError("layer_norm은 pre 또는 post.")
        if self.encoder_positional_encoding not in ("shared_sinusoidal_day", "none"):
            raise ValueError("encoder_positional_encoding: shared_sinusoidal_day | none")
        if self.decoder_positional_encoding not in ("sinusoidal_day", "none"):
            raise ValueError("decoder_positional_encoding: sinusoidal_day | none")
        if self.encoder_peers not in ("all", "target_only"):
            raise ValueError("encoder_peers: all | target_only")
        order = list(self.cross_attention_order)
        if len(set(order)) != len(order) or not set(order) <= set(CONTEXTS):
            raise ValueError(f"cross_attention_order는 {CONTEXTS}의 중복 없는 부분 순열이어야 한다.")

    @classmethod
    def from_dict(cls, cfg: dict) -> "ModelConfig":
        for key, value in FIXED.items():
            if key in cfg and cfg[key] != value:
                raise ValueError(f"SPEC 확정 항목 {key}={value!r}와 다르다: {cfg[key]!r}")
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def contexts(self) -> tuple[str, ...]:
        return tuple(self.cross_attention_order)


class DualContextTransformer(nn.Module):
    """Train/Test 공통 전체 모델. 입력은 전력량뿐이며 forward는 Building ID를 받지 않는다."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, norm_first = cfg.d_model, cfg.layer_norm == "pre"
        self.patch = DayPatchEmbedding(d)
        self.day_pe = SinusoidalDayPE(d, max_days=max(512, cfg.max_window_days))
        enc_pe = self.day_pe if cfg.encoder_positional_encoding != "none" else None
        layers = {"zone": cfg.zone_encoder_layers, "type": cfg.type_encoder_layers}
        # nn.ModuleDict의 "type" 키는 내장 메서드와 충돌하므로 `zone_encoder`/`type_encoder` 속성으로 등록한다.
        for c in cfg.contexts:
            patch = self.patch if cfg.share_patch_embedding else DayPatchEmbedding(d)
            setattr(self, f"{c}_encoder", JointDayEncoder(
                patch, enc_pe, d, cfg.n_heads, cfg.d_ff, layers[c], cfg.dropout, norm_first,
                cfg.target_role_embedding, cfg.same_building_attention_bias, cfg.same_building_bias_init,
            ))
        self.decoder = TemporalDecoder(
            d, cfg.n_heads, cfg.d_ff, cfg.decoder_layers, cfg.dropout, norm_first,
            cfg.contexts, cfg.ffn_after_each_attention, cfg.decoder_causal_mask,
        )
        self.head = HourlyAnomalyHead(d)

    def forward(
        self,
        target_daily: torch.Tensor,
        zone_daily: torch.Tensor | None = None,
        zone_mask: torch.Tensor | None = None,
        type_daily: torch.Tensor | None = None,
        type_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """target_daily: `[B, T, 24]`, {zone,type}_daily: `[B, T, G, 24]`, mask: `[B, T×G]` -> logits `[B, T, 24]`."""
        h = self.patch(target_daily)
        if self.cfg.decoder_positional_encoding != "none":
            h = h + self.day_pe(h.shape[1])
        given = {"zone": (zone_daily, zone_mask), "type": (type_daily, type_mask)}
        memories = {}
        for c in self.cfg.contexts:
            daily, mask = given[c]
            if daily is None or mask is None:
                raise ValueError(f"{c} Encoder가 켜져 있으면 {c}_daily/{c}_mask가 필요하다.")
            memories[c] = (getattr(self, f"{c}_encoder")(daily, mask), mask)
        return self.head(self.decoder(h, memories))


def model_inputs(batch: dict) -> dict:
    """Batch dict에서 모델 입력만 골라낸다. 레이블·observation_mask·행 정보는 전달하지 않는다."""
    keys = ("target_daily", "zone_daily", "zone_mask", "type_daily", "type_mask")
    return {k: batch[k] for k in keys if k in batch}
