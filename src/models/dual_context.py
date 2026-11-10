"""Dual-Context Transformer: Zone/Type Encoder + Temporal Decoder + Hourly Head (SPEC 4)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import torch
from torch import nn

from src.models.hourly_head import HourlyAnomalyHead
from src.models.patch_embedding import DayPatchEmbedding, SinusoidalDayPE
from src.models.set_encoder import CONTEXT_POOLINGS, JointDayEncoder, PerDayEncoder
from src.models.temporal_decoder import TemporalDecoder

CONTEXTS = ("zone", "type")
ENCODER_SCOPES = ("joint_days", "per_day_buildings", "per_hour_buildings")
# Token 단위 -> Token당 시간 수. hour = v5 (docs/LEAD_HourLevel_Transformer_v5_Change.md)
TOKEN_UNITS = {"day": 24, "hour": 1}

# SPEC 8.1에서 값이 하나로 확정된 키. 다른 값이 들어오면 구현과 SPEC이 어긋난 것이므로 거부한다.
FIXED = {
    "id_lookup_embedding": False,
    "decoder_window_days": "variable",
    "encoder_tokens_layout": "day_major",
    "target_first_per_day": True,
    "encoder_self_attention_temporal_mask": "none",
    "cross_attention_temporal_mask": "none",
    "cross_attention_fusion": "sequential",
    "exclude_target_from_peers": False,
}
# Encoder 방식에 따라 하나로 정해지는 키 (docs/LEAD_Encoder_CrossAttention_Change.md 6장).
SCOPE_FIXED = {
    "joint_days": {"encoder_joint_days": True},
    "per_day_buildings": {
        "encoder_joint_days": False,
        "encoder_internal_day_pe": False,
        "same_building_attention_bias": False,
    },
    # 시간마다 같은 시각의 건물 G개끼리만 Self-Attention. 구현은 per_day_buildings와 같고 Token만 1시간이다.
    "per_hour_buildings": {
        "token_unit": "hour",
        "encoder_joint_days": False,
        "encoder_internal_hour_pe": False,
        "same_building_attention_bias": False,
    },
}
# v5 Config의 시간 단위 키 -> 내부 필드 (같은 의미, Token 단위만 다름)
HOUR_KEY_ALIASES = {"encoder_context_hour_pe": "encoder_context_day_pe"}


@dataclass
class ModelConfig:
    token_unit: str = "day"  # day = 24시간 Token (v3까지) / hour = 1시간 Token (v5)
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
    # 기본값 joint_days = v1.0 Encoder라서 이전 Checkpoint도 그대로 불러온다.
    encoder_attention_scope: str = "joint_days"  # per_day_buildings = 날짜별 건물 Attention
    encoder_context_day_pe: str = "sinusoidal"  # per_day_buildings에서 추출한 날짜별 Context에 더하는 PE (none 가능)
    # per_day_buildings에서 Cross-Attention K/V로 넘길 날짜별 Context.
    # target_token = 대상 건물 1개 (KV T개), target_plus_peer_mean = 대상 + 유효 Peer 출력 평균 (KV 2T개),
    # all_buildings = 날짜별 전체 건물 출력 (KV T×G개)
    # (docs/LEAD_PeerSummary_2T_CrossAttention_Change.md). 기본값 target_token이라 이전 Checkpoint도 그대로 불러온다.
    encoder_context_pooling: str = "target_token"
    cross_attention_memory_layout: str = "day_major_target_peer"  # 2T Memory 배치 [t1, p1, t2, p2, ...]
    max_window_days: int = 64  # 최대 Window Token 수 (날짜 PE 표 크기). hour Token이면 시간 수

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model은 n_heads로 나누어떨어져야 한다.")
        if self.head_dim is not None and self.head_dim != self.d_model // self.n_heads:
            raise ValueError("head_dim은 d_model / n_heads와 같아야 한다.")
        if self.token_unit not in TOKEN_UNITS:
            raise ValueError(f"token_unit: {' | '.join(TOKEN_UNITS)}")
        if self.layer_norm not in ("pre", "post"):
            raise ValueError("layer_norm은 pre 또는 post.")
        if self.encoder_positional_encoding not in ("shared_sinusoidal_day", "none"):
            raise ValueError("encoder_positional_encoding: shared_sinusoidal_day | none")
        pe_name = f"sinusoidal_{self.token_unit}"
        if self.decoder_positional_encoding not in (pe_name, "none"):
            raise ValueError(f"token_unit={self.token_unit}이면 decoder_positional_encoding: {pe_name} | none")
        if (self.encoder_attention_scope == "per_hour_buildings") != (self.token_unit == "hour"):
            raise ValueError("hour Token은 encoder_attention_scope: per_hour_buildings와 함께 쓴다.")
        if self.encoder_attention_scope not in ENCODER_SCOPES:
            raise ValueError(f"encoder_attention_scope: {' | '.join(ENCODER_SCOPES)}")
        if self.encoder_context_day_pe not in ("sinusoidal", "none"):
            raise ValueError("encoder_context_day_pe: sinusoidal | none")
        if self.encoder_context_pooling not in CONTEXT_POOLINGS:
            raise ValueError(f"encoder_context_pooling: {' | '.join(CONTEXT_POOLINGS)}")
        if self.encoder_context_pooling != "target_token" and not self.per_day:
            raise ValueError("encoder_context_pooling은 per_day_buildings Encoder에서만 바꿀 수 있다.")
        if self.cross_attention_memory_layout != "day_major_target_peer":
            raise ValueError("cross_attention_memory_layout: day_major_target_peer")
        if self.per_day and self.same_building_attention_bias:
            raise ValueError("per_day_buildings Encoder는 날짜 간 Attention이 없어 동일 건물 Bias를 쓰지 않는다.")
        if self.encoder_peers not in ("all", "target_only"):
            raise ValueError("encoder_peers: all | target_only")
        order = list(self.cross_attention_order)
        if len(set(order)) != len(order) or not set(order) <= set(CONTEXTS):
            raise ValueError(f"cross_attention_order는 {CONTEXTS}의 중복 없는 부분 순열이어야 한다.")

    @classmethod
    def from_dict(cls, cfg: dict) -> "ModelConfig":
        cfg = dict(cfg)
        unit = cfg.get("token_unit", "day")
        if unit in TOKEN_UNITS:
            # Token 크기를 나타내는 키들은 token_unit과 같은 값이어야 한다 (v5 Config는 셋 다 적는다).
            hours = TOKEN_UNITS[unit]
            for key in ("patch_size_hours", "hour_projection_in_features", "anomaly_head_out_features"):
                if key in cfg and cfg[key] != hours:
                    raise ValueError(f"token_unit={unit}이면 {key}={hours}여야 한다: {cfg[key]!r}")
        for alias, name in HOUR_KEY_ALIASES.items():
            if alias in cfg:
                if unit != "hour":
                    raise ValueError(f"{alias}는 token_unit: hour에서만 쓴다.")
                cfg[name] = cfg.pop(alias)
        for key, value in FIXED.items():
            if key in cfg and cfg[key] != value:
                raise ValueError(f"SPEC 확정 항목 {key}={value!r}와 다르다: {cfg[key]!r}")
        scope = cfg.get("encoder_attention_scope", "joint_days")
        for key, value in SCOPE_FIXED.get(scope, {}).items():
            if key in cfg and cfg[key] != value:
                raise ValueError(f"encoder_attention_scope={scope}이면 {key}={value!r}여야 한다: {cfg[key]!r}")
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def per_day(self) -> bool:
        """시점(날짜 또는 시간)마다 같은 시점의 건물끼리만 Attention하는 Encoder인가."""
        return self.encoder_attention_scope in ("per_day_buildings", "per_hour_buildings")

    @property
    def patch_size_hours(self) -> int:
        return TOKEN_UNITS[self.token_unit]

    @property
    def contexts(self) -> tuple[str, ...]:
        return tuple(self.cross_attention_order)


class DualContextTransformer(nn.Module):
    """Train/Test 공통 전체 모델. 입력은 전력량뿐이며 forward는 Building ID를 받지 않는다."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, norm_first = cfg.d_model, cfg.layer_norm == "pre"
        p = cfg.patch_size_hours
        self.patch = DayPatchEmbedding(d, p)  # Linear(P, D): Day Linear(24,D) / Hour Linear(1,D)
        self.day_pe = SinusoidalDayPE(d, max_days=max(512, cfg.max_window_days))
        enc_pe = self.day_pe if cfg.encoder_positional_encoding != "none" else None
        context_pe = self.day_pe if cfg.encoder_context_day_pe != "none" else None
        layers = {"zone": cfg.zone_encoder_layers, "type": cfg.type_encoder_layers}
        # nn.ModuleDict의 "type" 키는 내장 메서드와 충돌하므로 `zone_encoder`/`type_encoder` 속성으로 등록한다.
        for c in cfg.contexts:
            patch = self.patch if cfg.share_patch_embedding else DayPatchEmbedding(d, p)
            if cfg.per_day:
                encoder = PerDayEncoder(
                    patch, context_pe, d, cfg.n_heads, cfg.d_ff, layers[c], cfg.dropout, norm_first,
                    cfg.target_role_embedding, cfg.encoder_context_pooling,
                )
            else:
                encoder = JointDayEncoder(
                    patch, enc_pe, d, cfg.n_heads, cfg.d_ff, layers[c], cfg.dropout, norm_first,
                    cfg.target_role_embedding, cfg.same_building_attention_bias, cfg.same_building_bias_init,
                )
            setattr(self, f"{c}_encoder", encoder)
        self.decoder = TemporalDecoder(
            d, cfg.n_heads, cfg.d_ff, cfg.decoder_layers, cfg.dropout, norm_first,
            cfg.contexts, cfg.ffn_after_each_attention, cfg.decoder_causal_mask,
        )
        self.head = HourlyAnomalyHead(d, p)  # Linear(D, P): Day 24 Logit / Hour 1 Logit per Token

    def forward(
        self,
        target_daily: torch.Tensor,
        zone_daily: torch.Tensor | None = None,
        zone_mask: torch.Tensor | None = None,
        type_daily: torch.Tensor | None = None,
        type_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """target_daily: `[B, T, P]`, {zone,type}_daily: `[B, T, G, P]`, mask: `[B, T×G]` -> logits `[B, T, P]`.

        P = Token당 시간 수 (Day 24, Hour 1). Hour Token의 logits `[B, T, 1]`은 시간마다 Logit 1개다.

        Cross-Attention Memory: joint_days = 전체 Token `[B, T×G, D]`,
        per_day_buildings = 날짜별 대상 Context `[B, T, D]` 또는 [대상, Peer 평균] `[B, 2T, D]`.
        """
        h = self.patch(target_daily)
        if self.cfg.decoder_positional_encoding != "none":
            h = h + self.day_pe(h.shape[1])
        given = {"zone": (zone_daily, zone_mask), "type": (type_daily, type_mask)}
        memories = {}
        for c in self.cfg.contexts:
            daily, mask = given[c]
            if daily is None or mask is None:
                raise ValueError(f"{c} Encoder가 켜져 있으면 {c}_daily/{c}_mask가 필요하다.")
            encoder = getattr(self, f"{c}_encoder")
            if self.cfg.per_day:  # Encoder가 Memory Mask를 함께 만든다 (대상 슬롯은 항상 유효)
                memories[c] = encoder(daily, mask)
            else:
                memories[c] = (encoder(daily, mask), mask)
        return self.head(self.decoder(h, memories))


def model_inputs(batch: dict) -> dict:
    """Batch dict에서 모델 입력만 골라낸다. 레이블·observation_mask·행 정보는 전달하지 않는다."""
    keys = ("target_daily", "zone_daily", "zone_mask", "type_daily", "type_mask")
    return {k: batch[k] for k in keys if k in batch}
