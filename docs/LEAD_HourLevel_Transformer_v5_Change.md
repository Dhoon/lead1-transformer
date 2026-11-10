# LEAD Dual-Context Transformer — Hour-level 전환 변경 명세 (v5)

> **문서 유형:** 기존 `SPEC.md` + `LEAD_Encoder_CrossAttention_Change.md`에 적용할 **부분 변경 명세**. 전체 SPEC을 대체하지 않는다.
>
> **기준 모델:** `v3_perday`(날짜별 Zone/Type Encoder, 날짜별 Target Context만 Cross-Attention K/V로 사용). `v3_allpeer`와 2T Peer Summary는 이번 기본 구현에 포함하지 않는다.
>
> **변경 목표:** 기존 **Day Token(24시간 → 1 Token)**을 **Hour Token(1시간 → 1 Token)**으로 변경하여 시간 단위 이상 탐지를 직접 수행한다. 기존 Zone/Type 간 역할 분담과 순차 Cross-Attention은 유지한다.
>
> **범위 밖:** 새로운 시간별 변화량·비율 Feature, CNN/추가 Branch, 날씨/캘린더 임베딩, Peer 평균 또는 전체 Peer K/V 전달은 이번 변경에서 **추가하지 않는다**. 향후 별도 Ablation으로 다룬다.

## 1. 확정 하이퍼파라미터 및 변경 요약

| 항목 | 기존 v3 (Day-level) | **v5 기본안 (Hour-level)** |
|---|---|---|
| Token 의미 | 건물 1개의 24시간 사용량 | **건물 1개의 1시간 사용량** |
| 입력 Projection | `Linear(24,128)` | **`Linear(1,64)`** |
| 최대 Sequence 길이 `T_max` | 64일 | **168시간 (7일)** |
| `d_model (D)` | 128 | **64** |
| Attention Head 수 `H` | 4 | **4** |
| Head 차원 `d_k = D/H` | 32 | **16** |
| FFN Hidden (`4D`) | 512 | **256** |
| Zone Encoder | 2 Layers, 날짜별 건물 간 Attention | **2 Layers, 시간별 건물 간 Attention** |
| Type Encoder | 2 Layers, 날짜별 건물 간 Attention | **2 Layers, 시간별 건물 간 Attention** |
| Decoder | 2 Layers, 날짜 간 Attention | **2 Layers, 시간 간 Attention** |
| Cross-Attention K/V | 날짜별 Target Context `[B,T,128]` | **시간별 Target Context `[B,T,64]`** |
| 최종 이상 탐지 Head | `Linear(128,24)` | **`Linear(64,1)`** |
| 이상 Logit 출력 | `[B,T,24]` | **`[B,T]`** (또는 `[B,T,1]` 후 squeeze) |

- **`T`의 단위는 모두 `hour`**이다. `T=168`은 168일이 아니라 **연속 168시간**이다.
- `H=4`는 **Multi-Head Attention Head 수**를 뜻하며, Encoder 출력 텐서의 기호 `H_enc`와 구별한다.
- Encoder/Decoder 각각의 가중치는 동일 시간들에서 공유되며, **168개의 별도 Encoder를 만들지 않는다**.

## 2. 입력: Day Token → Hour Token

건물 `b`의 타임스탬프 `t`에서 (기존 전처리를 거친) 전력 사용량 스칼라를 `x_{b,t}`라 한다.

\[
 x_{b,t}\in\mathbb R,\qquad
 e_{b,t}=W_{\mathrm{hour}}x_{b,t}+b_{\mathrm{hour}}\in\mathbb R^{64}.
\]

- **변경:** 기존 `Linear(24,128)`/Day Vector 생성 경로를 기본 모델에서 `Linear(1,64)`/시간별 스칼라 입력으로 바꾼다.
- **유지:** 기존 `meter_reading` 전처리·정규화 정책(실제 구현 설정 기준), 결측 `NaN→0`(Zero-fill), 결측 시간 **Loss 포함**, 별도 보관하는 `observation_mask`는 모델 입력에 넣지 않는 정책.
- **정렬 규칙:** 대상/Peer의 시간축은 반드시 **실제 동일 타임스탬프**로 정렬한다. 같은 `hour-of-day`이지만 날짜가 다른 관측값을 같은 Encoder 슬롯에 묶지 않는다.

### Positional Encoding (PE)

기존 v3에서 **Encoder 내부 시간축 PE 없음 → Target Context 추출 후 PE 적용** 규칙을 그대로 계승한다.

- Decoder 입력: `e_target,t + PE_hour(t)`.
- Zone/Type Encoder 입력: `e_building,t + E_role(target/peer)`; **Encoder 내부에 시간 PE를 추가하지 않음**.
- Encoder에서 Target Context 추출 후: `c_zone,t + PE_hour(t)` 및 `c_type,t + PE_hour(t)`.
- `PE_hour(t)`는 `D=64`인 **기존 Sinusoidal PE 수식**을 시간 인덱스 `t=1,…,T`에 적용한다. 동일 타임스탬프는 Zone/Type/Decoder에서 같은 인덱스를 사용한다.
- **건물 순서 PE / 건물 ID Embedding은 넣지 않는다.** Target/Peer Role Embedding은 유지한다. `Same-Building Attention Bias`는 v3처럼 사용하지 않는다.
- 실제 시각의 `hour-of-day`, `day-of-week` Embedding은 **이번 v5 확정 구현에 포함하지 않는다**. 필요하면 별도 실험으로 추가한다.

## 3. Zone/Type Encoder: 시간별 건물 간 Self-Attention

`g∈{Zone,Type}`에서 동일 그룹 건물 수를 `G_g`라 한다(대상 1개 + 유효 Peer `G_g−1`개).

각 시간 `t`에 대해:

\[
X_{g,t}=[e_{target,t},e_{peer_1,t},\ldots,e_{peer_{G_g-1},t}]
\in\mathbb R^{G_g\times64},
\]

\[
H_{g,t}=\operatorname{Encoder}_g\left(X_{g,t}+E_{\mathrm{role}}\right)
\in\mathbb R^{G_g\times64}.
\]

- **동일한 시간 `t`의 건물들끼리만** Self-Attention. 다른 시간에 있는 건물 Token은 이 Encoder Self-Attention에 들어가지 않는다.
- Zone=`site_id`, Type=`primary_use`; 두 Encoder 가중치는 **분리**한다.
- 각 그룹의 **첫 번째 건물은 항상 대상 건물**이며, Target/Peer Role Embedding 유지.
- 날짜별 Encoder(v3)와 마찬가지로 **최종 Encoder 출력의 대상 Token만 선택**한다.

\[
 c_{g,t}=H_{g,t}[0]\in\mathbb R^{64},\qquad
 C_g=[c_{g,1}+PE_{hour}(1),\ldots,c_{g,T}+PE_{hour}(T)]
 \in\mathbb R^{T\times64}.
\]

**배치 텐서 흐름:**

```text
zone_hourly [B,T,Z,1] -> Hour Projection + Role Embedding
                        -> [B*T,Z,64] -> Zone Encoder ×2
                        -> Target slot 0 추출 -> [B,T,64] + Hour PE
                        -> C_zone [B,T,64]

type_hourly [B,T,U,1] -> Hour Projection + Role Embedding
                        -> [B*T,U,64] -> Type Encoder ×2
                        -> Target slot 0 추출 -> [B,T,64] + Hour PE
                        -> C_type [B,T,64]
```

- 패딩된 Peer 건물은 Encoder Self-Attention의 **Key Padding Mask**에서 제외한다. Peer 0개여도 대상 건물 하나로 동작해야 한다.
- **값 결측(Zero-fill)과 건물 Padding은 다르다.** 결측 관측값 때문에 유효 건물 Token을 삭제해서는 안 된다.
- 같은 시간의 Peer 건물 순서만 바꾸고 Mask도 함께 바꾸면 Target Context와 최종 예측이 동일해야 한다(수치 오차 허용).

## 4. Temporal Decoder 및 Cross-Attention

Decoder는 대상 건물의 연속된 `T`개 **Hour Token**을 입력받는다.

\[
 H_D^{(0)}=[e_{target,1}+PE_{hour}(1),\ldots,e_{target,T}+PE_{hour}(T)]
 \in\mathbb R^{T\times64}.
\]

**Decoder 2 Layers**, 각 Layer의 연산 순서는 기존 v3를 그대로 유지한다.

```text
Causal Masked Self-Attention -> FFN
 -> Zone Cross-Attention -> FFN
 -> Type Cross-Attention -> FFN
```

각 Attention/FFN의 Pre-LN, Residual, Dropout 규칙도 기존 구현을 유지한다.

각 Encoder Context에 대한 **Cross-Attention Head 하나**는:

\[
Q\in\mathbb R^{T\times16},\quad
K_g,V_g\in\mathbb R^{T\times16},\quad
QK_g^\top\in\mathbb R^{T\times T},\quad g\in\{Zone,Type\}.
\]

- Self-Attention: 기존처럼 **Causal Mask 유지**(Query `t`는 Decoder 내부에서 미래 `>t`를 참조하지 않음).
- **Cross-Attention: 시간별 Causal Mask 없음**(Window 전체의 Context 참조 가능). 기존 오프라인 사후 이상 탐지 정책을 유지한다.
- 위 두 사항을 함께 적용하면 **전체 모델은 온라인 Causal 모델이 아님**에 주의한다. 이전/이후 시간 Context가 Cross-Attention을 통해 제공될 수 있다.
- Cross-Attention K/V는 v3 기본안인 **Target Context `T`개**이며, v4-B의 모든 건물 `T×G`개 전달 방식은 기본 v5에 반영하지 않는다.

## 5. 출력: Token당 24개 → Token당 1개 이상 Logit

\[
H_D^{out}\in\mathbb R^{B\times T\times64},\qquad
 A=\operatorname{Linear}(64,1)(H_D^{out})\in\mathbb R^{B\times T\times1}.
\]

\[
\boxed{\texttt{anomaly_logits}=\operatorname{squeeze}(A,-1)\in\mathbb R^{B\times T}}
\]

- 시간별 정답 Shape: `[B,T]` (`0/1`). Loss는 **기존 BCEWithLogits 계열**을 유지한다.
- Logit에 `sigmoid`를 적용한 결과가 **각 시간의 이상 확률 점수**다.
- 이상이 존재하는 시간도 학습에서 제외하지 않으며, 결측 `meter_reading` 시간도 기존 정책대로 **Loss에 포함**한다.
- 검증/테스트 출력은 각 원본 시간별 행과 대응시키며 **`row_id`당 이상 점수 정확히 1개**를 생성한다.

## 6. Window 샘플링 및 평가 단위

- `T_max=168`은 **hour 단위**로 정의한다.
- **기존 길이 샘플링 방침을 단위 변환하여 유지:** 학습 샘플마다 `T ∼ Uniform{1,2,…,168}`에서 길이를 선택하고, 연속된 `T`시간을 취한다. Encoder와 Decoder는 **같은 시간 구간**을 사용한다.
- 기본 검증·추론은 **연속 168시간 고정 Window**로 수행한다(마지막 남은 시간 구간은 별도 처리). 기존과 같이 겹치지 않는 Window를 쓰고 각 시간의 점수를 한 번 출력한다.
- Building 단위 5-Fold 분할, 검증/테스트의 **Train Building만 Peer로 사용**하는 규칙을 유지한다.
- 핵심 연구 평가지표는 **시간 단위 F1**이며, Precision/Recall, PR-AUC, ROC-AUC도 병행 기록한다. 비교 시 Threshold 및 Best Checkpoint 선택 정책을 기존 실험과 동일하게 명시한다.
- **설계상 한계:** 168개의 연속 시점 안에서는 `t`와 정확히 `t−168` 또는 `t+168`을 동시에 포함할 수 없다. 즉, 주간 패턴 하나를 볼 수는 있지만 **동일 시각의 정확한 일주일 전후 비교**는 이 Window만으로 불가능하다. 이는 `T=336` 이상 또는 별도 지연 참조를 넣는 **후속 실험**으로 분리한다.

## 7. 주요 구현 Config 변경안

아래 키/값은 **의미적 요구사항을 나타내는 예시**이며, 실제 코드의 Config Schema/키 이름에 맞춰 반영한다.

```yaml
model:
  token_unit: hour
  patch_size_hours: 1
  d_model: 64
  n_heads: 4
  head_dim: 16
  d_ff: 256
  zone_encoder_layers: 2
  type_encoder_layers: 2
  decoder_layers: 2
  dropout: 0.1
  hour_projection_in_features: 1
  anomaly_head_out_features: 1

  encoder_attention_scope: per_hour_buildings
  encoder_context_pooling: target_token
  encoder_internal_hour_pe: false
  encoder_context_hour_pe: sinusoidal
  decoder_positional_encoding: sinusoidal_hour
  target_role_embedding: true
  same_building_attention_bias: false
  decoder_causal_mask: true
  cross_attention_temporal_mask: none
  cross_attention_fusion: sequential
  cross_attention_order: [zone, type]
  ffn_after_each_attention: true

train:
  window_length_unit: hours
  window_length_sampling: uniform_integer
  min_window_hours: 1
  max_window_hours: 168
  contiguous_window: true
  objective: hourly_binary_cross_entropy
  loss_on: all_valid_hours

eval:
  window_length_unit: hours
  window_hours: 168
  stride_hours: null   # non-overlap
  kaggle_submission: row_id_level
```

**기존 Config 유지 항목:** `dropout=0.1`, Optimizer 및 Scheduler, 학습 Epoch/Seed, 건물별 Split·Peer 풀, 결측 처리, End-to-End 지도학습, 모델 선정/지표 정책(별도 실험 설계에서 정한 최신 값). `pos_weight`를 새로 도입하지 않는다.

**구형 이름/로직 대체 대상:** `target_daily`, `zone_daily`, `type_daily`, `day_patches`, `patch_size_hours:24`, `max_window_days`, `window_days`, `stride_days`, `Linear(D,24)`, 출력 `[B,T,24]`, `day_pe`, 날짜별 인덱싱 등 **시간 단위로 변경하지 않으면 의미가 충돌하는 부분**. 읽기 전용 이름만 바꾸는 게 아니라 데이터 샘플링·라벨 정렬·추론 병합까지 함께 갱신한다.

## 8. 텐서 Shape 및 필수 검증 (Acceptance Criteria)

| 텐서 | v5 Shape | 의미 |
|---|---|---|
| `target_hourly` | `[B,T,1]` | 대상 건물의 연속 T시간 사용량 |
| `zone_hourly` | `[B,T,Z,1]` | 동일 시각 Zone 건물들 (대상 첫 슬롯) |
| `type_hourly` | `[B,T,U,1]` | 동일 시각 Type 건물들 (대상 첫 슬롯) |
| `zone_encoder_input` | `[B*T,Z,64]` | 각 시간에 해당하는 Zone 건물들만 Attention |
| `type_encoder_input` | `[B*T,U,64]` | 각 시간에 해당하는 Type 건물들만 Attention |
| `zone_context` / `type_context` | `[B,T,64]` | 시간별 Target Context에 Hour PE 추가 |
| `decoder_input` / `decoder_output` | `[B,T,64]` | 대상 건물 시간별 Token |
| `zone_cross_scores` / `type_cross_scores` | `[B,4,T,T]` | Head별 Q/K 관계 |
| `anomaly_target` | `[B,T]` | 시간별 0/1 정답 |
| `anomaly_logits` | `[B,T]` | 시간별 이상 Logit |

- [ ] `T=168`, `D=64`, `H=4`에서 Head Dimension이 정확히 16이다.
- [ ] 같은 시각의 건물 Token만 Encoder Self-Attention에 들어간다. Zone/Type에 **시간축 전체 Flatten Self-Attention이 없어야** 한다.
- [ ] `[B,T,G,1] → [B*T,G,64] → 대상 Token 추출 [B,T,64]`을 확인한다.
- [ ] 각 Encoder의 `[B,T,64]` Context를 Decoder Cross-Attention K/V로 전달하고, Score가 `[B,4,T,T]`이다.
- [ ] Decoder 출력은 `[B,T]`로서 **시간당 Logit 하나**이며, `Linear(64,24)` 및 `24개 Logit/Token` 로직이 남아 있지 않다.
- [ ] `target_hourly[:,t]`와 `zone_hourly[:,t,0]`, `type_hourly[:,t,0]`의 값과 시각이 일치한다.
- [ ] Peer 순서 변경 불변성, 0개 Peer 대응, Padding Mask와 Zero-fill 구분을 확인한다.
- [ ] 마지막 짧은 Window에서도 유효 시간들만 Loss/추론 대상으로 삼고, `row_id` 중복·누락 없이 각 시간 점수 1개를 만든다.
- [ ] Validation/Test Peer는 Train Split 건물에서만 구성하고 이상 Label이 모델 입력에 유입되지 않는다.
- [ ] 기존 `v3_perday`와 비교 시 Fold/Seed, 데이터 범위, Threshold 선택 규칙을 일치시켜 **Hour Token 전환 효과**를 측정한다. 단, `D=128→64`, `T=64일→168시간`도 동시에 바뀌므로 결과 차이를 Token 단위만의 효과로 단정하지 않는다.

---

**이번 변경에서 유지하는 것:** Zone=`site_id` / Type=`primary_use` 구분, Target Role Embedding, Encoder별 시간 단위 건물 Attention과 Target Context 추출, Decoder의 Causal Self-Attention → FFN → Zone Cross-Attention → FFN → Type Cross-Attention → FFN, Pre-LN/Residual, Cross-Attention 시간 Mask 없음, Building 단위 검증, Train Building Peer 정책, 결측 Zero-fill 및 Loss 포함, End-to-End 지도학습.
