# LEAD Transformer — Peer-only Encoder + All-Peer K/V + Parallel Fusion 변경 명세

> **문서 유형:** 기존 전체 `SPEC.md`를 대체하지 않는 **부분 변경 명세**  
> **기준 구현:** `v3_pm` (Day Token, 날짜별 Encoder, Peer Missing Mask) + `v3_pm_parallel` (Site/Type 병렬 Cross-Attention)  
> **실험안 이름(제안):** `v3_pm_peeronly_allpeer_parallel`  
> **이번 변경:** Site/Type Encoder에서 **대상 건물(Target)을 제외**하고, 날짜별 **모든 유효 Peer Hidden State를 개별 K/V Token으로 유지**한다. 기존 병렬 Cross-Attention의 출력을 **Availability-aware Concat + Fusion MLP**로 결합한다.  
> **구현 상태 (2026-11-13):** 구현 완료. 설정 `configs/v3_pm_peeronly_allpeer_parallel.yaml`, 테스트 `tests/test_peeronly_allpeer.py`. 실제 Config 키는 6장 "설정"을 따른다.

## 1. 설계 목적

기존 모델의 서로 다른 세 관점을 명확히 분리한다.

| 모듈 | 입력/담당 | 학습 목적 |
|---|---|---|
| Decoder Self-Attention | **Target 건물만** | 대상 건물 자신의 날짜 간 전력 소비 관계 |
| Site(Zone) Encoder | 같은 `site_id`의 **Target 제외 Peer만** | 같은 Site 건물 사이의 소비 패턴 관계 |
| Type Encoder | 같은 `primary_use`의 **Target 제외 Peer만** | 같은 Type 건물 사이의 소비 패턴 관계 |
| Site/Type Cross-Attention | Target Query ↔ 각각의 Peer K/V | 내 건물의 변화가 동일 Site/Type 집단에도 나타나는지 비교하기 위한 표현 학습 |
| Fusion | Self + Site + Type | 세 관점을 바탕으로 시간별 이상 여부 판단 |

예: Target 건물의 소비량이 갑자기 증가해도 동일 Site/Type의 여러 건물에서 공통적으로 증가했다면, 이상으로 판정하지 않을 근거가 될 수 있다. **단, Cross-Attention Weight 그 자체가 소비량 차이 또는 이상 판정은 아니다.** 실제 비교 규칙은 최종 F1 Loss로 학습된다.

## 2. 변경 전 / 변경 후

### 변경 전 — `v3_pm_parallel`

```text
Day d, Site Encoder 입력: [Target, Peer 1, ..., Peer G-1]
                      └─ 날짜별 Self-Attention ─> Target 출력 한 개만 추출
                         Site K/V 길이 = T

Day d, Type Encoder 입력: [Target, Peer 1, ..., Peer U-1]
                      └─ 날짜별 Self-Attention ─> Target 출력 한 개만 추출
                         Type K/V 길이 = T

Decoder Self → Parallel(Site, Type) → Fusion
```

### 변경 후 — Peer-only + All-Peer

```text
Target 건물의 T개 Day Token ──> Decoder Causal Self-Attention ──> H_self
                                                               │
                              ┌────────────────────────────────┴──────────────┐
                              │                                               │
Site Peer만 [Peer 1..G_S] ─> 날짜별 Site Encoder ─> 전 Peer Hidden State  Type Peer만 [Peer 1..G_U]
                              │                                               │
                       Site K/V: T×G_S                            날짜별 Type Encoder
                              │                                               │
                    Site Cross-Attention                             전 Peer Hidden State
                      Q = LN(H_self)                                Type K/V: T×G_U
                              │                                               │
                             O_S                               Type Cross-Attention
                              │                                  Q = LN(H_self)
                              │                                               │
                              │                                              O_U
                              └──────────────────────┬────────────────────────┘
                                                     │
                                  Concat[Z; O_S; O_U; m_S; m_U]
                                                     │
                                              Fusion MLP → D
                                                     │
                                     H_self + Dropout(Fusion) → FFN
                                                     │
                                         Linear(D,24): 시간별 Logit
```

`G_S`, `G_U`는 **대상 건물을 제외한 Peer 건물 수**이다. 서로 달라도 된다.

**실제 그룹 크기 (train 200개 건물 기준):**

| 그룹 | 그룹 크기(대상 포함) | Peer 0개가 되는 경우 |
|---|---|---|
| Site | 1~52 (site 9: 52, site 14: 47, site 7·12·15: 1) | 건물이 1개뿐인 site 7, 12, 15 |
| Type | 1~80 (Education 80, Office 62, 1개뿐인 Type 3종) | Food sales and service, Other, Religious worship |

- Peer Pool은 기존대로 그 fold의 학습 건물(160개)이다. 그래서 fold마다 Peer 수가 위 값보다 작다. fold 0 로그 기준 최대 그룹 크기는 train 64, eval 65이다.
- 즉 Peer 0개인 Branch(5장)는 실제 데이터에서 일어난다.

## 3. Peer-only 날짜별 Encoder

### 3.1 입력

- 각 Day Token은 기존대로 `24시간 meter_reading → Linear(24,D)`로 만든다.
- Target 건물은 **Site/Type Encoder의 입력 Token, Self-Attention Q/K/V, Encoder Memory에서 전부 제외**한다.
- 단, Target의 `site_id`, `primary_use`, `building_id`는 Peer **선정 및 자기 자신 제외**를 위해 사용한다.
- Site와 Type은 동일 Encoder가 아니며 **독립된 학습 가중치**를 유지한다.

날짜 `d`, 그룹 `g ∈ {S,U}`에 대해:

\[
X_{g,d}^{\mathrm{peer}}=
[e_{p_1,d},\ldots,e_{p_{G_g},d}]
\in\mathbb R^{G_g\times D},
\qquad
H_{g,d}^{\mathrm{peer}}=\operatorname{Encoder}_g(X_{g,d}^{\mathrm{peer}}).
\]

- 날짜별 Encoder 내부 Attention은 기존 `v3_pm`과 같이 **해당 날짜 내 Peer ↔ Peer**에만 수행한다. 날짜 간 Encoder Self-Attention은 없다.
- `G_g=1`이면 Attention Softmax는 유효 Key 하나에 대해 1이지만 Q/K/V Projection, Residual, FFN을 통한 표현 변환은 가능하다.
- Target/Peer **Role Embedding은 Peer-only Encoder에서는 사용하지 않는다.** 모든 Encoder 입력이 Peer이므로 Target Role을 표시할 대상이 없다. 기존 모델의 구현은 Baseline 재현을 위해 보존한다.
- 기존 Same-Building Attention Bias는 계속 사용하지 않는다.
- Encoder의 **최종 LayerNorm을 모든 유효 Peer Hidden State에 먼저 적용**한 뒤, 모든 Token을 보존한다(Mean Pooling 없음).

### 3.2 Peer 결측 마스킹 — 기존 `v3_pm` 정책 유지

- `data.peer_missing_mask: true`.
- Peer의 해당 날짜 **24시간 전체가 결측**이면 그 Day Token을 해당 날짜 Encoder Attention에서 제외한다.
- 하루 중 **일부 시간만 결측**이면 기존과 같이 `NaN → 0`을 적용하되 해당 Peer Day Token은 유지한다.
- 그룹 크기 Padding도 별도로 Masking한다.
- 대상 건물의 결측 처리는 **Decoder의 기존 정책을 그대로 유지**한다. Target은 Encoder에 들어오지 않는다.
- Encoder Attention에서 제외한 결측 Peer Token은 **Decoder Cross-Attention K/V에서도 반드시 제외**한다. 결측 상태의 Hidden State를 새롭게 유효 Token으로 되살리지 않는다.

**구현상 필수:** 특정 `(배치 샘플, 날짜, 그룹)`에 유효 Peer가 하나도 없으면 해당 날짜 Encoder를 **전부 Mask된 행에 대한 Softmax로 호출하지 말 것**(NaN 위험). 유효 날짜만 계산하거나, 안전한 Dummy 입력으로 연산한 뒤 그 날짜의 K/V를 전부 무효 마스킹한다.

**구현:** 두 번째 방법을 쓴다. 유효 Peer가 0개인 `(샘플, 날짜)` 행은 Encoder 안에서만 Mask를 풀어 계산한다. 그 행의 출력은 Cross-Attention Memory Mask에서 모두 False라 Decoder에 전달되지 않는다. 배치 전체에 Peer 슬롯이 하나도 없으면(모든 샘플이 대상 단독 그룹) 0 벡터 Memory 1개를 Mask False로 넘긴다 (`PerDayEncoder._all_peers`).

## 4. All-Peer Cross-Attention Memory

### 4.1 Memory 구성

날짜별 Encoder 출력 `H_g`를 `[B,T,G_g,D]` 형태로 복구한다. **이 절의 Shape에서 `G_g`는 배치 안에서 Padding을 포함한 Peer 슬롯 수**(배치 최대 그룹 크기 − 1)다. 샘플별 실제 Peer 수와 결측 여부는 별도 Mask로 관리한다.

각 날짜의 **모든 유효 Peer Token에 동일한 Sinusoidal Day PE**를 추가한다.

\[
\tilde h_{g,d,j}=\operatorname{LN}_{\mathrm{final}}(h_{g,d,j})+PE(d).
\]

날짜 우선(`day-major`)으로 Flatten한다.

\[
C_g=[\tilde h_{g,1,1},\ldots,\tilde h_{g,1,G_g},
\tilde h_{g,2,1},\ldots,\tilde h_{g,T,G_g}]
\in\mathbb R^{B\times(TG_g)\times D}.
\]

- **Mean Pooling 없음**, Target Context 추출도 없음.
- `G_g`는 Site/Type 사이에 다를 수 있다.
- 배치 내부의 가변 Peer 수는 Padding하고 **Cross-Attention Key Padding Mask**에서 제외한다.
- 날짜 PE는 기존 Decoder와 동일한 날짜 인덱스/시간 정렬 규칙을 사용한다.
- **별도 건물 순서 PE/건물 ID Embedding을 추가하지 않는다.** Peer 슬롯 순서를 일관되게 재배치해도 예측은 불변이어야 한다(부동소수점 오차 허용).

### 4.2 병렬 Cross-Attention

`D=128`, Head `4`, `head_dim=32`는 기존 v3_pm 설정을 유지한다.

\[
H_{\mathrm{self}}\in\mathbb R^{B\times T\times D},\qquad
Z=\operatorname{LN}(H_{\mathrm{self}}).
\]

\[
\begin{aligned}
O_S &= \operatorname{CrossAttn}_S(Q=Z,\ K/V=C_S),\\
O_U &= \operatorname{CrossAttn}_U(Q=Z,\ K/V=C_U).
\end{aligned}
\]

- 두 Branch는 **동일한 `Z`를 Query 입력**으로 사용하되, Q/K/V 및 Output Projection은 **별도 파라미터**로 학습한다.
- Site 출력이 Type Query에 먼저 들어가지 않으며, 그 반대도 아니다.
- **기본 실험에서는 Cross-Attention 날짜 Causal Mask나 동일 날짜 Hard Mask를 추가하지 않는다.** 날짜 `d`의 Query는 각 그룹 Peer의 모든 유효 날짜 Token을 참조할 수 있다(오프라인 이상 탐지).

그룹별 Head 하나의 Score Shape는 다음과 같다.

\[
A_g\in\mathbb R^{B\times4\times T\times(TG_g)},
\qquad O_g\in\mathbb R^{B\times T\times128}.
\]

**예시:** `T=64`, Site Peer 3개, Type Peer 9개일 때:

| 구성 | Q Token | K/V Token | Head별 Score Shape |
|---|---:|---:|---|
| Site | 64 | 192 | `64 × 192` |
| Type | 64 | 576 | `64 × 576` |
| Decoder Self | 64 | 64 | `64 × 64` |

위 예시는 계산 설명용이다. 실제 Peer 수는 3장 표처럼 훨씬 크다. 예를 들어 `T=64`, Site Peer 51개, Type Peer 79개면 K/V는 Site 3,264개, Type 5,056개다.

- 기존 `train.max_tokens_per_batch: 16384`(배치 × T × 최대 그룹 크기)가 긴 T에서 배치 크기를 줄여 메모리를 제한한다.
- 같은 규모의 K/V(`T×G`)를 쓰는 `v3_allpeer`(대상 포함 전체 건물 K/V, 순차 구조)가 T4에서 fold당 약 19분에 문제없이 돌았다.

최종 Decoder Query 및 출력 Token 수는 **여전히 `T`개**다. Encoder의 시간별 출력 Head를 변경하지 않는다.

## 5. Peer 0개인 Site/Type의 처리 및 Fusion

### 5.1 Branch Availability

유효 Peer가 전체 Window 내에 **적어도 하나의 날짜에 존재하면** 해당 Branch를 사용한다.

\[
m_g^{(b)}=\mathbf1\left[\sum_{d,j}\operatorname{valid}_{g,b,d,j}>0\right],
\quad g\in\{S,U\}.
\]

- `m_S=0`이면 Site Cross-Attention **Skip**, `O_S=0`으로 고정.
- `m_U=0`이면 Type Cross-Attention **Skip**, `O_U=0`으로 고정.
- 한 Branch가 비었다고 **다른 Branch 출력의 스케일을 인위적으로 2배로 증폭하지 않는다.**
- 일부 날짜에만 유효 Peer가 없더라도 **기본 전체 날짜 Cross-Attention에서는 다른 유효 날짜의 Peer를 참조할 수 있다.** 해당 날짜 Key만 Mask한다.
- 어떤 배치 샘플에서는 Site가 비고 다른 샘플에서는 차 있을 수 있으므로, **샘플별 Skip/마스킹**을 적용한다. 모든 Key가 Mask된 샘플에 Softmax를 수행하지 않는다.
- **구현:** 배치 단위로 Attention을 한 번 호출하므로 샘플별로 실제로 건너뛰지는 않는다. `m_g=0`인 샘플은 Key 하나의 Mask만 풀어 계산한 뒤 출력을 `torch.where`로 0으로 바꾼다. 결과는 Skip과 같고, 그 샘플의 Branch 파라미터 Gradient도 0이다.
- `m_g`는 Decoder가 Memory Mask에서 직접 계산한다(`mask.any(dim=1)`). 별도 입력은 없다.

### 5.2 Availability-aware Concat + Fusion MLP

정규화된 Decoder 표현 `Z`, Site/Type Attention 출력, Branch 존재 플래그를 Concat한다.

\[
F=\operatorname{FusionMLP}([Z;\ m_SO_S;\ m_UO_U;\ m_S;\ m_U]),
\]

\[
\boxed{H_{\mathrm{fused}}=H_{\mathrm{self}}+\operatorname{Dropout}(F)}
\]

\[
H_{\mathrm{out}}=H_{\mathrm{fused}}+
\operatorname{Dropout}(\operatorname{FFN}(\operatorname{LN}(H_{\mathrm{fused}}))).
\]

`m_S`, `m_U`는 `[B,1,1]`에서 `[B,T,1]`로 Broadcast한다. Fusion 입력은 `[B,T,3D+2]`, 출력은 `[B,T,D]`다.

```python
# D=128; Fusion MLP 예시
self.context_fusion = nn.Sequential(
    nn.Linear(3 * d_model + 2, d_model),  # 386 -> 128
    nn.GELU(),
    nn.Linear(d_model, d_model),         # 128 -> 128
)

# Z: [B,T,D] = LN(H_self), O_S/O_U: [B,T,D]
# available_site / available_type: [B] boolean
ms = available_site[:, None, None].to(Z.dtype).expand(-1, Z.size(1), 1)
mu = available_type[:, None, None].to(Z.dtype).expand(-1, Z.size(1), 1)
F = self.context_fusion(torch.cat([
    Z, O_S * ms, O_U * mu, ms, mu
], dim=-1))
H_fused = H_self + self.dropout(F)
H_out = H_fused + self.dropout(self.ffn(self.ln_ffn(H_fused)))
```

- **Fusion MLP 입력은 `H_self`가 아니라 `Z=LN(H_self)`**이다. Residual에만 `H_self`를 사용한다. 이전 명세의 그림/수식 불일치를 여기서 해소한다. 기존 `v3_pm_parallel` 구현도 이미 `Z`를 넣고 있어 따로 바꿀 것은 없다.
- Site/Type Branch가 없는 샘플의 `O`는 Cross-Attention 출력 Projection Bias 때문에 비영(非零)이 되지 않도록 **Attention을 건너뛰고 출력까지 0으로 Mask**한다.
- 두 Branch 모두 없는 극단적 상황에서는 `[Z;0;0;0;0]`으로 **같은 Fusion MLP**를 수행한다. 입력 길이 때문에 별도 모델을 만들지 않는다.

## 6. 이번에 유지하는 기존 설정

| 항목 | 유지 값 |
|---|---|
| Token 단위 | 1 Day Token = 24시간 사용량 |
| Input Projection | 기존 `Linear(24,128)` 학습 방식 |
| `D`, Heads, FFN | `128`, `4`, `512` |
| Site / Type Encoder Layers | 각 `2` |
| Decoder Layers | `2` |
| Decoder Self-Attention | 기존 Causal Masked Self-Attention 유지 |
| Encoder Self-Attention | 날짜별 건물 간, Peer-only로 입력만 변경 |
| PE | 기존 Shared Sinusoidal Day PE; Encoder 내부 날짜 Attention은 없음 |
| Missing | Zero-fill / Peer 전체 결측 Day Mask 유지 / 기존 Loss 포함 정책 |
| Train/Validation/Test 분할 | 기존 건물 단위 5-Fold 및 훈련 건물 Peer Pool 규칙 유지 |
| 학습 Window | `T` 최대 64일, 기존 샘플링 규칙 유지 |
| Validation | `T=64`일 고정 Window |
| Output | `Linear(128,24)` → `[B,T,24]` 이상 Logits |
| 우선 지표 | `F1@0.5`; PR-AUC, ROC-AUC, 최적 Threshold F1도 보고 |

**주의:** 검증/테스트 Target이 훈련 건물과 다를 때 기존 Peer Pool 정책을 유지하되, **Peer 집합에 Target 자기 자신이 들어오지 않도록 항상 ID 기준으로 제외**한다. Peer Labels는 입력/집계에 사용하지 않는다.

- **현재 코드에서 이미 지켜진다.** `group_members`가 그룹을 `[대상] + [pool 중 같은 그룹이면서 p != 대상]`으로 만든다. 그래서 대상은 열 0에만 한 번 있다. Peer-only Encoder는 이 열 0을 뺀다.
- 검증 건물은 `peer_pool: train_only`라 원래 Pool에 없다.

### 설정 — 실제 Config 키 (`configs/v3_pm_peeronly_allpeer_parallel.yaml`)

`v3_pm_parallel.yaml`에서 아래 키만 바꿨다.

```yaml
model:
  encoder_attention_scope: per_day_buildings
  encoder_input_scope: peers_only              # 새 키. 기본값 target_and_peers
  encoder_context_pooling: all_peers           # 새 값. peers_only와 반드시 함께 쓴다
  cross_attention_memory_layout: day_major_peer  # all_peers일 때 이 값만 허용
  cross_attention_fusion: parallel_concat_mlp
  fusion_include_availability: true            # 새 키. 기본값 false. parallel_concat_mlp에서만 허용
  target_role_embedding: false                 # Peer-only에서는 true를 거부한다
  decoder_layers: 2

data:
  peer_missing_mask: true
```

처음 제안한 키 중 두 개는 새로 만들지 않았다.

| 제안 키 | 처리 |
|---|---|
| `cross_attention_date_mask: none` | 기존 확정 키 `cross_attention_temporal_mask: none`이 같은 뜻이다. `same_day`(7장)는 아직 구현하지 않았다. |
| `fusion_input_normalized_self: true` | 설정이 아니라 고정 동작이다. Fusion 입력은 항상 `Z=LN(H_self)`다. |

- 새 키는 모두 기본값이 기존 동작이다. `v3_pm`, `v3_pm_parallel` Config와 Checkpoint는 그대로 쓸 수 있다.
- 파라미터 수: 1,857,432개. `v3_pm_parallel`과 같다. Role Embedding 512개가 빠지고 Fusion 입력 2칸(층당 256개 × 2층)이 늘어서 우연히 같아졌다.

## 7. 이후 Ablation: 동일 날짜 Peer만 참조 (이번 기본 실험에서는 OFF)

연구 의도가 *"그 시점에 Site/Type의 다른 건물들도 함께 증가했는가"*를 확인하는 것이라면 Cross-Attention에 **Same-day Hard Mask**를 적용하는 Ablation도 가능하다.

- **기본(A):** `cross_attention_date_mask: none`. `d`일 Query가 모든 날짜 Peer K/V를 조회한다. Score `[T,T×G_g]`.
- **후속(B):** `cross_attention_date_mask: same_day`. `d`일 Query는 **동일 `d`일의 유효 Peer만** 조회한다. 실제 연산을 날짜별로 구현하면 Score `[T,1,G_g]`에 해당한다. 다른 날짜 Key는 완전히 Mask한다.
- (B)는 해당 `d`에 유효 Peer가 0개라면 **그 Query에 대한 Branch 출력만 0으로 Mask**해야 한다. 이때 Availability Flag도 `m_g[b,d]` 형태로 날짜별로 정의한다.
- 실험 시 Same-day 방식을 기본 All-day 방식과 혼합하지 않는다.

## 8. 구현 검증 및 비교 실험

### 필수 테스트

모두 `tests/test_peeronly_allpeer.py`에 있다. 전체 테스트 106개가 통과한다.

- [x] 모든 Site/Type Encoder 입력에서 Target이 제외됐는지 검증. 대상 열 값을 바꿔도 Encoder Memory가 그대로다 (`test_target_excluded_and_memory_shape`).
- [x] Decoder 입력은 여전히 Target만 사용하고 Decoder Causal Self-Attention과 시간별 Head가 유지되는지 검증 (`test_decoder_still_uses_target_and_is_causal`).
- [x] Cross Memory Shape `[B,T*G_peer,D]`, Memory Mask = 입력 Mask의 Peer 부분 (`test_target_excluded_and_memory_shape`).
- [x] 날짜별 Peer 출력에 Final LN을 적용하고 모든 Peer Token에 동일 날짜 PE를 더하는지 검증 (`test_final_ln_and_same_day_pe_no_pooling`).
- [x] **Mean Pooling이 발생하지 않는지** 검증 (같은 테스트).
- [x] 결측 Peer Day와 Padding Peer가 출력에 영향이 없는지 검증 (`test_missing_and_padding_peers_excluded`).
- [x] Site 0개, Type 0개, 둘 다 0개, 각 Peer 1개, 일부 날짜만 Peer 0개, 배치 전체 Peer 슬롯 0개에서 `NaN/Inf` 없이 Forward/Backward (`test_edge_cases_finite_forward_backward`, `test_group_with_no_peer_slots`).
- [x] Site/Type Attention 두 Branch가 같은 `Z`를 Query Source로 사용하고 파라미터는 공유하지 않는지 확인 (`test_branches_share_query_independent_weights`).
- [x] Fusion Input의 마지막 차원이 `3D+2`, 출력이 `D`, Peer 없는 Branch의 `O`와 `m`이 0인지 확인 (`test_availability_flags_and_zeroed_branch`).
- [x] Peer 건물의 입력 순서를 재배열해도 동일 출력인지 검증 (`test_peer_order_permutation_invariant`).
- [x] 예측 출력 Shape `[B,T,24]`. Label/row_id 정렬은 Dataset·평가 코드를 바꾸지 않아 그대로다.
- [x] 기존 Config가 그대로인지 확인 (`test_existing_parallel_unchanged`, 기존 테스트 전체 통과). 기존 학습 결과를 다시 돌려 재현하지는 않았다. 새 키의 기본값이 기존 코드 경로를 그대로 타기 때문이다.

### 실험 비교

| 모델 | Encoder 입력 | Decoder K/V | Site/Type 결합 | 목적 |
|---|---|---|---|---|
| `v3_pm` | Target + Peer | 날짜별 Target Context `T` | Sequential | 원래 Baseline |
| `v3_pm_parallel` | Target + Peer | 날짜별 Target Context `T` | Parallel Concat/MLP | 현재 비교 Baseline |
| **`v3_pm_peeronly_allpeer_parallel`** | **Peer만** | **모든 Peer `T×G_peer`** | **Parallel + Availability-aware Fusion** | 이번 실험 |
| `..._same_day` (후속) | Peer만 | 동일 날짜 Peer만 | Parallel + Availability-aware Fusion | 시간 정렬 제약 확인 |

- **5-Fold**, 동일 Seed/Window/학습 조건으로 비교한다. 특히 **F1**, Precision/Recall, PR-AUC, ROC-AUC, Fold별 차이, 학습 시간 및 GPU 메모리를 기록한다.
- **해석 주의:** 이번 실험은 **Target 제외 + K/V 확장 + Availability Flag 추가**를 동시에 변경한다. 성능이 달라져도 그 원인을 하나로 단정하지 말 것. 필요하면 `Peer-only+MeanPool`, `Target+AllPeer` 등의 분리 Ablation을 수행한다.
- `Target+AllPeer`에 가까운 결과는 이미 있다. `v3_allpeer`(대상 포함 전체 건물 K/V, 순차 구조, 결측 가리기 없음)의 5-fold 평균은 ROC 0.915 / PR 0.654로 `v3_perday`와 같았다(result_1110). 다만 순차 구조이고 결측 가리기가 없어서 이번 실험과 변수 하나만 다른 비교는 아니다.
- 이 변경안 자체가 **F1 0.90 달성을 보장하지 않는다**. 단순 구조 비교가 아니라 기존 연구 목적에 맞는 Target–Peer 분리가 효과적인지 검증하는 실험으로 취급한다.

---

**한 줄 요약:** `v3_pm` Day-level 구조와 Peer 결측 가리기는 유지하되, **Target은 Decoder에만 넣고 Site/Type Encoder에는 Peer만 입력**한다. 날짜별 Encoder의 **모든 유효 Peer Hidden State를 Mean Pooling 없이 K/V로 유지**하며, Decoder의 **병렬 Site/Type Cross-Attention + Availability-aware Fusion**으로 결합한다.
