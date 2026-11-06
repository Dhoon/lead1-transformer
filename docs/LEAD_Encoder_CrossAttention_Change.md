# LEAD Transformer — Encoder 및 Cross-Attention 구조 변경사항

> **적용 대상:** 기존 `SPEC(1).md` v1.0의 Encoder·Cross-Attention 관련 항목만 수정하기 위한 별도 변경 명세.  
> **범위:** 날짜별 건물 Attention → 대상 건물 Context 추출 → 날짜 간 Cross-Attention.  
> **주의:** 이 문서는 전체 SPEC이 아니며, 여기에 명시하지 않은 기존 설정은 그대로 유지한다.

## 1. 변경 이유

기존 Zone/Type Encoder는 `T`일 × 그룹 건물 `G`개의 Day Token을 모두 펼쳐 **한 번의 통합 Self-Attention**을 수행했다.

- 기존 입력: `[B, T×G, D]`
- 기존 Encoder Self-Attention 행렬: `[B, H, T×G, T×G]`
- 기존 Decoder Cross-Attention: 날짜별 Query `T`개가 Encoder의 `T×G`개 Key/Value 전체를 참조

이 방식은 Encoder가 건물 간 관계와 날짜 간 관계를 동시에 처리하며, 그룹 건물 수와 날짜 수가 커지면 Attention 연산 및 메모리 비용이 크게 증가한다.

**변경 목표:** Encoder는 **동일 날짜의 건물 간 관계**에 집중하고, Decoder의 Cross-Attention은 **날짜별로 요약된 건물 Context 사이의 관계**를 학습하도록 역할을 분리한다.

## 2. Zone/Type Encoder — 날짜별 독립 Self-Attention으로 변경

각 날짜 `d`에서 같은 Zone 또는 같은 Type에 속하는 `G`개 건물의 Day Token만 인코딩한다. 여기서 **첫 Token은 항상 Decoder 대상 건물**이며, 나머지는 Peer 건물이다.

\[
X_{g,d}=[e_{b,d},e_{p_1,d},\ldots,e_{p_{G-1},d}]
\in\mathbb R^{G\times D},\qquad g\in\{\mathrm{Zone},\mathrm{Type}\}.
\]

\[
H_{g,d}=\operatorname{Encoder}_{g}(X_{g,d}+E_{\mathrm{role}})
\in\mathbb R^{G\times D}.
\]

- **동일 날짜의 건물 `G`개끼리만 Self-Attention**한다. 서로 다른 날짜 Token은 같은 Encoder Attention 행렬에 넣지 않는다.
- 날짜별로 별도의 Encoder 모델을 만드는 것이 아니라, **같은 Zone Encoder 가중치 / 같은 Type Encoder 가중치를 모든 날짜에 반복 적용**한다.
- `E_role(target/peer)`는 **유지**한다. 대상 건물을 첫 번째 위치로 정렬하되, 건물 순서용 PE는 적용하지 않는다.
- **Encoder 내부에는 날짜 PE를 넣지 않는다.** 날짜별 결과를 추출한 후 Cross-Attention의 Key/Value로 전달하기 전에 날짜 PE를 추가한다.
- 기존의 **Same-Building Attention Bias `β·1[b_i=b_j]`는 제거**한다. 이제 Encoder에서 날짜 사이 Self-Attention을 하지 않으므로 그 Bias가 불필요하다.
- Peer 패딩은 날짜별 건물 Self-Attention에서 Key Padding Mask로 처리한다. `meter_reading` 결측으로 0 대체한 시간은 Attention Mask 대상이 아니다.

예시: Type A에 대상 건물을 포함한 80개 건물이 있고 `T=32`이면, `[80,D]` 단위 Self-Attention을 **32번** 수행한다(병렬 배치 처리 가능). `[2560,D]`를 한 번에 Self-Attention하지 않는다.

## 3. Encoder 출력 — 날짜마다 대상 건물 Context 1개 추출

각 날짜의 Encoder 출력 중 **대상 건물(첫 Token)의 Contextual Representation만** Decoder의 Cross-Attention Memory로 사용한다.

\[
c_{g,d}=H_{g,d}[0]\in\mathbb R^D.
\]

그런 다음 날짜 순서를 알 수 있도록 해당 날짜의 PE를 더한다.

\[
C_g=[c_{g,1}+PE(1),\ldots,c_{g,T}+PE(T)]
\in\mathbb R^{T\times D}.
\]

- Zone Context: `C_zone ∈ [B,T,D]`
- Type Context: `C_type ∈ [B,T,D]`
- 두 Encoder는 가중치를 공유하지 않는다(기존 원칙 유지).
- 같은 날짜의 Peer 정보를 반영한 **대상 건물 표현 1개**만 다음 단계로 전달한다. 다른 Peer의 출력은 Cross-Attention Memory로 직접 전달하지 않는다.

## 4. Decoder Cross-Attention — 건물×날짜가 아니라 날짜×날짜 비교

Decoder는 기존처럼 대상 건물의 `T`개 Day Token을 입력받아 Causal Masked Self-Attention을 수행한다. **Zone → Type 순차 Cross-Attention, 각 Attention 뒤 FFN, 2층 Decoder 기본 설정은 유지**한다.

Zone과 Type 각각의 Cross-Attention에서:

\[
Q=H_DW_Q\in\mathbb R^{T\times d_k},\qquad
K=C_gW_K\in\mathbb R^{T\times d_k},\qquad
V=C_gW_V\in\mathbb R^{T\times d_v}.
\]

\[
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
\qquad QK^\top\in\mathbb R^{T\times T}.
\]

예를 들어 `T=3`이면 Query `1·2·3일차` 각각이 Key `1·2·3일차 Context`를 비교한다. **Multi-Head Attention을 사용하며, 기본 오프라인 이상 탐지에서는 Cross-Attention에 날짜 Causal Mask를 추가하지 않는다.**

```text
날짜별 Zone Encoder → 대상 Context 1개/일 → C_zone [T,D] ─┐
                                                         ├─ Decoder:
날짜별 Type Encoder → 대상 Context 1개/일 → C_type [T,D] ─┘  Causal Self-Attn → FFN
                                                            → Zone Cross-Attn(T×T) → FFN
                                                            → Type Cross-Attn(T×T) → FFN
                                                            → Linear(D,24) → [T,24]
```

## 5. 텐서 Shape 및 Attention 비용 변경

| 항목 | 기존 | **변경 후** |
|---|---|---|
| Encoder 원본 입력 | `[B,T,G,24]` | **동일** |
| Encoder Self-Attention 입력 | `[B,T×G,D]` | **`[B×T,G,D]`** (날짜별 공유 가중치) |
| Encoder Self-Attention Score (Head당) | `(T×G)²` | **`T×G²`** (T일 합계) |
| Encoder에서 Decoder로 전달하는 Memory | `[B,T×G,D]` | **`[B,T,D]`** |
| Decoder Cross-Attention Score (Head당) | `T×(T×G)` | **`T×T`** |
| Decoder 출력 / Hourly Head | `[B,T,D]` → `[B,T,24]` | **동일** |

`T=32`, `G=80`이면 Encoder Self-Attention Score는 Head당 `6,553,600 → 204,800`개로 **32배 감소**한다. 이는 Attention Score 규모 비교이며, 전체 모델 실행 시간이 32배 빨라진다는 의미는 아니다.

## 6. 기존 SPEC에서 바꿀 설정과 설명

다음 항목만 교체한다. 나머지는 유지한다.

| 기존 항목 | 변경 지시 |
|---|---|
| 2장, 4.2, 4.3: `T×G` Token의 통합 Self-Attention | **날짜별 `G` Token Self-Attention**으로 교체 |
| 3.1: Encoder 모든 Token에 날짜 PE 추가 | **Encoder 내부 PE 제거, 추출된 날짜별 대상 Context에 PE 추가** |
| 4.2.1: Same-Building Attention Bias | **기본 모델에서 삭제**. 해당 Bias Ablation도 기본 실험 계획에서 제거 |
| 3.2/3.3: day-major Flatten 및 `zone_slot`·`type_slot` 기반 동일 건물 Bias | **날짜별 그룹 `[B×T,G,D]` 및 날짜별 Padding Mask**로 교체. Role을 위한 Target 인덱스 0만 유지 |
| 4.5: Cross-Attention K/V에 Encoder 전체 Token 사용 | **날짜별 대상 Context `[B,T,D]`만 K/V로 사용** |
| 8장 Config, 10장 Acceptance Test | 아래 설정/검증 기준으로 갱신 |

```yaml
model:
  encoder_attention_scope: per_day_buildings
  encoder_joint_days: false
  encoder_internal_day_pe: false
  encoder_context_day_pe: sinusoidal
  encoder_context_pooling: target_token
  target_role_embedding: true
  same_building_attention_bias: false
  cross_attention_fusion: sequential
  cross_attention_order: [zone, type]
  cross_attention_temporal_mask: none
```

### 구현 검증 기준

- [ ] 같은 날짜의 Peer 건물 순서를 바꿔도 해당 날짜의 대상 Context와 최종 Prediction이 동일하다(패딩 Mask도 함께 재배열).
- [ ] 날짜 `d`의 Encoder Self-Attention은 **날짜 `d`의 건물 Token에만** 접근한다.
- [ ] `[B,T,G,24]` → `[B×T,G,D]` → 대상 Token 추출 `[B,T,D]` → 날짜 PE → Cross-Attention K/V의 Shape를 만족한다.
- [ ] 두 Decoder Cross-Attention의 Head별 Score Shape는 모두 `[B,H,T,T]`이다.
- [ ] `T`를 1~64로 바꿔도 `Linear(D,24)` 출력은 `[B,T,24]`이며, 기존 시간별 정답과 일치한다.
- [ ] 기존 Same-Building Bias 및 날짜 전체 Flatten 코드가 실행 경로에 남아 있지 않다.

**변경하지 않는 사항:** Zone=`site_id`, Type=`primary_use` 그룹 구분, 대상 건물 포함/첫 Token 규약, Train/Test Peer 정책, Decoder Causal Self-Attention, Zone→Type 순차 순서, 각 Attention 뒤 FFN 및 Add/Norm, 1~64일 랜덤 학습, End-to-End 시간별 이상 지도학습, 결측 Zero-fill 및 결측 시간 Loss 포함, `[B,T,24]` 출력, F1 우선 모델 평가 정책.
