# LEAD Transformer — Target + Peer Summary 기반 2T Cross-Attention 변경안

> **문서 유형:** 기존 `SPEC.md` 및 `LEAD_Encoder_CrossAttention_Change.md`에 대한 **부분 변경/비교 실험 명세**. 전체 SPEC을 대체하지 않는다.  
> **적용 대상:** `v3_perday`의 **날짜별 Encoder 출력 요약 및 Decoder Cross-Attention K/V 구성**.  
> **목적:** 날짜별 Peer 정보를 별도 Context로 보존하면서, Cross-Attention Memory를 `T×G`개로 늘리지 않고 **`2T`개로 제한**한다.  
> **상태:** **비교 실험안**. 기존 v3 (`T`개 Target Context)는 Baseline으로 유지한다.

## 1. 기존 v3와의 차이

`T`일, 대상 건물 포함 `G`개 건물이라고 하자 (`G = 1 + Peer 수`). 날짜별 Encoder Self-Attention 방식은 **그대로 유지**한다.

```text
날짜 d 입력: [Target, Peer 1, ..., Peer G-1]  [G,D]
                        ↓
              날짜별 Encoder Self-Attention
                        ↓
         [h_target,d, h_peer1,d, ..., h_peer(G-1),d]  [G,D]
```

- **기존 v3:** 날짜마다 `h_target,d`만 추출 → `T`개 K/V Token.
- **변경안:** 날짜마다 `h_target,d` **1개 + 모든 Peer 출력의 평균 벡터 1개**를 추출 → `2T`개 K/V Token.
- **주의:** Peer 9개를 그대로 유지한다는 뜻이 아니다. Peer 9개의 Contextual Representation을 **하나로 집약(pooling)**하는 방식이다. 개별 Peer 9개를 모두 보존하려면 날짜당 10개, 전체 `10T`개 K/V Token이 필요하며, 이는 별도의 비교 구조다.

## 2. 날짜별 Target Context와 Peer Summary 생성

Zone 및 Type 각각에 대해 동일한 방법을 적용한다. `g ∈ {zone, type}`이며, 대상 건물은 날짜별 슬롯 `0`에 배치한다.

\[
H_{g,d}=\operatorname{Encoder}_{g}(X_{g,d})
   =[h_{g,d,0},h_{g,d,1},\ldots,h_{g,d,G-1}]
   \in\mathbb{R}^{G\times D}.
\]

**Target Context**는 기존과 동일하다.

\[
c^{\mathrm{target}}_{g,d}=h_{g,d,0}\in\mathbb{R}^{D}.
\]

**Peer Summary Context**는 유효한 Peer 출력만 평균한다.

\[
c^{\mathrm{peer}}_{g,d}
=\frac{\sum_{j=1}^{G-1}m_{g,d,j}\,h_{g,d,j}}
       {\sum_{j=1}^{G-1}m_{g,d,j}},
\qquad \sum_{j=1}^{G-1}m_{g,d,j}>0.
\]

- `m=1`: 실제 Peer 건물, `m=0`: Padding Token. **Padding은 평균에서 제외**한다.
- Pooling 대상은 **Encoder가 출력한 Peer Representation**이며, 원본 `meter_reading`의 평균이 아니다.
- Peer 수가 달라져도 Peer Summary는 날짜당 **항상 최대 1개의 벡터**를 만든다.
- **유효 Peer가 0개면** Peer Summary를 실제 토큰으로 쓰지 않는다. 구현상 슬롯은 유지하더라도 Cross-Attention **K/V Padding Mask**로 무효 처리한다. 값 0을 임의의 유효 Context로 취급하지 않는다.
- 기존의 Target/Peer Role Embedding은 **Encoder 입력에서 유지**하며, Encoder 내부에는 날짜 PE 및 Same-Building Bias를 넣지 않는 v3 규칙을 유지한다.

## 3. Cross-Attention Memory: 날짜당 2개 Token

Target Context와 Peer Summary 각각에 동일한 날짜 PE를 더한다.

\[
\tilde c^{\mathrm{target}}_{g,d}
=c^{\mathrm{target}}_{g,d}+PE(d),\qquad
\tilde c^{\mathrm{peer}}_{g,d}
=c^{\mathrm{peer}}_{g,d}+PE(d).
\]

날짜 우선으로 배치한다.

\[
C_g=[\tilde c^{\mathrm{target}}_{g,1},\tilde c^{\mathrm{peer}}_{g,1},
      \tilde c^{\mathrm{target}}_{g,2},\tilde c^{\mathrm{peer}}_{g,2},
      \ldots,
      \tilde c^{\mathrm{target}}_{g,T},\tilde c^{\mathrm{peer}}_{g,T}]
      \in\mathbb{R}^{2T\times D}.
\]

Decoder의 Query는 기존대로 대상 건물의 `T`개 날짜별 Representation이다.

\[
Q=H_DW_Q\in\mathbb{R}^{T\times d_k},\qquad
K=C_gW_K\in\mathbb{R}^{2T\times d_k},\qquad
V=C_gW_V\in\mathbb{R}^{2T\times d_v}.
\]

\[
\boxed{QK^\top\in\mathbb{R}^{T\times2T}}.
\]

예: `T=3`일 때 하나의 Cross-Attention Head가 아래 메모리를 비교한다.

| Decoder Query | Day 1 Target | Day 1 Peer Summary | Day 2 Target | Day 2 Peer Summary | Day 3 Target | Day 3 Peer Summary |
|---|---|---|---|---|---|---|
| Day 1 | 비교 | 비교 | 비교 | 비교 | 비교 | 비교 |
| Day 2 | 비교 | 비교 | 비교 | 비교 | 비교 | 비교 |
| Day 3 | 비교 | 비교 | 비교 | 비교 | 비교 | 비교 |

- Zone/Type **각 Cross-Attention에서 별도의** `C_zone`, `C_type` Memory를 사용한다.
- 기존 Decoder의 **Masked Self-Attention → FFN → Zone Cross-Attention → FFN → Type Cross-Attention → FFN** 순서는 변경하지 않는다.
- 현재 **오프라인 이상 탐지** 프로토콜에 따라 Cross-Attention에는 날짜 Causal Mask를 추가하지 않는다.
- 동일 날짜 내 Target Context와 Peer Summary의 구별이 부족하면, **KV Context Type Embedding**(target-summary/peer-summary 2종)을 추가하는 실험을 후속 옵션으로 둔다. 우선 구현에는 추가하지 않는다.

## 4. 텐서 Shape와 비용

| 항목 | 기존 v3 | **2T Peer Summary 변경안** | 전체 Peer 전달 방식(참고) |
|---|---|---|---|
| 날짜별 Encoder 입력 | `[B×T,G,D]` | **동일** | 동일 |
| 날짜별 Encoder 출력 | `[B×T,G,D]` | **동일** | 동일 |
| 날짜별 추출 Context | Target 1개 | **Target 1개 + Peer 평균 1개** | 모든 건물 G개 |
| Cross-Attention K/V | `[B,T,D]` | **`[B,2T,D]`** | `[B,T×G,D]` |
| Head별 Attention Score | `[B,H,T,T]` | **`[B,H,T,2T]`** | `[B,H,T,T×G]` |
| 최종 Decoder 출력 | `[B,T,D]` | **동일** | 동일 |
| Hourly Anomaly Head | `[B,T,24]` | **동일** | 동일 |

예: `T=32`, `G=10`이면 K/V Token 수는 **32 → 64**로 증가한다. 모든 Peer를 전달하는 방식의 **320개**보다 작다. **Encoder Self-Attention 계산량은 기존 v3와 동일**하고, Cross-Attention Score 수만 약 2배가 된다(전체 모델 연산량이 2배라는 뜻은 아님).

## 5. 구현 시 변경할 사항만 정리

```yaml
model:
  encoder_attention_scope: per_day_buildings  # v3 유지
  encoder_context_pooling: target_plus_peer_mean  # 기존 target_token에서 변경
  cross_attention_memory_layout: day_major_target_peer
  encoder_context_day_pe: sinusoidal           # 각 날짜의 두 Context에 동일 PE
  cross_attention_temporal_mask: none          # 오프라인 방식 유지
```

**Forward 연산 개요:**

```python
# h: [B, T, G, D] — 날짜별 Encoder 출력, slot 0 = target
# peer_valid: [B, T, G-1] — 패딩이 아닌 Peer 표시

target = h[:, :, 0, :]                          # [B,T,D]
peer_h = h[:, :, 1:, :]                         # [B,T,G-1,D]
peer_count = peer_valid.sum(dim=2)               # [B,T]
peer_sum = (peer_h * peer_valid[..., None]).sum(dim=2)
peer_mean = peer_sum / peer_count.clamp_min(1)[..., None]  # [B,T,D]

memory = torch.stack([target, peer_mean], dim=2)  # [B,T,2,D]
memory = memory + day_pe[:, :, None, :]         # 날짜별 두 슬롯에 같은 PE
memory = memory.reshape(B, 2*T, D)             # [B,2T,D]

# peer_count == 0이면 각 날짜의 두 번째 슬롯을 K/V Mask로 가린다.
# decoder query [B,T,D] -> cross-attention memory [B,2T,D]
```

## 6. 비교 실험 및 검증

**비교 실험:**

- **A — v3 Baseline:** 날짜별 Target Context만 전달 → KV `T`개.
- **B — 변경안:** 날짜별 Target Context + Peer Mean Context 전달 → KV `2T`개.
- **C — 참고 실험:** 날짜별 개별 Peer Context 전체 전달 → KV `T×G`개 (필요할 때만 수행).

모두 동일한 Fold, 학습 설정, 평가 Window와 **F1 중심 평가**로 비교한다. Peer Mean은 개별 Peer 정보를 손실시킬 수 있으므로 **B가 반드시 A보다 좋다는 가정은 하지 않는다**.

**필수 검증 항목:**

- [ ] Target Context는 Encoder 출력의 슬롯 0에서 추출한다.
- [ ] Peer 평균에는 유효 Peer만 포함하며 Padding이 평균에 들어가지 않는다.
- [ ] Peer 순서를 바꾸어도 Peer Summary 및 최종 Prediction이 수치 허용 오차 내에서 동일하다.
- [ ] Peer 수가 0개인 날짜의 Peer Summary 슬롯은 Cross-Attention에서 마스킹된다.
- [ ] 각 날짜의 Target/Peer 두 Context에 **동일한 날짜 PE**를 적용한다.
- [ ] 두 Encoder의 K/V shape는 각각 `[B,2T,D]`, Attention Score는 `[B,H,T,2T]`이다.
- [ ] `T∈[1,64]`에서도 최종 이상 점수는 `[B,T,24]`로 나온다.
- [ ] 기존 v3의 날짜별 Encoder Self-Attention, Zone→Type 순차 Cross-Attention 및 결측 처리 정책을 변경하지 않는다.

**변경하지 않는 사항:** Day Token=`24시간`, Zone/Type 그룹 정의와 Train/Test Peer 구성, Encoder 가중치와 깊이, Target Role Embedding, Encoder 내부 날짜 PE 미사용, Encoder Same-Building Bias 미사용, Decoder 구조/깊이, `T=1~64` 학습, End-to-End 지도학습, Zero-fill 및 결측 시간 Loss 포함, `Linear(D,24)` 시간별 출력.
