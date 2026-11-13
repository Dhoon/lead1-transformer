# LEAD Transformer — v3_pm 기반 Site/Type 병렬 Cross-Attention 변경 명세

> **문서 유형:** 기존 `SPEC.md` 전체를 대체하지 않는 **부분 변경 명세**  
> **적용 기준:** `v3_pm` = `v3_perday` + `data.peer_missing_mask: true`  
> **실험안 이름(제안):** `v3_pm_parallel`  
> **변경 범위:** **Temporal Decoder 각 Layer의 Site(Zone) → Type 순차 Cross-Attention 부분만** 병렬 Cross-Attention + Concat/MLP Fusion으로 변경한다.  
> **변경하지 않는 것:** 두 Encoder, Day Token, 대상 건물 Context 추출, Peer 결측 마스킹, 데이터 분할, 학습 및 평가 프로토콜.

## 1. 변경 목적 및 판단 관점

모델의 원래 의도는 **세 가지 서로 다른 비교 관점**을 조합해 시간별 이상을 판단하는 것이다.

1. **Decoder Self-Attention:** 대상 건물 자체의 날짜 간 소비 패턴 학습 — *내 건물의 평소 패턴과 다른가?*
2. **Site(Zone) Encoder:** 같은 `site_id`의 건물들 간 소비 패턴 관계 학습 — *동일 Site 건물들도 비슷하게 변하는가?*
3. **Type Encoder:** 같은 `primary_use`의 건물들 간 소비 패턴 관계 학습 — *같은 용도 건물들도 비슷하게 변하는가?*

예를 들어 대상 건물의 전력 사용량이 증가했어도, 같은 Site 또는 같은 Type의 다른 건물들도 공통적으로 증가했다면 이것이 반드시 이상을 뜻하지는 않는다. 반대로 대상 건물만 독립적으로 달라졌다면 이상 가능성을 높이는 비교 근거가 될 수 있다.

기존 순차 구조에서는 **Type Cross-Attention의 Query가 이미 Site Context를 반영**한다. 이번 실험은 **Site와 Type을 대상 건물의 동일한 시점 표현에서 각각 독립적으로 조회**한 뒤 융합한다. 이 설계가 F1을 높일지는 **검증할 가설**이며 보장되지 않는다.

## 2. 현재 v3_pm에서 유지할 부분

### 2.1 날짜별 Site / Type Encoder — 변경 없음

- 동일 날짜의 그룹 건물 `G`개끼리만 Self-Attention: 입력 `[B×T,G,D]`.
- Site Encoder와 Type Encoder는 서로 다른 학습 파라미터 사용.
- 건물 Role Embedding(target / peer) 유지. 대상 건물은 슬롯 `0`.
- 같은 날짜의 Peer 정보를 반영한 **대상 건물의 Encoder 출력 하나만** 추출하고 날짜 PE를 더한다.
- 따라서 Decoder Cross-Attention Memory는 다음처럼 기존과 동일하다.

\[
C_S\in\mathbb R^{B\times T\times D},\qquad
C_U\in\mathbb R^{B\times T\times D}.
\]

- **`v3_allpeer`로 전환하지 않는다.** `encoder_context_pooling: target_token` 유지. K/V 크기는 `T×G`가 아니라 `T`.
- `data.peer_missing_mask: true` 유지: **Peer의 하루 24시간이 모두 결측이면 그 날짜의 Peer Token을 Encoder Self-Attention에서 제외**한다. 일부 시간만 결측인 Peer는 기존 정책 유지, 대상 건물은 결측이어도 마스킹하지 않는다.
- 결측값 Zero-fill, 시간별 결측 정답 Loss 포함 정책, Pre-LN, Encoder 레이어 수와 Day Token 표현은 유지한다.

### 2.2 Decoder Self-Attention — 변경 없음

대상 건물 `T`개의 Day Token을 입력받고, 기존 **Causal Masked Self-Attention → FFN**을 수행한다.

\[
H_{\mathrm{self}}\in\mathbb R^{B\times T\times D}.
\]

기존의 Day Token Projection `Linear(24,128)`, 공유 Sinusoidal Day PE, `D=128`, `n_heads=4`, `d_ff=512`, `decoder_layers=2`, `dropout=0.1`, 출력 `Linear(128,24)`을 유지한다.

## 3. Decoder Cross-Attention 변경

### 3.1 Before — Site → Type 순차 수행

```text
Decoder Layer input
  → Causal Self-Attention → FFN
  → Site Cross-Attention → FFN
  → Type Cross-Attention → FFN
Decoder Layer output
```

Type Query는 Site 단계에서 갱신된 Decoder Representation에서 만들어진다. 따라서 Site/Type의 처리 순서에 따른 의존성이 있다.

### 3.2 After — 동일 Query에서 Site / Type 병렬 수행 후 Fusion

```text
Decoder Layer input
          │
          ▼
  Causal Self-Attention
          │
         FFN
          │
          ▼
     H_self [B,T,D]
          │
     ┌────┴────┐
     │         │
     ▼         ▼
 Site Cross   Type Cross
 Attention    Attention
 Q: H_self    Q: H_self
 KV: C_S      KV: C_U
     │         │
    O_S       O_U        (각각 [B,T,D])
     └────┬────┘
          │
   Concat [H_self; O_S; O_U]  [B,T,3D]
          │
    Fusion MLP: 3D → D
          │
   Residual + Dropout
          │
         FFN
          │
Decoder Layer output [B,T,D]
```

**핵심 조건:** 두 Cross-Attention의 Query Source는 **완전히 같은 `H_self`**이고, Site branch 출력을 Type branch의 입력으로 사용하거나 그 반대로 전달하지 않는다.

### 3.3 수식 (Decoder Layer 한 개)

먼저 기존 방식으로 Self-Attention과 그 뒤의 FFN을 수행해 `H_self`를 얻는다. 아래의 `LN`은 **각 서브레이어의 Pre-LN**을 뜻한다.

\[
Z=\operatorname{LN}(H_{\mathrm{self}}).
\]

\[
\begin{aligned}
O_S&=\operatorname{CrossAttn}_S(Q=Z,\ K/V=C_S),\\
O_U&=\operatorname{CrossAttn}_U(Q=Z,\ K/V=C_U).
\end{aligned}
\]

두 Branch는 **독립적인** Q/K/V 및 Output Projection 파라미터를 갖는다. `Z` 입력 텐서는 공유하지만 각 Branch의 `W_Q`는 공유하지 않는다.

\[
F=\operatorname{FusionMLP}([Z;O_S;O_U])\in\mathbb R^{B\times T\times D}.
\]

\[
\begin{aligned}
H_{\mathrm{fused}}&=H_{\mathrm{self}}+\operatorname{Dropout}(F),\\
H_{\mathrm{out}}&=H_{\mathrm{fused}}
+\operatorname{Dropout}\!\left(\operatorname{FFN}(\operatorname{LN}(H_{\mathrm{fused}}))\right).
\end{aligned}
\]

**초기 Fusion MLP(실험 고정):**

```python
self.context_fusion = nn.Sequential(
    nn.Linear(3 * d_model, d_model),  # 384 → 128
    nn.GELU(),
    nn.Linear(d_model, d_model),      # 128 → 128
)
```

- 단순 Concat 결과 `3D`를 Decoder의 다음 Layer로 넘기지 않는다. `D`로 재투영해 **기존 Decoder 출력 차원 유지**.
- `H_self`는 **Fusion MLP 입력**이자 **Residual 경로**로 사용한다. 이것은 의도된 구성이다.
- 각 Cross-Attention branch에 별도 Residual을 순차로 적용하지 않는다. **두 Context를 융합한 결과에 하나의 Residual**을 적용한다.
- 기존의 Cross-Attention 사이 FFN을 그대로 두지 않는다. 대신 위 그림처럼 **병렬 Fusion 뒤 FFN 하나**를 둔다. 따라서 이 실험은 **Attention 수행 순서뿐 아니라 Decoder 내부 FFN 배치/파라미터 수도 변경**한다는 점을 결과에서 명시한다.

## 4. Attention Shape · Mask · 시간 관계

`T=64일`, `D=128`, `H=4`, `head_dim=32` 기준:

| 구성 | Q | K/V | Head별 Score |
|---|---|---|---|
| Decoder Self-Attn | `[B,64,128]` | `[B,64,128]` | `[B,4,64,64]` |
| Site Cross-Attn | `[B,64,128]` | `[B,64,128]` | `[B,4,64,64]` |
| Type Cross-Attn | `[B,64,128]` | `[B,64,128]` | `[B,4,64,64]` |
| Concat | — | — | `[B,64,384]` |
| Fusion MLP 출력 | — | — | `[B,64,128]` |
| 최종 시간별 이상 Logits | — | — | `[B,64,24]` |

- **Decoder Self-Attention의 Causal Mask 유지.**
- Site/Type Cross-Attention에는 기존 v3_pm의 **날짜 Causal Mask를 추가하지 않음**(오프라인 이상 탐지).
- Window Padding / Key Padding Mask는 **각 Branch의 기존 Memory Mask**를 그대로 사용한다. Peer 결측 가리기는 Encoder 내부에 적용되는 것이며, K/V의 날짜 수 `T`를 제거하는 규칙이 아니다.
- Site와 Type Branch는 서로 다른 Key/Value를 참고하지만 **Query 날짜 위치는 동일**하다.
- 별도의 날짜-일치 Hard Mask를 **이번 실험에서 추가하지 않는다**. 즉, 날짜 `d`의 Query는 `T`일의 Site/Type Context에 각각 Attention할 수 있다.

## 5. 기존 대비 바뀌는 코드 범위

**변경:**

1. `TemporalDecoderLayer.forward()`의 `Site Cross-Attn → FFN → Type Cross-Attn → FFN` 호출부를 **동일한 `H_self` 기반 병렬 두 Cross-Attn**으로 교체.
2. Branch 출력 `O_S`, `O_U`와 `H_self`를 마지막 Feature 차원에서 Concat 후 **Fusion MLP `3D→D`** 적용.
3. 위 수식과 같이 Fusion Residual 및 뒤쪽 FFN을 배치.
4. 기존 Sequential 구현은 제거하지 말고 **설정으로 분기**해 Ablation에서 재현 가능하게 유지.

**변경하지 않음:**

- DataLoader / Fold 분할 / Zone·Type Peer 선정 / Encoder 날짜별 Attention / Peer Missing Mask
- `encoder_context_pooling: target_token` / Encoder의 날짜별 Target Context + PE
- Decoder Self-Attention의 Causal Mask / Day Token / Hourly Output Head
- 학습 Loss, 학습 Window `T∈[1,64]`, 64일 고정 검증, Checkpoint 기준(F1@0.5)

설정 키 이름은 **추가 제안**이며 프로젝트에 이미 존재하는 키와 충돌하면 기존 명명 규칙에 맞춘다.

```yaml
model:
  encoder_attention_scope: per_day_buildings
  encoder_context_pooling: target_token
  cross_attention_fusion: parallel_concat_mlp  # 기존: sequential
  cross_attention_parallel_query: shared_decoder_self
  fusion_include_self: true
  fusion_hidden_dim: 128
  fusion_residual: true
  d_model: 128
  n_heads: 4
  decoder_layers: 2

data:
  peer_missing_mask: true
```

**주의:** 실제 YAML 스키마에서 `d_model`, `n_heads`, `decoder_layers`가 어느 경로에 정의돼 있는지는 기존 설정 파일을 따를 것. 위 코드는 변경 목표를 보여 주는 예시이며 새 키와 기존 키의 위치를 무조건 강제하지 않는다.

## 6. 구현 검증 / 실험 비교

### 6.1 단위 테스트

- [ ] Site Cross-Attention과 Type Cross-Attention이 **동일한 Decoder Self-Attn+FFN 결과**를 입력으로 받는지 검증.
- [ ] 각 Branch의 Q/K/V Projection 가중치가 독립인지 검증.
- [ ] `C_S`, `C_U` Shape가 각각 `[B,T,D]`이며 `T×G` 전체 Peer Memory를 사용하지 않는지 확인.
- [ ] `concat([H_self,O_S,O_U], dim=-1)`의 Shape가 `[B,T,3D]`; Fusion 출력은 `[B,T,D]`인지 확인.
- [ ] 모든 Decoder Layer 출력이 `[B,T,D]`, 최종 Logits가 `[B,T,24]`인지 확인.
- [ ] Site 및 Type Memory를 **각각 따로** 변경했을 때 해당 Branch 출력이 달라지는지 확인(독립 경로 존재 여부).
- [ ] Padding Key 및 결측 Peer 마스킹이 기존 `v3_pm`과 동일하게 적용되는지 확인.
- [ ] Causal Self-Attention Mask는 유지되고 Cross-Attention에는 새 Causal Mask가 생기지 않는지 확인.
- [ ] 기존 `sequential` 모드 실행 결과가 변경 전 `v3_pm`과 동일한지 회귀 테스트.

### 6.2 비교 실험

| 모델 | Encoder / 결측 처리 | Decoder Site·Type 결합 | 목적 |
|---|---|---|---|
| **Baseline `v3_pm`** | 날짜별 Target Context + Peer 결측 Mask | Site → Type Sequential | 기존 결과 재현 |
| **`v3_pm_parallel`** | **완전히 동일** | **Site ∥ Type → Concat[Self;Site;Type] → MLP + Residual → FFN** | 병렬 Fusion의 효과 측정 |

- 같은 5-Fold, Seed, 학습 스텝/Window, 평가 기간과 프로토콜로 비교한다.
- **최우선 지표:** `F1@0.5`; 함께 PR-AUC, ROC-AUC, Precision, Recall, Fold별 성능 및 Epoch당 시간 기록.
- 기존 `v3_pm` 참고 결과: 5-Fold 평균 **ROC 0.916 / PR 0.673 / F1@0.5 0.676** (`result_1112.md`). 이 수치는 신규 모델의 성능이 아니다.
- **해석 유의:** 병렬화를 하면서 FFN 배치와 Fusion MLP가 동시에 바뀌므로, 결과 개선 시 **병렬화 그 자체의 효과라고만 단정하지 않는다**. 필요하면 같은 파라미터 예산/FFN 수를 맞춘 추가 Ablation으로 분리한다.

## 7. 이 변경에서 하지 않는 것

- `v3_allpeer`의 `T×G` Memory로 변경하지 않는다.
- `2T` Peer Mean Summary를 추가하지 않는다.
- Hour Token(v5) 또는 Day/Hour Hybrid로 변경하지 않는다.
- 새로운 날짜-일치 Cross-Attention Hard Mask, Learned Gate, 건물 ID Embedding, 외부 Feature를 추가하지 않는다.
- Encoder와 Decoder의 Layer 수, Embedding 차원, 학습 프로토콜을 임의 변경하지 않는다.

**최종 한 줄:** `v3_pm`의 날짜별 Site/Type Encoder와 Peer 결측 Mask는 유지하고, Decoder의 **Site → Type 순차 Cross-Attention을 동일 Decoder Query 기반 병렬 Site/Type Cross-Attention으로 변경한 후 `[Self;Site;Type]`를 MLP로 융합**한다.
