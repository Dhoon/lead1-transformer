# SPEC — Kaggle LEAD 기반 Dual-Context Transformer 이상 탐지 연구

> **문서 상태:** 연구 설계 초안 (Draft v0.9 — 평가 건물 Peer를 학습 건물에서 매칭)  
> **기준일:** 2026-11-02  
> **주요 목표:** Kaggle LEAD 전력 소비 시계열에서 **Zone/Type 관계와 일별 소비 패턴**을 함께 학습하여, Train과 Test에서 **동일한 전체 Transformer**로 시간별 이상 점수를 예측한다.  
> **표기:** **[확정]** = 대화에서 합의된 설계, **[초기 설정]** = 첫 실험에서 사용하기로 정한 하이퍼파라미터(최적 성능을 의미하지 않음), **[제안]** = 실험 옵션, **[미정]** = 후속 연구에서 결정할 사항.

---

## 1. 프로젝트 개요

### 1.1 목표 및 범위

- **목표:** Transformer를 활용하여 건물의 시간별 전력 소비 데이터에서 **비정상적인 전력 소비 패턴(Anomaly)**을 탐지한다.
- **데이터:** **Kaggle LEAD Competition**에서 제공하는 건물별 1시간 간격 전력 소비량과 학습용 이상 레이블을 사용한다.
- **핵심 접근:** 하루 24시간 전력량을 **Day Token**으로 변환하고, 같은 Zone·같은 Type의 건물 간 관계와 대상 건물의 날짜 간 패턴을 함께 학습하는 **Dual-Context Transformer**를 설계한다.
- **학습 절차:** `train.csv`의 시간별 이상 레이블을 사용하여 **Zone Encoder + Type Encoder + Temporal Decoder + Hourly Anomaly Head 전체를 End-to-End 지도학습**한다. 별도 Building ID 분류 Pretraining/Downstream 분리는 기본 설계에서 사용하지 않는다.
- **최종 출력:** Kaggle LEAD 평가 방식에 맞춰 **건물별·시간별 이상 점수/확률**을 산출한다.

### 1.2 데이터 출처 및 참고 문헌

**데이터 출처**

- Kaggle, *Large-scale Energy Anomaly Detection (LEAD) Competition*: <https://www.kaggle.com/competitions/energy-anomaly-detection>

**참고 논문**

1. Manoj Gulati, Pandarasamy Arjunan, *LEAD1.0: A Large-scale Annotated Dataset for Energy Anomaly Detection in Commercial Buildings* (2022). [논문 원문 (arXiv)](https://arxiv.org/abs/2203.17256).
2. Chun Fu, Pandarasamy Arjunan, Clayton Miller, *Trimming outliers using trees: Winning solution of the Large-scale Energy Anomaly Detection (LEAD) competition*, ACM BuildSys 2022. [논문 DOI](https://doi.org/10.1145/3563357.3566147).

### 1.3 사용 데이터 및 학습 목표

| 구분 | 내용 |
|---|---|
| 학습 데이터 | Kaggle LEAD `train.csv`의 **200개 건물**, 총 **1,048,574개 시간별 관측 행**(사용 데이터 기준), 정상/이상 레이블 제공 |
| 테스트 데이터 | Kaggle LEAD `test.csv`의 **206개 건물**, 이상 레이블 비공개 |
| 학습 방식 | **전체 Dual-Encoder + Decoder 모델을 시간별 이상 레이블로 End-to-End 지도학습** |
| 추론 방식 | Train/Test에서 동일 아키텍처를 사용하며, Test 대상 건물의 Context는 **같은 Zone/Type의 Train 건물 전력량(레이블 미사용)**에서 매칭해 구성 |
| 출력 단위 | **Day Token 1개 → 시간별 이상 점수 24개**, 입력 `T`일 → `T×24`개 점수 |
| 공식 평가 지표 | **ROC-AUC** |

> 본 프로젝트는 **Kaggle LEAD 대회 데이터**를 실험 대상으로 한다. 원 논문의 전체 데이터셋 규모와 대회용 데이터 규모를 별도 연구 대상으로 병렬 관리하지 않는다.

---

## 2. 핵심 아이디어

**[확정]** 아키텍처는 **2개의 관계 Encoder + 1개의 시간 Decoder**로 구성한다.

1. **Zone Encoder**: 같은 Zone 건물들의 **T일치 Day Token 전체를 날짜 순서대로 연결한 단일 Token 집합**을 입력받아 건물 간·날짜 간 소비 관계를 학습한다.
2. **Type Encoder**: 같은 Type(`primary_use` 등) 건물들의 **T일치 Day Token 전체를 하나로 연결**하여 유형 내 건물 간·날짜 간 소비 관계를 학습한다.
3. **Temporal Decoder**: 관심 건물 하나의 **하루 24시간 소비량을 1개 토큰**으로 만들어, 여러 날짜(그제·어제·오늘 등)의 순서 있는 시퀀스를 처리한다.
4. Decoder는 **Causal Masked Self-Attention**을 사용하며, 두 Encoder 출력과 **Cross-Attention**을 수행한다.
5. Decoder의 **각 날짜별 Representation에서 24개의 시간별 이상 Logit**을 출력한다. 시간별 레이블로 모델 전체를 함께 학습하며, Test에도 두 Encoder를 유지한다.

### 2.1 확정된 설계 원칙

- **[확정]** 하루의 24개 `meter_reading`을 `Linear(24,D)`로 투영하여 **Day Token 1개**로 만든다.
- **[확정]** Zone Encoder/Type Encoder는 대상 건물을 포함한 그룹의 동일 `T`일 데이터를 **날짜 우선(day-major)으로 Flatten**하여 각각 통합 Self-Attention을 수행한다.
- **[확정]** 각 Encoder의 **날짜별 첫 번째 Token은 Decoder 대상 건물의 해당 날짜 Day Token**이며, 동일 날짜 건물들에는 **동일한 날짜 PE**를 적용한다. 건물 순서용 PE는 없다.
- **[확정]** Decoder는 대상 건물의 Day Token 시퀀스를 Causal Masked Self-Attention으로 처리하고, **Zone → Type(Building Type) 순서**로 Cross-Attention을 적용한다. 각 Attention 뒤에는 FFN을 둔다. **Type → Zone**은 후속 Ablation 옵션이다.
- **[확정]** Train과 Test에서 **Zone Encoder + Type Encoder + Decoder 전체를 사용**한다. `train.csv`의 이상 레이블로 **시간별 이상 탐지 End-to-End 지도학습**을 수행한다. Building ID Classification Pretraining 및 Decoder-only 전이는 기본 설계에서 제외한다.
- **[확정]** `building_id`는 입력 데이터의 정렬, 동일 건물 추적, Peer 구성 및 Building 단위 평가 Split에만 사용하고, **Building ID Lookup Embedding 또는 숫자 ID Feature로 투입하지 않는다.** `D`는 건물 수와 독립적이다.
- **[확정]** 입력 기간은 샘플마다 `T ∼ Uniform{1,…,64}`이며, Encoder와 Decoder는 **동일한 연속 날짜 범위**를 사용한다.
- **[확정]** `T`개의 Day Token에서 각 날짜별 24개 시간에 대한 Anomaly Score를 출력한다: `[B,T,24]`.

---

## 3. 입력 데이터 및 토큰화

### 3.1 하루 = 토큰 하나

건물 `b`, 날짜 `d`, 시간 `h ∈ {1,...,24}`에서의 전력 계측값을 `x[b,d,h]`로 정의한다.

\[
\mathbf x_{b,d}=[x_{b,d,1},\ldots,x_{b,d,24}]\in\mathbb R^{24}.
\]

**[확정]** Day Token은:

\[
\mathbf e_{b,d}=W_{\text{patch}}\mathbf x_{b,d}+\mathbf b_{\text{patch}}\in\mathbb R^D,
\quad W_{\text{patch}}\in\mathbb R^{D\times24}.
\]

**[확정]** Encoder와 Decoder 모두 **날짜(day)에 대해서만 Positional Encoding(PE)**을 적용한다:

\[
\mathbf z_{b,d}=\mathbf e_{b,d}+PE(d).
\]

- **Zone/Type Encoder:** 같은 날짜 `d`의 서로 다른 건물 Token에는 **동일한** `PE(d)`를 더하고, 다른 날짜에는 **서로 다른** `PE(d)`를 더한다. 건물의 나열 순서를 나타내는 PE는 넣지 않는다.
- **Temporal Decoder:** 대상 건물의 연속된 Day Token에 날짜 순서대로 `PE(d)`를 더한다.
- **[초기 설정]** Sinusoidal PE를 사용하며, `D=128`에서 날짜마다 길이 128의 벡터를 만든다. 예를 들어 1일차의 4개 건물에는 모두 **같은** `PE(1)` 벡터를 더한다. 2일차에는 네 건물 모두 `PE(2)`를 더한다. 이는 '1일차에는 cos만, 2일차에는 sin만'을 더하는 것이 아니라, 매 날짜별로 sin/cos가 교차하는 **D차원 벡터**를 생성한다.

\[
PE(d,2i)=\sin\left(\frac{d}{10000^{2i/D}}\right),\qquad
PE(d,2i+1)=\cos\left(\frac{d}{10000^{2i/D}}\right).
\]

- **[초기 설정]** `D=128`을 사용한다. 200개 건물이라고 해서 `D=200`일 필요는 없다. `D=64/256`은 후속 비교 실험으로 둔다.
- 하루 24개 값의 시간별 위치는 **선형 투영의 24개 입력 인덱스**에 의해 구별된다. 날짜 PE는 **Day Token 간의 시간 정보**를 나타내며, 같은 날 건물들의 순서를 인코딩하지 않는다.
- **[초기 설정 — V1]** `meter_reading`이 결측인 시간은 **0으로 대체(Zero-fill)**하여 길이 24의 Day Vector를 구성한다. 이는 **값 대체(value masking)**이며, 해당 시간/날짜를 Self-Attention에서 삭제하는 **Attention Mask가 아니다**. 결측이 아닌 실제 `0` 소비량은 변경하지 않는다.
- **[비교 실험 — V2]** LEAD 대회 우승 솔루션을 참고하여 **건물별 관측값 중앙값(median)**으로 결측 시간을 대체하는 별도 버전을 구현한다. V1(0 대체)과 V2(중앙값 대체)의 이상 탐지 성능을 비교하며, 어느 방법이 우수한지는 **미정**이다.
- **[초기 설정]** 결측 여부를 `observation_mask`로 전처리 단계에 보관하되, 두 버전 모두 **모델 입력 특징으로 직접 전달하지 않는다**. 값 대체 방식이 Building/Anomaly Shortcut이 되는지는 검증한다. 이상 레이블이 존재하는 시간이나 날짜를 **라벨에 근거해 제외·마스킹하지 않는다**.
- **[미정]** 결측 `meter_reading`이 있는 행의 **지도학습 Loss 포함 여부** 및 관측 여부 Mask를 추가 입력으로 제공하는 확장 실험은 추후 결정한다. Padding용 Attention Mask와 결측 관측 Mask는 구분한다.

### 3.2 Zone/Type 구분

- **[제안]** LEAD 데이터에서는 `site_id`를 **Zone의 대리 변수(proxy)**로 사용한다. `site_id`가 실제 지리적 Zone과 완전히 동일하다는 주장은 하지 않는다.
- **[제안]** `primary_use`를 Building Type으로 사용한다.
- Train과 Test 모두 Zone/Type 그룹 구성용 메타데이터를 사용한다. 그룹을 구성하기 위한 **건물 고유 ID 자체는 모델 입력 Feature로 넣지 않는다.**
- **[확정]** 두 Encoder의 입력 단위도 24시간짜리 **Day Token**이며, Decoder가 사용하는 `T`일과 **같은 날짜 범위**를 사용한다. `T`일 × 그룹 건물 수만큼의 Token을 **하나의 Encoder Self-Attention 시퀀스**로 합친다. 각 날짜의 그룹에서 **대상 건물을 첫 번째**에 둔다. 시간(hour)별 별도 Encoder는 기본 설계에서 사용하지 않는다.

### 3.3 입력/출력 Tensor 규약 (MVP 기본안)

| Tensor | Shape | 의미 |
|---|---|---|
| `target_daily` | `[B, T, 24]` | 대상 건물별 연속 T일 전력량 |
| `zone_daily` | `[B, T, Z, 24]` | 같은 Zone의 건물들(매 날짜 첫 번째는 대상 건물) |
| `type_daily` | `[B, T, U, 24]` | 같은 Type의 건물들(매 날짜 첫 번째는 대상 건물) |
| `zone_tokens` | `[B, T×Z, D]` | Day Projection + 날짜별 공유 PE + day-major Flatten |
| `type_tokens` | `[B, T×U, D]` | Day Projection + 날짜별 공유 PE + day-major Flatten |
| `zone_mask` / `type_mask` | `[B, T×Z]` / `[B, T×U]` | 그룹 크기 차이 및 패딩을 위한 Attention 유효성 Mask |
| `observation_mask` | `[B, T, 24]` | 실제 meter reading 관측 여부(전처리 보관용; MVP 모델 입력에는 전달하지 않음) |
| `anomaly_target` | `[B, T, 24]` | Train에서만 제공되는 시간별 0/1 이상 레이블 |
| `anomaly_logits` | `[B, T, 24]` | Train/Test 공통: 시간별 이상 Logit |

- `B`: 배치 크기, `T`: 1~64 사이에서 샘플링한 날짜 수, `Z`/`U`: Zone/Type 그룹 건물 수(배치 내 최대값에 맞춰 Padding 가능).
- Flatten 순서: `[day 1: target, peer 1, ...] → [day 2: target, peer 1, ...] → ...`. **같은 날짜의 Token에는 동일한 `PE(d)`**를 더한다.
- `target_daily`와 Encoder의 날짜별 첫 Token은 **동일 건물·동일 날짜·동일 원본 24시간 전력량**을 공유한다.
- 학습에서는 `anomaly_target`으로 Loss를 계산하고, Test에서는 라벨 없이 `anomaly_logits`를 예측한다. 최종 제출은 **실제 `test.csv`에 있는 시간별 행에 대응하는 점수**로 `row_id` 기준 매핑한다.
- 결측 `meter_reading`은 **V1에서 0으로 채운 뒤** Day Projection에 투입한다(V2에서는 건물별 관측 중앙값 사용). `observation_mask`는 원래 관측 여부만 보관하며 모델 입력에 포함하지 않는다. `zone_mask`/`type_mask`는 **패딩용 Token Mask**로서 결측 시간을 뜻하지 않는다. 제출 대상 행은 누락하지 않는다.

---

## 4. 모델 아키텍처

### 4.1 전체 데이터 흐름

```text
       ZONE: T days × Z buildings          TYPE: T days × U buildings
        target first in each day            target first in each day
                  |                                   |
            Linear(24,D)                        Linear(24,D)
                  |                                   |
            Shared Day PE                       Shared Day PE
                  |                                   |
          Flatten [T×Z,D]                      Flatten [T×U,D]
                  |                                   |
          Zone Encoder × 2                     Type Encoder × 2
                  |                                   |
                  +-----> Decoder cross-attention <---+
                                    ^
                                    |
Target: T days × 24 -> Linear(24,D) + Day PE
                                    |
                     Decoder × 2 (each layer):
                     Masked Self-Attn -> FFN
                          Zone Cross-Attn -> FFN
                          Type Cross-Attn -> FFN
                                    |
                             Output [T,D]
                                    |
                         Linear(D,24) (logits)
                                    |
                        Hourly scores [T,24]
```

- **Train:** 전체 네트워크를 시간별 이상 레이블로 End-to-End 지도학습한다.
- **Test:** **동일한 Encoder + Decoder + Head**를 그대로 사용해 미학습 건물의 시간별 이상 점수를 예측한다.

### 4.2 Zone Encoder

**[확정]** 대상 건물 `b`의 같은 Zone에 속한 건물 집합을 `\mathcal Z(b)`라고 할 때, Decoder와 동일한 `T`개 날짜에서 얻은 **모든 Day Token을 하나의 Self-Attention 시퀀스**로 처리한다.

\[
X_Z=\big[\underbrace{z_{b,1},z_{2,1},z_{3,1},z_{4,1}}_{\text{1일차}},
\underbrace{z_{b,2},z_{2,2},z_{3,2},z_{4,2}}_{\text{2일차}},
\ldots,
\underbrace{z_{b,T},z_{2,T},z_{3,T},z_{4,T}}_{\text{T일차}}\big]
\]

여기서 `z_{i,d} = Linear(x_{i,d}) + PE(d)`이며, 예시는 같은 Zone에 **대상 포함 4개 건물**이 있는 상황이다.

- 날짜별로 Encoder를 따로 호출하지 않고 **`T×Z`개 토큰을 통합해 Self-Attention**을 수행한다.
- 각 날짜 블록의 **첫 번째 Token은 해당 날짜의 Decoder 대상 건물 원본 전력량**을 투영한 벡터다.
- **같은 날짜의 모든 건물 Token에 동일한 PE**, 서로 다른 날짜에는 서로 다른 PE를 더한다.
- 건물 Token의 **나열 순서에는 별도 PE를 부여하지 않는다.** 따라서 같은 날짜 내부의 건물 순서를 바꿔도 (올바른 마스크와 함께) 학습 결과는 순열 등변/불변 특성을 유지한다.
- 날짜와 건물 모두를 통합하므로 **건물 간 관계와 서로 다른 날짜 간 관계를 동시에 학습**한다.

### 4.3 Type Encoder

Type Encoder도 같은 방식을 적용한다. 대상 건물 `b`와 같은 유형의 건물 집합 `\mathcal U(b)`에 대해:

\[
X_U=[\{z_{i,1}:i\in\mathcal U(b)\},\{z_{i,2}:i\in\mathcal U(b)\},\ldots,\{z_{i,T}:i\in\mathcal U(b)\}]
\]

- 입력 크기는 `[T×U,D]`; **각 날짜 블록 첫 번째는 Decoder 대상 건물의 동일 날짜 Token**이다.
- **날짜 공유 PE 사용**, 건물 나열 순서에 대한 PE는 사용하지 않는다.
- `T×U` 전체에 대한 Joint Self-Attention을 수행한다.
- Zone/Type Encoder는 **서로 다른 Transformer 가중치**를 사용한다. **[제안]** 입력 `Linear(24,D)`만 공유하는 모델부터 시작하고, Projection을 분리하는 Ablation을 수행한다.

### 4.4 Temporal Decoder

입력은 대상 건물의 **날짜별 Day Token** 시퀀스다.

\[
H^{(0)}_b=[P(\mathbf x_{b,d-T+1})+PE(1),\ldots,P(\mathbf x_{b,d})+PE(T)].
\]

Causal Self-Attention은 날짜 인덱스 `i`가 날짜 `j > i`를 볼 수 없도록:

\[
M_{ij}=\begin{cases}0,&j\le i\\-\infty,&j>i\end{cases}
\]

를 적용한다. 예시:

```text
           Keys / days
            1  2  3  4  5  6
Query  1    O  X  X  X  X  X
       2    O  O  X  X  X  X
       3    O  O  O  X  X  X
       4    O  O  O  O  X  X
       5    O  O  O  O  O  X
       6    O  O  O  O  O  O
```

- **오늘 Token은 오늘·어제·그제 등의 기록을 함께 참조**할 수 있다.
- 여기서 Causal Mask는 정보 접근 제약이지, **다음 시점 예측을 반드시 수행해야 한다는 의미가 아니다.**

### 4.5 두 Encoder와 순차 Cross-Attention

Decoder의 날짜별 상태를 Query로, **T일 전체를 통합 인코딩한 Zone/Type Encoder 출력**을 Key/Value로 사용한다.

\[
H_Z=\operatorname{Enc}_{Z}(X_Z)\in\mathbb R^{(TZ)\times D},\qquad
H_U=\operatorname{Enc}_{U}(X_U)\in\mathbb R^{(TU)\times D}.
\]

**[확정]** 각 Decoder Layer는 아래의 **순차 구조**로 구성하며, 각 Attention 직후 FFN을 배치한다(Attention/FFN에 Residual Connection 및 LayerNorm 적용).

```text
Decoder layer input
       ↓
Causal Masked Self-Attention → FFN
       ↓
Zone Cross-Attention (K,V = H_Z) → FFN
       ↓
Type Cross-Attention (K,V = H_U) → FFN
       ↓
Decoder layer output
```

- **[확정]** 기본 순서는 **Zone → Type(Building Type)**. **Type → Zone** 반대 순서는 Ablation Study에서 비교한다. 두 Cross-Attention의 병렬 융합이나 Learned Gate는 현재 기본 설계가 아니다.
- **[확정]** Encoder는 날짜별 분리 출력이 아니라 **전체 날짜·전체 Peer가 통합된 메모리**를 제공한다.
- **[확정]** 기본 Encoder는 해당 Window **T일 전체에 대해 양방향 Self-Attention**을 수행한다. Encoder 날짜 Causal Mask나 Cross-Attention 날짜 Mask는 기본 모델에 사용하지 않는다.
- **[주의 — 시간적 인과성]** Decoder의 Causal Self-Attention만으로 모델 전체가 인과적이 되는 것은 아니다. **Encoder가 T일 전체를 보므로 초기 날짜의 예측에도 해당 Window의 이후 날짜 소비 정보가 반영될 수 있다.** 이는 Window 전체를 관측한 다음 수행하는 **오프라인/사후 이상 탐지**를 기본 평가 설정으로 정할 때 허용된다. 추후 온라인 탐지를 목표로 하면 Encoder Prefix 또는 날짜 기준 시간 제약이 필요하다.
- **[확정]** 대상 건물이 각 그룹에 포함되므로 대상 Token이 그룹의 날짜별 첫 번째다. Peer 건물이 하나도 없다면 대상 건물 단독 Context가 입력된다.
- **[확정]** Peer는 항상 **학습 건물 집합**에서 가져온다. 학습 시에는 대상 건물을 제외한 같은 Zone/Type의 학습 건물, 평가(검증/테스트) 시에는 같은 Zone/Type의 학습 건물 전체가 Peer다(6.1 참조).

### 4.6 Transformer 레이어 구성 및 초기 하이퍼파라미터

**[초기 설정]** 합의한 첫 번째 구현은 **Zone Encoder 2층 + Type Encoder 2층 + Temporal Decoder 2층(2–2–2)** 구조다. 이는 우선 실험할 **기본 모델**이며, 최적 레이어 수가 검증되었다는 의미는 아니다.

| 구성요소 | 초기 값 | 한 층의 구성 |
|---|---:|---|
| Zone Encoder | **2 layers** | Multi-Head Self-Attention → FFN |
| Type Encoder | **2 layers** | Multi-Head Self-Attention → FFN |
| Temporal Decoder | **2 layers** | Causal Self-Attention → FFN → Zone Cross-Attention → FFN → Type Cross-Attention → FFN |
| Hidden Dimension `d_model` | **128** | Day Token 및 Attention의 기본 표현 차원 |
| Attention Heads `n_heads` | **4** | Head당 차원 `128/4 = 32` |
| FFN Hidden Dimension `d_ff` | **512** | `4 × d_model` |
| Dropout | **0.1** | Attention/FFN의 일반화 정규화 시작값 |
| Hourly Anomaly Head | **24 logits/day** | 날짜별 `Linear(128,24)`; `T`일 입력 시 `[B,T,24]` |

- Zone/Type Encoder는 동일 기간의 **건물 간 관계**를 학습하므로 각각 2층을 초기값으로 두고, Decoder는 날짜 간 관계와 두 종류의 Cross-Attention을 결합하므로 2층에서 출발한다.
- Decoder **각 층에서 Masked Self-Attention + FFN → Zone Cross-Attention + FFN → Type Cross-Attention + FFN을 순차 적용**한다. Type → Zone 순서는 비교 실험 옵션이다.
- 모든 Attention과 FFN에는 Residual Connection과 Layer Normalization을 사용한다. **[제안]** 첫 구현의 LayerNorm 배치는 Pre-LN으로 둔다.
- `D=128`은 건물 수와 독립적인 내부 표현 차원이다. Hourly Anomaly Head는 건물 ID 수에 관계없이 **24개 시간별 Logit**을 출력한다.

**[제안] 레이어 깊이 Ablation**

| 모델 | Zone Encoder | Type Encoder | Decoder | 목적 |
|---|---:|---:|---:|---|
| 경량 | 1 | 1 | 1 | 작은 모델의 기준점 |
| **기본 모델** | **2** | **2** | **2** | 첫 구현 및 기본 비교군 |
| Decoder 확장 | 2 | 2 | 3 | 날짜 간 패턴 학습을 더 깊게 했을 때의 효과 |
| 전체 확장 | 3 | 3 | 3 | 전체 깊이 증가의 효과 |

우선순위는 **2–2–2 vs 2–2–3** 비교이며, 이후 필요에 따라 **1–1–1**, **3–3–3**을 추가한다. 깊이뿐 아니라 모델 표현 차원의 영향을 확인하기 위해 `D ∈ {64, 128, 256}`도 별도 Ablation으로 검토한다.

---

## 5. End-to-End 지도학습

### 5.1 학습 목표

**[확정]** `train.csv`의 **시간별 Anomaly Label**을 이용해 Zone Encoder, Type Encoder, Decoder, Hourly Anomaly Head를 **처음부터 함께 학습**한다.

- 별도 **Building ID Classification Pretraining**을 기본 파이프라인에서 수행하지 않는다.
- 이상 데이터를 정답으로 활용해야 하므로 **이상 시간이 포함된 날짜를 라벨 기준으로 제거하거나 Masking하지 않는다.** Train과 Test에서 동일한 전처리 규칙을 사용한다.
- 학습 대상은 건물의 특정 날짜 **24시간 각각의 이상 여부**이며, Test에서도 Zone/Type Encoder를 유지한다.

### 5.2 시간별 출력 및 Loss

대상 건물 `b`의 날짜 `d`에 대한 Decoder 출력이 `h[b,d] ∈ R^D`이면,

\[
\mathbf a_{b,d}=W_a\mathbf h_{b,d}+\mathbf c_a\in\mathbb R^{24},\qquad
\widehat{\mathbf y}_{b,d}=\sigma(\mathbf a_{b,d}).
\]

- **[확정]** 날짜별 `Linear(D,24)` Head를 적용한다. 전체 Window의 출력은 **`[B,T,24]`**이며 입력 `T`에 따라 최종 시간별 출력 개수도 변한다.
- **[초기 설정]** 시간별 Binary Cross-Entropy with Logits(`BCEWithLogitsLoss`)를 사용한다. 클래스 불균형에 대한 가중치/샘플링은 검증 후 결정한다.
- **[확정]** Loss는 해당 Window에서 레이블이 존재하는 **모든 유효한 날짜·시간**을 대상으로 계산한다. Padding 및 실제 정답이 없는 시점은 Loss에서 제외한다.
- 정상/이상 모두 학습에 쓰므로 정상 데이터만 선별하는 별도 사전학습 정책 및 Building ID Classifier(`Linear(D,N)`)는 현재 기본 구조에 없다.

### 5.3 정보 누출과 Shortcut 통제

- **[확정]** `building_id` 자체, 정답 레이블, 이상 여부에 기반한 Mask를 모델 입력으로 전달하지 않는다. `site_id`, `primary_use`는 Encoder Peer 그룹을 구성하기 위한 메타데이터다.
- 실측 결측(`meter_reading`이 NaN)은 이상 라벨과 **별개 문제**로 취급한다. **V1은 0 대체**, **V2는 건물별 관측 중앙값 대체**로 비교한다. 결측 자체와 0 대체 패턴이 Shortcut에 미치는 영향은 실험으로 확인하며, 관측 여부 Mask는 초기 모델 입력으로 사용하지 않는다.
- 입력 날짜 전체를 결합해 처리하는 Encoder는 후속 날짜의 값을 이용할 수 있으므로 현재 모델의 시간별 예측은 **Window를 다 관측한 뒤 수행하는 사후 이상 탐지**로 정의한다. 온라인/실시간 인과적 탐지와 구분한다.
- 학습 시 사용한 전처리 및 Zone/Type 그룹 구성 규칙을 Test에서도 동일하게 적용하되, Test 정답은 참조하지 않는다.

---

## 6. 추론 및 Kaggle 평가

### 6.1 Test 입력과 출력

- **[확정]** Test에서도 **Zone Encoder + Type Encoder + Temporal Decoder + Hourly Anomaly Head 전체**를 그대로 사용한다.
- **[확정]** Train/Test는 날짜 범위가 같고 건물은 서로 겹치지 않는다. 평가 대상 건물의 Encoder 그룹은 날짜별 첫 Token을 대상 건물로 두고, 나머지 Peer는 **학습 건물 중 같은 Zone/Type 건물의 동일 날짜 전력량**으로 구성한다(`peer_pool: train_only`). 학습 건물의 **레이블은 사용하지 않는다.**
  - 학습 시와 같은 Peer 분포·그룹 크기를 유지하기 위한 선택이다. 예: Type 최대 그룹은 학습 80개, Test 81개(대상 포함).
  - 학습 건물에 같은 Zone/Type이 없는 Test 건물(예: site 6, Retail, Utility, Warehouse/storage)은 해당 Encoder에서 대상 건물 단독 Context가 된다.
  - 평가 건물끼리 Peer로 묶는 `within_split`, 학습+평가 건물을 모두 쓰는 `with_train`은 비교 옵션으로만 둔다.
- Train의 레이블은 Loss 계산용이고, Test에는 레이블 없이 `meter_reading` 및 메타데이터만 사용한다.
- Day Token 하나에서 시간별 24개 점수를 예측하며, 데이터셋의 실제 시간별 행으로 되돌려 **대회 제출 파일의 `row_id`와 이상 점수를 1:1 대응**시킨다.

### 6.2 평가 방식

- **[확정]** 공식 주지표는 **Hour-level ROC-AUC**다. 추가적으로 PR-AUC, F1, Precision, Recall을 내부 검증셋에서 측정한다.
- Kaggle 테스트 정답 레이블은 공개되어 있지 않지만, **Late Submission에서 Public/Private Score가 실제 반환됨**을 확인했다. 다만 이 점수는 최종 외부 평가로 사용하고, 하이퍼파라미터 튜닝은 Train 기반 검증에서 수행한다.
- 동일 행이 여러 Overlapping Window에 포함될 수 있으므로, 추론 시 **한 행당 최종 점수 1개**를 만드는 Window 선택/점수 집계 규칙을 고정해야 한다. 예: 해당 날짜가 Window 마지막에 오도록 Context를 구성하거나, 다중 Window의 점수를 사전에 정한 방식으로 통합한다. **[미정]** 최종 규칙.
- Missing `meter_reading`이 있는 Test 행도 제출 대상에서 누락하지 않는다. **V1의 Test 입력도 Train과 동일하게 0으로 대체**하고(비교 V2에서는 건물별 중앙값 사용), 원래 `row_id`에 해당하는 이상 점수를 제출한다.

---

## 7. 데이터 전처리 및 Split

### 7.1 입력 Feature 정책

- **[확정] MVP의 핵심 입력은 시간별 `meter_reading`을 모아 만든 24차원 Day Vector**다.
- **[제안]** `log1p(meter_reading)`과 학습 구간에서 계산한 정규화 파라미터를 사용한다.
- **[초기 설정 — 결측 처리 V1]** 관측되지 않은 `meter_reading`은 **0으로 값 대체**한다. 하루 중 일부 시간만 결측이어도 나머지 관측값은 유지하고, 24시간 길이의 Day Vector를 만든다. 이 단계에서 결측 시간이나 날짜를 Attention에서 제거하지 않는다.
- **[비교 실험 — 결측 처리 V2]** 결측값을 해당 **건물의 유효 관측값 중앙값**으로 대체한다(LEAD 우승 솔루션 방식 참고). 검증·테스트 건물의 경우에도 **정답 라벨 없이 해당 건물의 관측값**에서 중앙값을 계산하는 오프라인 프로토콜을 사용한다. 유효 관측이 0개인 건물의 대체값은 예외 처리 정책을 추후 결정한다.
- `observation_mask`(원본이 실제로 관측됐는지 여부)는 두 버전 모두 따로 보관하되, **MVP에서 Day Token의 추가 특징으로 넣지 않는다**. `0`은 실제 소비량일 수도 있으므로 Zero-fill을 '결측을 완벽히 숨기는 Mask'로 가정하지 않으며, Shortcut 가능성을 비교한다.
- **[미정]** 관측값이 없는 시간의 Loss 제외 여부 및 관측 여부를 입력 특징으로 제공하는 추가 Ablation은 별도 결정한다.
- **[미정]** 날씨·캘린더·건물 규모 등의 외생 Feature를 추가할지 여부. 첫 실험에는 추가하지 않아 모델 자체의 효과를 측정한다.
- **[주의]** LEAD 우승 논문의 `value_chg_diff/ratio` 및 미래 시점 Shift Feature는 본 모델의 필수 입력이 아니며, 비교 Baseline에서만 사용한다. 기본 모델은 오프라인 사후 탐지로 정의하며, 향후 온라인 평가로 확장할 경우 미래 정보 접근을 통제한다.

### 7.2 학습·검증 전략

**[확정]** Train 200개 건물과 Test 206개 건물은 서로 다른 건물 집합이므로, 건물 일반화 성능을 검증하기 위해 **Building ID 기준** Train/Validation 분할을 사용한다.

1. **내부 평가:** `GroupKFold` 또는 Building 단위 Holdout으로 `train.csv`의 건물을 분리한다. Validation Building의 이상 레이블은 학습에 쓰지 않는다.
2. **전체 모델 학습:** 각 Fold의 Training Building에 대해 **Zone/Type Encoder + Decoder + Hourly Head를 공동 지도학습**하고, Validation Building을 대상 건물로 두되 Peer는 **해당 Fold의 Training Building**에서 매칭하여(Test와 동일 규칙) 평가한다.
3. **최종 제출:** 내부 검증이 끝난 뒤 Train 200개 건물 전체로 모델을 학습하고 Test 206개 건물의 시간별 이상 점수를 제출한다.

- Train/Test의 **날짜 범위가 동일하다**고 해서 건물 ID를 섞어 학습하거나 정답 레이블을 Test에서 이용한다는 뜻은 아니다. 학습/검증/테스트 분할별 정답 정보의 경계를 지킨다.
- 스케일러, Imputation 통계 및 임계값 등 학습 가능한 설정은 **Training Fold에서만 추정**한다(미학습 Test Building에 대한 비지도/개별 통계 정책은 별도 명시하여 평가 프로토콜을 일관되게 유지한다).
- **[후속 검토]** 실시간 날짜별 탐지가 필요해지면 Encoder 전체 Window의 미래 관측값 접근을 제한해야 한다. 지금의 기본 목표는 **Window 전체를 확보한 뒤 시간별 이상을 판정하는 오프라인/사후 탐지**다.

### 7.3 입력 기간

- **[확정]** Decoder는 여러 날짜(예: 그제·어제·오늘)를 나란히 받아 일별 패턴을 비교한다.
- **[확정]** Zone/Type Encoder도 **Decoder와 같은 `T`일을 사용**하여 날짜별 Peer 그룹을 하나의 시퀀스로 Flatten한다.
- **[초기 설정]** 각 **지도학습 샘플마다** 입력 기간 `T`를 1~64일 사이의 정수로 균등 무작위 샘플링한다.

  \[
  T \sim \operatorname{Uniform}\{1,2,\ldots,64\}.
  \]

- **[확정]** 시작 날짜도 선택하여 **연속된 `T`일**을 입력한다. Decoder의 대상 건물과 Zone/Type Encoder의 Peer 건물들은 **완전히 같은 날짜 구간**을 사용한다.
- **[초기 설정]** PE는 매 Window 내 날짜 상대 인덱스 `d=1,...,T`를 사용하며, 같은 날짜의 모든 건물에 동일한 날짜 PE를 더한다.
- **[구현 주의]** 배치 내 `T`가 다르면 날짜 Padding과 유효 Token Attention Mask를 사용하거나, 같은 길이끼리 Batch를 구성한다. **이상 레이블이 있는 날짜도 샘플에 유지**한다. 미관측 `meter_reading`은 **V1에서는 0 대체, V2에서는 건물별 중앙값 대체**로 처리한다.
- **[제안]** 평가할 때는 `T ∈ {1, 7, 14, 32, 64}` 등 고정된 길이별 성능을 비교한다. 긴 `T`일수록 Encoder Token 수(`T×Z`, `T×U`)가 커져 Self-Attention 비용이 증가한다.

---

## 8. 구현 구성안

```text
project/
├── SPEC.md
├── configs/
│   └── train.yaml
├── src/
│   ├── data/
│   │   ├── lead_loader.py           # CSV 로딩·건물/시간 정렬
│   │   ├── day_patches.py           # 시간별 -> 24시간 Day Vector
│   │   ├── peer_groups.py           # Zone/Type 날짜 동기화 그룹 구성
│   │   └── splits.py               # Building-aware CV
│   ├── models/
│   │   ├── patch_embedding.py      # Linear(24,D)
│   │   ├── set_encoder.py          # Joint T-day Zone/Type Encoder + Shared Day PE
│   │   ├── temporal_decoder.py     # Masked Self-Attn+FFN, Zone Cross+FFN, Type Cross+FFN
│   │   └── hourly_head.py          # Linear(D,24)
│   ├── train/
│   │   └── train_supervised.py     # 전체 모델 End-to-End 지도학습
│   └── eval/
│       ├── metrics.py
│       └── kaggle_submission.py
└── tests/
    ├── test_shapes.py
    ├── test_causal_mask.py
    ├── test_encoder_permutation.py
    ├── test_no_id_input.py
    ├── test_split_leakage.py
    └── test_submission_shape.py
```

### 8.1 초기 Config [초기 설정 + 일부 제안/미정 옵션]

```yaml
seed: 42
model:
  d_model: 128
  n_heads: 4
  head_dim: 32                      # d_model / n_heads
  d_ff: 512                         # 4 × d_model
  zone_encoder_layers: 2
  type_encoder_layers: 2
  decoder_layers: 2
  dropout: 0.1
  layer_norm: pre                   # [제안] Pre-LN 구현
  patch_size_hours: 24
  decoder_window_days: variable     # T를 샘플마다 1~64일 선택
  id_lookup_embedding: false
  encoder_positional_encoding: shared_sinusoidal_day
  decoder_positional_encoding: sinusoidal_day
  encoder_joint_days: true
  encoder_tokens_layout: day_major
  target_first_per_day: true
  encoder_self_attention_temporal_mask: none  # 오프라인 Window 전체 참조
  decoder_causal_mask: true
  cross_attention_temporal_mask: none          # 오프라인 Window 전체 참조
  cross_attention_fusion: sequential
  cross_attention_order: [zone, type]          # 반대 순서는 Ablation
  ffn_after_each_attention: true
  exclude_target_from_peers: false
data:
  missing_value_strategy: zero_fill       # MVP V1: NaN -> 0 (Attention Mask 아님)
  missing_value_ablation: building_median # V2: 건물별 유효 관측 중앙값
  preserve_observation_mask: true         # 전처리/진단용
  observation_mask_as_feature: false      # MVP 모델 입력에 결측 플래그 미전달
  missing_reading_loss_policy: to_be_decided
train:
  mode: supervised_end_to_end
  objective: hourly_binary_cross_entropy
  window_length_sampling: uniform_integer
  min_window_days: 1
  max_window_days: 64
  contiguous_window: true
  match_encoder_decoder_dates: true
  loss_on: all_valid_hours
  normal_only: false
  optimizer: adamw                  # [제안]
  output_hours_per_day: 24
eval:
  metric_primary: roc_auc
  metrics_secondary: [f1, precision, recall, pr_auc]
  split_unit: building_id
  peer_pool: train_only                   # 평가 건물 Peer는 학습 건물에서 매칭 (within_split / with_train은 비교 옵션)
  inference_window_aggregation: to_be_decided
  kaggle_submission: row_id_level
```

**핵심 초기값:** `d_model=128`, `n_heads=4`, `d_ff=512`, **Zone/Type Encoder 각 2층 + Decoder 2층**, `dropout=0.1`, `T ∼ Uniform{1,…,64}`. 모든 Attention 뒤에 FFN을 사용하며 **Zone → Type Cross-Attention을 순차 적용**한다. Train/Test에서 전체 Encoder+Decoder를 유지하며 시간별 이상을 직접 학습·예측한다. 결측값 대체의 **기본 구현은 V1(0 대체)**이며 **V2(건물별 중앙값 보정)**를 비교한다. 결측 시간 Loss 처리, Overlapping Window 평가 집계, 온라인 확장 여부는 미정이다.

---

## 9. 실험 및 Ablation Plan [제안]

| ID | 실험 | 목적 |
|---|---|---|
| E0 | LEAD 우승 솔루션 재현 | 기존 지도학습 기준 성능 확보 |
| E1 | Decoder-only Supervised | Zone/Type Context 없는 기준선 |
| E2 | Decoder + Zone Encoder | Zone Context의 기여 평가 |
| E3 | Decoder + Type Encoder | Type Context의 기여 평가 |
| E4 | **Decoder + Zone + Type (전체 모델)** | 제안 아키텍처의 기본 성능 |
| E5 | Cross-Attention 순서 **Zone → Type vs Type → Zone** | 순차 결합 순서의 효과 |
| E6 | 가변 `T ∈ [1,64]` 학습 및 `T=1/7/14/32/64` 평가 | 입력 길이별 이상 탐지 성능과 Attention 비용 |
| E7 | Peer 순서 셔플·Context 제거 | 집합 순서 및 관계 학습 기여 검증 |
| E8 | Encoder 날짜 공유 PE 사용/제거 | 날짜 정보가 관계 학습에 주는 효과 |
| E9 | Layer depth: 1–1–1 / 2–2–2 / 2–2–3 / 3–3–3 | 모델 깊이/복잡도 비교 |
| E10 | `D`: 64 / 128 / 256 | 표현 차원의 효과 |
| E11 | **Missing Value V1(0 대체) vs V2(건물별 중앙값 대체)** | 이상 탐지 ROC-AUC, 결측률별 성능, Zero-fill Shortcut 가능성 비교 |
| E12 | Hourly Anomaly Head 구조 비교 | `Linear(D,24)` vs 원본 24시간 입력을 결합한 Head의 시간별 판별 성능 |

### 9.1 평가지표

- **Kaggle 공식:** ROC-AUC (Hour-level).
- **추가:** Hour-level F1, Precision, Recall, PR-AUC.
- **일반화:** 미학습 건물 기준 성능을 별도 보고한다.
- **자원 성능:** 모델 파라미터 수, 추론 시간, 모델 크기 및 그룹 규모(`T×Z`, `T×U`)에 따른 메모리 비용 측정.

---

## 10. 필수 검증 테스트 / Acceptance Criteria

- [ ] `target_daily.shape == [B,T,24]`, Patch Output `[B,T,D]`, Zone/Type Encoder 입력은 `[B,T×Z,D]`, `[B,T×U,D]`이다.
- [ ] 매 학습 샘플에서 `T`를 1~64 정수로 Uniform Sampling하고 Encoder/Decoder는 동일한 연속 날짜 구간을 사용한다. Padding은 Attention 및 Loss에서 제외한다.
- [ ] 각 Zone/Type 날짜 묶음의 첫 Token은 대상 건물의 동일 날짜 원본 24시간 전력량을 투영한 값이다.
- [ ] 같은 날짜의 건물 Token에 동일한 PE, 다른 날짜에 서로 다른 PE가 적용된다.
- [ ] Decoder 각 Layer가 **Causal Self-Attn→FFN→Zone Cross-Attn→FFN→Type Cross-Attn→FFN** 순서를 따른다.
- [ ] 초기 설정은 **Zone 2층 / Type 2층 / Decoder 2층**, `d_model=128`, `n_heads=4`, `head_dim=32`, `d_ff=512`, `dropout=0.1`이다.
- [ ] **Input-side Building ID Lookup Embedding이 없고** 이상 레이블 또는 라벨 기반 Mask가 입력 Feature로 들어가지 않는다.
- [ ] Train에서는 `[B,T,24]` Logit에 대해 시간별 라벨로 Loss를 계산한다. **이상이 발생한 시간/날짜도 학습 샘플에 포함**한다.
- [ ] Test에서도 Encoder/Decoder **전체가 작동**하고 `[B,T,24]` Score를 생성한다.
- [ ] 날짜 `d`의 Decoder **Causal Self-Attention 단독**에서는 미래 날짜를 참조하지 않는다. 단, 전체 모델은 Encoder 때문에 **오프라인/비인과적 모델**임을 테스트/문서에 명시한다.
- [ ] 같은 날짜의 Peer 입력 순서를 섞어도 마스크를 올바르게 동반한 경우 최종 Prediction이 동일하다(수치 오차 범위).
- [ ] Train/Validation 건물의 레이블이 분리되고, Test 레이블 없이 그룹 Context와 예측값을 구성한다.
- [ ] **V1에서는 NaN을 0으로만 대체**, **V2에서는 건물별 유효 관측 중앙값으로 대체**하며, 둘 다 입력 NaN이 없고 `test.csv`의 모든 제출 대상 `row_id`에 점수 1개가 매핑된다. `observation_mask`를 입력 Feature로 넘기거나 결측 시간을 Attention Mask로 제거하지 않는다.
- [ ] 중복 Window가 동일 행에 점수를 생성하면 사전에 정의한 규칙에 따라 하나로 통합한다.
- [ ] 최종 Kaggle 제출 파일이 시간별 행 수와 출력 형식에 일치한다.

---

## 11. 아직 확정되지 않은 핵심 설계 결정

1. **입력 기간 `T`의 최적화:** 기본은 1~64일 Uniform Sampling으로 시작하며, 성능에 따라 최대 길이와 Sampling 분포 조정.
2. **실제 결측값 처리:** 기본 구현 V1(0 대체)과 비교 실험 V2(건물별 중앙값 대체)는 정했으나, **더 나은 방식·결측 시간의 Loss 처리·관측 여부 Mask를 Feature로 추가할지**는 미정. Missingness Shortcut을 검증한다.
3. **중복 Window 추론:** 동일 `row_id`에 여러 Window가 대응할 때 최종 시간별 점수 선택·집계 방식.
4. **시간별 이상 탐지 Head:** 초기 `Linear(D,24)`와 원본 24시간 입력을 결합하는 Head 등 추가 구조 비교.
5. **평가 및 운영 범위:** 현재는 Encoder가 Window 전체를 보는 **오프라인/사후 탐지**. 미래 정보를 사용하지 않는 실시간/온라인 탐지는 후속 확장 여부 결정.
6. **그룹 컨텍스트 설계:** Zone(`site_id`) / Type(`primary_use`)의 Proxy 적합성, Group이 작거나 데이터가 누락된 건물에 대한 처리.

**이미 확정한 항목:** Train/Test 동일 전체 모델, Hourly Supervised Loss, Zone → Type 순차 Cross-Attention, Attention마다 FFN, 날짜별 공유 PE, `T∈[1,64]` Uniform Sampling, **MVP 결측 0 대체 및 중앙값 보정 비교 버전**, **평가 건물 Peer는 학습 건물에서 매칭**. 위 항목들을 다시 미정으로 표시하지 않는다.

---

## 12. 본 설계의 한 문장 정의

> **하루 24시간 전력량을 1개의 Day Token으로 투영하고, 동일 Zone·Type 건물의 T일치 Token을 날짜별 공유 PE와 함께 두 개의 통합 Encoder에 넣어 관계를 학습한 다음, 대상 건물의 T일 시퀀스를 처리하는 Decoder에서 Zone → Type 순차 Cross-Attention으로 결합하고, 전체 모델을 시간별 이상 레이블로 End-to-End 지도학습하여 각 Day Token마다 24개의 이상 점수를 출력하는 Dual-Context Transformer.**
