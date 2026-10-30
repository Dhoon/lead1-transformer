# SPEC — LEAD1.0 기반 Dual-Context Transformer 이상 탐지 연구

> **문서 상태:** 연구 설계 초안 (Draft v0.2 — 초기 모델 깊이/차원 반영)  
> **기준일:** 2026-10-30  
> **주요 목표:** LEAD1.0 전력 소비 시계열을 활용하여, **Zone/Type 관계 및 일별 소비 패턴**을 학습하는 Transformer를 정상 데이터 기반으로 사전학습하고, 이후 전력 이상 탐지에 전이한다.  
> **표기:** **[확정]** = 대화에서 합의된 설계, **[초기 설정]** = 첫 실험에서 사용하기로 정한 하이퍼파라미터(최적 성능을 의미하지 않음), **[제안]** = 실험 옵션, **[미정]** = 후속 연구에서 결정할 사항.

---

## 1. 연구 배경과 범위

### 1.1 연구 맥락

- 상위 과제: **AI 기반 차세대 보안 전력 인프라 고도화**.
- 아주대학교 연구 범위 중 **Self-supervised Learning 기반 시계열 이상 탐지 모델** 개발을 담당한다.
- 실제 SMGW/전력망 보안 텔레메트리 데이터가 아직 충분하지 않으므로, 초기 알고리즘 연구·재현·벤치마킹에 **LEAD1.0 / Kaggle LEAD Competition** 데이터를 이용한다.
- 본 단계에서 탐지하는 것은 **전력 소비 패턴의 이상**이다. 이 데이터로 얻은 탐지 성능을 KCMVP 공격이나 비인가 접근에 대한 사이버 위협 탐지 성능으로 동일시하지 않는다.

### 1.2 참조 자료

1. Manoj Gulati, Pandarasamy Arjunan, *LEAD1.0: A Large-scale Annotated Dataset for Energy Anomaly Detection in Commercial Buildings* (2022). 업로드 파일: `2203.17256v1.pdf`.
2. Chun Fu, Pandarasamy Arjunan, Clayton Miller, *Trimming outliers using trees: Winning solution of the Large-scale Energy Anomaly Detection (LEAD) competition*, ACM BuildSys 2022. 업로드 파일: `2022-buildsys-lead-winning-solution.pdf`.
3. Kaggle LEAD Competition: <https://www.kaggle.com/competitions/energy-anomaly-detection>
4. LEAD1.0 데이터셋: <https://github.com/samy101/lead-dataset>
5. 우승 솔루션 공개 코드: <https://github.com/buds-lab/LEAD-1st-solution>

### 1.3 원 데이터와 대회 데이터의 구별

| 구분 | 내용 |
|---|---|
| LEAD1.0 전체 데이터셋 | 1,413개 전기 계량기 시계열, 2016년 시간별 계측값, 정상·이상 주석 |
| Kaggle LEAD 대회 | 학습용 200개 건물, 숨겨진 이상 레이블을 예측하는 테스트용 206개 건물 |
| 대회 태스크 | **시간별 정상/이상 이진 판정**, 공식 평가 ROC-AUC |
| 본 연구 Pretraining | 정상 소비 패턴으로부터 **Building ID 분류**를 수행하는 표현 학습 |
| 본 연구 Downstream | 학습한 표현을 **시간별 이상 탐지**에 전이 |

> LEAD1.0 원 논문의 24시간 구간 단위 비지도 벤치마크와 Kaggle 우승 논문의 시간별 지도학습 결과는 평가 단위·프로토콜이 달라 직접 비교하지 않는다.

---

## 2. 핵심 아이디어

**[확정]** 아키텍처는 **2개의 관계 Encoder + 1개의 시간 Decoder**로 구성한다.

1. **Zone Encoder**: 같은 Zone에 있는 여러 건물의 **동일 기간** 전력 소비 패턴을 입력받아 건물 간 공통·상대적 특성을 학습한다.
2. **Type Encoder**: 같은 건물 유형(`primary_use` 등)에 있는 건물의 **동일 기간** 전력 소비 패턴을 입력받아 유형별 공통·상대적 특성을 학습한다.
3. **Temporal Decoder**: 관심 건물 하나의 **하루 24시간 소비량을 1개 토큰**으로 만들어, 여러 날짜(그제·어제·오늘 등)의 순서 있는 시퀀스를 처리한다.
4. Decoder는 **Causal Masked Self-Attention**을 사용하며, 두 Encoder 출력과 **Cross-Attention**을 수행한다.
5. 사전학습 시 최종 출력은 **학습 대상 Building ID의 확률 분포**이며, 이후 이 분류 Head를 이상 탐지용 Head로 교체/확장한다.

### 2.1 기존 대화에서 정리된 핵심 수정

- **[확정] Input-side Building ID Lookup Embedding은 사용하지 않는다.**
- **[확정]** ViT의 Patch Embedding과 유사하게 하루의 24개 계측값을 `Linear(24, D)`로 직접 투영한다.
- **[확정]** 모델의 내부 표현 차원 `D`와 Building ID의 개수 `N`은 서로 독립적이다. 예: `D=128`, `N=200`.
- **[확정]** `building_id`는 데이터 정렬·그룹 구성·정답 Label로만 사용한다. 모델의 입력 Feature 또는 ID Lookup Embedding으로 전달하지 않는다.
- **[확정]** `Linear(D, N) + Softmax`를 통해 **모든 N개 건물에 대한 분류 확률**을 산출한다.
- **[확정]** ID 분류는 최종 태스크가 아닌 **Pretext Task**이며, 최종 목적은 Anomaly Detection이다.

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

Decoder 입력 시에만 날짜 순서를 나타내는 Positional Encoding을 더한다:

\[
\mathbf z_{b,d}=\mathbf e_{b,d}+PE(d).
\]

- **[초기 설정]** `D=128`을 사용한다. 200개 건물이라고 해서 `D=200`일 필요는 없다. `D=64/256`은 후속 비교 실험으로 둔다.
- 하루 24개 값의 시간별 위치는 **선형 투영의 24개 입력 인덱스**에 의해 구별된다. Decoder Positional Encoding은 **날짜 토큰들 사이의 순서**를 나타낸다.
- **[미정]** 결측 시간이나 하루 24시간 미만 구간의 처리 방법. 기본 구현은 Missing Mask를 함께 보관하고 결측 대체값을 사용한다.

### 3.2 Zone/Type 구분

- **[제안]** LEAD 데이터에서는 `site_id`를 **Zone의 대리 변수(proxy)**로 사용한다. `site_id`가 실제 지리적 Zone과 완전히 동일하다는 주장은 하지 않는다.
- **[제안]** `primary_use`를 Building Type으로 사용한다.
- 그룹을 구성하기 위한 메타데이터는 사용하되, **ID 자체를 모델에 입력하지 않는다.**
- **[미정]** 동일 시점의 의미: 동일 **24시간 날짜 창**을 기준으로 다른 빌딩을 묶는 구현을 **MVP 기본안**으로 삼는다. 당초 구상처럼 **매 시간(hour)마다 건물 집합을 인코딩**하는 해상도는 별도 실험 옵션으로 남긴다.

### 3.3 입력/출력 Tensor 규약 (MVP 기본안)

| Tensor | Shape | 의미 |
|---|---|---|
| `target_daily` | `[B, T, 24]` | 대상 건물 B개의 연속 T일 전력량 |
| `zone_daily` | `[B, T, Z, 24]` | 대상 건물과 동일 Zone에 있는 건물 Z개의 해당 날짜 전력량 |
| `type_daily` | `[B, T, U, 24]` | 대상 건물과 동일 Type에 있는 건물 U개의 해당 날짜 전력량 |
| `zone_mask` | `[B, T, Z]` | Zone 그룹 패딩 여부 |
| `type_mask` | `[B, T, U]` | Type 그룹 패딩 여부 |
| `day_valid_mask` | `[B, T, 24]` | 원래 관측된 시간값 여부 |
| `building_target` | `[B]` 또는 `[B, T]` | Pretraining 정답 Building ID 클래스 인덱스 |

- `B`: 배치 크기, `T`: Decoder 입력 날짜 수.
- `Z`, `U`: 배치별 최대 그룹 크기. 가변 집합 크기는 Padding + Attention Mask로 처리.
- `target_daily`에는 대상 건물 ID가 아니라 **전력 소비량 24개**만 들어간다.

---

## 4. 모델 아키텍처

### 4.1 전체 데이터 흐름

```text
                      SAME-DAY PEER CONTEXTS
        same-zone buildings              same-type buildings
          [Z, 24] each day                 [U, 24] each day
                 |                                |
           Linear(24,D)                       Linear(24,D)
                 |                                |
      Zone Transformer Encoder          Type Transformer Encoder
        (no positional encoding)        (no positional encoding)
                 |                                |
             [Z, D]                           [U, D]
                 |                                |
                 +-------- Cross-Attention --------+
                                    ^
                                    |
Target building days: [T,24] -> Linear(24,D) + Day PE
                                    |
                       Causal Self-Attention
                                    |
                   Decoder (day-level states [T,D])
                                    |
                    PRETRAIN: Linear(D,N)+Softmax
                      DOWNSTREAM: Anomaly Head
```

### 4.2 Zone Encoder

**[확정]** 동일 Zone, 동일 기간에 속한 복수 건물의 전력 패턴이 토큰 집합이 된다.

\[
H^{(z)}_d=\mathrm{Enc}_{\mathrm{zone}}\left(\{P(\mathbf x_{i,d})\mid i\in\mathcal Z(b,d)\}\right).
\]

- **건물 순서에 의미가 없으므로 Positional Encoding을 넣지 않는다.**
- Self-Attention으로 동시점 지역 내 건물들의 관계를 표현한다.
- Encoder의 Self-Attention은 **순열 등변(permutation-equivariant)**이며, Decoder가 동일하게 순열된 K/V에 Cross-Attention하면 출력은 그 순열에 무관하다.
- **[제안]** 대상 건물 `b`를 Peer Context에서 제외하는 버전을 초기 설정으로 두고, 포함하는 버전과 비교한다. 이는 원 설계에서 확정한 사항이 아닌 검증용 선택이다.

### 4.3 Type Encoder

\[
H^{(u)}_d=\mathrm{Enc}_{\mathrm{type}}\left(\{P(\mathbf x_{i,d})\mid i\in\mathcal U(b,d)\}\right).
\]

- **Positional Encoding 없음.**
- 동일 건물 유형 간 소비 패턴의 공통성 및 차이를 모델링한다.
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

### 4.5 두 Encoder와 Cross-Attention

날짜 `d`에 해당하는 Decoder 상태를 Query로, 같은 날짜의 Zone/Type Encoder 출력을 각각 Key/Value로 사용한다.

\[
C^{(z)}_{b,d}=\operatorname{CrossAttn}(Q=h_{b,d},K=H^{(z)}_d,V=H^{(z)}_d)
\]

\[
C^{(u)}_{b,d}=\operatorname{CrossAttn}(Q=h_{b,d},K=H^{(u)}_d,V=H^{(u)}_d).
\]

- **[제안]** 첫 구현은 **병렬 Cross-Attention 후 결합**한다. 예: `LayerNorm(h + W_fuse [C_zone ; C_type])`.
- **[미정]** 두 Cross-Attention을 순차 적용할지, 병렬 융합할지, Learned Gate를 사용할지.
- **[제안]** 동일 날짜 Context만 참조하여 후속 날짜의 정보가 과거 날짜 판정에 유입되지 않도록 한다. 과거의 Zone/Type 정보까지 모으는 확장 구조는 별도 검증한다.
- **[제안]** Peer 그룹이 비면 빈 Attention을 수행하지 않고 학습 가능한 일반 `NULL_CONTEXT`를 사용한다. 이 벡터는 Building ID를 인코딩하지 않는다.

### 4.6 Transformer 레이어 구성 및 초기 하이퍼파라미터

**[초기 설정]** 합의한 첫 번째 구현은 **Zone Encoder 2층 + Type Encoder 2층 + Temporal Decoder 2층(2–2–2)** 구조다. 이는 우선 실험할 **기본 모델**이며, 최적 레이어 수가 검증되었다는 의미는 아니다.

| 구성요소 | 초기 값 | 한 층의 구성 |
|---|---:|---|
| Zone Encoder | **2 layers** | Multi-Head Self-Attention → FFN |
| Type Encoder | **2 layers** | Multi-Head Self-Attention → FFN |
| Temporal Decoder | **2 layers** | Causal Self-Attention → Zone Cross-Attention → Type Cross-Attention → FFN |
| Hidden Dimension `d_model` | **128** | Day Token 및 Attention의 기본 표현 차원 |
| Attention Heads `n_heads` | **4** | Head당 차원 `128/4 = 32` |
| FFN Hidden Dimension `d_ff` | **512** | `4 × d_model` |
| Dropout | **0.1** | Attention/FFN의 일반화 정규화 시작값 |
| Pretraining 출력 클래스 `N` | **200** | 전체 200개 빌딩을 학습에 사용할 때 `Linear(128, 200)` |

- Zone/Type Encoder는 동일 기간의 **건물 간 관계**를 학습하므로 각각 2층을 초기값으로 두고, Decoder는 날짜 간 관계와 두 종류의 Cross-Attention을 결합하므로 2층에서 출발한다.
- Decoder **각 층에서** 두 Encoder에 대한 Cross-Attention을 수행하는 구성이 초기안이다. 두 Cross-Attention의 결합 방식(병렬/순차/Gate)은 4.5절의 **[미정]** 항목이며, 구체적 실행 시 현재 Config의 `parallel`을 우선 적용한다.
- 모든 Attention과 FFN에는 Residual Connection과 Layer Normalization을 사용한다. **[제안]** 첫 구현의 LayerNorm 배치는 Pre-LN으로 둔다.
- `N`은 분류할 빌딩 수, `D`는 내부 표현 차원이므로 별개다. 학습 빌딩이 달라질 때 `N`은 변경될 수 있지만 `D=128`은 유지할 수 있다.

**[제안] 레이어 깊이 Ablation**

| 모델 | Zone Encoder | Type Encoder | Decoder | 목적 |
|---|---:|---:|---:|---|
| 경량 | 1 | 1 | 1 | 작은 모델의 기준점 |
| **기본 모델** | **2** | **2** | **2** | 첫 구현 및 기본 비교군 |
| Decoder 확장 | 2 | 2 | 3 | 날짜 간 패턴 학습을 더 깊게 했을 때의 효과 |
| 전체 확장 | 3 | 3 | 3 | 전체 깊이 증가의 효과 |

우선순위는 **2–2–2 vs 2–2–3** 비교이며, 이후 필요에 따라 **1–1–1**, **3–3–3**을 추가한다. 깊이뿐 아니라 모델 표현 차원의 영향을 확인하기 위해 `D ∈ {64, 128, 256}`도 별도 Ablation으로 검토한다.

---

## 5. 사전학습 (Pretraining)

### 5.1 학습 목적

**[확정]** 정상적인 일별 전력 소비 패턴을 입력하고 **어느 Building ID에 속하는지** 분류한다.

- 정상 데이터만 사용하는 것이 연구 의도다.
- Building ID는 **정답 레이블**이며, Input-side ID Embedding은 사용하지 않는다.
- 같은 건물의 서로 다른 날들은 **같은 Building 클래스**로 학습된다.

날짜별 Decoder 표현 `h[b,d] ∈ R^D`에서:

\[
\mathbf z_{b,d}=W_{\mathrm{id}}\mathbf h_{b,d}+\mathbf c,
\qquad W_{\mathrm{id}}\in\mathbb R^{N\times D}.
\]

\[
P(i\mid b,d)=\operatorname{Softmax}(\mathbf z_{b,d})_i,
\qquad i=1,\ldots,N.
\]

\[
\mathcal L_{\mathrm{ID}}=-\frac1{|\mathcal M|}\sum_{(b,d)\in\mathcal M}\log P(b\mid b,d).
\]

- **[제안]** 유효한 모든 날짜 Token에 Classification Loss를 적용한다. 비교 대상으로 마지막 날짜 Token만 분류하는 버전도 둔다.
- `N`은 **현재 사전학습 Fold에서 사용하는 학습 건물의 수**다. Kaggle 학습용 200개를 모두 쓸 경우 `N=200`.
- `D=128`, `N=200`이면 분류기는 단순히 `Linear(128,200)`이다.

### 5.2 “모든 빌딩에 Loss를 뿌린다”의 수학적 의미

정답 Building이 `k`일 때,

\[
\frac{\partial\mathcal L}{\partial z_i}=P(i)-\mathbf 1[i=k].
\]

즉, **단일 Cross-Entropy Loss에서도 전체 `N`개 클래스의 Logit 및 출력 분류 가중치에 Gradient가 전달된다.**

정확히는 모든 Building마다 서로 독립적인 Loss를 계산하는 것은 아니다. 또한 **현재 입력하지 않은 다른 건물의 입력 토큰을 직접 갱신한다는 의미도 아니다.** 공유 Backbone과 모든 출력 클래스의 가중치가 학습되는 것이다.

### 5.3 Shortcut Learning 통제

- **[확정]** 대상 Building ID의 고유 Lookup Embedding은 Decoder/Encoder 입력에 넣지 않는다.
- `building_id`를 숫자 Feature로 추가하지 않는다.
- `site_id`, `primary_use`는 Peer 집합 구성에만 사용한다. **[제안]** Zone/Type ID를 직접 입력하는 옵션은 초기에 끈다.
- 건물 그룹 구성만으로도 분류에 유리한 정보를 얻을 수 있으므로 **Context-only**, **Target-only**, **Context permutation** 비교 실험을 수행한다.
- 사전학습의 Building ID 정확도가 높다고 곧바로 이상 탐지 표현이 우수하다고 판단하지 않는다.

---

## 6. 이상 탐지 Downstream

### 6.1 원칙

- **[확정]** Building ID 분류는 Pretraining에만 사용한다.
- **[확정]** 이후 Encoder/Decoder Representation을 재사용해 Anomaly Detection을 수행한다.
- **[미정]** 최종 Downstream 학습이 완전 비지도/정상 전용인지, 일부 레이블을 활용하는 Fine-tuning인지와 Anomaly Scoring 방법은 아직 정해지지 않았다.

### 6.2 초기 구현 제안: Hour-level Classification Head

Kaggle의 **시간별 이상 여부**와 평가 단위를 맞추기 위한 **[제안]** 기본선:

\[
\mathbf a_{b,d}=W_a\mathbf h_{b,d}+\mathbf c_a\in\mathbb R^{24},
\qquad \widehat{\mathbf y}_{b,d}=\sigma(\mathbf a_{b,d}).
\]

- Day Token 표현 하나에서 **그 날짜의 24개 시간별 이상 확률**을 출력한다.
- 정답 레이블은 `y[b,d,h] ∈ {0,1}`.
- 레이블을 쓰는 경우: `BCEWithLogitsLoss`(클래스 불균형을 고려한 가중치 검토).
- Pretraining의 `Linear(D,N)` 헤드는 제거하거나 Freeze된 비교용으로만 보관한다.
- **주의:** Day Token 기반 24개 출력의 시간별 위치 식별 성능은 실험으로 검증해야 한다. 필요하면 입력 원시 전력량을 연결하는 Hour-level Skip/Head를 별도 Ablation으로 검토한다.

### 6.3 정상 전용 학습 대안 [미정]

프로젝트 계획서의 Self-supervised / 정상 학습 방향을 엄격하게 유지하려면, 이상 레이블을 사용한 Classification Head 외에 다음 대안의 필요성을 평가한다.

- 정상 시계열 복원 또는 마스킹 구간 복원 기반 점수.
- 정상 Representation의 분포/거리 기반 이상 점수.
- 정상 구간에 대해 학습한 예측 또는 Consistency 기반 이상 점수.

이 대안들의 세부 Objective와 점수 결합식은 **현재 확정된 설계가 아니다**. 단순히 하루 전체를 입력하고 같은 하루를 재구성하면 정보 복사 문제가 생길 수 있으므로 입력 마스킹 또는 정보 병목을 명시적으로 설계해야 한다.

---

## 7. 데이터 전처리 및 Split

### 7.1 입력 Feature 정책

- **[확정] MVP의 핵심 입력은 시간별 `meter_reading`을 모아 만든 24차원 Day Vector**다.
- **[제안]** `log1p(meter_reading)`과 학습 구간에서 계산한 정규화 파라미터를 사용한다.
- **[제안]** 결측 데이터의 원본 유효성 Mask를 보관하고, Leakage가 없는 대체 정책을 적용한다.
- **[미정]** 날씨·캘린더·건물 규모 등의 외생 Feature를 추가할지 여부. 첫 실험에는 추가하지 않아 모델 자체의 효과를 측정한다.
- **[주의]** LEAD 우승 논문의 `value_chg_diff/ratio` 및 미래 시점 Shift Feature는 본 모델의 필수 입력이 아니며, 비교 Baseline에서만 사용한다. 미래 시점 참조는 실시간 운영 제약과 충돌할 수 있다.

### 7.2 학습·검증 전략

**[제안]** 서로 다른 실험을 다음과 같이 구분한다.

1. **Pretraining 내부 검증:** ID 분류 Head가 다룰 수 있는 **동일한 학습 건물 집합** 내에서 날짜/기간을 나눠 ID 분류 학습 상태를 검증한다.
2. **Downstream 일반화 검증:** Building ID 기준으로 Train/Validation 건물을 분리한다(`GroupKFold` 또는 Group holdout). 검증 건물의 ID는 사전학습 분류 클래스에 포함시키지 않는다.
3. **공식 Kaggle 테스트:** 최종적으로 대회 학습용 200개 건물로 학습하고 테스트용 206개 건물에 대한 시간별 이상 확률을 예측해 제출한다. 공개 테스트 레이블은 전제하지 않는다.

예시: 5-Fold Group CV에서는 각 Fold의 160개 Training Building으로 ID 사전학습을 수행하므로 ID Head의 출력 크기는 **200이 아닌 160**이다. 이후 나머지 40개 빌딩은 이상 탐지 일반화 검증에 사용한다.

- 스케일러 및 임계값 튜닝에 검증/테스트 정답을 이용하지 않는다.
- **[미정]** 추론 시 미학습 건물의 비라벨 전력량을 Peer Context로 사용하는 Transductive 설정을 허용할지. 공정한 비교를 위해 이 정책을 실험별로 기록한다.
- 특정 날짜의 판정은 그 날짜까지 관측된 데이터만으로 가능하도록 설정한다. **하루 24시간을 입력으로 사용하므로 실시간 시간 단위 판정보다는 하루 종료 후 배치 분석을 우선 가정한다.**

### 7.3 입력 기간

- **[확정]** Decoder는 여러 날짜(예: 그제·어제·오늘)를 나란히 받아 일별 패턴을 비교한다.
- **[미정]** Decoder의 `T`는 고정값이 아니다.
- **[제안]** 초기 실험: `T ∈ {3, 7, 14, 28}`로 비교하고, 일주일 주기성을 고려한 7일을 출발점으로 삼는다.

---

## 8. 구현 구성안

```text
project/
├── SPEC.md
├── configs/
│   ├── pretrain.yaml
│   └── downstream.yaml
├── src/
│   ├── data/
│   │   ├── lead_loader.py           # 데이터 로딩·건물/시간 정렬
│   │   ├── day_patches.py          # hourly -> 24-hour Day Vector
│   │   ├── peer_groups.py          # Zone/Type 동기식 그룹 구성
│   │   └── splits.py               # Group-aware CV 및 기간 분리
│   ├── models/
│   │   ├── patch_embedding.py      # Linear(24,D)
│   │   ├── set_encoder.py          # Zone/Type Encoder (no PE)
│   │   ├── temporal_decoder.py     # Causal Self-Attn + Cross-Attn
│   │   └── heads.py                # ID 분류 / 이상 탐지 Head
│   ├── train/
│   │   ├── pretrain.py
│   │   └── finetune.py
│   └── eval/
│       ├── metrics.py
│       └── kaggle_submission.py
└── tests/
    ├── test_shapes.py
    ├── test_causal_mask.py
    ├── test_encoder_permutation.py
    ├── test_no_id_input.py
    ├── test_no_future_leakage.py
    └── test_submission_shape.py
```

### 8.1 초기 Config [초기 설정 + 일부 제안/미정 옵션]

```yaml
seed: 42
model:
  d_model: 128
  n_heads: 4
  head_dim: 32                     # d_model / n_heads
  d_ff: 512                        # 4 × d_model
  zone_encoder_layers: 2
  type_encoder_layers: 2
  decoder_layers: 2
  dropout: 0.1
  layer_norm: pre                  # [제안] Pre-LN 구현
  patch_size_hours: 24
  decoder_window_days: 7
  id_lookup_embedding: false
  encoder_positional_encoding: false
  decoder_positional_encoding: true
  decoder_causal_mask: true
  cross_attention_fusion: parallel
  exclude_target_from_peers: true
pretrain:
  objective: building_id_cross_entropy
  normal_only: true
  loss_on: all_valid_days
  num_classes: auto_from_training_buildings
  optimizer: adamw
downstream:
  mode: supervised_hourly_head  # [제안] 아직 확정되지 않음
  output_hours_per_day: 24
  metric_primary: roc_auc   # Kaggle 공식 지표
  metrics_secondary: [f1, precision, recall, pr_auc]
```

**`d_model=128`, `n_heads=4`, `d_ff=512`, Zone/Type Encoder 각 2층, Decoder 2층, `dropout=0.1`은 이번 논의에서 선정한 초기 실험 설정이다.** 다만 검증된 최적값은 아니며 레이어 깊이·D는 Ablation을 수행한다. `decoder_window_days=7`, Cross-Attention 융합법, Peer 포함 정책, 학습 Objective의 세부 구현 및 Optimizer는 여전히 제안/미정 사항이다.

---

## 9. 실험 및 Ablation Plan [제안]

| ID | 실험 | 목적 |
|---|---|---|
| E0 | LEAD 우승 솔루션 재현 | 공식 대회형 지도학습 비교 기준 확보 |
| E1 | Temporal Decoder only, ID Pretraining | 일별 시간 표현의 단독 기여 확인 |
| E2 | Temporal + Zone Encoder | 지역 Context 기여 확인 |
| E3 | Temporal + Type Encoder | 유형 Context 기여 확인 |
| E4 | Temporal + Zone + Type | 제안 모델 전체 성능 확인 |
| E5 | Without ID Pretraining | 사전학습의 실제 효용 검증 |
| E6 | Day Token 길이 T 변경 | 장기·주간 패턴의 영향 확인 |
| E7 | Peer Context 제외/포함·셔플 | 관계 학습의 유효성 및 Shortcut 가능성 확인 |
| E8 | Hour-level Encoder Context | 동일 시각(hour) 집합과 동일 일(day) 집합 비교 |
| E9 | Layer depth: 1–1–1 / 2–2–2 / 2–2–3 / 3–3–3 | Encoder/Decoder 깊이 영향과 복잡도 비교 |
| E10 | `D`: 64 / 128 / 256 | 내부 표현 차원에 따른 성능·자원 비용 비교 |

### 9.1 평가지표

- **Kaggle 공식:** ROC-AUC (Hour-level).
- **추가:** Hour-level F1, Precision, Recall, PR-AUC.
- **과제 목표 연계:** 아주대 계획서의 서버 이상 탐지 **F1 Score 90% 이상**을 목표로 참고하되, LEAD 대회 원 데이터·라벨 정의·평가 프로토콜과 동일한 목표라고 가정하지 않는다.
- **일반화:** 미학습 건물 기준 성능을 별도 보고한다.
- **자원 성능:** 모델 파라미터 수, 추론 시간, 모델 크기 측정. 향후 SMGW Knowledge Distillation 적용 시 경량 모델 성능 저하를 별도로 검증한다.

---

## 10. 필수 검증 테스트 / Acceptance Criteria

- [ ] `target_daily.shape == [B,T,24]`, Patch Output `[B,T,D]`.
- [ ] Pretraining Logits는 `[B,T,N]`(또는 마지막 날짜만 사용하는 경우 `[B,N]`).
- [ ] `D=128`, `N=200`을 독립적으로 설정해 정상 실행된다.
- [ ] 초기 모델이 **Zone 2층 / Type 2층 / Decoder 2층**, `n_heads=4`, `head_dim=32`, `d_ff=512`, `dropout=0.1`로 구성된다.
- [ ] **입력 Pipeline에 Building ID Lookup Embedding이 없다.** 정답 `building_target`은 Loss 계산에만 전달된다.
- [ ] Building 1의 Cross-Entropy가 200개 출력 클래스 모두의 Logit에 Gradient를 발생시킨다(수치 오차 범위 내).
- [ ] Decoder 날짜 `d`의 상태는 `d+1` 이후 날짜 입력을 변경해도 변하지 않는다(Causal Mask 테스트).
- [ ] Zone/Type Encoder의 건물 입력 순서를 섞어도, 유효한 마스크를 적용한 최종 Prediction이 바뀌지 않는다(순열 테스트).
- [ ] 동일 시점 Zone/Type Context로부터 미래 날짜 정보가 유입되지 않는다.
- [ ] 학습과 검증 건물의 분리를 확인하고, 데이터 정규화/튜닝 단계의 누수를 검사한다.
- [ ] Downstream의 Hour-level Prediction이 건물·날짜·시간으로 원래 Row에 정확히 매핑된다.
- [ ] 공식 제출 형식에 맞는 1시간 단위 이상 확률을 산출한다.

---

## 11. 아직 확정되지 않은 핵심 설계 결정

1. **Zone/Type Encoder 입력의 시간 해상도:** 동일한 하루의 24시간 벡터를 건물별 토큰으로 묶을지(MVP 기본안), 동일한 1시간의 빌딩별 스칼라를 묶을지(초기 구상의 엄밀한 hour-wise 구현).
2. **대상 건물을 Peer Encoder의 집합에 포함할지:** 두 구성을 비교하되, 초기 구현은 자기 자신 제외.
3. **두 Cross-Attention 결합 방식:** 병렬 융합 / 순차 적용 / Gating.
4. **사전학습 손실 부여 위치:** 매 날짜의 Decoder 상태 / 마지막 날짜 상태.
5. **Downstream Objective:** 시간별 레이블 활용 분류 / 정상 데이터 기반 점수 / 이들의 조합.
6. **운영 모드:** 하루 종료 후 배치 탐지와 추후 시간 단위 온라인 탐지의 구분.
7. **테스트 건물의 비라벨 측정값을 Peer Context로 활용할 수 있는지에 대한 평가 프로토콜.**

---

## 12. 본 설계의 한 문장 정의

> **하루 24시간 전력 소비량을 1개 Day Token으로 투영하고, 같은 Zone·같은 Type 건물의 동시점 패턴을 각각 별도의 순서 없는 Encoder로 학습한 뒤, 여러 날짜의 소비 패턴을 Causal Decoder로 통합하여 Building ID Classification으로 사전학습하고, 학습된 표현을 시간별 전력 이상 탐지에 전이하는 Dual-Context Transformer.**

