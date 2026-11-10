# LEAD-Transformer MVP (2026-11-10 기준)

현재 기본 모델(**v3_perday**)의 구조, 설정, 실행 방법, 성능을 한 문서에 모은다.
- 설계 원문: [SPEC.md](SPEC.md)
- 변경 명세: [LEAD_Encoder_CrossAttention_Change.md](LEAD_Encoder_CrossAttention_Change.md)
- 실험 기록: [result_1104.md](result_1104.md) → [result_1106.md](result_1106.md) → [result_1110.md](result_1110.md)

## 한 줄 정의

하루 24시간 전력량을 Token 하나로 만든다. 대상 건물의 T일 Token 열을 Decoder에 넣는다. 같은 날 같은 Zone·Type 건물들의 Token을 날짜별 Encoder로 요약한다. 이 요약을 Cross-Attention으로 참조해 시간별 이상 확률 24개를 낸다.

## 1. 문제와 데이터

- 데이터: Kaggle LEAD 1.0
  - Train: 200개 건물, Test: 206개 건물
  - 두 집합의 날짜 범위는 같고(366일), 건물은 겹치지 않는다.
- 목표: 시간별 이상 여부(`anomaly`)를 예측한다. 이상 비율은 약 2%다.
- 대회 지표: 시간 단위 ROC-AUC. 내부 검증에서는 PR-AUC, F1, 최고 F1도 함께 본다.

## 2. 구조

### 2.1 전체 흐름

Encoder 두 개가 주변 건물(Peer) 정보를 날짜별로 요약한다. Decoder는 대상 건물의 T일을 읽으면서 이 요약을 참고한다.

```text
  같은 Zone 건물들의 T일          같은 Type 건물들의 T일
           │                               │
           ▼                               ▼
   ┌───────────────┐               ┌───────────────┐
   │ Zone Encoder  │               │ Type Encoder  │
   └───────┬───────┘               └───────┬───────┘
           │ 날짜별 Zone 요약               │ 날짜별 Type 요약
           │ [B, T, D]                      │ [B, T, D]
           │                               │
           │    대상 건물의 T일             │
           │          │                    │
           │          ▼                    │
           │  ┌─────────────────┐          │
           └─►│     Decoder     │◄─────────┘
              │  (Cross-Attn로  │
              │   두 요약 참조) │
              └────────┬────────┘
                       ▼
                ┌─────────────┐
                │ Hourly Head │
                └──────┬──────┘
                       ▼
          대상 건물 T일 × 24시간 이상 확률
```

- Day Token: 하루 24시간 전력량을 `Linear(24, D=128)`로 바꾼 벡터 하나다. Encoder와 Decoder가 같은 Linear를 공유한다.
- Zone: 같은 site의 건물들이다. Type: 같은 용도(primary use)의 건물들이다.
- Encoder와 Decoder는 같은 T일 구간을 본다.

### 2.2 공통 단위 블록 (Norm·Add)

원 논문 Transformer 그림의 "Add & Norm" 상자는 Attention **뒤**에 Norm이 있다(Post-LN). 이 모델은 Norm이 Attention **앞**에 있는 Pre-LN(`layer_norm: pre`)이다. 아래 Encoder·Decoder 그림의 `Norm → ○○ → Add` 한 줄은 다음 순서를 뜻한다.

**Encoder 한 층을 풀어 쓴 예**

```text
          x
          │
          ├────────────────┐
          ▼                │
     LayerNorm             │
          ▼                │ x를 그대로 넘김
   Self-Attention          │
          ▼                │
         (+) ◄─────────────┘   Add  ← x + Attention 결과
          │
          ├────────────────┐
          ▼                │
     LayerNorm             │
          ▼                │ 위 결과를 그대로 넘김
         FFN               │
          ▼                │
         (+) ◄─────────────┘   Add  ← 위 결과 + FFN 결과
          │
       다음 층
```

즉 `LayerNorm → Attention → Add → LayerNorm → FFN → Add` 순서다. Post-LN(원 논문)은 `Attention → Add → LayerNorm → FFN → Add → LayerNorm`이다. 두 방식의 차이는 LayerNorm이 Attention·FFN의 앞에 오느냐 뒤에 오느냐뿐이다.

- 옆으로 넘기는 Residual 경로에는 Norm이 없다. 층을 지날수록 값이 커질 수 있어서, Encoder와 Decoder 맨 끝에 최종 Norm을 한 번 더 둔다.

### 2.3 Encoder (Zone, Type 각각)

**하는 일**: 날짜마다 "그날 대상 건물이 주변 건물들과 비교해 어떤가"를 벡터 하나로 요약한다.

```text
  입력: 날짜마다 [대상 건물, Peer 1, Peer 2, ..., Peer G-1]의 Day Token
        → 날짜를 Batch 쪽으로 펼쳐 [B×T, G, D]
           │
   + Role Embedding (대상 / Peer 구분)
           │
     ┌─────┴──────────────────────┐
     │ Norm → Self-Attention → Add │   같은 날짜의 건물 G개끼리만
     ├────────────────────────────┤
     │ Norm → FFN → Add            │
     └─────┬──────────────────────┘
           │   × 2층
           ▼
   날짜마다 대상 건물 Token(첫 번째)만 꺼냄 → [B, T, D]
           ▼
     최종 LayerNorm
           ▼
     + 날짜 PE
           ▼
  출력: 날짜별 요약 Memory [B, T, D]
```

- 서로 다른 날짜끼리는 Attention하지 않는다. 날짜 간 관계는 Decoder가 맡는다.
- 건물 순서에는 PE가 없다. Role Embedding으로 대상 건물만 표시하므로 Peer 순서를 바꿔도 결과가 같다.
- 한 배치 안에서 그룹 크기(G)가 샘플마다 다르면, 가장 큰 G에 맞춰 빈 자리를 채우고 Padding Mask로 가린다. 결측 날짜는 0으로 채워 그대로 들어가므로 날짜 단위로 빠지는 Peer는 없다.
- 같은 Zone·Type의 Peer가 아예 없으면 대상 건물 혼자 들어간다.
- Zone Encoder와 Type Encoder는 구조가 같고 가중치는 따로다.
- 코드: [set_encoder.py](../src/models/set_encoder.py) `PerDayEncoder`

### 2.4 Decoder

**하는 일**: 대상 건물의 T일을 날짜 순서대로 보면서, 날짜마다 Zone·Type 요약을 참고해 그날의 상태를 만든다.

```text
  입력: 대상 건물 Day Token [B, T, D] + 날짜 PE
           │
     ┌─────┴──────────────────────────────┐
     │ Norm → Causal Self-Attention → Add  │   오늘은 오늘과 과거 날짜만 봄
     ├────────────────────────────────────┤
     │ Norm → FFN → Add                    │
     ├────────────────────────────────────┤
     │ Norm → Zone Cross-Attention → Add   │ ◄── Zone Encoder 출력 (K, V)
     ├────────────────────────────────────┤
     │ Norm → FFN → Add                    │
     ├────────────────────────────────────┤
     │ Norm → Type Cross-Attention → Add   │ ◄── Type Encoder 출력 (K, V)
     ├────────────────────────────────────┤
     │ Norm → FFN → Add                    │
     └─────┬──────────────────────────────┘
           │   × 2층
           ▼
     최종 LayerNorm
           ▼
     Hourly Head: Linear(D, 24)
           ▼
  출력: 시간별 이상 Logit [B, T, 24] → sigmoid → 확률
```

- Cross-Attention의 Query는 Decoder의 날짜 T개, Key·Value는 Encoder 요약의 날짜 T개다(T×T). Encoder 쪽에는 날짜 Mask가 없어서 Window 안의 미래 날짜 요약도 볼 수 있다(오프라인 탐지).
- Cross-Attention에서 Norm은 Query에만 걸린다. Memory는 Encoder 최종 Norm을 이미 거쳤다.
- Zone을 먼저, Type을 나중에 참조한다.
- 코드: [temporal_decoder.py](../src/models/temporal_decoder.py) `TemporalDecoderLayer`, [hourly_head.py](../src/models/hourly_head.py)

### 2.5 v2_222와 다른 점

- v2는 Encoder가 `T×G`개 Token 전체에 Self-Attention했다.
- v3는 날짜마다 그 날짜의 건물 G개끼리만 Self-Attention한다.
- 그 결과 Encoder 비용이 `(T·G)²`에서 `T·G²`로 줄었다.
- 같은 건물 Attention Bias와 Encoder 내부 날짜 PE는 필요 없어져서 뺐다.
- 날짜 간 관계는 Decoder가 맡는다(Causal Self-Attention, 그리고 T×T Cross-Attention).

### 2.6 하이퍼파라미터와 파라미터 수

| 항목 | 값 |
|---|---|
| `d_model` / heads / `d_ff` | 128 / 4 / 512 |
| 층 수 (Zone - Type - Decoder) | 2 - 2 - 2 |
| Dropout / LayerNorm | 0.1 / Pre-LN |
| Patch Embedding `Linear(24,128)` | Zone·Type·Decoder 공유, 3,200 |
| Zone Encoder | 400,256 |
| Type Encoder | 400,256 |
| Decoder | 1,189,888 |
| Head `Linear(128,24)` | 3,096 |
| **합계** | **1,990,296 (약 1.99M)** |

## 3. 입력과 전처리

- 입력은 `meter_reading`만 쓴다. 날씨, 캘린더, 메타데이터 Feature는 쓰지 않는다.
- 결측값은 0으로 대체한다(`zero_fill`). 관측 여부 Mask는 입력에 넣지 않는다.
- 정규화: `log1p` 후 학습 Fold 통계로 표준화한다.
- Peer 구성(`peer_pool: train_only`)
  - Peer는 항상 학습 건물 중 같은 Zone(site)과 같은 Type(primary use)의 건물이다.
  - 학습 때는 대상 건물을 뺀다. 평가·Test 때는 학습 건물 전체를 쓴다.
  - Peer의 레이블은 쓰지 않는다.
  - 같은 Zone·Type의 학습 건물이 없으면 대상 건물 단독 Context가 된다.

## 4. 학습

| 항목 | 값 |
|---|---|
| Loss | 시간별 BCE. 결측 시간도 포함(`include`). `pos_weight` 없음 |
| Optimizer | AdamW, lr 3e-4, weight decay 0.01, grad clip 1.0 |
| LR | 2% 선형 Warmup 후 Cosine으로 0까지 |
| Window | 샘플마다 T ~ Uniform{1..64}일, 연속 구간. Encoder와 Decoder가 같은 날짜를 쓴다 |
| 배치 | 최대 32, Token 상한 16,384 |
| 학습량 | 10 epoch × 20,000 샘플 |
| Checkpoint | `best.pt` = T=64 평가의 F1@0.5 최고 epoch, `last.pt` = 마지막 epoch |
| 시간 | T4 1장 기준 epoch당 약 1.7분, fold당 약 17분 (v2_222는 epoch당 10.5분) |

## 5. 평가와 제출

- 내부 검증: 건물 단위 5-fold(`select_fold`, seed 42). 학습 160개, 검증 40개 건물이다.
- 추론: 366일을 겹치지 않는 64일 Window로 자르고, 한 행에 점수 1개를 낸다.
- 제출: 5개 fold의 `best.pt` 확률을 평균한 앙상블이 전체 200개 건물 학습보다 낫다(result_1106).

## 6. 성능

**5-fold 검증 (T=64, best.pt)**

Baseline은 E0 LightGBM(우승 솔루션 재현, 후처리 전)이다.

| fold | Baseline ROC | Baseline PR | Baseline F1 | Baseline 최고 F1 | v3 ROC | v3 PR | v3 F1 | v3 최고 F1 (thr) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.956 | 0.708 | 0.480 | 0.699 | 0.899 | 0.662 | 0.682 | 0.682 (0.47) |
| 1 | 0.981 | 0.839 | 0.614 | 0.783 | 0.941 | 0.722 | 0.715 | 0.718 (0.41) |
| 2 | 0.938 | 0.752 | 0.606 | 0.749 | 0.900 | 0.658 | 0.684 | 0.688 (0.43) |
| 3 | 0.974 | 0.802 | 0.550 | 0.763 | 0.939 | 0.678 | 0.630 | 0.648 (0.73) |
| 4 | 0.956 | 0.718 | 0.549 | 0.681 | 0.899 | 0.597 | 0.611 | 0.616 (0.42) |
| **평균** | **0.961** | **0.764** | 0.560 | **0.735** | 0.916 | 0.663 | **0.664** | 0.670 |

- F1@0.5만 v3가 높다. v3는 확률이 잘 보정되어 있다(최적 threshold 0.41~0.47, fold 3만 0.73). E0는 1:1 Downsampling 때문에 최적 threshold가 0.997 이상이다.
- ROC, PR, 최고 F1은 E0가 모든 fold에서 높다.

**Kaggle (v2_222 기준, 같은 성능의 이전 구조)**

| 제출 | Private | Public |
|---|---|---|
| v2_222 5-fold 앙상블 | 0.9426 | 0.9212 |
| v2_222 전체 학습 | 0.9372 | 0.9165 |
| 우승 솔루션 (참고) | 0.9866 | – |

v3_perday의 Kaggle 제출은 아직 하지 않았다.

## 7. 확정·기각된 결정

| 결정 | 근거 |
|---|---|
| 결측 0 대체 (`zero_fill`) | 건물 중앙값 대체보다 PR 0.613 vs 0.554 (result_1104) |
| 결측 시간 Loss 포함 (`include`) | `exclude_missing`은 결측 구간에 오탐이 쏟아짐, PR 0.19 (result_1104) |
| 2-2-2 + Cosine LR | 고정 LR의 epoch 간 급락이 사라짐 (result_1104) |
| 날짜별 Encoder (v3_perday) | v2와 성능이 같고 6배 빠름 (result_1110) |
| `pos_weight` 쓰지 않음 | ROC 불변, PR 같거나 낮음, threshold만 0.8~0.9로 밀림. v2와 v3 모두 같은 결론 (result_1104, result_1110) |

## 8. 실행

```bash
# 테스트
python -m pytest tests

# fold k 학습 → runs/v3_perday/fold<k>/{best.pt, last.pt, metrics.json}
python -m src.train.train_supervised --config configs/v3_perday.yaml --fold 0

# threshold별 F1 곡선 → best.threshold_T64.json
python -m src.eval.threshold_curve runs/v3_perday/fold0/best.pt --window 64

# 5-fold 앙상블 Test 제출
python -m src.eval.predict_test runs/v3_perday/fold{0,1,2,3,4}/best.pt --out runs/v3_perday/ensemble5/submission.csv

# Baseline E0 (CPU)
python -m src.baseline.e0_lgbm
```

- 원시 데이터 `data/raw/`는 git에 올리지 않는다. 서버에는 scp로 옮긴다.
- 학습은 Brev T4(`upset-gray-vulture-hoon`)에서 돌린다. 끌 때는 `brev stop`만 쓴다.

## 9. 코드 위치

| 역할 | 파일 |
|---|---|
| 데이터 로드·Day Patch·정규화 | `src/data/lead_loader.py`, `src/data/day_patches.py` |
| Peer 그룹 (Zone/Type) | `src/data/peer_groups.py` |
| 건물 단위 Fold | `src/data/splits.py` |
| 학습 샘플·배치 | `src/data/dataset.py` |
| 전체 모델·Config 검증 | `src/models/dual_context.py` |
| 날짜별 Encoder (`PerDayEncoder`) | `src/models/set_encoder.py` |
| Decoder / Head / Patch Embedding | `src/models/temporal_decoder.py`, `src/models/hourly_head.py`, `src/models/patch_embedding.py` |
| 학습 루프 | `src/train/train_supervised.py`, `src/train/common.py` |
| 지표·threshold 곡선·제출 | `src/eval/metrics.py`, `src/eval/threshold_curve.py`, `src/eval/predict_test.py`, `src/eval/kaggle_submission.py` |
| Baseline E0 | `src/baseline/e0_lgbm.py` |

## 10. 한계와 다음 단계

- **E0와의 격차**: ROC −0.045, PR −0.10이다.
  - E0 중요도 상위는 1시간 전후 대비 변화와 1주일 전후 대비 변화다.
  - 하루를 Token 하나로 압축하면서 이런 시간 단위 급변 신호가 흐려진 것으로 본다.
- **best.pt 선택 기준**: F1@0.5는 확률 높이에 민감하다. 확률 분포를 바꾸는 실험에서는 비교가 왜곡된다. PR-AUC로 바꿀지 결정이 필요하다.
- **다음 후보**
  1. 시간 단위 급변 입력이나 시간 축 Conv를 추가한다. SPEC 7.1의 "Value-change Feature는 Baseline 전용" 정책을 바꿀지 먼저 정해야 한다.
  2. 사이트 단위 fold에서 E0와 v3를 비교한다. Peer 문맥이 이점을 가질 수 있는 조건이다.
  3. v3_perday 5-fold 앙상블을 Kaggle에 제출한다.
