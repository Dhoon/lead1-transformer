# 학습 결과 요약 (2026-11-06)

[result_1104.md](result_1104.md) 이후의 결과다.
- Kaggle 제출 점수
- 우승 논문 수치 검토
- E0(우승 솔루션 재현) Baseline과 Transformer 비교

## 핵심 결론

1. **Kaggle 점수: 5-fold 앙상블이 전체 학습보다 낫다.**
   - 앙상블은 Private 0.9426 / Public 0.9212다.
   - 우승 솔루션은 Private 0.9866이다.
2. **우승 논문의 F1 0.89는 논문 자신의 Confusion Matrix와 맞지 않는다.**
   - 행렬로 다시 계산하면 F1은 0.707이다(2절).
3. **같은 건물 5-fold에서 E0 LightGBM이 Transformer보다 모든 fold, 모든 지표에서 높다.**
   - ROC 0.961 vs 0.916
   - PR 0.764 vs 0.664
   - 최고 F1 0.735 vs 0.672
   - "트리 모델은 건물을 외워서 이긴다"는 가설은 건물 단위 분할에서 맞지 않았다.

## 1. Kaggle 제출 (Transformer 2-2-2 + cosine LR)

대회 지표는 시간 단위 **ROC-AUC 하나**다(SPEC 9.1). PR-AUC와 F1은 검증 fold에서만 잰다.

| 제출 | Private | Public | 평균 확률 | 0.5 이상 비율 |
|---|---|---|---|---|
| **5-fold 앙상블** (`runs/v2_222/ensemble5`) | **0.9426** | **0.9212** | 0.0222 | 1.52% |
| 전체 200개 건물 학습 (`runs/v2_222/full`) | 0.9372 | 0.9165 | 0.0230 | 1.68% |

- 두 제출의 확률 상관은 0.959다.
- 0.5 기준 판정은 99.6% 행에서 같다. 다만 이상으로 판정한 행끼리 겹치는 비율(Jaccard)은 0.79다.
- 5-fold 검증 ROC 평균 0.916 ± 0.023과 비교하면 다음과 같다.
  - Public은 이 범위 안이다.
  - Private은 조금 더 높다.
  - 검증이 Test 점수를 과대평가하지 않는다.

## 2. 우승 논문 수치 검토 (Fu et al., BuildSys '22)

- 공저자 Arjunan은 LEAD1.0 데이터셋 저자이자 대회 주최 측이다. 그래서 Test 정답으로 Confusion Matrix(Figure 3)를 그릴 수 있었다.
- 행렬의 행은 902,594개로 Test 1,800,567행의 50.1%다. Private 리더보드 절반으로 보인다. 이 절반의 이상 비율은 2.2%다.

| | 본문 | Figure 3 행렬로 계산 |
|---|---|---|
| Precision | 98.7% | 18036 / (18036 + 12949) = **0.582** |
| Recall | 81.9% | 18036 / (18036 + 1995) = **0.900** |
| F1 | ≈ 0.89 | **0.707** |

- 정상:이상을 1:1로 Downsampling하면 확률이 위로 쏠린다. 그래서 0.5로 자르면 오탐이 많다. 행렬의 모양이 이와 맞으므로 행렬 쪽이 맞는 값으로 본다.
- 우리 F1은 검증 fold 값이고, 논문 F1은 Test 절반 값이다. 기준이 달라 직접 비교하지 않는다.

## 3. E0: 우승 솔루션 재현 (LightGBM, 같은 건물 5-fold)

구현은 `src/baseline/e0_lgbm.py`이고 CPU에서 수 분 걸린다.

**Fold와 지표는 Transformer와 같다.**
- Fold: `select_fold`, seed 42
- 지표: `hourly_metrics`, threshold 0.5
- 평가 대상: 검증 건물 40개의 전체 Row

**논문과 같은 점**
- 결측값을 건물별 평균으로 대체한다.
- 정상:이상을 1:1로 Downsampling한다.
- 대회 제공 Feature(날씨, 시간, 메타데이터, target encoding)를 쓴다.
- Value-change Feature 32개를 쓴다.
  - 시간 격자 기준으로 계산한다. 차이는 `X(t) − X(t−s)`, 비율은 `(X(t)+1)/(X(t−s)+1)`이다.
  - s = ±1, 2, 3, 23, 24, 48, 72, 168시간이다. 음수 s는 미래 시점과 비교한다.

**논문과 다른 점**
- LightGBM 하나만 쓴다. 논문은 4개 트리 모델을 평균했고, 단일 모델 대비 +0.002였다.
- 건물 ID가 들어간 조합 문자열 Feature는 뺐다.

| fold | E0 ROC | E0 PR | E0 F1 (0.5) | E0 최고 F1 | Transformer ROC / PR / F1 | Transformer 최고 F1 (threshold) |
|---|---|---|---|---|---|---|
| 0 | 0.956 | 0.708 | 0.480 | 0.699 | 0.893 / 0.630 / 0.640 | 0.644 (0.46) |
| 1 | 0.981 | 0.839 | 0.614 | 0.783 | 0.942 / 0.712 / 0.705 | 0.707 (0.46) |
| 2 | 0.938 | 0.752 | 0.606 | 0.749 | 0.910 / 0.670 / 0.685 | 0.691 (0.41) |
| 3 | 0.974 | 0.802 | 0.550 | 0.763 | 0.937 / 0.723 / 0.726 | 0.728 (0.44) |
| 4 | 0.956 | 0.718 | 0.549 | 0.681 | 0.896 / 0.583 / 0.586 | 0.589 (0.43) |
| **평균** | **0.961 ± 0.015** | **0.764 ± 0.050** | 0.560 ± 0.048 | **0.735** | 0.916 / 0.664 / **0.669** | 0.672 |

- **E0가 5개 fold 모두에서 ROC +0.03~0.06, PR +0.08~0.13, 최고 F1 +0.04~0.09 높다.**
- threshold 0.5의 F1만 Transformer가 높다.
  - E0는 1:1 Downsampling 때문에 확률이 위로 쏠려서, 최적 threshold가 0.995 이상이다.
  - Transformer는 최적 threshold가 0.41~0.46으로 확률 보정이 잘 되어 있다.
- 후처리(`meter_reading == 1` → 1, 건물별 첫·마지막 시각 → 0)의 효과는 작다. ROC +0.0004, PR +0.009다.
  - train에서 `meter_reading == 1`인 17,059행 중 99.96%가 이상이다. 라벨링 방식이 남긴 흔적이라 기본 비교는 후처리 전 값으로 한다.

**Feature 중요도 (gain, 5-fold 평균 비중)**

| 순위 | Feature | 비중 |
|---|---|---|
| 1 | `meter_reading` | 22.4% |
| 2 | `value_chg_ratio_-168` (일주일 뒤 대비) | 6.3% |
| 3 | `value_chg_ratio_168` (일주일 전 대비) | 5.2% |
| 4 | `building_id` | 4.3% |
| 5 | `value_chg_ratio_1` (1시간 전 대비) | 3.2% |
| 6 | `value_chg_ratio_-1` (1시간 뒤 대비) | 3.0% |

- 논문과 같은 경향이다. 핵심은 1시간과 1주일 전후 대비 변화다.
- `building_id`는 처음 보는 건물에서는 의미 없는 번호다. 그런데도 4위다. 우연한 상관일 수 있으므로 빼고 다시 확인할 필요가 있다.

## 4. 해석과 다음 단계

- 현재 Transformer는 시간 단위 급변(1시간·1주일 전후 대비) 신호를 트리 모델 + 수작업 Feature만큼 잡지 못한다.
  - 하루 24값을 한 Token으로 압축하면서 시간 단위 세부 신호가 흐려졌을 가능성이 있다.
- Transformer의 차별점이 남아 있을 수 있는 조건은 **사이트 단위 일반화**다.
  - 사이트 단위 일반화는 우승 논문이 Future work로 남긴 문제다.
  - E0의 target encoding과 `building_id`는 같은 사이트 정보를 이용할 수 있다.
- 후보는 다음과 같다.
  1. E0 Kaggle 제출(`runs/e0_lgbm/submission_raw.csv`)로 재현 수준을 확인한다.
  2. 사이트 단위 fold에서 E0와 Transformer를 비교한다.
  3. Transformer에 시간별 차이·비율 입력이나 시간 축 Conv를 추가한다. Value-change Feature는 SPEC 7.1에서 Baseline 전용으로 정해 두었으므로 결정이 필요하다.
  4. 하이브리드: Transformer 출력을 E0 Feature로 넣어 Peer 문맥의 추가 정보량을 측정한다.

## 결과 파일

| 실험 | 경로 |
|---|---|
| Transformer 5-fold 앙상블 제출 | `runs/v2_222/ensemble5/submission.csv` |
| Transformer 전체 학습 제출 | `runs/v2_222/full/submission.csv`, `runs/v2_222/full/full.log` |
| E0 fold별 지표·검증 확률·중요도 | `runs/e0_lgbm/fold<k>/{metrics.json, val_prob.npz, importance.csv}` |
| E0 5-fold 요약 | `runs/e0_lgbm/summary.json`, `runs/e0_lgbm/run.log` |
| E0 Test 제출 (5개 fold 모델 평균) | `runs/e0_lgbm/submission_raw.csv`, `runs/e0_lgbm/submission_post.csv` |
