#!/bin/bash
# pos_weight 실험 (fold 0): 결측 처리 실험 3개 중 최고 설정 + pos_weight
# 사용: nohup bash scripts/queue_posweight.sh [기다릴 PID] [pos_weight] > runs/queue_posw.log 2>&1 &
set -eo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
FOLD=0
POS_WEIGHT=${2:-45}  # 음성/양성 비율 약 49 (학습 anomaly 2.1%)

stamp() { while IFS= read -r l; do echo "$(date +%H:%M:%S) $l"; done; }

if [ -n "$1" ]; then
    echo "$(date +%H:%M:%S) waiting for PID $1"
    while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi

# 실행 폴더 -> config. T=64 ROC-AUC 최고값이 가장 높은 설정을 고른다.
BASE=$(python - "$FOLD" <<'EOF'
import json, sys
fold = sys.argv[1]
runs = {
    "runs": "configs/train.yaml",
    "runs/exclude": "configs/exclude.yaml",
    "runs/median": "runs/median/config.yaml",
}
scores = {}
for d, cfg in runs.items():
    try:
        scores[cfg] = max(r["T=64"]["roc_auc"] for r in json.load(open(f"{d}/fold{fold}/metrics.json")))
    except FileNotFoundError:
        pass
for cfg, s in scores.items():
    print(f"{cfg} {s:.4f}", file=sys.stderr)
print(max(scores, key=scores.get))
EOF
)
echo "$(date +%H:%M:%S) base config: $BASE"

mkdir -p runs/posw
sed -e "s#^output_dir: .*#output_dir: runs/posw#" \
    -e "s#pos_weight: [a-z0-9.]*#pos_weight: $POS_WEIGHT#" \
    "$BASE" > runs/posw/config.yaml
grep -nE "^output_dir|missing_value_strategy|missing_reading_loss_policy|pos_weight" runs/posw/config.yaml
echo "$(date +%H:%M:%S) start pos_weight=$POS_WEIGHT"
python -u -m src.train.train_supervised --config runs/posw/config.yaml --fold $FOLD 2>&1 | stamp > runs/posw/fold$FOLD.log
echo "$(date +%H:%M:%S) queue done"
