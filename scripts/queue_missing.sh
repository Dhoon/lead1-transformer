#!/bin/bash
# 결측 처리 실험 대기열 (fold 0): include(runs/fold0) -> exclude -> 우승 정책 + building_median
# 사용: nohup bash scripts/queue_missing.sh [기다릴 PID] > runs/queue.log 2>&1 &
set -eo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
FOLD=0

stamp() { while IFS= read -r l; do echo "$(date +%H:%M:%S) $l"; done; }

# T=64 ROC-AUC 최고값
best() {
    python -c "import json,sys; print(max(r['T=64']['roc_auc'] for r in json.load(open(sys.argv[1]))))" "$1"
}

if [ -n "$1" ]; then
    echo "$(date +%H:%M:%S) waiting for PID $1"
    while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi

if [ ! -f runs/exclude/fold$FOLD/last.pt ]; then
    echo "$(date +%H:%M:%S) start exclude"
    mkdir -p runs/exclude
    python -u -m src.train.train_supervised --config configs/exclude.yaml --fold $FOLD 2>&1 | stamp > runs/exclude/fold$FOLD.log
fi

INC=$(best runs/fold$FOLD/metrics.json)
EXC=$(best runs/exclude/fold$FOLD/metrics.json)
WIN=$(python -c "import sys; print('include' if float(sys.argv[1]) >= float(sys.argv[2]) else 'exclude_missing')" "$INC" "$EXC")
echo "$(date +%H:%M:%S) best T=64 roc_auc include=$INC exclude=$EXC -> $WIN"

mkdir -p runs/median
sed -e "s#^output_dir: .*#output_dir: runs/median#" \
    -e "s#missing_value_strategy: zero_fill#missing_value_strategy: building_median#" \
    -e "s#missing_reading_loss_policy: [a-z_]*#missing_reading_loss_policy: $WIN#" \
    configs/train.yaml > runs/median/config.yaml
echo "$(date +%H:%M:%S) start building_median ($WIN)"
python -u -m src.train.train_supervised --config runs/median/config.yaml --fold $FOLD 2>&1 | stamp > runs/median/fold$FOLD.log
echo "$(date +%H:%M:%S) best T=64 roc_auc median($WIN)=$(best runs/median/fold$FOLD/metrics.json)"
echo "$(date +%H:%M:%S) queue done"
