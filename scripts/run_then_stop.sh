#!/bin/bash
# 학습 명령을 실행하고, 성공·실패·중단과 관계없이 끝나면 Brev 인스턴스를 끈다 (GPU 유휴 과금 방지).
# 사용: nohup bash scripts/run_then_stop.sh <instance> <log> <명령...> > /dev/null 2>&1 &
# 예:   nohup bash scripts/run_then_stop.sh upset-gray-vulture-hoon runs/v1/fold0.log \
#           python -u -m src.train.train_supervised --config configs/train.yaml --fold 0 > /dev/null 2>&1 &
# 서버에서 brev login --skip-browser가 되어 있어야 한다. 로그 시각은 한국시간(KST)이다. 인스턴스를 끄지 않으려면 runs/NO_STOP 파일을 만든다.
INSTANCE=$1
LOG=$2
shift 2
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p "$(dirname "$LOG")"

stop() {
    echo "$(TZ=Asia/Seoul date '+%m-%d %H:%M:%S') finished; stopping $INSTANCE" >> "$LOG"
    if [ -f runs/NO_STOP ]; then
        echo "runs/NO_STOP exists; keep running" >> "$LOG"
        return
    fi
    # OS shutdown은 Brev 상태와 어긋나 인스턴스가 다시 켜지지 않으므로 쓰지 않는다. brev stop만 쓴다.
    if ! brev stop "$INSTANCE" >> "$LOG" 2>&1; then
        echo "WARNING: brev stop failed (brev login --skip-browser 필요); instance still running" >> "$LOG"
    fi
}
trap stop EXIT

"$@" 2>&1 | while IFS= read -r l; do echo "$(TZ=Asia/Seoul date '+%m-%d %H:%M:%S') $l"; done >> "$LOG"
