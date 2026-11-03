#!/bin/bash
# 실행 중인 run_then_stop.sh 뒤에 다음 학습을 이어 붙인다. 앞 작업이 인스턴스를 끄지 않게 막고, 마지막 작업이 끈다.
# 사용: nohup bash scripts/queue_then_stop.sh <기다릴 PID> <instance> <log> <명령...> > /dev/null 2>&1 &
# 여러 개를 이어 붙이려면 앞 queue_then_stop.sh의 PID를 기다리게 한다.
WAIT=$1
shift
cd "$(dirname "$0")/.."
mkdir -p runs
# 앞 작업의 run_then_stop.sh가 끝날 때 인스턴스를 끄지 않는다. 3개 이상 이을 때 앞 대기열이 지운 NO_STOP을 되살리도록 기다리는 동안 계속 만든다.
while kill -0 "$WAIT" 2>/dev/null; do touch runs/NO_STOP; sleep 30; done
rm -f runs/NO_STOP
exec bash scripts/run_then_stop.sh "$@"
