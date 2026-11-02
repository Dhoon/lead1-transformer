# LEAD-Transformer

작업 전 항상 @docs/SPEC.md 를 읽고 그 내용을 따른다. [확정] 항목은 임의로 바꾸지 않고, [제안]/[미정] 항목은 Config 옵션으로 노출한다.

- 원시 데이터(`data/raw/`)는 git에 올리지 않는다. 메타데이터 캐시는 `data/processed/`, 학습 산출물은 `runs/`.
- 테스트: `python -m pytest tests`
- 내부 검증(미학습 건물): `python -m src.train.train_supervised --config configs/train.yaml --fold 0` → `runs/fold0/metrics.json`
- 최종 제출: `--fold -1` → `runs/full/submission.csv`
- `configs/train.yaml`의 SPEC 확정 키(`cross_attention_fusion` 등)는 `ModelConfig.from_dict`가 다른 값을 거부한다.
