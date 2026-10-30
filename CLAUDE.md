# LEAD-Transformer

작업 전 항상 @docs/SPEC.md 를 읽고 그 내용을 따른다. [확정] 항목은 임의로 바꾸지 않고, [제안]/[미정] 항목은 Config 옵션으로 노출한다.

- 원시 데이터(`data/raw/`)는 git에 올리지 않는다. 메타데이터 캐시는 `data/processed/`, 학습 산출물은 `runs/`.
- 테스트: `python -m pytest tests`
- 사전학습: `python -m src.train.pretrain --config configs/pretrain.yaml --fold 0` → `runs/pretrain/fold0/best.pt`
- Fine-tuning(미학습 건물 검증): `python -m src.train.finetune --config configs/downstream.yaml --fold 0` → `runs/downstream/fold0/metrics.json`
- Kaggle 제출: 위 두 명령을 `--fold -1`로 실행하고 downstream의 `init_from`을 `runs/pretrain/full/best.pt`로 바꾼다 → `runs/downstream/full/submission.csv`
- Fine-tuning은 같은 fold/seed/n_splits의 사전학습 체크포인트만 받는다 (검증 건물 누수 방지).
