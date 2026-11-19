# baseline LightGBM 학습

Feast training set(`features/feast/data/training_set.parquet`)으로 PCTR baseline LightGBM을 학습한다.

입력은 일반 9열 + `u_emb` 768 + `a_emb` 768 + `label` + `split`이다. `label`은 impression 기준 클릭 여부다.

## 실행

```bash
uv sync --frozen --extra train
uv run --frozen --extra train python training/train_lgbm.py \
  --training-set features/feast/data/training_set.parquet \
  --model-out models/lgbm.txt
```

- `models/`는 gitignore라 모델 파일은 커밋하지 않는다.
- `--rounds`, `--early-stopping`, `--learning-rate`, `--num-leaves`, `--feature-fraction`으로 조정한다.
- cold `u_emb`는 NaN으로 두고 LightGBM의 결측 처리에 맡긴다.

## MLflow (선택)

`MLFLOW_TRACKING_URI`가 있으면 실험 기록과 모델 등록을 한다. `MODEL_NAME` 기본값은 `pctr`이고, 등록 뒤 `champion` alias를 붙인다.

```bash
MLFLOW_TRACKING_URI=http://mlflow.mlflow.svc.cluster.local:5000 \
  MODEL_NAME=pctr \
  uv run --frozen --extra train python training/train_lgbm.py
```

MLflow 서버는 클러스터 안에서만 접근된다. 로컬에서는 `MLFLOW_TRACKING_URI`를 비워 두면 모델 파일만 저장한다.
