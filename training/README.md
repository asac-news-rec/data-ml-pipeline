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
- `fit`(train의 앞 90%)은 negative를 클릭과 1:1로 줄여 학습한다. `calib`(train의 최근 10%)과 `valid`는 줄이지 않는다.
- sampling 때문에 커진 확률은 `calib`으로 보정한다(Platt scaling). 보정값은 모델의 leaf 값에 넣어서 `predict()`가 보정된 pctr을 낸다.
- early stopping도 `calib`으로 하고, 최종 점수는 `valid`로만 잰다.
- `fit`·`calib`·`valid` 중 하나라도 클릭이 20건(`--min-clicks`)보다 적으면 학습하지 않고 종료 코드 99로 끝난다.
- cold `u_emb`는 NaN으로 두고 LightGBM의 결측 처리에 맡긴다.

## MLflow (선택)

`MLFLOW_TRACKING_URI`가 있으면 실험 기록과 모델 등록을 한다. `MODEL_NAME` 기본값은 `pctr`이고, 등록 뒤 같은 `valid`로 잰 AUC가 이전 champion보다 높을 때만 `champion` alias를 옮긴다. champion이 7일(`--champion-max-age-days`, 임시값)보다 오래되면 낮아도 옮긴다.

```bash
MLFLOW_TRACKING_URI=http://mlflow.mlflow.svc.cluster.local:5000 \
  MODEL_NAME=pctr \
  uv run --frozen --extra train python training/train_lgbm.py
```

MLflow 서버는 클러스터 안에서만 접근된다. 로컬에서는 `MLFLOW_TRACKING_URI`를 비워 두면 모델 파일만 저장한다.

## 매일 자동 학습 (`dags/train_pctr.py`)

매일 02:30 KST에 돈다. 사람이 하는 단계는 없다.

1. `training/build_training_set.py`: `clean.served`·`clean.actions`로 label을 만들고 Feast offline에서 served 시점 피처를 붙인다.
   최근 14일, 마지막 24시간은 valid, 끝 3시간은 클릭이 도착하지 않았을 수 있어 뺀다(모두 임시값).
2. `training/train_lgbm.py`: 학습하고 MLflow에 등록한다. champion과 비교한다.
3. champion이 바뀌었으면 추론 서버를 롤링 재시작하고 새 Pod가 준비될 때까지 기다린다.

| 종료 코드 | 뜻 | Airflow |
|---|---|---|
| 0 | champion을 바꿨다 | 추론 서버 재시작 |
| 98 | 새 모델이 더 낫지 않다 | skipped |
| 99 | 클릭이 부족하다 | skipped |

가상 독자 로그(`model_version='simulator'`)는 기본으로 뺀다. `train_include_simulator=true`일 때만 넣고, 넣은 행 수를 MLflow `simulated_rows`에 남긴다.

재시작 task는 Airflow 스케줄러 안에서 돈다(LocalExecutor). 스케줄러 계정의 `inference` 네임스페이스 deployment `get`·`patch` 권한이 필요하다.
추론 서버는 `MODEL_URI=models:/pctr@champion`으로 떠 있어야 한다.
