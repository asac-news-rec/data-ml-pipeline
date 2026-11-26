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

`MLFLOW_TRACKING_URI`가 있으면 실험 기록과 모델 등록을 한다. `MODEL_NAME` 기본값은 `pctr`이고, 등록 뒤 같은 `valid`로 잰 AUC가 이전 champion보다 높을 때만 `champion` alias를 옮긴다. champion이 7일(`--champion-max-age-days`, 임시값)보다 오래되면 AUC 하락이 0.02(`--max-auc-drop`) 안일 때만 옮긴다. 새 모델 AUC가 0.55(`--min-auc`)보다 낮으면 첫 모델이어도 옮기지 않는다(상류 피처가 깨진 모델 차단).

입력 행렬(열 순서·결측 채우기)은 `serving/matrix.py` 하나를 추론 서버와 같이 쓴다. 학습 이미지는 저장소 전체를 `/app`에 두므로 `serving/matrix.py`도 들어 있다.

```bash
MLFLOW_TRACKING_URI=http://mlflow.mlflow.svc.cluster.local:5000 \
  MODEL_NAME=pctr \
  uv run --frozen --extra train python training/train_lgbm.py
```

MLflow 서버는 클러스터 안에서만 접근된다. 로컬에서는 `MLFLOW_TRACKING_URI`를 비워 두면 모델 파일만 저장한다.
