# 추론 서버 `/v1/rank`

뉴스 백엔드가 후보 기사를 보내면 Feast online store 피처로 PCTR을 계산해 순위를 돌려준다. 계약은 청사진 규약 ②(https://blueprint.yoonmo335.workers.dev/overview#c2)를 따른다.

## 실행

로컬(SQLite online store + 로컬 모델):

```bash
uv sync --frozen --extra serving
FEAST_FS_YAML=feature_store.yaml \
MODEL_PATH=models/lgbm.txt \
uv run --frozen --extra serving uvicorn serving.app:app --host 0.0.0.0 --port 8080
```

GCP(Valkey online store + MLflow 모델):

```bash
FEAST_FS_YAML=feature_store.gcp.yaml \
FEAST_REGISTRY_PATH=gs://final-project-513508-model-artifacts/feast/registry.pb \
VALKEY_HOST=valkey.valkey.svc.cluster.local \
MLFLOW_TRACKING_URI=http://mlflow.mlflow.svc.cluster.local:5000 \
MODEL_URI=models:/pctr@champion \
uv run --frozen --extra serving uvicorn serving.app:app --host 0.0.0.0 --port 8080
```

MLflow 서버가 아직 없으면 `MLFLOW_TRACKING_URI`·`MODEL_URI` 대신 GCS 파일을 읽는다(임시):

```bash
MODEL_PATH=gs://final-project-513508-model-artifacts/models/pctr/lgbm.txt
```

## 이미지

```bash
docker build -f serving/Dockerfile -t <registry>/inference-server:<커밋 SHA 7자리> .
```

모델은 이미지에 넣지 않는다. Python 3.12, 포트 8080, 비루트 사용자(uid 10001)로 돈다.

## 확인

```bash
curl localhost:8080/healthz
curl localhost:8080/readyz
curl localhost:8080/v1/model
curl -X POST localhost:8080/v1/rank -H 'content-type: application/json' -d '{
  "request_id": "req-001",
  "reader_id": "<reader_id>",
  "candidates": ["news-...", "news-..."],
  "context": {"request_time": "2026-11-06T12:00:00+09:00", "device": "mobile"},
  "top_k": 3
}'
```

## 설계 메모

- 모델 입력은 일반 9 + `u_emb` 768 + `a_emb` 768 = 1,545열이다. `position`은 요청 후보 순서를 1부터 센다.
- 독자 피처가 전부 없으면 `READER_FEATURES_NOT_FOUND`, 기사 임베딩이 없으면 `ARTICLE_FEATURES_NOT_FOUND`로 `defaulted`에 넣는다.
- `news-[0-9a-f]{32}` 형식이 아닌 후보는 `UNKNOWN_ARTICLE`로 `unscored`에 넣는다. 실제 기사 ID 형식이 바뀌면 `ARTICLE_ID` 정규식을 고친다.
- `features_as_of`는 온라인 store의 피처 기준 시각을 읽을 수 있게 되면 바꾼다. 지금은 `request_time`으로 둔다.
- `context.device`는 계약에 있지만 현재 피처에 없어서 받기만 하고 쓰지 않는다.
- 모델 로드와 store 연결이 안 되면 `/readyz`와 `/v1/rank`가 503을 돌려준다.
