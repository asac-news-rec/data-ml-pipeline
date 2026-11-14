# Feast 피처 스토어 (로컬 프로토타입)

> 상태: **[현재]** 로컬(parquet + SQLite)에서 끝까지 돈다. 오프라인은 BigQuery(`dev_yoonmo`)로도 돈다(학습 표까지).
> **[계획]** 온라인을 Valkey로 옮기고, 원천 표를 Airflow가 `feat` 데이터셋에 만든다.

PCTR LightGBM 모델의 피처(일반 9 + 유저 임베딩 768 + 기사 임베딩 768)를 Feast로 정의한다.
학습 표와 추론 서버가 **같은 정의**에서 피처를 꺼내게 하는 것이 목적이다.

```
원천 표(parquet / BigQuery dev_yoonmo.* / [계획] feat.*)
   ├─ get_historical_features(entity_df=노출 목록) → 학습 표 (point-in-time join)
   └─ materialize → 온라인 저장소(SQLite / [계획] Valkey)
                       └─ get_online_features(reader 1명 + 후보 기사 N건) → 추론 서버
```

## 실행 (WSL)

`uv`와 `requirements.txt`는 이 프로토타입에서만 쓴다. 저장소 전체의 패키지 관리 도구는 아직 [미정]이다.

```bash
cd features/feast
uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python prepare_sources.py --src <final-project/virtual_reader 경로>   # 원천 표 5개 (약 1.5분), --valid-from으로 검증 시작 시각 지정
cd feature_repo && ../.venv/bin/feast apply && cd ..                           # 정의 등록
.venv/bin/python build_training_set.py --ref <virtual_reader/features/features.npz>  # 학습 표 + 대조 (약 1분)
.venv/bin/python online_demo.py                                                # materialize + 온라인 조회, --now로 '지금' 지정
```

### 오프라인을 BigQuery로 (WSL)

원천 표만 BigQuery로 바뀌고 피처 정의는 같다. registry와 온라인 저장소는 로컬 파일(`data/*_bq.db`)이라 위 로컬 실행과 섞이지 않는다.

준비 (한 번):
- `gcloud auth application-default login` (WSL 안에서). Python 라이브러리는 이 인증(ADC)을 쓴다. `gcloud auth login`만으로는 안 된다.
- 계정에 `dev_yoonmo` 표 만들기·쿼리 권한이 있어야 한다 (infra#29).
- 위 로컬 실행의 `prepare_sources.py`까지 먼저 해서 `data/*.parquet`가 있어야 한다.

```bash
cd features/feast
.venv/bin/python upload_to_bq.py                     # 원천 표 4개 → final-project-513508.dev_yoonmo (덮어쓴다, 행 수 확인), --dataset으로 바꿈
export FEAST_SOURCE=bigquery                         # definitions.py가 FileSource 대신 BigQuerySource를 만든다
cd feature_repo && ../.venv/bin/feast -f feature_store.bq.yaml apply && cd ..
.venv/bin/python build_training_set.py --yaml feature_store.bq.yaml --ref <virtual_reader/features/features.npz>
                                                     # 결과: data/training_set_bigquery.parquet (약 13분, 아래 '알아 둘 것')
```

- `FEAST_SOURCE`를 켠 셸에서 로컬 설정으로 `feast apply`하면 로컬 registry에 BigQuery 원천이 등록된다. 로컬로 돌아갈 땐 `unset FEAST_SOURCE`.
- 다른 데이터셋을 쓰면 `upload_to_bq.py --dataset`, `FEAST_BQ_DATASET`(원천 표 위치), `feature_store.bq.yaml`의 `dataset`(entity_df 임시 표 위치)을 같이 바꾼다.

## 파일

| 파일 | 하는 일 |
| --- | --- |
| `prepare_sources.py` | 가짜 로그 → 원천 표 parquet 5개. [계획] Airflow 집계 단계가 BigQuery에 같은 표를 만든다 |
| `feature_repo/definitions.py` | entity 2개, feature view 4개, on-demand feature view 1개, feature service 1개 |
| `feature_repo/feature_store.yaml` | 로컬 설정 (file + sqlite) |
| `feature_repo/feature_store.bq.yaml` | 오프라인만 BigQuery(`dev_yoonmo`), registry·온라인은 로컬 파일 |
| `upload_to_bq.py` | 원천 parquet 4개 → BigQuery 표 (exposures는 entity_df라 올리지 않는다) |
| `feature_repo/feature_store.gcp.yaml.example` | [계획] BigQuery + Valkey 설정 예시 (미실행) |
| `build_training_set.py` | `get_historical_features`로 학습 표를 만들고 `make_features.py` 결과와 대조 |
| `online_demo.py` | `materialize` → `get_online_features` → 같은 시각 오프라인 값과 비교 |

## 피처 정의

| feature view | entity | 원천 표 1행 = | 피처 |
| --- | --- | --- | --- |
| `article_profile` | article | 기사 1건 (시각 = 발행 시각) | `a_emb`(768), `category`, `published_ts` |
| `reader_profile` | reader | 클릭 1건 (그 직후 상태) | `u_emb`(768), `hist_clicks`, `last_click_ts` |
| `reader_activity` | reader | 유효 노출 1건 (그 직후 상태) | `hist_exposures` |
| `reader_article` | reader + article | 상태가 바뀐 사건 1건 | `already_opened`, `already_seen` |
| `request_context` (on-demand) | — | 요청마다 계산 | `article_age_h`, `hour`, `hours_since_last_click` |

- 피처 행 시각 = 사건 시각 + 1µs. 노출과 같은 시각의 사건은 그 노출에 붙지 않는다.
- `u_emb`는 클릭 때만 바뀐다. 타임 디케이 가중치 `0.5 ** (Δt/24h)`는 분자·분모에 같이 곱해져 약분된다.
- `position`은 요청값이다. 학습 땐 entity_df 열, 서빙 땐 요청에서 받는다.
- 행이 없으면 null이다. `hist_*`, `already_*`는 0으로, `u_emb`는 NaN(cold)으로 채운다 (`build_training_set.to_matrix`).
- ttl은 모두 0(무제한)이다. 피처가 전부 '지금까지' 누적값이라, 기간으로 자르면 오래 쉬다 온 독자가 0으로 학습된다.
- 같은 (entity 키, 시각) 행이 여러 개면 Feast가 어느 행을 고를지 정해져 있지 않아, 원천 표를 쓸 때 마지막 행만 남긴다.

## 결과 (2026-11-13, 가짜 로그 11/3~11/6 12시)

- 학습 표 40,763행 × 1,545열. `make_features.py` 결과와 모든 열이 **차이 0**으로 같다. cold 41.5%.
- materialize 약 3초. 독자 1명 + 후보 10건 온라인 조회 약 23ms(로컬 SQLite, 두 번째 호출부터).
- 같은 시각의 온라인 값과 오프라인 값이 같다 (training-serving skew 없음).
- (2026-11-14) BigQuery 오프라인으로도 40,763행, 모든 열 **차이 0**. 걸린 시간 약 13분(대부분 결과 내려받기).

## 알아 둘 것

- Feast 0.68 file 오프라인 저장소는 feature view를 붙일 때마다 '지금까지 나온 join key + 시각'으로 중복을 지운다.
  다른 독자가 같은 기사를 같은 ms에 본 노출 2건이 1건으로 합쳐졌다(40,763행 중 1쌍).
  `build_training_set.historical()`이 그런 행을 따로 호출해 피한다. BigQuery 저장소에는 이 문제가 없어(한 번에 불러도 40,763행) 나눠 부르지 않는다.
  회피책이 못 막는 경우를 위해 `check_rows()`가 결과 행 수와 row_id를 항상 검사하고, 다르면 멈춘다.
- materialize 범위는 view의 ttl보다 짧으면 안 된다. 짧으면 오프라인엔 있고 온라인엔 없는 값이 생긴다.
  `online_demo.py`는 원천 표의 가장 이른 시각부터 전체를 쓴다. 서빙 DAG는 첫 적재만 전체, 이후 `materialize_incremental`로 한다.
- BigQuery 결과는 보통 Storage Read API로 빠르게 받는데, 그 권한(`bigquery.readsessions.create`, 역할 `roles/bigquery.readSessionUser`)이
  2026-11-14 기준 없어 403이 난다. `build_training_set.use_rest_download()`가 임시로 REST로 받게 해서 느리다(약 13분).
  권한이 생기면 이 함수를 지운다.
- on-demand feature view는 Feast가 "experimental"로 표시한다. 오프라인에서 행이 많으면 느릴 수 있다.
