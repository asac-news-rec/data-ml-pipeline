# Feast 피처 스토어 (로컬 프로토타입)

> 상태: **[현재]** 로컬(parquet + SQLite)에서 끝까지 돈다. 오프라인은 BigQuery(`dev_yoonmo`)로도 돈다(training set까지).
> **[계획]** 온라인을 Valkey로 옮기고, source table을 Airflow가 `feat` 데이터셋에 만든다.

PCTR LightGBM 모델의 피처(일반 9 + 유저 임베딩 768 + 기사 임베딩 768)를 Feast로 정의한다.
training set과 추론 서버가 **같은 정의**에서 피처를 꺼내게 하는 것이 목적이다.

```
source table(parquet / BigQuery dev_yoonmo.* / [계획] feat.*)
   ├─ get_historical_features(entity_df=impression 목록(기본)) → training set (point-in-time join)
   └─ materialize → 온라인 저장소(SQLite / [계획] Valkey)
                       └─ get_online_features(reader 1명 + 후보 기사 N건) → 추론 서버
```

## 실행 (WSL)

로컬 프로토타입은 `requirements.txt`를 쓴다. Airflow 작업 이미지는 저장소 루트의 `pyproject.toml`(의존성)과 `uv.lock`(버전 고정)으로 만든다.

```bash
cd features/feast
uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python prepare_sources.py --src <final-project/virtual_reader 경로>   # source table 5개 (약 1.5분). --base impression(기본) 또는 exposure, --valid-from으로 검증 시작 시각 지정
cd feature_repo && ../.venv/bin/feast apply && cd ..                           # 정의 등록
.venv/bin/python build_training_set.py --ref <virtual_reader/features/features.npz>  # training set + 대조 (약 1분). --ref는 --base exposure(40,763행)일 때만 쓴다
.venv/bin/python online_demo.py                                                # materialize + 온라인 조회, --now로 '지금' 지정
```

### 오프라인을 BigQuery로 (WSL)

source table만 BigQuery로 바뀌고 피처 정의는 같다. registry와 온라인 저장소는 로컬 파일(`data/*_bq.db`)이라 위 로컬 실행과 섞이지 않는다.

준비 (한 번):
- `gcloud auth application-default login` (WSL 안에서). Python 라이브러리는 이 인증(ADC)을 쓴다. `gcloud auth login`만으로는 안 된다.
- 계정에 `dev_yoonmo` 표 만들기·쿼리 권한이 있어야 한다 (infra#29).
- 위 로컬 실행의 `prepare_sources.py`까지 먼저 해서 `data/*.parquet`가 있어야 한다.

```bash
cd features/feast
.venv/bin/python upload_to_bq.py                     # source table 4개 → final-project-513508.dev_yoonmo (덮어쓴다, 행 수 확인), --dataset으로 바꿈. dev_로 시작하지 않는 데이터셋은 --allow-non-dev가 필요하다
export FEAST_SOURCE=bigquery                         # definitions.py가 FileSource 대신 BigQuerySource를 만든다
cd feature_repo && ../.venv/bin/feast -f feature_store.bq.yaml apply && cd ..
.venv/bin/python build_training_set.py --yaml feature_store.bq.yaml --ref <virtual_reader/features/features.npz>
                                                     # 결과: data/training_set_bigquery.parquet (약 13분, 아래 '알아 둘 것')
```

- `FEAST_SOURCE`를 켠 셸에서 로컬 설정으로 `feast apply`하면 로컬 registry에 BigQuery 원천이 등록된다. 로컬로 돌아갈 땐 `unset FEAST_SOURCE`.
- 다른 데이터셋을 쓰면 `upload_to_bq.py --dataset`, `FEAST_BQ_DATASET`(source table 위치), `feature_store.bq.yaml`의 `dataset`(entity_df 임시 표 위치)을 같이 바꾼다.

### Airflow에서 materialize (GCP)

Feast는 Airflow 이미지에 넣지 않는다. Airflow 이미지의 `google-cloud-storage` 3.17.1과 충돌하고 Python 버전도 다르기 때문이다. 대신 Python 3.12 작업 이미지를 `KubernetesPodOperator`로 띄운다.

- 작업 이미지: 저장소 루트 `pyproject.toml`의 `airflow-dag` 묶음 + `uv.lock`. 빌드는 인프라가 맡는다 (`uv sync --frozen --extra airflow-dag`).
- 작업 이미지는 저장소를 `/app`에 넣고, 이미지의 `python`이 그 환경을 가리키게 한다(예: `ENV PATH=/app/.venv/bin:$PATH`). DAG는 `python /app/features/feast/materialize.py`를 실행한다.
- 이미지 태그는 Airflow Variable `feast_work_image`로 받는다. 같은 태그를 덮어쓰지 말고 새 태그를 만든다(`image_pull_policy: IfNotPresent`라 옛 이미지가 남는다).
- DAG: `dags/materialize_feast.py`, 스크립트: `features/feast/materialize.py`
- 설정: `feature_store.gcp.yaml` (GCS registry + Valkey online store)
- Airflow 이미지에 `apache-airflow-providers-cncf-kubernetes`가 필요하다. 로컬 Airflow 3.4.0에는 없어서 DAG import는 클러스터에서 확인한다.
- Pod는 Kubernetes service account `airflow-task`으로 뜬다(`FEAST_WORK_SERVICE_ACCOUNT`로 바꿀 수 있다). GCP 권한은 이 계정의 Workload Identity에 붙는다.
- 자원 요청: CPU 500m / 메모리 2Gi, 한도: CPU 2 / 메모리 4Gi. 첫 실행 때 실측해 조정한다.

Airflow Variables (`feast_work_image`나 `feast_registry_path`가 없으면 파드를 만들기 전에 실패한다. 나머지 값이 비면 파드가 뜬 뒤 몇 초 안에 무엇이 비었는지 출력하고 끝난다. 채우기 전에는 DAG를 켜지 않는다):

| Variable | 값 |
| --- | --- |
| `feast_work_image` | 작업 이미지 태그 (필수, DAG가 템플릿으로 읽는다) |
| `feast_registry_path` | `gs://.../feast/registry.pb` (필수, 읽기·쓰기) |
| `valkey_host` | `valkey.valkey.svc.cluster.local` |
| `gcp_project_id` | `final-project-513508` |
| `feast_bq_dataset` | `final-project-513508.dev_yoonmo` (source table 위치) |
| `feast_bq_temp_dataset` | `dev_yoonmo` (entity_df 임시 표) |
| `feast_full_materialize` | `true`면 전체 적재. 첫 적재와 Valkey가 비었을 때만 |
| `feast_materialize_start` | 전체 적재 시작 시각 `2026-11-03T00:00:00+09:00` |
| `feast_materialize_lookback_hours` | 늦게 온 원천 행을 덮어쓸 구간 (기본 3, 0이면 안 함) |

적재 방식 (`materialize.py`):

- 기본은 증분이다. registry에 기록된 feature view별 마지막 적재 끝에서 Airflow interval 끝까지 이어서 적재한다. 그래서 run이 몇 시간 실패하거나 스케줄이 비어도 그 구간이 빠지지 않는다.
- 증분 적재 뒤에 `end - lookback` ~ `end`를 한 번 더 적재한다. 원천 행이 늦게 들어와도 lookback 안이면 Valkey 값이 고쳐진다. lookback보다 늦게 온 행은 반영되지 않는다.
- `end`가 이미 적재한 끝보다 이르면(늦게 재시도된 run) 아무것도 하지 않고 끝난다. 온라인 값을 뒤로 돌리지 않기 위해서다.
- 적재 이력이 없는 registry에서 증분을 부르면 Feast가 52주 전부터 읽는다. 그래서 이력이 없으면 스크립트가 멈추고 전체 적재를 안내한다.

첫 적재: `feast_full_materialize=true`, `feast_materialize_start=2026-11-03T00:00:00+09:00`로 한 번 돌리고, 이후 `false`로 되돌린다.

Valkey는 디스크가 없다. 노드를 끄면 값이 비므로, 노드를 다시 켠 뒤에도 같은 방법으로 전체 적재를 한 번 돌린다. 노드가 꺼져 있는 동안 매시 DAG는 실패한다.

지금은 `dev_yoonmo`를 쓴다. `dev_feast` 이름 변경은 MVP 이후.

#### registry는 CI와 DAG가 같이 쓴다

Feast는 materialize가 끝나면 적재 구간을 registry에 append한다(`FeatureStore.materialize` → `registry.apply_materialization`). 그래서:

- 작업 파드의 서비스 계정에는 registry 읽기와 쓰기가 모두 필요하다.
- GCS file registry는 파일을 통째로 덮어쓴다. 매시 materialize와 main 머지 시 `feast apply`가 겹치면 나중에 쓴 쪽이 이기고, 진 쪽의 변경(정의 또는 적재 구간)이 사라진다. `feast-apply.yml`의 `concurrency`는 apply끼리만 막는다.

정의 변경 배포 순서: `feast_materialize` DAG 일시 중지 → 정의 PR 머지 → `feast-apply.yml`의 apply 성공 확인 → DAG 재개. apply가 실패했는데 DAG가 켜져 있으면 다음 materialize가 옛 정의를 다시 쓸 수 있다.

## 파일

| 파일 | 하는 일 |
| --- | --- |
| `materialize.py` | Airflow 작업 이미지에서 증분(또는 전체) materialize를 실행하고, 늦게 온 행 구간을 덮어쓴다 |
| `prepare_sources.py` | 가짜 로그 → source table parquet 5개 + entity_df. `--base impression`(기본)이면 impression 1건 = 1행, `exposure`면 유효 노출 1건 = 1행. [계획] Airflow 집계 단계가 BigQuery에 같은 표를 만든다 |
| `feature_repo/definitions.py` | entity 2개, feature view 4개, on-demand feature view 1개, feature service 1개 |
| `feature_repo/feature_store.yaml` | 로컬 설정 (file + sqlite) |
| `feature_repo/feature_store.bq.yaml` | 오프라인만 BigQuery(`dev_yoonmo`), registry·온라인은 로컬 파일 |
| `upload_to_bq.py` | 원천 parquet 4개 → BigQuery 표 (exposures는 entity_df라 올리지 않는다) |
| `feature_repo/feature_store.gcp.yaml.example` | [계획] BigQuery + Valkey 설정 예시 (미실행) |
| `feature_repo/feature_store.ci.yaml` | GitHub Actions의 `feast plan`/`apply`용 설정 (GCS registry, BigQuery offline, SQLite online) |
| `.github/workflows/feast-apply.yml` | PR이면 `feast plan`, main 머지면 GCS registry에 `feast apply`. Settings variables를 넣기 전에는 건너뛴다 |
| `feature_repo/feature_store.gcp.yaml` | GCP 운영 설정 (GCS registry + Valkey online) |
| `../../dags/materialize_feast.py` | `KubernetesPodOperator`로 materialize 작업 이미지를 띄우는 DAG (이미지는 `feast_work_image` Variable) |
| `../../pyproject.toml` | 작업 이미지 의존성 (`airflow-dag`/`train` 묶음) |
| `../../uv.lock` | 작업 이미지 의존성 고정 |
| `../../uv.lock` | 그 의존성의 버전 고정. 작업 이미지와 CI가 같이 쓴다 |
| `build_training_set.py` | `get_historical_features`로 training set을 만들고 `make_features.py` 결과와 대조 |
| `online_demo.py` | `materialize` → `get_online_features` → 같은 시각 오프라인 값과 비교 |

## 피처 정의

| feature view | entity | source table 1행 = | 피처 |
| --- | --- | --- | --- |
| `article_profile` | article | 기사 1건 (시각 = 발행 시각) | `a_emb`(768), `category`, `published_ts` |
| `reader_profile` | reader | 클릭 1건 (그 직후 상태) | `u_emb`(768), `hist_clicks`, `last_click_ts` |
| `reader_activity` | reader | 유효 노출 1건 (그 직후 상태) | `hist_exposures` |
| `reader_article` | reader + article | 상태가 바뀐 사건 1건 | `already_opened`, `already_seen` |
| `request_context` (on-demand) | — | 요청마다 계산 | `article_age_h`, `hour`, `hours_since_last_click` |

- 피처 행 시각 = 사건 시각 + 1µs. 노출과 같은 시각의 사건은 그 노출에 붙지 않는다.
- `u_emb`는 클릭 때만 바뀐다. 타임 디케이 가중치 `0.5 ** (Δt/24h)`는 분자·분모에 같이 곱해져 약분된다.
- `position`은 요청값이다. 학습 땐 entity_df 열, 서빙 땐 요청에서 받는다.
- label(11/15 멘토링): PCTR = P(click | impression). impression마다 `article_open`이 있으면 1, 없으면 0이다. 화면에 안 보인 impression(exposure 없음)도 0으로 넣는다. exposure 기준(11/15 이전)은 유효 노출이 있는 impression만 썼다.
- `exposures.parquet`의 `exposed`는 분석용이다. `training_set.parquet`에서는 뺀다. 모델 입력으로 쓰지 않는다.
- fraud click(11/15 멘토링): 그 impression의 첫 exposure(없으면 `served_at`) 뒤 0.1초 안 `article_open`은 사람이 누른 클릭이 아니므로 label과 피처에서 뺀다. `prepare_sources.remove_fraud_clicks()`가 맡고, [임시] 실서비스에서는 정제 단계가 맡는다.
- 행이 없으면 null이다. `hist_*`, `already_*`는 0으로, `u_emb`는 NaN(cold)으로 채운다 (`build_training_set.to_matrix`).
- ttl은 모두 0(무제한)이다. 피처가 전부 '지금까지' 누적값이라, 기간으로 자르면 오래 쉬다 온 독자가 0으로 학습된다.
- 같은 (entity 키, 시각) 행이 여러 개면 Feast가 어느 행을 고를지 정해져 있지 않아, source table을 쓸 때 마지막 행만 남긴다.

## 결과 (가짜 로그 11/3~11/6 12시)

- (2026-11-16, impression 기준, 로컬 file) training set 103,244행 × 1,545열. label 1 = 2,510건(2.43%), 학습 93,852행 / 검증 9,392행. 유효 노출이 있는 impression은 40,763건이고, 나머지 62,481건은 exposure 없이 label 0이다.
- (2026-11-18, impression 기준, BigQuery) training set 103,244행 × 1,545열. label 1 = 2,510건(2.43%), 학습 93,852행 / 검증 9,392행, cold(u_emb 없음) 50.6%. Storage Read API로 80.8초 걸렸다. `exposed` 열은 뺐다.
- (2026-11-13, exposure 기준) training set 40,763행 × 1,545열. `make_features.py` 결과와 모든 열이 **차이 0**으로 같다. cold 41.5%. 11/15 멘토링 전 방식이다.
- materialize 약 3초. 독자 1명 + 후보 10건 온라인 조회 약 23ms(로컬 SQLite, 두 번째 호출부터).
- 같은 시각의 온라인 값과 오프라인 값이 같다 (training-serving skew 없음).
- (2026-11-14) BigQuery 오프라인으로도 exposure 기준 40,763행, 모든 열 **차이 0**. 그때는 REST로 받아 약 13분 걸렸다.

## 알아 둘 것

- Feast 0.68 file 오프라인 저장소는 feature view를 붙일 때마다 '지금까지 나온 join key + 시각'으로 중복을 지운다.
  exposure 기준에서는 다른 독자가 같은 기사를 같은 ms에 본 노출 2건이 1건으로 합쳐졌다(40,763행 중 1쌍).
  impression 기준에서는 한 요청의 카드가 최대 20장이라 (reader_id, 시각)이 겹치고, (article_id, 시각)도 1,039행이 겹친다.
  `build_training_set.historical()`이 두 키가 겹치는 행을 40개 호출로 나눠 부른다(한 호출 안에서는 겹치지 않는다). BigQuery 저장소에는 이 문제가 없어 한 번에 부른다.
  회피책이 못 막는 경우를 위해 `check_rows()`가 결과 행 수와 row_id를 항상 검사하고, 다르면 멈춘다.
- materialize 범위는 view의 ttl보다 짧으면 안 된다. 짧으면 오프라인엔 있고 온라인엔 없는 값이 생긴다.
  `online_demo.py`는 source table의 가장 이른 시각부터 전체를 쓴다. 서빙 DAG는 첫 적재만 전체, 이후 `materialize_incremental`로 한다.
- BigQuery 결과는 Storage Read API로 받는다. `bigquery.readsessions.create`(`roles/bigquery.readSessionUser`) 권한이 필요하고, 2026-11-18에 프로젝트 수준으로 부여됐다.
- on-demand feature view는 Feast가 "experimental"로 표시한다. 오프라인에서 행이 많으면 느릴 수 있다.
