# data-ml-pipeline
Data pipeline (ingest, clean, features) and ML training

## 현재 구성

- `dags/`: Airflow DAG. 연결 확인용 `hello_airflow.py`(#12), 행동 로그 GCS 내보내기 `export_behavior_logs.py`(#14), Feast materialize `materialize_feast.py`(#16), GCS → BigQuery raw 적재 `load_raw.py`(#32), raw → clean 정제 `clean_events.py`(#34), 기사 임베딩 `embed_articles.py`(#36), Feast 원천 표 `build_feast_sources.py`(#38)가 있다.
- `transform/`: 정제 코드. `clean_events.py`가 `sql/clean_events.sql`로 `clean.*`을 만든다. `build_feast_sources.py`는 `clean.actions` + `feat.article_*`로 Feast 원천 표 4개를 매시 다시 만든다(SQL `transform/sql/build_feast_sources.sql`, `prepare_sources.py`와 같은 계산).
- `ingest/`: 적재 코드. `load_raw.py`가 GCS 행동 로그를 `raw.served`·`raw.actions`에 넣는다. `embed_articles.py`가 Cloud SQL 새 기사를 multilingual-e5-base로 임베딩해 `feat.article_embedding`·`feat.article_meta`에 넣는다.
- `features/feast/`: Feast 피처 스토어. 로컬(parquet + SQLite), BigQuery 오프라인, GCS registry + Valkey 온라인을 쓴다. 실행 방법은 `features/feast/README.md`.
- `tests/`: `test_export_bounds.py`(unittest). 행동 로그 내보내기 구간 입력과 업로드 → 정리 의존성을 검증한다. `test_load_raw.py`는 raw 적재의 시간 구간·중복 건너뛰기를, `test_clean_events.py`는 정제 범위·표 이름·파티션 필터를 검증한다.
- `training/`, `serving/`, `docs/`: [계획] 아직 없다.

## 개발 환경

- Python 3.12. 의존성과 버전은 저장소 루트 `pyproject.toml`과 `uv.lock`으로 고정한다.
- Feast materialize 작업 이미지: `uv sync --frozen --extra airflow-dag`
- 기사 임베딩 작업 이미지: `uv sync --frozen --extra embed` (torch는 CPU 빌드). 모델 `intfloat/multilingual-e5-base`를 이미지에 미리 내려받아 두고 Pod는 `HF_HUB_OFFLINE=1`로 돈다.
- 학습 Job 이미지: `uv sync --frozen --extra train`
- lint·테스트 도구는 [미정]이다. 지금은 `python -m unittest discover -s tests -v`로 테스트를 돌린다.

## GCS → BigQuery raw 적재 (`load_raw`)

`dags/load_raw.py`가 매시 작업 이미지(`airflow-dag` 묶음)로 `ingest/load_raw.py`를 실행한다.

- 입력: `export_behavior_logs`가 쓴 `{prefix}/served|actions/interval_start={UTC}/interval_end={UTC}/part-{job_id}.jsonl.gz`.
- 범위: `data_interval_start`가 든 UTC 한 시간과 앞 `raw_load_lookback_hours`(기본 3)시간에 **시작하는** 구간의 파일.
- 출력: `raw.served`(`impression_id`, `request_id`, `served_at`), `raw.actions`(`event_id`, `event_type`, `event_time`, `impression_id`, `event_type` 클러스터링). 두 표 모두 `payload`(줄 전체 JSON), `source_file`, `load_date`, `loaded_at`을 가진다. 열은 청사진 overview.html 인터페이스 규약 ⑤를 따른다.
- `load_date`는 파일 구간 시작의 한국 날짜이고 파티션 열이다.
- 중복 방지: GCS 객체는 덮어쓰지 않으므로 `source_file`이 이미 표에 있으면 건너뛴다. 같은 시간을 다시 돌려도 행이 늘지 않는다. 새 파일은 load job 하나로 넣어 실패하면 아무 행도 남지 않는다.
- 표가 없으면 첫 적재 때 만든다.
- Airflow Variable: `feast_work_image`, `gcp_project_id`, `raw_gcs_bucket`(필수), `raw_gcs_prefix`, `raw_bq_dataset`(기본 `raw`), `raw_load_lookback_hours`.
- 권한: 작업 Pod 서비스 계정에 로그 버킷 객체 나열·읽기, `raw` 데이터셋 쓰기, BigQuery job 실행.

## raw → clean 정제 (`clean_events`)

`dags/clean_events.py`가 같은 시간의 `load_raw`가 끝나길 기다린 뒤 `transform/clean_events.py`를 작업 이미지로 실행한다. 범위는 `load_raw`와 같다(그 시간 + 앞 3시간에 시작한 내보내기 구간).

| 표 | 한 행 | 규칙 |
|---|---|---|
| `clean.served` | impression 1건 | `impression_id`로 중복 제거. 파티션 `served_date`(한국 날짜) |
| `clean.actions` | 행동 1건 | `event_id`(소문자)로 중복 제거, `impression_id`로 served 값을 붙임. 파티션 `event_date` |
| `clean.orphan_actions` | 짝 served가 아직 없는 행동 | 다음 실행마다 다시 보고, 짝이 생기면 `clean.actions`로 옮김 |
| `clean.rejected_actions` | 버린 행동 + 이유 | `fraud_click`: 첫 exposure 시작(없으면 `served_at`) 뒤 0.1초 안의 article_open (11/15 멘토링). `invalid_event`: 계약 밖 값 |

- 쓰기는 모두 MERGE라 같은 시간을 다시 돌려도 행이 늘지 않는다.
- `is_bot`은 봇 기준이 정해지지 않아 모두 false다.
- 열은 청사진 overview.html 인터페이스 규약 ⑤를 따른다. 규약 밖으로 `clean.served.received_at`을 더 둔다(늦게 온 행 찾기).
- 시험: `CLEAN_BQ_DATASET=dev_<이름> CLEAN_TABLE_PREFIX=clean_ RAW_BQ_DATASET=dev_<이름>`로 개인 dataset에 만든다.

## 기사 임베딩 (`embed_articles`)

`dags/embed_articles.py`가 매시 임베딩 작업 이미지로 `ingest/embed_articles.py`를 실행한다.

- 입력: Cloud SQL `public.news_articles`(backend `generate/load_postgres.py`). 접속은 `cloud-sql-proxy`, 비밀번호는 Secret Manager.
- 글: `passage: 제목 부제 요약` (로컬 `article_embedding/embed.py`와 같다). 512토큰을 넘으면 잘린다(MVP 뒤에 다룬다).
- 출력: `feat.article_embedding`(article_id, embedding 768칸·길이 1, model_name, model_version, created_at), `feat.article_meta`(article_id, category, subcategory, tags, publisher_id, published_at, clickbait_score, copied_at). 열은 청사진 overview.html 인터페이스 규약 ⑤를 따르고 `subcategory`를 더 둔다.
- 중복 방지: 같은 모델·버전으로 이미 임베딩한 기사는 건너뛴다. 메타는 `article_id`가 없을 때만 넣는다.
- Airflow Variable: `embed_work_image`, `gcp_project_id`, `embed_db_user`, `embed_db_password_secret`(필수), `embed_db_name`, `feat_bq_dataset`, `embed_model_revision`, `embed_limit`.
- 권한: 작업 Pod 계정에 Secret 읽기, `feat` 쓰기, BigQuery job 실행. DB는 `news_articles` 읽기 전용 계정.

## 행동 로그 GCS 내보내기

`dags/export_behavior_logs.py`는 PostgreSQL에 커밋된 행동 로그를 GCS에
gzip JSONL과 manifest로 저장합니다. PostgreSQL 수신 시각(`received_at`) 기준으로
UTC 구간의 시작은 포함하고 끝은 제외합니다. 별도 태스크는 내보내기가 확인된
PostgreSQL 레코드를 기존 보관 정책에 따라 정리합니다.

### Airflow 실행 환경

- [프런트엔드 #38의 머지 커밋](https://github.com/asac-news-rec/frontend/commit/8bacd79cef622b67aef9cc18ee9bae7bcdfce75c)에서
  `services/logging-server/logging_service/`를 태스크 실행 이미지의 Python 경로에
  포함합니다. 패키지 내 SQL 파일도 함께 포함합니다. 같은 이미지에
  `psycopg[binary]==3.3.8`과 `google-cloud-storage==3.17.1`을 설치합니다.
- 태스크에 `LOGGING_DATABASE_URL`을 Secret으로 주입하고 `GCS_BUCKET`,
  `GOOGLE_CLOUD_PROJECT`, 필요 시 `GCS_PREFIX`를 설정합니다. 태스크에서
  PostgreSQL에 연결할 수 있어야 하며 로깅 API가 스키마를 먼저 초기화해야 합니다.
- 태스크의 서비스 계정이 Workload Identity와 ADC로 대상 버킷에 접근하도록
  구성합니다. 업로드, 응답 유실 후 검증, 삭제 후속 처리에 객체 생성·읽기·삭제
  권한이 필요합니다. [인프라 요청 #37](https://github.com/asac-news-rec/infra/issues/37)

### 검증과 활성화

Airflow가 설치된 환경에서 `python -m unittest discover -s tests -v`로 구간 입력과
업로드 → 정리 의존성을 검증합니다. 실제 PostgreSQL·GCS 내보내기는 아래 순서로 확인합니다.

1. Airflow의 DAG import 오류를 확인하고 `export_behavior_logs`가 등록되었는지 확인합니다.
2. 검증 환경에서 로그가 들어 있는 UTC 구간을 수동 실행 설정에 넣습니다.
   예시는 `{"interval_start":"2026-11-16T00:00:00Z","interval_end":"2026-11-16T00:05:00Z"}`입니다.
3. `upload`와 `prune_postgresql` 상태를 확인합니다. 해당 구간 경로에 gzip JSONL과
   manifest가 생성되고 같은 구간 재실행에서 중복 객체가 생기지 않는지 확인합니다.

자동 실행을 시작할 때 DAG 파서에 `BEHAVIOR_EXPORT_CRON`(UTC)과
`BEHAVIOR_EXPORT_START`(UTC ISO 8601)를 함께 설정합니다. DAG는 처음에 일시
정지 상태로 등록됩니다. `catchup=True`이므로 처리할 첫 데이터 구간에 맞춰
시작 시각을 정한 뒤 활성화합니다.

수동 입력은 두 경계를 함께 요구하며 시간대, 밀리초 정렬, `start < end`를
검증합니다. 잘못된 입력은 `AirflowFailException`으로 재시도 없이 실패합니다.
오프셋이 있는 값은 UTC로 정규화합니다. `upload` 성공 후에만 `prune_postgresql`을 실행합니다.

이미지의 `PostgresStore.prune_exported_records()`는 `uploaded` 상태와 GCS generation
receipt가 있는 작업에 포함된 레코드만 정리하고, 수신·발생 시각 모두 24시간이 지난
행을 대상으로 합니다. 작업 ID와 객체명·receipt는 보존합니다.
`Exporter.flush()`는 데이터와 manifest 저장 뒤 작업을 uploaded로 표시합니다.
일부 업로드 후 실패하면 고정된 객체명·내용으로 재시도하며, `GCSSink.put()`은
generation 조건부 생성과 기존 내용 비교로 중복·충돌을 검사합니다.
구간은 발생 시각이 아닌 `received_at` 기준이므로 발생 시각만 늦은 이벤트는 실제
수신 시각이 속한 구간에 포함됩니다. 원래 발생 시각은 JSONL 레코드에 유지하며
객체 경로는 수신 구간을 사용합니다.
다만 `received_at`이 이미 내보낸 과거 구간에 속하는 레코드가 내보내기 조회 이후
커밋되는 경우, 다음 구간의 필터로는 수집되지 않습니다. `catchup=True`도 성공한
구간을 자동 재실행하지 않으므로 해당 과거 구간을 같은 경계로 수동 재실행해야 합니다.
재실행 시 `prepare_exports()`는 `export_members`에 없는 레코드만 새 작업으로 만들고,
기존 미완료 작업은 저장된 객체명과 내용으로 재시도합니다. 이미 완료된 작업은
다시 업로드하지 않으며, 추가 레코드는 별도 객체와 manifest로 저장합니다.

## 에이전트 하네스

AI 코딩 도구가 이 저장소에서 일할 때 따르는 규칙은 `AGENTS.md`에
있고, 사람용 협업 규칙은 `CONTRIBUTING.md`에 있습니다.
이 규칙들은 우리 [infra](https://github.com/asac-news-rec/infra) 레포의 하네스를 가져와
이 레포에 맞게 줄이고 수정한 것입니다 (경량 구성: 상세 가이드 폴더 `.claude/docs/`는 아직 넣지 않음).

## 뉴스 생성 DAG

[현재] `dags/generate_news.py`는 KubernetesPodOperator로 생성기 Pod를 실행합니다.
Pod 설정은 함께 동기화되는 `dags/pod_templates/news-generator.yaml`에서 읽습니다.
생성기 소스·CSV·패키지는 백엔드 저장소에서 빌드한 이미지에 포함됩니다.

### 실행 설정

| 항목 | 현재 값 |
| --- | --- |
| 일정 | 한국 시간 기준 5분마다 (`*/5 * * * *`) |
| 생성 건수 | 실행당 기사 1건 |
| 썸네일 | 보관 백업의 누적 기사 수 기준 10건당 1장 |
| 동시 실행 | `max_active_runs=1` |
| 재시도 | 최대 2회, 간격 5분 |
| 과거 예약 자동 실행 | `catchup=False` |
| 최초 등록 | 일시 중지 (`is_paused_upon_creation=True`) |
| 실행 제한 | 태스크당 1시간 |
| Namespace | `airflow` |
| Kubernetes 서비스 계정 | `airflow-task` |
| 생성기 컨테이너 이름 | `base` |
| PVC | `news-output` (2Gi, ReadWriteOnce) |
| 마운트·출력 경로 | `/mnt/news-output` |
| 기사 모델 | `google/gemini-2.5-flash` |
| 이미지 모델 | `black-forest-labs/flux.2-klein-4b` |

배치 날짜는 예약 실행의 `data_interval_end`를 한국 시간으로 변환해 사용합니다.
수동 실행에서 이 값이 없거나 null이면 해당 DagRun의 고정된 `run_after`를 사용합니다.
현재 시각을 다시 계산하지 않으므로 재시도·동일 DagRun Clear에서도 배치 날짜가 유지됩니다.
실행 ID는 DAG ID와 Airflow `run_id`를 연결해 전달하므로 같은 태스크의 재시도에서는
같은 날짜·실행 ID로 백업을 재사용합니다. JSONL에 같은 `generation.run_id`와 순번의
기사가 있으면 기사·이미지 생성 API를 다시 호출하지 않고 GCS 확인과 DB UPSERT를 재시도합니다.
DB의 동일 기사 ID는 UPDATE하므로 중복 INSERT하지 않습니다. 해당 순번이 저장되기 전에
실패했거나 백업이 7일 정책·수동 삭제·PVC 교체로 없어졌다면 새 기사 생성과 API 호출이 발생할 수
있습니다. Clear도 같은 DagRun일 때만 이 조건을 따르며 새로 트리거한 DagRun은 별도 생성 작업입니다.
작업이 끝나면 Pod는 삭제되지만 PVC 파일은 유지됩니다.

### 이미지 버전 수동 반영

[현재] 백엔드 생성기 CD는 이미지 빌드·검사·Artifact Registry 업로드까지 수행합니다.
pipeline 저장소의 이슈·PR은 자동 생성하지 않으며 `PIPELINE_REPO_TOKEN`은 필요하지 않습니다.

이미지 주소는 `dags/generate_news.py`의 `GENERATOR_IMAGE` 한 곳에서 관리합니다.
Pod YAML은 실행 계정·환경변수·볼륨 등의 병합용 템플릿이며 `image` 값을 두지 않습니다.
실제 Pod에는 KubernetesPodOperator의 `image=GENERATOR_IMAGE` 값이 적용됩니다.
현재 주소는 다음과 같습니다.

```text
asia-northeast3-docker.pkg.dev/final-project-513508/news-rec/news-generator:54317e07df0dc8076c2949f8d66d0368bbdd4af6
```

생성기를 갱신할 때는 다음 순서로 진행합니다.

1. 백엔드 Actions → generator-cd → 성공한 실행 → Summary의 이미지 주소를 복사합니다.
2. `dags/generate_news.py`의 `GENERATOR_IMAGE`를 새 전체 주소로 바꿉니다.
3. 이 저장소의 이슈 → 연결 브랜치 → PR → 승인·머지 절차로 반영합니다.

머지 후 기존 git-sync가 DAG와 템플릿을 Airflow에 전달합니다. 최초 등록한 DAG는
일시 중지 상태이므로 인프라 준비와 실행 검증 후 활성화합니다.

### Secret·GCS·DB 연결

비밀값 자체를 YAML에 넣지 않습니다. 생성기가 Workload Identity/ADC 인증으로
아래 Secret Manager 버전 경로를 Google SDK에서 조회합니다.

| 환경변수 | 값 |
| --- | --- |
| `OPENROUTER_API_KEY_SECRET` | `projects/final-project-513508/secrets/openrouter-api-key/versions/latest` |
| `PGPASSWORD_SECRET` | `projects/final-project-513508/secrets/news-db-password/versions/latest` |
| `PGHOST` | `cloud-sql-proxy` |
| `PGPORT` | `5432` |
| `PGDATABASE` | `newsdb` |
| `PGUSER` | `news_app` |
| `THUMBNAIL_BUCKET` | `final-project-513508-news-thumbnails` |
| DAG의 `IMAGE_BASE_URL` | `https://storage.googleapis.com/final-project-513508-news-thumbnails/images` |

Pod는 같은 `airflow` namespace의 공유 Cloud SQL Proxy Service를 사용합니다.
생성기 Pod 안에 별도 Proxy 사이드카는 없습니다. Cloud SQL 접속 IAM 권한은 공유
Proxy 실행 계정이 사용하며, 생성기는 DB 비밀번호와 기사 적재 권한이 있는 DB 계정을 사용합니다.
기사 테이블은 사전에 준비되어 있어야 합니다.

썸네일은 PVC에 저장한 뒤 GCS의 `images/YYYY-MM-DD/images/<기사ID>.png`로 업로드합니다.
이미지가 없는 기사의 `image_url`은 null일 수 있습니다. 업로드된 URL을 Cloud SQL에 적재하며,
재시도에서 같은 객체가 있으면 내용을 확인하고 재사용합니다.

[현재: 인프라 담당자의 11월 20일 이슈 #59 회신 기준] `openrouter-api-key` 등록과
GCP 서비스 계정 `airflow`의 두 Secret 읽기 권한, 썸네일 버킷 `images/` 아래 객체 생성·읽기
권한 설정은 완료됐습니다. PVC 이름·용량은 확정됐지만 생성은 클러스터 가동 시 적용 예정입니다.
`airflow/airflow-task`와 GCP 계정 `airflow`의 연결도 인프라 적용이 필요합니다.
실제 Pod의 Secret 조회·GCS 업로드·DB 적재 성공을 확인한 상태는 아닙니다.
관련 요청: https://github.com/asac-news-rec/infra/issues/59

### 백업 보관

생성기는 정상 생성·업로드·DB 저장 후 한국 시간 기준 오늘 포함 최근 7일의 날짜 폴더만
보관합니다. 기간이 지난 PVC 폴더의 JSONL·PNG를 함께 삭제하고 삭제 경로를 로그에 남깁니다.
GCS 이미지와 DB 기사는 삭제하지 않습니다. 정리 실패는 경고로 표시하고 생성 성공은 유지합니다.
실행이 중지된 동안에는 정리하지 않으며 다음 성공 실행에서 정리합니다.

과거 날짜를 수동 지정해 보관 범위를 벗어나면 생성 성공 직후 그 백업도 삭제됩니다.
백업 삭제는 썸네일 순번의 집계 기준을 바꿀 수 있습니다. 예를 들어 29건 중 3건의 백업을
삭제하면 다음 썸네일은 신규 1건째가 아니라 신규 4건째에 생성됩니다.

### 검증 및 운영

커밋 전 Python 문법과 공백 오류를 확인합니다.

```bash
python -m py_compile dags/generate_news.py
git diff --check
```

Airflow와 Kubernetes provider가 설치된 환경에서는 다음으로 실제 DAG를 import하고
날짜 인자를 렌더링합니다. 예약 실행, 수동 실행의 null·누락 값, 한국 날짜 경계,
같은 실행의 재시도·Clear 인자 유지와 이미지 설정을 외부 API·클러스터 접속 없이 검사합니다.

```bash
python -m unittest discover -s tests -p test_generate_news.py
```

실제 Airflow 환경에는 `airflow.sdk`를 지원하는 버전과
`apache-airflow-providers-cncf-kubernetes`가 필요합니다. 로컬 개발용 의존성 파일을
올리는 것만으로 인프라의 Airflow 이미지에 패키지가 설치되지는 않습니다.
Airflow 환경에서 다음 명령으로 DAG 로드 상태를 확인합니다.

```bash
airflow dags list-import-errors
airflow dags list
```

클러스터 가동 후 PVC Bound 상태, 실행 계정 연결, 스케줄러의 Pod 생성·조회·로그·삭제 권한,
공유 Proxy 상태와 DB 계정의 INSERT·UPDATE 권한을 확인합니다.
일시 중지 상태에서 수동으로 1건을 생성해 JSONL·DB 적재와 재시도 시 기사 재사용을 검증합니다.
썸네일은 10건마다 생성되므로 첫 1건에 PNG가 없는 것은 정상입니다. 썸네일 생성 대상까지
누적 실행한 뒤 GCS 업로드와 화면의 이미지 표시를 확인하고 예약 실행을 활성화합니다.

장애 시 DAG를 일시 중지하고 `generate_news.py`의 `GENERATOR_IMAGE`를 이전 버전으로
되돌립니다. 이미 삭제한 PVC 파일은 이미지 롤백으로 복구되지 않습니다.
OpenRouter 사용료와 이미지 저장·전송 비용은 실행 주기와 생성 건수에 따라 발생합니다.

### 로컬 검증 파일과 커밋 범위

로컬 Airflow용 Compose·설정·로그와 개발용 `pyproject.toml`·`uv.lock`은 이 뉴스 생성 DAG
배포에 포함하지 않으며 개인 저장소의 `.git/info/exclude`로 제외합니다. 파일은 PC에 유지됩니다.
공유 `.gitignore`에서는 `pyproject.toml`·`uv.lock`·설정 디렉터리를 차단하지 않습니다.
`dags/hello_airflow.py`는 기존 연결 확인 DAG이므로 이번 뉴스 생성 PR에 로컬 실험 변경을
함께 넣지 않습니다. 이미 추적 중인 파일은 `.gitignore`로 제외되지 않으므로 파일을 지정해 추가합니다.

이슈 연결 브랜치를 준비한 뒤 이번 배포 파일만 스테이징합니다.

```bash
git add README.md dags/generate_news.py dags/pod_templates/news-generator.yaml tests/test_generate_news.py
git diff --cached --stat
git diff --cached --check
```
