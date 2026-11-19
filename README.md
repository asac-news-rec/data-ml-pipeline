# data-ml-pipeline
Data pipeline (ingest, clean, features) and ML training

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
