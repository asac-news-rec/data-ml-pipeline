# data-ml-pipeline
Data pipeline (ingest, clean, features) and ML training

## 현재 구성

- `dags/`: Airflow DAG. 연결 확인용 `hello_airflow.py`(#12)와 Feast materialize `materialize_feast.py`(#16)가 있다.
- `features/feast/`: Feast 피처 스토어. 로컬(parquet + SQLite), BigQuery 오프라인, GCS registry + Valkey 온라인을 쓴다. 실행 방법은 `features/feast/README.md`.
- `training/`, `serving/`, `docs/`: [계획] 아직 없다.

## 개발 환경

- Python 3.12. 의존성과 버전은 저장소 루트 `pyproject.toml`과 `uv.lock`으로 고정한다.
- Feast materialize 작업 이미지: `uv sync --frozen --extra airflow-dag`
- 학습 Job 이미지: `uv sync --frozen --extra train`
- lint·테스트 도구는 [미정]이다.

## 에이전트 하네스

AI 코딩 도구가 이 저장소에서 일할 때 따르는 규칙은 `AGENTS.md`에
있고, 사람용 협업 규칙은 `CONTRIBUTING.md`에 있습니다.
이 규칙들은 우리 [infra](https://github.com/asac-news-rec/infra) 레포의 하네스를 가져와
이 레포에 맞게 줄이고 수정한 것입니다 (경량 구성: 상세 가이드 폴더 `.claude/docs/`는 아직 넣지 않음).
