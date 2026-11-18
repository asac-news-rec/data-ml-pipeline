"""[파이프라인 구간: 피처 → 서빙 내보내기] Feast online store(Valkey)에 최신 피처를 적재한다.

Airflow의 KubernetesPodOperator가 이 파일을 Python 3.12 작업 이미지에서 실행한다.
Airflow 이미지에는 Feast를 넣지 않는다.

환경변수:
  FEAST_REGISTRY_PATH   gs://final-project-513508-model-artifacts/feast/registry.pb
  VALKEY_HOST           valkey.valkey.svc.cluster.local (기본)
  FEAST_BQ_DATASET      source table이 있는 데이터셋 (기본 final-project-513508.dev_yoonmo)
  FEAST_BQ_TEMP_DATASET entity_df 임시 표를 만드는 데이터셋 (기본 dev_yoonmo)
  MATERIALIZE_START     Airflow data_interval_start (ISO 8601)
  MATERIALIZE_END       Airflow data_interval_end (ISO 8601)
  FEAST_FULL_MATERIALIZE true면 FEAST_MATERIALIZE_START부터 전체를 다시 적재
  FEAST_MATERIALIZE_START 전체 적재 시작 시각 (ISO 8601, full일 때만)
"""
import os
import time
from datetime import datetime
from pathlib import Path

from feast import FeatureStore

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("FEAST_REPO_PATH", HERE / "feature_repo"))
FS_YAML = os.environ.get("FEAST_FS_YAML", "feature_store.gcp.yaml")


def parse_time(name: str) -> datetime:
    text = os.environ.get(name, "").strip()
    if not text:
        raise SystemExit(f"{name} 환경변수가 없다")
    return datetime.fromisoformat(text)


def main() -> None:
    os.environ.setdefault("FEAST_SOURCE", "bigquery")
    os.environ.setdefault("FEAST_BQ_DATASET", "final-project-513508.dev_yoonmo")
    os.environ.setdefault("GCP_PROJECT_ID", "final-project-513508")
    os.environ.setdefault("FEAST_BQ_TEMP_DATASET", "dev_yoonmo")
    os.environ.setdefault("VALKEY_HOST", "valkey.valkey.svc.cluster.local")
    if not os.environ.get("FEAST_REGISTRY_PATH"):
        raise SystemExit("FEAST_REGISTRY_PATH 환경변수가 없다")

    full = os.environ.get("FEAST_FULL_MATERIALIZE", "false").lower() == "true"
    end = parse_time("MATERIALIZE_END")
    start = parse_time("FEAST_MATERIALIZE_START") if full else parse_time("MATERIALIZE_START")

    store = FeatureStore(repo_path=str(REPO), fs_yaml_file=str(REPO / FS_YAML))
    print(f"materialize {start.isoformat()} ~ {end.isoformat()} (full={full})")
    started = time.time()
    store.materialize(start_date=start, end_date=end)
    print(f"materialize 완료: {time.time() - started:.1f}초")


if __name__ == "__main__":
    main()
