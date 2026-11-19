"""[파이프라인 구간: 피처 → 서빙 내보내기] Feast online store(Valkey)에 최신 피처를 적재한다.

Airflow의 KubernetesPodOperator가 이 파일을 Python 3.12 작업 이미지에서 실행한다.
Airflow 이미지에는 Feast를 넣지 않는다.

- 첫 실행, 노드 재시작 뒤, 늦게 도착한 원천 행을 반영할 때는 full로 돌린다.
- 평시에는 materialize_incremental로 registry의 마지막 적재 지점부터 이어서 적재하고,
  그 뒤 end - lookback ~ end를 한 번 더 적재해 늦게 온 원천 행을 덮어쓴다.
- end가 이미 적재한 끝보다 이르면 아무것도 하지 않는다(온라인 값을 뒤로 돌리지 않는다).

환경변수:
  FEAST_REGISTRY_PATH   gs://final-project-513508-model-artifacts/feast/registry.pb
  VALKEY_HOST           valkey.valkey.svc.cluster.local (기본)
  FEAST_BQ_DATASET      source table이 있는 데이터셋 (기본 final-project-513508.dev_yoonmo)
  FEAST_BQ_TEMP_DATASET entity_df 임시 표를 만드는 데이터셋 (기본 dev_yoonmo)
  MATERIALIZE_END       Airflow data_interval_end (ISO 8601)
  FEAST_MATERIALIZE_LOOKBACK_HOURS 늦게 온 원천 행을 덮어쓸 구간 (기본 3, 0이면 안 함)
  FEAST_FULL_MATERIALIZE true면 FEAST_MATERIALIZE_START부터 전체를 다시 적재
  FEAST_MATERIALIZE_START 전체 적재 시작 시각 (ISO 8601, full일 때만)
"""
import os
import time
from datetime import datetime, timedelta, timezone
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


def aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def main() -> None:
    os.environ.setdefault("FEAST_SOURCE", "bigquery")
    os.environ.setdefault("FEAST_BQ_DATASET", "final-project-513508.dev_yoonmo")
    os.environ.setdefault("GCP_PROJECT_ID", "final-project-513508")
    os.environ.setdefault("FEAST_BQ_TEMP_DATASET", "dev_yoonmo")
    os.environ.setdefault("VALKEY_HOST", "valkey.valkey.svc.cluster.local")
    if not os.environ.get("FEAST_REGISTRY_PATH"):
        raise SystemExit("FEAST_REGISTRY_PATH 환경변수가 없다")

    try:
        lookback_hours = float(os.environ.get("FEAST_MATERIALIZE_LOOKBACK_HOURS", "3"))
    except ValueError as exc:
        raise SystemExit("FEAST_MATERIALIZE_LOOKBACK_HOURS가 숫자가 아니다") from exc

    full = os.environ.get("FEAST_FULL_MATERIALIZE", "false").lower() == "true"
    end = aware(parse_time("MATERIALIZE_END"))
    store = FeatureStore(repo_path=str(REPO), fs_yaml_file=str(REPO / FS_YAML))
    started = time.time()

    if full:
        start = aware(parse_time("FEAST_MATERIALIZE_START"))
        print(f"materialize(full) {start.isoformat()} ~ {end.isoformat()}")
        store.materialize(start_date=start, end_date=end)
    else:
        ends = [
            aware(end_time)
            for view in store.list_feature_views()
            for _, end_time in getattr(view, "materialization_intervals", [])
        ]
        if not ends:
            raise SystemExit(
                "materialization 이력이 없다. feast_full_materialize=true로 먼저 full 적재한다."
            )
        latest = max(ends)
        if end <= latest:
            print(f"이미 {latest.isoformat()}까지 적재했다. end={end.isoformat()}라 아무것도 하지 않는다.")
            return

        print(f"materialize_incremental ~ {end.isoformat()}")
        store.materialize_incremental(end_date=end)
        if lookback_hours > 0:
            lookback_start = end - timedelta(hours=lookback_hours)
            print(f"lookback materialize {lookback_start.isoformat()} ~ {end.isoformat()}")
            store.materialize(start_date=lookback_start, end_date=end)

    print(f"materialize 완료: {time.time() - started:.1f}초")


if __name__ == "__main__":
    main()
