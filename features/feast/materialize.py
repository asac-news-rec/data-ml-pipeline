"""[파이프라인 구간: 피처 → 서빙 내보내기] Feast online store(Valkey)에 최신 피처를 적재한다.

Airflow의 KubernetesPodOperator가 이 파일을 Python 3.12 작업 이미지에서 실행한다.
Airflow 이미지에는 Feast를 넣지 않는다.

- 첫 실행, 노드 재시작 뒤에는 full로 돌린다.
- 평시에는 materialize_incremental로 feature view별 마지막 적재 지점부터 이어서 적재하고,
  그 뒤 end - lookback ~ end를 한 번 더 적재해 늦게 온 원천 행을 덮어쓴다.
- 적재는 view별로 넘긴다. 뒤처진 view만 증분하고, 이미 end보다 앞선 view는
  lookback에서도 빼서 온라인 값을 뒤로 돌리지 않는다.
- 같은 end로 재시도하면 증분은 건너뛰고 lookback만 다시 실행한다. 증분이 성공한 뒤
  lookback이 실패해도 다음 시도에서 그 구간을 덮는다.

환경변수(전부 필수, 기본값 없음):
  FEAST_REGISTRY_PATH   gs://final-project-513508-model-artifacts/feast/registry.pb
  GCP_PROJECT_ID        BigQuery 프로젝트
  VALKEY_HOST           valkey.valkey.svc.cluster.local
  FEAST_BQ_DATASET      source table이 있는 데이터셋
  FEAST_BQ_TEMP_DATASET entity_df 임시 표를 만드는 데이터셋
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


def require_env(names: list[str]) -> None:
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise SystemExit(f"필수 환경변수가 없다: {', '.join(missing)}")


def view_end_times(store: FeatureStore) -> dict[str, datetime | None]:
    """feature view별 마지막 적재 끝 시각. 적재 이력이 없으면 None."""
    states: dict[str, datetime | None] = {}
    for view in store.list_feature_views():
        ends = [
            aware(end_time)
            for _, end_time in getattr(view, "materialization_intervals", [])
        ]
        states[view.name] = max(ends) if ends else None
    return states


def main() -> None:
    os.environ.setdefault("FEAST_SOURCE", "bigquery")
    require_env([
        "FEAST_REGISTRY_PATH",
        "GCP_PROJECT_ID",
        "VALKEY_HOST",
        "FEAST_BQ_DATASET",
        "FEAST_BQ_TEMP_DATASET",
    ])

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
        states = view_end_times(store)
        if not states:
            raise SystemExit("feature view가 없다. feast apply를 먼저 실행한다.")

        missing = sorted(name for name, latest in states.items() if latest is None)
        if missing:
            raise SystemExit(
                "적재 이력이 없는 feature view가 있다: "
                + ", ".join(missing)
                + ". feast_full_materialize=true로 먼저 full 적재한다."
            )

        # 모든 view가 end보다 앞서 있으면 온라인 값을 뒤로 돌리지 않는다.
        if all(latest > end for latest in states.values()):
            print(
                f"이미 {min(states.values()).isoformat()} 이후까지 적재했다. "
                f"end={end.isoformat()}라 아무것도 하지 않는다."
            )
            return

        behind = sorted(name for name, latest in states.items() if latest < end)
        at_end = sorted(name for name, latest in states.items() if latest == end)

        if behind:
            print(f"materialize_incremental ~ {end.isoformat()} (view: {', '.join(behind)})")
            store.materialize_incremental(end_date=end, feature_views=behind)

        # at_end는 이 end까지 증분이 끝났지만 lookback이 실패했을 수 있는 view다.
        # 같은 end로 재시도해도 lookback을 다시 실행해 늦게 온 행을 덮는다.
        targets = behind + at_end
        if lookback_hours > 0 and targets:
            lookback_start = end - timedelta(hours=lookback_hours)
            print(
                f"lookback materialize {lookback_start.isoformat()} ~ {end.isoformat()} "
                f"(view: {', '.join(targets)})"
            )
            store.materialize(
                start_date=lookback_start, end_date=end, feature_views=targets
            )

    print(f"materialize 완료: {time.time() - started:.1f}초")


if __name__ == "__main__":
    main()
