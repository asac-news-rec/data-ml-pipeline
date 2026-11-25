"""[파이프라인 구간: 피처 → 학습 데이터] clean 로그와 Feast offline(BigQuery)으로 training set을 만든다.

impression(clean.served) 1건 = 1행. label은 그 impression에 article_open이 있으면 1이다.
피처는 served_at 시점 값만 붙인다(get_historical_features, point-in-time join).

환경변수:
  GCP_PROJECT_ID        BigQuery 프로젝트
  TRAIN_END             학습 데이터의 끝 (Airflow data_interval_end, 시간대 포함 ISO 8601)
  TRAIN_DAYS            며칠치를 볼지 (기본 14, 임시값)
  TRAIN_VALID_HOURS     마지막 몇 시간을 valid로 둘지 (기본 24)
  TRAIN_LABEL_WAIT_HOURS  클릭이 도착할 때까지 기다리는 시간. 이보다 최근 impression은 뺀다 (기본 3)
  CLEAN_BQ_DATASET      기본 clean
  TRAINING_SET          출력 parquet (기본 features/feast/data/training_set.parquet)
  FEAST_FS_YAML, FEAST_REGISTRY_PATH, FEAST_BQ_TEMP_DATASET, VALKEY_HOST  Feast 설정
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
NOT_ENOUGH_CLICKS = 99   # training/train_lgbm.py, dags/train_pctr.py와 같다

QUERY = """
SELECT s.impression_id, s.reader_id, s.article_id, s.served_at AS event_timestamp, s.position,
       IF(o.impression_id IS NULL, 0, 1) AS label
FROM `{clean}.served` AS s
LEFT JOIN (
  SELECT DISTINCT impression_id FROM `{clean}.actions`
  WHERE event_date BETWEEN @first_date AND @action_last_date
    AND event_type = 'article_open' AND event_time < @action_end
) AS o USING (impression_id)
WHERE s.served_date BETWEEN @first_date AND @last_date
  AND s.served_at >= @start AND s.served_at < @stop
ORDER BY s.served_at, s.reader_id, s.position, s.impression_id
"""


def window(end: datetime, days: int, valid_hours: int, wait_hours: int) -> dict:
    """served 범위 [start, stop), valid 시작, 파티션 날짜. 파티션은 KST와 UTC 어느 쪽 날짜여도 들어오게 잡는다."""
    if end.utcoffset() is None:
        raise ValueError("TRAIN_END에 시간대가 없다")
    end = end.astimezone(timezone.utc)
    stop = end - timedelta(hours=wait_hours)
    start = stop - timedelta(days=days)
    valid_from = stop - timedelta(hours=valid_hours)
    if not start < valid_from < stop:
        raise ValueError("TRAIN_VALID_HOURS는 TRAIN_DAYS보다 짧고 0보다 커야 한다")
    return {"start": start, "stop": stop, "valid_from": valid_from, "action_end": end,
            "first_date": start.date(), "last_date": stop.astimezone(KST).date(),
            "action_last_date": end.astimezone(KST).date()}


def entity_frame(rows, valid_from: datetime):
    import pandas as pd

    df = rows.copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    df["request_ts"] = (df["event_timestamp"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    df["split"] = ["valid" if t >= valid_from else "train" for t in df["event_timestamp"]]
    df["row_id"] = range(len(df))
    return df


def main() -> None:
    missing = [n for n in ("GCP_PROJECT_ID", "TRAIN_END") if not os.environ.get(n)]
    if missing:
        sys.exit(f"환경변수가 없다: {', '.join(missing)}")
    from feast import FeatureStore
    from google.cloud import bigquery

    sys.path.insert(0, str(ROOT / "features/feast"))
    import build_training_set as feast_set

    project = os.environ["GCP_PROJECT_ID"]
    w = window(datetime.fromisoformat(os.environ["TRAIN_END"]), int(os.environ.get("TRAIN_DAYS", "14")),
               int(os.environ.get("TRAIN_VALID_HOURS", "24")), int(os.environ.get("TRAIN_LABEL_WAIT_HOURS", "3")))
    query = QUERY.format(clean=f"{project}.{os.environ.get('CLEAN_BQ_DATASET', 'clean')}")
    types = {"start": "TIMESTAMP", "stop": "TIMESTAMP", "action_end": "TIMESTAMP",
             "first_date": "DATE", "last_date": "DATE", "action_last_date": "DATE"}
    config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter(k, t, w[k]) for k, t in types.items()])
    rows = bigquery.Client(project=project).query(query, job_config=config).to_dataframe()
    entity = entity_frame(rows, w["valid_from"])
    print(f"impression {len(entity):,}행, 클릭 {int(entity['label'].sum()):,}, "
          f"{w['start']:%m-%d %H:%M} ~ {w['stop']:%m-%d %H:%M} UTC, split {entity['split'].value_counts().to_dict()}")
    if entity.empty or set(entity["split"]) != {"train", "valid"} or not entity["label"].sum():
        print("impression이나 클릭이 부족하다. 학습하지 않는다.")
        sys.exit(NOT_ENOUGH_CLICKS)

    repo = ROOT / "features/feast/feature_repo"
    store = FeatureStore(repo_path=str(repo), fs_yaml_file=repo / os.environ.get("FEAST_FS_YAML", "feature_store.gcp.yaml"))
    df = feast_set.historical(store, entity).sort_values("row_id").reset_index(drop=True)
    feast_set.check_rows(entity, df)
    out = Path(os.environ.get("TRAINING_SET", ROOT / "features/feast/data/training_set.parquet"))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"저장: {out} {df.shape}")


if __name__ == "__main__":
    main()
