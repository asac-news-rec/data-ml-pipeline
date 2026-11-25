"""[파이프라인 구간: 정제] raw.served·raw.actions → clean.served·clean.actions를 매시 만든다.

`transform/sql/clean_events.sql` 스크립트 하나를 BigQuery에서 실행한다. 처리 범위는 load_raw와 같다:
LOAD_HOUR_START가 든 UTC 한 시간 + 앞 lookback 시간에 시작하는 내보내기 구간. 모든 쓰기가 MERGE라서
같은 범위를 다시 돌려도 행이 늘지 않고, 늦게 들어온 raw 파일은 다음 실행이 lookback으로 줍는다.

환경변수:
  GCP_PROJECT_ID        BigQuery 프로젝트
  LOAD_HOUR_START       처리할 시간의 시작 (Airflow data_interval_start, 시간대 포함 ISO 8601)
  LOAD_LOOKBACK_HOURS   앞 몇 시간을 함께 볼지 (기본 3, load_raw와 같게)
  RAW_BQ_DATASET        raw 표가 있는 데이터셋 (기본 raw)
  CLEAN_BQ_DATASET      clean 표를 쓸 데이터셋 (기본 clean)
  CLEAN_TABLE_PREFIX    clean 표 이름 앞에 붙일 말 (기본 빈 값. 개인 dataset 시험 때 clean_)
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
SQL = Path(__file__).parent / "sql" / "clean_events.sql"
CLEAN_TABLES = ("served", "actions", "orphan_actions", "rejected_actions")
# clean 쪽을 읽을 때 이보다 이른 파티션은 보지 않는다. 행동은 받은 시각 앞 24시간 안에서만
# 로깅 서버가 받으므로(frontend logging_service/schema.py) lookback + 2일이면 충분하다.
CLEAN_LOOKBACK_DAYS = 2


def iso_ms(value: datetime) -> str:
    """GCS 경로의 interval_start와 같은 모양 (UTC, 밀리초, Z)."""
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def window(hour_start: datetime, lookback_hours: int) -> dict:
    """처리 범위를 쿼리 매개변수 값으로 만든다."""
    if hour_start.utcoffset() is None:
        raise ValueError("LOAD_HOUR_START에 시간대가 없다")
    end = hour_start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    start = end - timedelta(hours=lookback_hours + 1)
    first, last = start.astimezone(KST).date(), (end - timedelta(milliseconds=1)).astimezone(KST).date()
    load_dates = [first + timedelta(days=d) for d in range((last - first).days + 1)]
    return {
        "start": iso_ms(start),
        "end": iso_ms(end),
        "load_dates": load_dates,
        "min_date": first - timedelta(days=CLEAN_LOOKBACK_DAYS),
    }


def render(project: str, raw: str, clean: str, prefix: str) -> str:
    names = {f"raw_{s}": f"{project}.{raw}.{s}" for s in ("served", "actions")}
    names |= {f"clean_{t}": f"{project}.{clean}.{prefix}{t}" for t in CLEAN_TABLES}
    text = SQL.read_text(encoding="utf-8")
    for key, table in names.items():   # format()은 주석의 중괄호도 건드리므로 정한 이름만 바꾼다
        text = text.replace(f"`{{{key}}}`", f"`{table}`")
    return text


def params(values: dict) -> list:
    from google.cloud import bigquery

    return [
        bigquery.ScalarQueryParameter("start", "STRING", values["start"]),
        bigquery.ScalarQueryParameter("end", "STRING", values["end"]),
        bigquery.ArrayQueryParameter("load_dates", "DATE", values["load_dates"]),
        bigquery.ScalarQueryParameter("min_date", "DATE", values["min_date"]),
    ]


def main() -> None:
    missing = [n for n in ("GCP_PROJECT_ID", "LOAD_HOUR_START") if not os.environ.get(n)]
    if missing:
        sys.exit(f"환경변수가 없다: {', '.join(missing)}")
    from google.cloud import bigquery

    project = os.environ["GCP_PROJECT_ID"]
    values = window(datetime.fromisoformat(os.environ["LOAD_HOUR_START"]),
                    int(os.environ.get("LOAD_LOOKBACK_HOURS", "3")))
    sql = render(project, os.environ.get("RAW_BQ_DATASET", "raw"),
                 os.environ.get("CLEAN_BQ_DATASET", "clean"), os.environ.get("CLEAN_TABLE_PREFIX", ""))
    print(f"clean_events 구간 {values['start']} ~ {values['end']}, raw 파티션 {values['load_dates']}")
    client = bigquery.Client(project=project)
    rows = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params(values))).result()
    for row in rows:   # 스크립트 마지막 SELECT: 판정별 행동 수
        print(f"  {row.verdict}: {row.n}")


if __name__ == "__main__":
    main()
