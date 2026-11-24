"""[파이프라인 구간: 적재] GCS 행동 로그(gzip JSONL)를 BigQuery raw.served·raw.actions에 넣는다.

입력은 `export_behavior_logs` DAG가 쓴 GCS 객체다 (frontend `services/logging-server/README.md`):
  {prefix}/served/interval_start={UTC}/interval_end={UTC}/part-{job_id}.jsonl.gz
  {prefix}/actions/interval_start={UTC}/interval_end={UTC}/part-{job_id}.jsonl.gz

한 번 실행하면 [LOAD_HOUR_START - lookback, LOAD_HOUR_START + 1시간) 안에서 시작하는 구간의
파일을 나열하고, raw 표에 아직 없는 파일만 load job 하나로 넣는다.
- 줄 전체는 payload(JSON)에 그대로 두고 키 몇 개만 꺼낸다. 로그 필드가 늘어도 적재는 멈추지 않는다.
- GCS 객체는 덮어쓰지 않는다(if_generation_match=0). 그래서 source_file이 같으면 같은 내용이고,
  이미 넣은 파일을 건너뛰는 것만으로 같은 시간을 다시 돌려도 행이 두 배가 되지 않는다.
- load_date는 파일 구간 시작(interval_start)의 한국 날짜다. 적재 시각이 아니라 파일에서 정해지므로
  다시 돌려도 같은 파티션에 들어가고, 중복 확인도 그 파티션만 읽는다.
- 한 load job은 원자적이다. 실패하면 아무 행도 들어가지 않고, 다음 실행이 같은 파일을 다시 넣는다.

환경변수:
  GCP_PROJECT_ID        BigQuery·GCS 프로젝트
  RAW_GCS_BUCKET        행동 로그 버킷 (export_behavior_logs의 GCS_BUCKET과 같은 값)
  RAW_GCS_PREFIX        버킷 안 접두어 (export_behavior_logs의 GCS_PREFIX, 없으면 빈 값)
  RAW_BQ_DATASET        적재할 데이터셋 (기본 raw, 시험은 dev_<이름>)
  LOAD_HOUR_START       적재할 시간의 시작 (Airflow data_interval_start, 시간대 포함 ISO 8601)
  LOAD_LOOKBACK_HOURS   늦게 내보내진 파일을 위해 앞 몇 시간을 함께 볼지 (기본 3)
"""
from __future__ import annotations

import gzip
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

KST = timezone(timedelta(hours=9))
INTERVAL_RE = re.compile(r"/interval_start=([0-9T:\-.]+Z)/")


@dataclass(frozen=True)
class Stream:
    name: str                       # GCS 폴더 이름이자 raw 표 이름
    keys: tuple[tuple[str, str], ...]  # payload에서 꺼내 열로 둘 키와 BigQuery 타입
    cluster: tuple[str, ...] = ()


# 열 목록은 청사진 overview.html 규약 ⑤ raw 층 표를 따른다.
STREAMS = (
    Stream("served", (("impression_id", "STRING"), ("request_id", "STRING"),
                      ("served_at", "TIMESTAMP"))),
    Stream("actions", (("event_id", "STRING"), ("event_type", "STRING"),
                       ("event_time", "TIMESTAMP"), ("impression_id", "STRING")),
           cluster=("event_type",)),
)


def hour_prefixes(hour_start: datetime, lookback_hours: int) -> list[str]:
    """interval_start 경로 앞부분(UTC 'YYYY-MM-DDTHH')을 오래된 시간부터 돌려준다."""
    if hour_start.utcoffset() is None:
        raise ValueError("LOAD_HOUR_START에 시간대가 없다")
    base = hour_start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return [(base - timedelta(hours=h)).strftime("%Y-%m-%dT%H")
            for h in range(lookback_hours, -1, -1)]


def interval_start_of(uri: str) -> datetime:
    match = INTERVAL_RE.search(uri)
    if not match:
        raise ValueError(f"interval_start가 없는 경로: {uri}")
    return datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))


def load_date_of(uri: str) -> date:
    """파일 구간 시작의 한국 날짜. raw 표의 파티션이다."""
    return interval_start_of(uri).astimezone(KST).date()


def object_prefix(prefix: str, stream: str, hour: str) -> str:
    head = prefix.strip("/")
    return f"{head}/{stream}/interval_start={hour}" if head else f"{stream}/interval_start={hour}"


def list_uris(storage_client, bucket: str, prefix: str, stream: str,
              hours: Iterable[str]) -> list[str]:
    uris = []
    for hour in hours:
        for blob in storage_client.list_blobs(bucket, prefix=object_prefix(prefix, stream, hour)):
            if blob.name.endswith(".jsonl.gz"):
                uris.append(f"gs://{bucket}/{blob.name}")
    return sorted(uris)


def to_rows(uri: str, text: str, loaded_at: str, stream: Stream) -> list[dict]:
    """JSONL 한 파일을 raw 표 행으로 바꾼다. 빈 줄은 건너뛰고, 깨진 줄은 실패시킨다."""
    rows = []
    load_date = load_date_of(uri).isoformat()
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{uri} {number}번째 줄이 JSON이 아니다") from error
        row = {key: payload.get(key) for key, _ in stream.keys}
        row.update(payload=payload, source_file=uri, load_date=load_date, loaded_at=loaded_at)
        rows.append(row)
    return rows


def schema(stream: Stream):
    from google.cloud import bigquery

    fields = [bigquery.SchemaField(key, kind) for key, kind in stream.keys]
    fields += [
        bigquery.SchemaField("payload", "JSON", description="로그 한 줄 전체"),
        bigquery.SchemaField("source_file", "STRING", description="GCS 객체 경로"),
        bigquery.SchemaField("load_date", "DATE", description="파일 구간 시작의 한국 날짜"),
        bigquery.SchemaField("loaded_at", "TIMESTAMP"),
    ]
    return fields


def already_loaded(bq_client, table: str, uris: list[str]) -> set[str]:
    """raw 표에 이미 있는 source_file. 표가 없으면 빈 집합."""
    from google.api_core.exceptions import NotFound
    from google.cloud import bigquery

    dates = sorted({load_date_of(uri) for uri in uris})
    sql = (f"SELECT DISTINCT source_file FROM `{table}` "
           "WHERE load_date IN UNNEST(@dates) AND source_file IN UNNEST(@uris)")
    config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("dates", "DATE", dates),
        bigquery.ArrayQueryParameter("uris", "STRING", uris),
    ])
    try:
        return {row.source_file for row in bq_client.query(sql, job_config=config).result()}
    except NotFound:
        return set()


def load_stream(bq_client, storage_client, *, project: str, dataset: str, bucket: str,
                prefix: str, stream: Stream, hours: list[str], now: datetime) -> int:
    from google.cloud import bigquery

    table = f"{project}.{dataset}.{stream.name}"
    uris = list_uris(storage_client, bucket, prefix, stream.name, hours)
    if not uris:
        print(f"[{stream.name}] 파일 없음 ({hours[0]} ~ {hours[-1]})")
        return 0
    loaded = already_loaded(bq_client, table, uris)
    new = [uri for uri in uris if uri not in loaded]
    print(f"[{stream.name}] 파일 {len(uris)}개 중 새 파일 {len(new)}개")
    if not new:
        return 0

    loaded_at = now.astimezone(timezone.utc).isoformat()
    rows = []
    for uri in new:
        blob = storage_client.bucket(bucket).blob(uri.removeprefix(f"gs://{bucket}/"))
        text = gzip.decompress(blob.download_as_bytes()).decode("utf-8")
        rows += to_rows(uri, text, loaded_at, stream)

    body = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows).encode("utf-8")
    config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        schema=schema(stream),
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
        time_partitioning=bigquery.TimePartitioning(field="load_date"),
        clustering_fields=list(stream.cluster) or None,
    )
    bq_client.load_table_from_file(io.BytesIO(body), table, job_config=config).result()
    print(f"[{stream.name}] {len(rows)}행 적재 → {table}")
    return len(rows)


def require_env(names: list[str]) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        sys.exit(f"환경변수가 없다: {', '.join(missing)}")


def main() -> None:
    require_env(["GCP_PROJECT_ID", "RAW_GCS_BUCKET", "LOAD_HOUR_START"])
    from google.cloud import bigquery, storage

    project = os.environ["GCP_PROJECT_ID"]
    hours = hour_prefixes(datetime.fromisoformat(os.environ["LOAD_HOUR_START"]),
                          int(os.environ.get("LOAD_LOOKBACK_HOURS", "3")))
    bq_client = bigquery.Client(project=project)
    storage_client = storage.Client(project=project)
    for stream in STREAMS:
        load_stream(bq_client, storage_client, project=project,
                    dataset=os.environ.get("RAW_BQ_DATASET", "raw"),
                    bucket=os.environ["RAW_GCS_BUCKET"],
                    prefix=os.environ.get("RAW_GCS_PREFIX", ""),
                    stream=stream, hours=hours, now=datetime.now(timezone.utc))


if __name__ == "__main__":
    main()
