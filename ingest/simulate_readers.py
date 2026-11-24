"""[파이프라인 구간: 수집] 가상 독자의 한 시간치 행동 로그를 GCS에 로깅 서버 형식으로 쓴다.

한 바퀴 시험(11/26)용 임시 코드다. 실제 독자가 생기면 DAG와 함께 지운다.
로깅 서버 API를 거치지 않고, 로깅 서버 내보내기(`export_behavior_logs`)가 남기는 파일과
같은 경로·같은 줄 형식으로 GCS에 바로 쓴다. 그래서 `load_raw` 이후 단계는 진짜 로그와
똑같이 읽는다.

- 경로 (frontend `services/logging-server/logging_service/interval_export.py`):
    {prefix}/{served|actions}/interval_start={UTC}/interval_end={UTC}/part-{job_id}.jsonl.gz
  구간은 한 시간이다. 모든 줄의 received_at이 그 구간 안에 들어간다.
- 줄 (같은 저장소 `store.py`의 `prepare_exports`, `schema.py`의 `served`·`action`):
    served  = request_id, impression_id, reader_id, article_id, position, served_at,
              score, model_version, features_as_of, received_at
    actions = event_id, event_type, event_time, impression_id, page_type,
              (exposure만) duration_ms, ended_at, received_at,
              source, schema_version, measurement_version
  가짜 표시: served는 model_version="simulator", actions는 source="simulator".
- 기사: feat.article_meta에서 그 시간 끝 전에 발행된 최신 기사 (embed_articles가 채운다).
- 같은 시간은 언제 돌려도 같은 내용이 나온다(시간을 시드로 쓴다). 객체 이름도 고정이고
  덮어쓰지 않으므로(if_generation_match=0), 다시 돌리면 이미 있는 파일은 건너뛴다.

환경변수:
  GCP_PROJECT_ID     BigQuery·GCS 프로젝트
  RAW_GCS_BUCKET     행동 로그 버킷 (load_raw와 같은 값)
  RAW_GCS_PREFIX     버킷 안 접두어 (load_raw와 같은 값, 없으면 빈 값)
  SIM_HOUR_START     만들 시간의 시작 (Airflow data_interval_start, 시간대 포함 ISO 8601)
  FEAT_BQ_DATASET    article_meta가 있는 데이터셋 (기본 feat)
  FEAT_TABLE_PREFIX  표 이름 앞에 붙일 말 (시험용, 기본 빈 값)
  SIM_READERS        가상 독자 수 (기본 200)
  SIM_VISIT_SCALE    방문 확률에 곱할 값 (기본 1.0)
"""
from __future__ import annotations

import gzip
import json
import math
import os
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
HOUR = timedelta(hours=1)
PAGE_SIZE = 20            # 피드 한 페이지 기사 수
ARTICLE_POOL = 200        # article_meta에서 읽어 올 최신 기사 수
LAST_VISIT_MINUTE = 40    # 이 분 전에 시작한 방문만 만든다. 나머지 시간은 행동이 끝날 여유다
MIN_EXPOSURE_MS = 1000    # 유효 노출: 연속 1초 이상
SKIP_CARD_PROB = 0.15     # 빠르게 지나가 유효 노출이 안 된 카드 비율
FILTER_PROB = 0.2         # 좋아하는 카테고리 필터를 걸고 들어오는 방문 비율
BASE_VISIT = 0.03         # 페르소나의 주 방문 시간이 아닐 때 방문 확률
BASE_OPEN = 0.02          # 페르소나 취향에 없는 카테고리를 열어 볼 확률
SIM_NAMESPACE = uuid.UUID("6f1b3c1e-2a8d-4c55-9a51-5f0a3b7d9e21")
MEASUREMENT_VERSION = "stationary-visible-v1"   # 로깅 서버 store.py가 넣는 값과 같다

# visit: {한국 시각(시): 그 시간 방문 확률}, open: {카테고리: 노출된 카드를 열어 볼 확률}
# depth: 한 번 방문에서 평균 몇 번째 카드까지 보나
PERSONAS = {
    "경제직장인": {"share": 0.25, "depth": 8, "visit": {8: 0.6, 12: 0.4, 19: 0.4},
              "open": {"경제": 0.35, "증권": 0.40, "금융": 0.30, "기업": 0.30, "부동산": 0.25}},
    "테크관심": {"share": 0.15, "depth": 10, "visit": {9: 0.3, 13: 0.4, 21: 0.6},
             "open": {"테크·과학": 0.50, "기업": 0.20, "증권": 0.10}},
    "스포츠연예": {"share": 0.20, "depth": 12, "visit": {12: 0.4, 18: 0.4, 22: 0.6},
              "open": {"스포츠": 0.50, "연예": 0.40, "문화": 0.10}},
    "시사형": {"share": 0.20, "depth": 9, "visit": {7: 0.6, 20: 0.5},
            "open": {"정치": 0.45, "사회": 0.30, "국제": 0.35}},
    "문화건강": {"share": 0.10, "depth": 7, "visit": {10: 0.3, 15: 0.4, 21: 0.4},
             "open": {"문화": 0.40, "건강": 0.45, "연예": 0.15}},
    "탐색형": {"share": 0.10, "depth": 16, "visit": {h: 0.12 for h in range(9, 24)}, "open": {}},
}


def utc_ms(when: datetime) -> str:
    """로깅 서버와 같은 시각 문자열: UTC, 밀리초, Z."""
    return when.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def uuid7(rng: random.Random, when: datetime) -> str:
    """시간순 정렬되는 UUID v7. 같은 시드면 같은 값이 나오도록 rng로 만든다."""
    ms = int(when.timestamp() * 1000)
    value = (ms << 80) | (0x7 << 76) | (rng.getrandbits(12) << 64)
    value |= (0b10 << 62) | rng.getrandbits(62)
    return str(uuid.UUID(int=value))


def make_readers(count: int) -> list[dict]:
    """가상 독자 목록. 독자 i는 실행마다 같은 reader_id·페르소나·취향을 가진다."""
    names = list(PERSONAS)
    weights = [PERSONAS[n]["share"] for n in names]
    readers = []
    for i in range(count):
        rng = random.Random(f"reader-{i}")
        persona = rng.choices(names, weights)[0]
        noise = {}
        readers.append({
            "reader_id": str(uuid.uuid5(SIM_NAMESPACE, f"reader-{i}")),
            "persona": persona,
            "noise": noise,
            "seed": i,
        })
    return readers


def taste(reader: dict, category: str) -> float:
    """독자가 이 카테고리 카드를 열어 볼 기본 확률. 같은 페르소나여도 사람마다 조금 다르다."""
    if category not in reader["noise"]:
        rng = random.Random(f"taste-{reader['seed']}-{category}")
        reader["noise"][category] = math.exp(rng.gauss(0, 0.4))
    base = PERSONAS[reader["persona"]]["open"].get(category, BASE_OPEN)
    return min(0.8, base * reader["noise"][category])


def feed(articles: list[dict], at: datetime, category: str | None) -> list[dict]:
    """피드 API 흉내: 그 시각까지 발행된 기사를 최신순으로 한 페이지 돌려준다."""
    pool = [a for a in articles if a["published_at"] <= at and category in (None, a["category"])]
    pool.sort(key=lambda a: (a["published_at"], a["article_id"]), reverse=True)
    return pool[:PAGE_SIZE]


def received(rng: random.Random, when: datetime, end: datetime) -> str:
    """로깅 서버가 받은 시각. 보낸 뒤 조금 늦게 받되 구간 끝을 넘지 않는다."""
    at = min(when + timedelta(milliseconds=rng.randint(80, 1500)), end - timedelta(milliseconds=1))
    return utc_ms(at)


def action(rng, when, end, event_type, impression_id, page_type, duration_ms=None) -> dict:
    row = {"event_id": uuid7(rng, when), "event_type": event_type, "event_time": utc_ms(when),
           "impression_id": impression_id, "page_type": page_type}
    sent = when
    if duration_ms is not None:
        ended = when + timedelta(milliseconds=duration_ms)
        row.update(duration_ms=duration_ms, ended_at=utc_ms(ended))
        sent = ended                     # 노출은 끝난 뒤에 보낸다
    row.update(received_at=received(rng, sent, end), source="simulator", schema_version=1,
               measurement_version=MEASUREMENT_VERSION)
    return row


def visit(rng, reader, articles, start, end) -> tuple[list[dict], list[dict]]:
    """방문 한 번: 피드 한 페이지를 받고 위에서부터 내려가며 노출·클릭·체류·좋아요를 남긴다."""
    category = None
    if rng.random() < FILTER_PROB and PERSONAS[reader["persona"]]["open"]:
        category = rng.choice(list(PERSONAS[reader["persona"]]["open"]))
    page = feed(articles, start, category)
    if not page:
        return [], []
    request_id = uuid7(rng, start)
    served, actions = [], []
    for position, article in enumerate(page, start=1):
        served.append({
            "request_id": request_id, "impression_id": uuid7(rng, start),
            "reader_id": reader["reader_id"], "article_id": article["article_id"],
            "position": position, "served_at": utc_ms(start), "score": None,
            "model_version": "simulator", "features_as_of": None,
            "received_at": received(rng, start, end),
        })

    stop_at = end - timedelta(minutes=2)          # 체류가 구간 밖으로 넘치지 않게 여유를 둔다
    depth = max(1, int(rng.expovariate(1 / PERSONAS[reader["persona"]]["depth"])) + 1)
    now = start + timedelta(milliseconds=rng.randint(300, 1500))
    for row, article in list(zip(served, page))[:depth]:
        if now >= stop_at:
            break
        if rng.random() < SKIP_CARD_PROB:
            now += timedelta(milliseconds=rng.randint(100, 400))
            continue
        shown_ms = rng.randint(MIN_EXPOSURE_MS, 4000)
        actions.append(action(rng, now, end, "exposure", row["impression_id"], "list", shown_ms))
        now += timedelta(milliseconds=shown_ms)
        interest = taste(reader, article["category"])
        if rng.random() < interest / (1 + 0.06 * (row["position"] - 1)):
            opened = now + timedelta(milliseconds=rng.randint(200, 900))
            actions.append(action(rng, opened, end, "article_open", row["impression_id"], "list"))
            dwell = int(rng.lognormvariate(math.log(15000 + 60000 * interest), 0.6))
            dwell = max(1500, min(dwell, 300000))
            closed = min(opened + timedelta(milliseconds=dwell), stop_at)
            if (closed - opened).total_seconds() >= 11 and rng.random() < min(0.5, 0.03 + 0.6 * interest):
                liked = opened + timedelta(seconds=rng.uniform(10, (closed - opened).total_seconds() - 0.5))
                actions.append(action(rng, liked, end, "like", row["impression_id"], "detail"))
            actions.append(action(rng, closed, end, "article_close", row["impression_id"], "detail"))
            now = closed + timedelta(milliseconds=rng.randint(300, 1200))
    return served, actions


def simulate_hour(readers: list[dict], articles: list[dict], start: datetime,
                  visit_scale: float = 1.0) -> tuple[list[dict], list[dict]]:
    """[start, start+1시간) 동안의 served·actions 줄. 같은 start면 늘 같은 결과다."""
    end = start + HOUR
    rng = random.Random(f"hour-{utc_ms(start)}")
    hour_kst = start.astimezone(KST).hour
    served, actions = [], []
    for reader in readers:
        prob = PERSONAS[reader["persona"]]["visit"].get(hour_kst, BASE_VISIT) * visit_scale
        if rng.random() >= min(prob, 1.0):
            continue
        at = start + timedelta(seconds=rng.uniform(0, LAST_VISIT_MINUTE * 60))
        at = at.replace(microsecond=at.microsecond // 1000 * 1000)
        s, a = visit(rng, reader, articles, at, end)
        served += s
        actions += a
    served.sort(key=lambda r: r["impression_id"])
    actions.sort(key=lambda r: r["event_id"])
    return served, actions


def object_names(prefix: str, start: datetime) -> dict[str, str]:
    """스트림별 GCS 객체 이름. job_id 자리는 시간에서 정해지는 고정 값이라 다시 돌려도 같다."""
    s, e = utc_ms(start), utc_ms(start + HOUR)
    job = f"simulator-{uuid.uuid5(SIM_NAMESPACE, s)}"
    base = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
    return {stream: f"{base}{stream}/interval_start={s}/interval_end={e}/part-{job}.jsonl.gz"
            for stream in ("served", "actions")}


def to_gzip(rows: list[dict]) -> bytes:
    """로깅 서버 gcs.py와 같이 compact JSON 줄 + gzip(mtime=0)."""
    text = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
    return gzip.compress(text.encode("utf-8"), mtime=0)


def write(bucket, name: str, rows: list[dict]) -> bool:
    """객체가 없을 때만 쓴다. 이미 있으면 False."""
    from google.api_core.exceptions import PreconditionFailed

    if not rows:
        return False
    try:
        bucket.blob(name).upload_from_string(to_gzip(rows), content_type="application/gzip",
                                             if_generation_match=0)
    except PreconditionFailed:
        return False
    return True


def fetch_articles(client, table: str, end: datetime) -> list[dict]:
    from google.cloud import bigquery as bq

    sql = (f"SELECT article_id, category, published_at FROM `{table}` "
           "WHERE published_at < @end ORDER BY published_at DESC, article_id DESC LIMIT @n")
    job = client.query(sql, job_config=bq.QueryJobConfig(query_parameters=[
        bq.ScalarQueryParameter("end", "TIMESTAMP", end),
        bq.ScalarQueryParameter("n", "INT64", ARTICLE_POOL)]))
    return [{"article_id": r["article_id"], "category": r["category"],
             "published_at": r["published_at"]} for r in job.result()]


def main() -> None:
    from google.cloud import bigquery, storage

    project = os.environ["GCP_PROJECT_ID"]
    start = datetime.fromisoformat(os.environ["SIM_HOUR_START"])
    if start.utcoffset() is None:
        raise ValueError("SIM_HOUR_START에 시간대가 없다")
    start = start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    dataset = os.environ.get("FEAT_BQ_DATASET", "feat")
    table = f"{project}.{dataset}.{os.environ.get('FEAT_TABLE_PREFIX', '')}article_meta"

    articles = fetch_articles(bigquery.Client(project=project), table, start + HOUR)
    if not articles:
        print(f"{table}에 {utc_ms(start + HOUR)} 전에 발행된 기사가 없어 만들지 않는다")
        return
    readers = make_readers(int(os.environ.get("SIM_READERS", "200")))
    served, actions = simulate_hour(readers, articles, start,
                                    float(os.environ.get("SIM_VISIT_SCALE", "1.0")))
    bucket = storage.Client(project=project).bucket(os.environ["RAW_GCS_BUCKET"])
    names = object_names(os.environ.get("RAW_GCS_PREFIX", ""), start)
    for stream, rows in (("served", served), ("actions", actions)):
        wrote = write(bucket, names[stream], rows)
        status = "씀" if wrote else ("없음" if not rows else "이미 있음")
        print(f"{stream}: {len(rows)}줄 {status} gs://{bucket.name}/{names[stream]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
