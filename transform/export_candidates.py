"""[파이프라인 구간: 피처 → 서빙 내보내기] 추천 후보 목록(최신 기사)을 GCS JSON으로 쓴다.

추천 서버(`serving/recommend.py`의 `CandidatePool`)가 이 파일을 5분마다 읽어 분류별 최신 200개에 점수를 매긴다.
리트리벌 모델은 MVP 이후이고, 지금 후보 규칙은 "최근 CANDIDATE_DAYS일 안에 발행된 기사, 최신순"이다.

환경변수:
  GCP_PROJECT_ID     BigQuery 프로젝트
  CANDIDATES_PATH    쓸 위치 (gs://버킷/경로.json 또는 로컬 경로)
  FEAT_BQ_DATASET    article_meta가 있는 데이터셋 (기본 feat)
  CANDIDATE_DAYS     며칠 안의 기사까지 (기본 7)
  CANDIDATE_MAX      최대 기사 수 (기본 5000)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

QUERY = """
SELECT article_id, category, subcategory, published_at
FROM `{table}`
WHERE published_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
  AND published_at <= CURRENT_TIMESTAMP()
QUALIFY ROW_NUMBER() OVER (PARTITION BY article_id ORDER BY copied_at DESC) = 1
ORDER BY published_at DESC, article_id DESC
LIMIT @max_rows
"""


def to_payload(rows, now: datetime) -> dict:
    articles = [{
        "article_id": row["article_id"],
        "category": row["category"],
        "subcategory": row["subcategory"],
        "published_at": row["published_at"].astimezone(timezone.utc).isoformat(),
    } for row in rows]
    return {"generated_at": now.astimezone(timezone.utc).isoformat(), "articles": articles}


def write(path: str, text: str) -> None:
    if path.startswith("gs://"):
        from google.cloud import storage

        bucket, _, name = path.removeprefix("gs://").partition("/")
        storage.Client().bucket(bucket).blob(name).upload_from_string(text, content_type="application/json")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def main() -> None:
    missing = [k for k in ("GCP_PROJECT_ID", "CANDIDATES_PATH") if not os.environ.get(k)]
    if missing:
        sys.exit(f"환경변수가 없다: {', '.join(missing)}")
    from google.cloud import bigquery

    project = os.environ["GCP_PROJECT_ID"]
    table = f"{project}.{os.environ.get('FEAT_BQ_DATASET', 'feat')}.article_meta"
    config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("days", "INT64", int(os.environ.get("CANDIDATE_DAYS", "7"))),
        bigquery.ScalarQueryParameter("max_rows", "INT64", int(os.environ.get("CANDIDATE_MAX", "5000"))),
    ])
    rows = list(bigquery.Client(project=project).query(QUERY.format(table=table), job_config=config).result())
    payload = to_payload(rows, datetime.now(timezone.utc))
    write(os.environ["CANDIDATES_PATH"], json.dumps(payload, ensure_ascii=False))
    print(f"후보 {len(payload['articles']):,}개 → {os.environ['CANDIDATES_PATH']}")


if __name__ == "__main__":
    main()
