"""[파이프라인 구간: 피처 → 서빙 내보내기] 추천 후보 목록(최신 기사)을 GCS JSON으로 쓴다.

추천 서버(`serving/recommend.py`의 `CandidatePool`)가 이 파일을 5분마다 읽어 분류별 최신 200개에 점수를 매긴다.
리트리벌 모델은 MVP 이후이고, 지금 후보 규칙은 "최근 CANDIDATE_DAYS일 안에 발행된 기사, 최신순"이다.

환경변수:
  GCP_PROJECT_ID     BigQuery 프로젝트
  FEAST_REGISTRY_PATH  Feast registry 위치. CANDIDATES_PATH가 없으면 같은 폴더의 candidates.json에 쓴다
  CANDIDATES_PATH    (선택) 쓸 위치를 직접 정할 때 (gs://버킷/경로.json 또는 로컬 경로)
  FEAT_BQ_DATASET    article_meta가 있는 데이터셋 (기본 feat)
  FEAT_TABLE_PREFIX  article_meta 표 이름 앞말 (개인 dataset 시험용, 기본 없음)
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


def candidates_path(env=os.environ) -> str:
    """쓸 위치. serving/recommend.py의 default_path와 같은 규칙이다(테스트가 둘을 비교한다).

    registry 파일 이름이 registry.pb가 아니어도 부모 폴더 기준으로 만든다. registry와 같은 경로면 멈춘다.
    """
    registry = env.get("FEAST_REGISTRY_PATH", "")
    path = env.get("CANDIDATES_PATH") or (
        registry.rsplit("/", 1)[0] + "/candidates.json" if "/" in registry else "candidates.json")
    if registry and path.rstrip("/") == registry.rstrip("/"):
        sys.exit(f"후보 목록 경로가 Feast registry와 같다: {path}. registry를 덮어쓰지 않도록 멈춘다.")
    return path


def write(path: str, text: str) -> None:
    if path.startswith("gs://"):
        from google.cloud import storage

        bucket, _, name = path.removeprefix("gs://").partition("/")
        storage.Client().bucket(bucket).blob(name).upload_from_string(text, content_type="application/json")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def main() -> None:
    if not os.environ.get("GCP_PROJECT_ID"):
        sys.exit("환경변수가 없다: GCP_PROJECT_ID")
    if not (os.environ.get("CANDIDATES_PATH") or os.environ.get("FEAST_REGISTRY_PATH")):
        sys.exit("환경변수가 없다: FEAST_REGISTRY_PATH 또는 CANDIDATES_PATH")
    path = candidates_path()
    from google.cloud import bigquery

    project = os.environ["GCP_PROJECT_ID"]
    table = f"{project}.{os.environ.get('FEAT_BQ_DATASET', 'feat')}.{os.environ.get('FEAT_TABLE_PREFIX', '')}article_meta"
    config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("days", "INT64", int(os.environ.get("CANDIDATE_DAYS", "7"))),
        bigquery.ScalarQueryParameter("max_rows", "INT64", int(os.environ.get("CANDIDATE_MAX", "5000"))),
    ])
    rows = list(bigquery.Client(project=project).query(QUERY.format(table=table), job_config=config).result())
    payload = to_payload(rows, datetime.now(timezone.utc))
    write(path, json.dumps(payload, ensure_ascii=False))
    print(f"후보 {len(payload['articles']):,}개 → {path}")


if __name__ == "__main__":
    main()
