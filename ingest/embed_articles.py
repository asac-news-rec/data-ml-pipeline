"""[파이프라인 구간: 피처] Cloud SQL 새 기사 → BigQuery feat.article_embedding·feat.article_meta.

매 실행마다 BigQuery에 이 모델·버전으로 아직 임베딩이 없는 기사만 골라 넣는다. 그래서 같은 기사를
두 번 임베딩하지 않고, 실패하면 다음 실행이 같은 기사를 다시 시도한다.
- 입력 글: "passage: " + 제목 + 부제 + 요약 (로컬 article_embedding/embed.py와 같다. e5는 문서 앞에
  "passage: "를 붙이도록 학습됐다). 512토큰을 넘는 부분은 잘린다(MVP 뒤에 다룬다).
- 벡터: multilingual-e5-base 768칸, 길이 1로 정규화.
- 열: 청사진 overview.html 인터페이스 규약 ⑤의 feat.article_meta, feat.article_embedding.

환경변수:
  GCP_PROJECT_ID          BigQuery 프로젝트
  FEAT_BQ_DATASET         쓸 데이터셋 (기본 feat)
  FEAT_TABLE_PREFIX       표 이름 앞에 붙일 말 (기본 빈 값. 개인 dataset 시험 때 feat_)
  PGHOST, PGPORT, PGDATABASE, PGUSER   Cloud SQL 접속 (클러스터에서는 cloud-sql-proxy)
  PGPASSWORD 또는 PGPASSWORD_SECRET     비밀번호 또는 Secret Manager 버전 경로
  EMBED_MODEL             기본 intfloat/multilingual-e5-base
  EMBED_MODEL_REVISION    모델 버전 표시 (기본 main. 이미지에 넣은 스냅샷과 맞춘다)
  EMBED_LIMIT             한 번에 임베딩할 최대 기사 수 (기본 500, 최신 발행순)
"""
from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime, timezone
from typing import Callable, Iterable

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DIM = 768
ARTICLE_SQL = """
SELECT id, title, subtitle, summary, category, subcategory, tags, publisher, published_at
FROM public.news_articles
"""

Embedder = Callable[[list[str]], list[list[float]]]


def passage(article: dict) -> str:
    parts = [article.get("title"), article.get("subtitle"), article.get("summary")]
    return "passage: " + " ".join(p for p in parts if p)


def meta_row(article: dict, copied_at: str) -> dict:
    tags = article.get("tags") or []
    if isinstance(tags, str):
        tags = json.loads(tags)
    published = article["published_at"]
    return {
        "article_id": article["id"],
        "category": article.get("category"),
        "subcategory": article.get("subcategory"),
        "tags": list(tags),
        "publisher_id": article.get("publisher"),
        "published_at": published.isoformat() if hasattr(published, "isoformat") else published,
        "clickbait_score": None,   # 아직 만들지 않는다 (청사진 「결정 필요」: 자극도 점수)
        "copied_at": copied_at,
    }


def embedding_rows(articles: list[dict], vectors: list[list[float]], model: str,
                   revision: str, created_at: str) -> list[dict]:
    if len(articles) != len(vectors):
        raise ValueError("기사 수와 벡터 수가 다르다")
    rows = []
    for article, vector in zip(articles, vectors):
        if len(vector) != DIM:
            raise ValueError(f"{article['id']} 벡터가 {len(vector)}칸이다 ({DIM}칸이어야 한다)")
        rows.append({"article_id": article["id"], "embedding": [float(v) for v in vector],
                     "model_name": model, "model_version": revision, "created_at": created_at})
    return rows


def pick_new(articles: Iterable[dict], done: set[str], limit: int) -> list[dict]:
    """아직 임베딩이 없는 기사를 최신 발행순으로 limit개."""
    todo = [a for a in articles if a["id"] not in done]
    todo.sort(key=lambda a: str(a["published_at"]), reverse=True)
    return todo[:limit]


# ── BigQuery ────────────────────────────────────────────────────────────
def schemas():
    from google.cloud import bigquery as bq

    meta = [bq.SchemaField("article_id", "STRING", "REQUIRED"), bq.SchemaField("category", "STRING"),
            bq.SchemaField("subcategory", "STRING"), bq.SchemaField("tags", "STRING", "REPEATED"),
            bq.SchemaField("publisher_id", "STRING"), bq.SchemaField("published_at", "TIMESTAMP"),
            bq.SchemaField("clickbait_score", "FLOAT64"), bq.SchemaField("copied_at", "TIMESTAMP")]
    emb = [bq.SchemaField("article_id", "STRING", "REQUIRED"),
           bq.SchemaField("embedding", "FLOAT64", "REPEATED"),
           bq.SchemaField("model_name", "STRING", "REQUIRED"),
           bq.SchemaField("model_version", "STRING", "REQUIRED"),
           bq.SchemaField("created_at", "TIMESTAMP", "REQUIRED")]
    return meta, emb


def embedded_ids(client, table: str, model: str, revision: str) -> set[str]:
    from google.api_core.exceptions import NotFound
    from google.cloud import bigquery

    sql = (f"SELECT DISTINCT article_id FROM `{table}` "
           "WHERE model_name = @model AND model_version = @revision")
    config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("model", "STRING", model),
        bigquery.ScalarQueryParameter("revision", "STRING", revision)])
    try:
        return {row.article_id for row in client.query(sql, job_config=config).result()}
    except NotFound:
        return set()


def append(client, table: str, rows: list[dict], schema) -> None:
    from google.cloud import bigquery

    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode("utf-8")
    config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON, schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
        clustering_fields=["article_id"])
    client.load_table_from_file(io.BytesIO(body), table, job_config=config).result()


def write(client, meta_table: str, emb_table: str, metas: list[dict], embs: list[dict]) -> None:
    """임베딩을 먼저 쓰고 메타를 쓴다. 메타는 article_id가 없을 때만 넣는다(MERGE)."""
    from google.cloud import bigquery

    meta_schema, emb_schema = schemas()
    append(client, emb_table, embs, emb_schema)
    staging = f"{meta_table}__staging"
    client.load_table_from_file(
        io.BytesIO("\n".join(json.dumps(m, ensure_ascii=False) for m in metas).encode("utf-8")),
        staging, job_config=bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON, schema=meta_schema,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)).result()
    cols = ", ".join(f.name for f in meta_schema)
    client.query(f"""
        CREATE TABLE IF NOT EXISTS `{meta_table}` LIKE `{staging}`;
        MERGE `{meta_table}` T USING `{staging}` S ON T.article_id = S.article_id
        WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ({cols});
        DROP TABLE `{staging}`;
    """).result()


# ── Cloud SQL · 모델 ─────────────────────────────────────────────────────
def db_password() -> str:
    if os.environ.get("PGPASSWORD"):
        return os.environ["PGPASSWORD"]
    from google.cloud import secretmanager

    name = os.environ["PGPASSWORD_SECRET"]
    return secretmanager.SecretManagerServiceClient().access_secret_version(
        name=name).payload.data.decode("utf-8")


def fetch_articles() -> list[dict]:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(host=os.environ["PGHOST"], port=os.environ.get("PGPORT", "5432"),
                         dbname=os.environ["PGDATABASE"], user=os.environ["PGUSER"],
                         password=db_password(), row_factory=dict_row) as conn:
        return list(conn.execute(ARTICLE_SQL))


def e5_embedder(model: str, revision: str) -> Embedder:
    """처음 부를 때 모델을 불러온다. 새 기사가 없는 실행은 모델을 메모리에 올리지 않는다."""
    loaded = {}

    def embed(texts: list[str]) -> list[list[float]]:
        if "encoder" not in loaded:
            from sentence_transformers import SentenceTransformer

            loaded["encoder"] = SentenceTransformer(
                model, revision=None if revision == "main" else revision, device="cpu")
        return loaded["encoder"].encode(texts, batch_size=8, normalize_embeddings=True,
                                        show_progress_bar=False).tolist()
    return embed


def run(client, articles: list[dict], embed: Embedder, *, project: str, dataset: str,
        prefix: str, model: str, revision: str, limit: int, now: datetime) -> int:
    meta_table = f"{project}.{dataset}.{prefix}article_meta"
    emb_table = f"{project}.{dataset}.{prefix}article_embedding"
    todo = pick_new(articles, embedded_ids(client, emb_table, model, revision), limit)
    print(f"기사 {len(articles)}건 중 새로 임베딩할 기사 {len(todo)}건")
    if not todo:
        return 0
    stamp = now.astimezone(timezone.utc).isoformat()
    vectors = embed([passage(a) for a in todo])
    write(client, meta_table, emb_table, [meta_row(a, stamp) for a in todo],
          embedding_rows(todo, vectors, model, revision, stamp))
    print(f"{len(todo)}건 적재 → {emb_table}, {meta_table}")
    return len(todo)


def main() -> None:
    missing = [n for n in ("GCP_PROJECT_ID", "PGHOST", "PGDATABASE", "PGUSER") if not os.environ.get(n)]
    if not (os.environ.get("PGPASSWORD") or os.environ.get("PGPASSWORD_SECRET")):
        missing.append("PGPASSWORD 또는 PGPASSWORD_SECRET")
    if missing:
        sys.exit(f"환경변수가 없다: {', '.join(missing)}")
    from google.cloud import bigquery

    project = os.environ["GCP_PROJECT_ID"]
    model = os.environ.get("EMBED_MODEL", DEFAULT_MODEL)
    revision = os.environ.get("EMBED_MODEL_REVISION", "main")
    run(bigquery.Client(project=project), fetch_articles(), e5_embedder(model, revision),
        project=project, dataset=os.environ.get("FEAT_BQ_DATASET", "feat"),
        prefix=os.environ.get("FEAT_TABLE_PREFIX", ""), model=model, revision=revision,
        limit=int(os.environ.get("EMBED_LIMIT", "500")), now=datetime.now(timezone.utc))


if __name__ == "__main__":
    main()
