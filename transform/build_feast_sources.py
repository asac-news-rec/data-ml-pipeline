"""[파이프라인 구간: 집계 → 피처] clean + feat 기사 표 → Feast 원천 표 4개를 매시 다시 만든다.

`transform/sql/build_feast_sources.sql` 스크립트 하나를 BigQuery에서 실행한다. 결과 표 이름은 Feast
정의(`features/feast/feature_repo/definitions.py`)가 읽는 이름과 같다: article_profile, reader_profile,
reader_activity, reader_article. 매번 CREATE OR REPLACE로 전체를 다시 만들어서 다시 돌려도 결과가 같다.

환경변수:
  GCP_PROJECT_ID          BigQuery 프로젝트
  CLEAN_BQ_DATASET        clean.actions가 있는 데이터셋 (기본 clean)
  FEAT_BQ_DATASET         article_meta·article_embedding이 있는 데이터셋 (기본 feat)
  FEAST_SOURCE_DATASET    결과 표를 쓸 데이터셋 (기본 feat. Feast의 FEAST_BQ_DATASET과 같은 값)
  CLEAN_TABLE_PREFIX, FEAT_TABLE_PREFIX, FEAST_SOURCE_PREFIX   표 이름 앞말 (개인 dataset 시험용)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SQL = Path(__file__).parent / "sql" / "build_feast_sources.sql"
OUTPUTS = ("article_profile", "reader_profile", "reader_activity", "reader_article")


def table_names(project: str, env: dict) -> dict[str, str]:
    clean = f"{project}.{env.get('CLEAN_BQ_DATASET', 'clean')}.{env.get('CLEAN_TABLE_PREFIX', '')}"
    feat = f"{project}.{env.get('FEAT_BQ_DATASET', 'feat')}.{env.get('FEAT_TABLE_PREFIX', '')}"
    # Feast 정의의 FEAST_BQ_DATASET은 '프로젝트.데이터셋' 형태라서 같은 값을 그대로 받을 수 있게 한다
    out_ds = env.get("FEAST_SOURCE_DATASET", "feat")
    out = f"{out_ds if '.' in out_ds else f'{project}.{out_ds}'}.{env.get('FEAST_SOURCE_PREFIX', '')}"
    names = {"clean_actions": f"{clean}actions",
             "article_meta": f"{feat}article_meta",
             "article_embedding": f"{feat}article_embedding"}
    names |= {f"out_{t}": f"{out}{t}" for t in OUTPUTS}
    return names


def render(names: dict[str, str]) -> str:
    text = SQL.read_text(encoding="utf-8")
    for key, table in names.items():   # format()은 주석의 중괄호도 건드리므로 정한 이름만 바꾼다
        text = text.replace(f"`{{{key}}}`", f"`{table}`")
    return text


def run(client, names: dict[str, str]) -> dict[str, int]:
    counts = {row.t: row.n for row in client.query(render(names)).result()}
    for name, n in counts.items():
        print(f"  {name}: {n:,}행")
    return counts


def main() -> None:
    if not os.environ.get("GCP_PROJECT_ID"):
        sys.exit("환경변수가 없다: GCP_PROJECT_ID")
    from google.cloud import bigquery

    project = os.environ["GCP_PROJECT_ID"]
    names = table_names(project, dict(os.environ))
    print("Feast 원천 표 다시 만들기 →", ", ".join(names[f"out_{t}"] for t in OUTPUTS))
    run(bigquery.Client(project=project), names)


if __name__ == "__main__":
    main()
