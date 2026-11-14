"""[파이프라인 구간: 피처 원천 → BigQuery] prepare_sources.py가 만든 원천 parquet 4개를 BigQuery에 올린다.

실행:  .venv/bin/python upload_to_bq.py [--dataset final-project-513508.dev_yoonmo]
입력:  data/{article_profile,reader_profile,reader_activity,reader_article}.parquet
결과:  <dataset>.<표 이름> 4개 (있으면 덮어쓴다). 임베딩 열은 ARRAY<FLOAT64>, event_timestamp는 TIMESTAMP.
exposures(entity_df)는 올리지 않는다. get_historical_features가 DataFrame을 받아 임시 표로 직접 올린다.
인증: gcloud auth application-default login (ADC).
"""
import argparse
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

DATA = Path(__file__).parent / "data"
TABLES = ["article_profile", "reader_profile", "reader_activity", "reader_article"]


def upload(client: bigquery.Client, dataset: str, name: str) -> None:
    df = pd.read_parquet(DATA / f"{name}.parquet")
    job = client.load_table_from_dataframe(
        df, f"{dataset}.{name}",
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"))
    job.result()
    table = client.get_table(f"{dataset}.{name}")
    cols = ", ".join(f"{f.name}:{f.field_type}{'[]' if f.mode == 'REPEATED' else ''}"
                     for f in table.schema)
    print(f"  {name}: {table.num_rows:,}행  ({cols})")
    if table.num_rows != len(df):
        raise RuntimeError(f"{name}: 로컬 {len(df):,}행인데 BigQuery {table.num_rows:,}행")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="final-project-513508.dev_yoonmo",
                   help="올릴 BigQuery 데이터셋 (project.dataset)")
    args = p.parse_args()
    client = bigquery.Client(project=args.dataset.split(".")[0])
    print(f"BigQuery {args.dataset}에 올린다")
    for name in TABLES:
        upload(client, args.dataset, name)


if __name__ == "__main__":
    main()
