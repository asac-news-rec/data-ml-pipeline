"""[파이프라인 구간: 피처 원천 → BigQuery] prepare_sources.py가 만든 원천 parquet 4개를 BigQuery에 올린다.

실행:  .venv/bin/python upload_to_bq.py [--dataset final-project-513508.dev_yoonmo] [--allow-non-dev]
입력:  data/{article_profile,reader_profile,reader_activity,reader_article}.parquet
결과:  <dataset>.<표 이름> 4개 (있으면 덮어쓴다). 임베딩 열은 ARRAY<FLOAT64>, event_timestamp는 TIMESTAMP.
exposures(entity_df)는 올리지 않는다. get_historical_features가 DataFrame을 받아 임시 표로 직접 올린다.
인증: gcloud auth application-default login (ADC).

안전장치: --dataset의 데이터셋 이름이 dev_로 시작하지 않으면 덮어쓰지 않는다.
raw·clean·agg·feat·train 같은 공용 데이터셋을 실수로 지우지 않게 하려는 것이다.
의도적으로 다른 데이터셋에 쓸 때만 --allow-non-dev를 붙인다.
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
                   help="올릴 BigQuery 데이터셋 (project.dataset). 기본값은 가짜 로그 실험용 dev 데이터셋")
    p.add_argument("--allow-non-dev", action="store_true",
                   help="dev_로 시작하지 않는 데이터셋에도 덮어쓴다 (기본은 거부)")
    args = p.parse_args()
    if "." not in args.dataset:
        raise SystemExit(f"--dataset은 project.dataset 형식으로 준다: {args.dataset}")
    project, dataset_name = args.dataset.split(".", 1)
    if not dataset_name.startswith("dev_") and not args.allow_non_dev:
        raise SystemExit(
            f"안전을 위해 dev_로 시작하는 데이터셋만 덮어쓴다: {args.dataset} "
            "(의도했으면 --allow-non-dev를 붙인다)")
    client = bigquery.Client(project=project)
    print(f"BigQuery {args.dataset}에 올린다")
    for name in TABLES:
        upload(client, args.dataset, name)


if __name__ == "__main__":
    main()
