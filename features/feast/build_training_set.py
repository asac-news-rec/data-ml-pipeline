"""[파이프라인 구간: 피처 → 학습 데이터] Feast get_historical_features로 training set을 만든다.

실행:  .venv/bin/python build_training_set.py [--ref <features.npz>]
입력:  data/exposures.parquet   entity_df (reader_id, article_id, event_timestamp=노출 시각,
                                request_ts, position, label, split)
결과:  data/training_set.parquet  일반 9열 + u_emb(768 배열) + a_emb(768 배열) + label + split
       --ref를 주면 기존 피처 행렬 X(40,763 x 1,545)와 열마다 비교해 결과를 출력한다.
행 수 검사(entity_df 행 = 결과 행, row_id 고유)는 --ref와 상관없이 항상 한다.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from feast import FeatureStore

HERE = Path(__file__).parent
DATA = HERE / "data"
DIM = 768
HEAD = ["position", "article_age_h", "category", "hour", "hist_clicks", "hist_exposures",
        "hours_since_last_click", "already_opened", "already_seen"]
FILL_ZERO = ["hist_clicks", "hist_exposures", "already_opened", "already_seen"]
SERVICE = "ctr_lgbm_v1"


def use_rest_download() -> None:
    """[임시] BigQuery 결과를 Storage Read API 대신 REST로 내려받게 한다.

    Feast는 결과를 QueryJob.to_arrow()로 받고, 이 함수는 기본으로 Storage Read API를 쓴다.
    그 API는 bigquery.readsessions.create 권한(roles/bigquery.readSessionUser)이 필요한데
    2026-11-14 기준 우리 계정에 없어 403이 난다. 권한이 생기면 이 함수를 지운다.
    REST는 느리다 (임베딩 1,536열 × 4만 행이면 수 분).
    """
    from google.cloud.bigquery.job import QueryJob
    original = QueryJob.to_arrow
    if getattr(original, "_rest_only", False):
        return

    def to_arrow(self, *args, **kwargs):
        kwargs["create_bqstorage_client"] = False
        return original(self, *args, **kwargs)

    to_arrow._rest_only = True
    QueryJob.to_arrow = to_arrow


def historical(store: FeatureStore, entity_df: pd.DataFrame) -> pd.DataFrame:
    """get_historical_features. 로컬 file 오프라인 저장소의 중복 제거 문제를 피해 나눠 부른다.

    Feast 0.68 file(dask) 저장소는 feature view를 하나씩 붙이면서 '지금까지 나온 join key + 노출 시각'으로
    중복을 지운다. article_profile이 먼저 붙으면 (article_id, 시각)만 보게 되어,
    다른 독자가 같은 기사를 같은 ms에 본 노출 2건이 1건으로 합쳐진다 (가짜 로그 40,763행 중 1쌍).
    그래서 (article_id, 시각)이 겹치는 행은 다른 호출로 보낸다.
    붙는 순서가 바뀌어 다른 키 조합으로 합쳐지면 이 회피책이 막지 못한다 → check_rows가 잡는다.
    BigQuery 저장소는 이 문제가 없어 한 번에 부른다 (check_rows로 확인).
    """
    if store.config.offline_store.type != "file":
        use_rest_download()
        return store.get_historical_features(
            entity_df=entity_df, features=store.get_feature_service(SERVICE)).to_df()
    rank = entity_df.groupby(["article_id", "event_timestamp"]).cumcount()
    parts = []
    for r in sorted(rank.unique()):
        part = entity_df[rank == r]
        parts.append(store.get_historical_features(
            entity_df=part, features=store.get_feature_service(SERVICE)).to_df())
    return pd.concat(parts, ignore_index=True)


def check_rows(entity_df: pd.DataFrame, df: pd.DataFrame) -> None:
    """Feast가 행을 조용히 합치거나 늘리지 않았는지 확인한다."""
    if len(df) != len(entity_df) or not df.row_id.is_unique \
            or set(df.row_id) != set(entity_df.row_id):
        lost = sorted(set(entity_df.row_id) - set(df.row_id))[:5]
        raise RuntimeError(
            f"get_historical_features 행 수 불일치: entity_df {len(entity_df):,}행 → 결과 {len(df):,}행, "
            f"row_id 고유={df.row_id.is_unique}, 빠진 row_id 예={lost}")


def to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Feast 결과 → make_features.py와 같은 열 순서의 float32 행렬 (cold u_emb = NaN)."""
    head = df[HEAD].astype("float64").copy()
    head[FILL_ZERO] = head[FILL_ZERO].fillna(0)   # 행이 없으면 '아직 0번'

    def stack(col: str) -> np.ndarray:
        nan = np.full(DIM, np.nan, dtype=np.float32)
        return np.vstack([np.asarray(v, dtype=np.float32) if v is not None and len(v) else nan
                          for v in df[col]])

    return np.hstack([head.to_numpy(np.float32), stack("u_emb"), stack("a_emb")])


def compare(X: np.ndarray, ref_path: Path) -> None:
    ref = np.load(ref_path)
    R = ref["X"]
    print(f"\n대조: Feast {X.shape} vs make_features {R.shape}")
    if X.shape != R.shape:
        raise RuntimeError(f"모양이 다르다: Feast {X.shape}, {ref_path} {R.shape}")
    names = HEAD + ["u_emb(768)", "a_emb(768)"]
    blocks = [slice(i, i + 1) for i in range(len(HEAD))]
    blocks += [slice(len(HEAD), len(HEAD) + DIM), slice(len(HEAD) + DIM, len(HEAD) + 2 * DIM)]
    for name, sl in zip(names, blocks):
        a, b = X[:, sl], R[:, sl]
        nan_same = np.array_equal(np.isnan(a), np.isnan(b))
        both = ~np.isnan(a) & ~np.isnan(b)
        diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        rows_bad = int(np.sum(np.any((np.abs(np.nan_to_num(a) - np.nan_to_num(b)) > 1e-4)
                                     | (np.isnan(a) != np.isnan(b)), axis=1)))
        print(f"  {name:24s} NaN 위치 같음={nan_same!s:5s} 최대 차이={diff:.2e} 다른 행={rows_bad}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=Path, help="make_features.py가 만든 features.npz")
    parser.add_argument("--yaml", default="feature_store.yaml",
                        help="feature_repo 안의 설정 파일. BigQuery는 feature_store.bq.yaml "
                             "(FEAST_SOURCE=bigquery와 같이 쓴다)")
    args = parser.parse_args()

    repo = HERE / "feature_repo"
    store = FeatureStore(repo_path=str(repo), fs_yaml_file=repo / args.yaml)
    kind = store.config.offline_store.type
    out = DATA / ("training_set.parquet" if kind == "file" else f"training_set_{kind}.parquet")
    entity_df = pd.read_parquet(DATA / "exposures.parquet")
    print(f"오프라인 저장소: {kind}, entity_df: {len(entity_df):,}행 (유효 노출 1건 = 1행)")

    start = time.time()
    df = historical(store, entity_df).sort_values("row_id").reset_index(drop=True)
    check_rows(entity_df, df)
    print(f"get_historical_features: {time.time() - start:.1f}초, 결과 {df.shape} (행 수 검사 통과)")

    df.to_parquet(out, index=False)
    print(f"  저장: {out.name}")
    cold = df["u_emb"].isna() | df["u_emb"].map(lambda v: v is None or len(v) == 0)
    print(f"  학습 {int((df.split == 'train').sum()):,}행 / 검증 {int((df.split == 'valid').sum()):,}행,"
          f" CTR {df.label.mean():.3%}, cold(u_emb 없음) {cold.mean():.1%}")
    print("\n앞 3행 (임베딩 제외):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[["reader_id", "article_id", "event_timestamp", *HEAD, "label"]].head(3)
              .assign(reader_id=lambda d: d.reader_id.str[:8], article_id=lambda d: d.article_id.str[:12]))

    if args.ref:
        compare(to_matrix(df), args.ref)


if __name__ == "__main__":
    main()
