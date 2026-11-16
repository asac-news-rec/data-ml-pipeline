"""[파이프라인 구간: 서빙 내보내기] materialize → get_online_features: 추론 서버가 요청 1건에 하는 일을 흉내 낸다.

실행:  .venv/bin/python online_demo.py [--now 2026-11-06T12:00:00+09:00] [--reader <reader_id>] [--k 10]
순서:
  1. materialize: 오프라인 표에서 entity별 '가장 최근 값'만 골라 온라인 저장소(SQLite/Valkey)에 쓴다.
     ttl이 무제한이므로 범위는 source table의 가장 이른 시각부터 --now까지 전체다.
     (실서비스는 첫 적재만 전체로 하고 이후엔 materialize_incremental로 지난 적재 뒤 변경분만 쓴다.)
  2. get_online_features: 독자 1명 + 후보 기사 k건의 피처를 온라인 저장소에서 읽는다.
  3. 같은 (독자, 기사, 시각)을 get_historical_features로도 뽑아 두 값이 같은지 본다
     (학습 때 본 피처 = 서빙 때 보는 피처인지 확인. 다르면 training-serving skew).
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from feast import FeatureStore

from build_training_set import DIM, HEAD, SERVICE, to_matrix

HERE = Path(__file__).parent
DATA = HERE / "data"
SOURCES = ["article_profile", "reader_profile", "reader_activity", "reader_article"]


def earliest_source_time() -> datetime:
    """source table 4개 중 가장 이른 event_timestamp. materialize 시작점으로 쓴다."""
    first = min(pd.read_parquet(DATA / f"{n}.parquet", columns=["event_timestamp"]).event_timestamp.min()
                for n in SOURCES)
    return first.to_pydatetime()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", default="2026-11-06T12:00:00+09:00",
                        help="'지금'으로 둘 시각 (ISO 8601, 시간대 포함). 기본값 = 가짜 로그 끝")
    parser.add_argument("--reader")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    now = datetime.fromisoformat(args.now)
    if now.tzinfo is None:
        raise SystemExit(f"--now에 시간대가 없다: {args.now}")

    store = FeatureStore(repo_path=str(HERE / "feature_repo"))

    start = time.time()
    since = earliest_source_time()
    store.materialize(start_date=since, end_date=now)
    print(f"materialize ({since:%Y-%m-%d %H:%M} ~ {now:%Y-%m-%d %H:%M}): {time.time() - start:.1f}초")

    profile = pd.read_parquet(DATA / "reader_profile.parquet")
    articles = pd.read_parquet(DATA / "article_profile.parquet")
    pairs = pd.read_parquet(DATA / "reader_article.parquet")
    reader_id = args.reader or profile.reader_id.value_counts().index[0]
    opened = set(pairs[(pairs.reader_id == reader_id) & (pairs.already_opened == 1)].article_id)
    # 후보: 지금까지 발행된 기사 중 일부(열어 본 기사 2건 + 안 연 기사). 실서비스에선 후보 생성 단계가 정한다.
    published = articles[articles.event_timestamp <= pd.Timestamp(now)]
    unopened = published[~published.article_id.isin(opened)].article_id.tolist()
    candidates = sorted(opened)[:2] + unopened[: args.k - 2]
    t = now.timestamp()
    rows = [{"reader_id": reader_id, "article_id": a, "request_ts": t, "position": i + 1}
            for i, a in enumerate(candidates)]

    times = []
    for _ in range(5):   # 첫 호출은 registry 읽기 등 준비 시간이 섞여서 여러 번 잰다
        start = time.time()
        online = store.get_online_features(
            features=store.get_feature_service(SERVICE), entity_rows=rows,
        ).to_df()
        times.append((time.time() - start) * 1000)
    # position은 요청값이라 저장소에서 오지 않는다. 서버가 요청에서 받은 값을 그대로 붙인다.
    online["position"] = [r["position"] for r in rows]
    print(f"\nget_online_features: 독자 {reader_id[:8]}… + 후보 {len(rows)}건, "
          f"첫 호출 {times[0]:.0f}ms / 이후 {np.median(times[1:]):.0f}ms")
    print(f"  반환 열 {len(online.columns)}개: {list(online.columns)}")
    view = online[["article_id", *HEAD]].copy()
    view["article_id"] = view.article_id.str[5:17]
    view["u_emb[:3]"] = online.u_emb.map(lambda v: np.round(v[:3], 3) if v is not None else None)
    view["a_emb[:3]"] = online.a_emb.map(lambda v: np.round(v[:3], 3) if v is not None else None)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(view.to_string(index=False))

    # 같은 요청을 오프라인(point-in-time)으로도 뽑아 비교
    entity_df = pd.DataFrame(rows).assign(event_timestamp=pd.Timestamp(now))
    offline = store.get_historical_features(
        entity_df=entity_df, features=store.get_feature_service(SERVICE),
    ).to_df()
    offline = offline.set_index("article_id").loc[candidates].reset_index()
    A, B = to_matrix(online), to_matrix(offline)
    same = np.allclose(np.nan_to_num(A, nan=-9), np.nan_to_num(B, nan=-9), atol=1e-5)
    print(f"\n온라인 vs 오프라인(같은 시각): {A.shape} 행렬 일치 = {same}")
    print(f"  모델 입력 1행 = 일반 {len(HEAD)} + u_emb {DIM} + a_emb {DIM} = {A.shape[1]}열")
    if not same:
        raise RuntimeError("온라인 피처와 오프라인 피처가 다르다 (training-serving skew)")


if __name__ == "__main__":
    main()
