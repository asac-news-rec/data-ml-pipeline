"""Feast online feature 조회. 입력 행렬 조립은 serving/matrix.py에 있다."""
import re

import numpy as np
import pandas as pd
from feast import FeatureStore

from .matrix import DIM, FILL_ZERO, HEAD, feature_names, is_missing, to_matrix  # noqa: F401  (다른 모듈이 여기서 가져간다)

SERVICE = "ctr_lgbm_v1"
ARTICLE_ID = re.compile(r"^news-[0-9a-f]{32}$")


def fetch_online(store: FeatureStore, reader_id: str, candidates: list[str], request_time,
                 positions: list[int] | None = None) -> pd.DataFrame:
    """독자 1명 + 후보 기사 N건의 온라인 피처를 읽는다. position은 후보 순서다."""
    if positions is None:
        positions = list(range(1, len(candidates) + 1))
    rows = [
        {"reader_id": reader_id, "article_id": article_id,
         "request_ts": request_time.timestamp(), "position": position}
        for article_id, position in zip(candidates, positions)
    ]
    online = store.get_online_features(
        features=store.get_feature_service(SERVICE), entity_rows=rows,
    ).to_df()
    online["position"] = [row["position"] for row in rows]
    return online


def missing_flags(online: pd.DataFrame) -> tuple[bool, list[bool]]:
    """독자 피처가 전부 없는지, 기사별 임베딩이 없는지."""
    reader_missing = all(is_missing(value) for value in online["u_emb"])
    article_missing = [is_missing(value) for value in online["a_emb"]]
    return reader_missing, article_missing
