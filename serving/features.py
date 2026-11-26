"""Feast online feature 조회와 LightGBM 입력 행렬 조립."""
import re

import numpy as np
import pandas as pd
from feast import FeatureStore

HEAD = ["position", "article_age_h", "category", "hour", "hist_clicks", "hist_exposures",
        "hours_since_last_click", "already_opened", "already_seen"]
FILL_ZERO = ["hist_clicks", "hist_exposures", "already_opened", "already_seen"]
DIM = 768
SERVICE = "ctr_lgbm_v1"
ARTICLE_ID = re.compile(r"^news-[0-9a-f]{32}$")


def feature_names() -> list[str]:
    return HEAD + [f"u_emb_{i}" for i in range(DIM)] + [f"a_emb_{i}" for i in range(DIM)]


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    try:
        return len(value) == 0
    except TypeError:
        return False


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


def to_matrix(df: pd.DataFrame) -> np.ndarray:
    """일반 9 + u_emb 768 + a_emb 768. cold 임베딩은 NaN으로 둔다."""
    head = df[HEAD].astype("float64").copy()
    head[FILL_ZERO] = head[FILL_ZERO].fillna(0)

    def stack(col: str) -> np.ndarray:
        nan = np.full(DIM, np.nan, dtype=np.float32)
        return np.vstack([
            np.asarray(value, dtype=np.float32) if not is_missing(value) else nan
            for value in df[col]
        ])

    return np.hstack([head.to_numpy(np.float32), stack("u_emb"), stack("a_emb")])


def missing_flags(online: pd.DataFrame) -> tuple[bool, list[bool]]:
    """독자 피처가 전부 없는지, 기사별 임베딩이 없는지."""
    reader_missing = all(is_missing(value) for value in online["u_emb"])
    article_missing = [is_missing(value) for value in online["a_emb"]]
    return reader_missing, article_missing
