"""학습과 추론 서버가 같이 쓰는 LightGBM 입력 행렬 조립.

training/train_lgbm.py와 serving/features.py가 이 파일 하나만 쓴다. 열 순서나 결측 채우기를
바꾸면 학습과 서빙이 함께 바뀐다. feast를 import하지 않는다(학습 이미지에서도 가볍게 쓰려고).
"""
import numpy as np
import pandas as pd

HEAD = ["position", "article_age_h", "category", "hour", "hist_clicks", "hist_exposures",
        "hours_since_last_click", "already_opened", "already_seen"]
FILL_ZERO = ["hist_clicks", "hist_exposures", "already_opened", "already_seen"]
DIM = 768


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


def to_matrix(df: pd.DataFrame) -> np.ndarray:
    """일반 9 + u_emb 768 + a_emb 768. cold 임베딩은 NaN으로 둔다."""
    head = df[HEAD].astype("float64").copy()
    head[FILL_ZERO] = head[FILL_ZERO].fillna(0)

    def stack(col: str) -> np.ndarray:
        nan = np.full(DIM, np.nan, dtype=np.float32)
        return np.vstack([
            nan if is_missing(value) else np.asarray(value, dtype=np.float32)
            for value in df[col]
        ])

    return np.hstack([head.to_numpy(np.float32), stack("u_emb"), stack("a_emb")])
