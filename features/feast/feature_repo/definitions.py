"""[파이프라인 구간: 피처 → 학습 데이터·서빙 내보내기] Feast 피처 정의.

`feast apply`가 이 파일을 읽어 registry에 등록한다.
entity 2개(reader, article) → feature view 4개 + on-demand feature view 1개.
열 이름과 뜻은 make_features.py의 HEAD_COLS와 같다.

ttl은 모두 0(무제한)이다. 모든 피처가 '지금까지' 누적값이거나 고정값이라,
마지막 사건이 오래됐다고 null(→ 0)로 바꾸면 make_features.py의 정의(전체 누적)와 어긋난다.
'최근 N일' 같은 피처가 필요하면 원천 표에서 따로 계산해 새 피처로 둔다.
"""
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from feast import (BigQuerySource, Entity, FeatureService, FeatureView, Field, FileSource,
                   RequestSource)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Array, Float32, Float64, Int64
from feast.value_type import ValueType

DATA = Path(__file__).resolve().parent.parent / "data"
KST_OFFSET_S = 9 * 3600
NO_TTL = timedelta(0)   # Feast에서 ttl 0 = 기간 제한 없음

# ── entity: 피처를 찾는 열쇠 ─────────────────────────────────────────
reader = Entity(name="reader", join_keys=["reader_id"], value_type=ValueType.STRING,
                description="독자 (로그의 reader_id)")
article = Entity(name="article", join_keys=["article_id"], value_type=ValueType.STRING,
                 description="기사 (로그의 article_id)")


# 원천 표 위치. FEAST_SOURCE=bigquery면 BigQuery 표(upload_to_bq.py로 올린 것), 아니면 로컬 parquet.
# 정의(피처 이름·타입·entity)는 같고 원천만 바뀐다. BigQuery 쪽 설정은 feature_store.bq.yaml.
SOURCE_KIND = os.environ.get("FEAST_SOURCE", "file")
BQ_DATASET = os.environ.get("FEAST_BQ_DATASET", "final-project-513508.dev_yoonmo")


def source(name: str):
    if SOURCE_KIND == "bigquery":
        return BigQuerySource(name=f"{name}_src", table=f"{BQ_DATASET}.{name}",
                              timestamp_field="event_timestamp")
    return FileSource(name=f"{name}_src", path=str(DATA / f"{name}.parquet"),
                      timestamp_field="event_timestamp")


# ── feature view: 원천 표 1개 = 피처 묶음 1개 ───────────────────────────
article_profile = FeatureView(
    name="article_profile",
    entities=[article],
    ttl=NO_TTL,
    schema=[
        Field(name="a_emb", dtype=Array(Float32), description="e5-base 기사 임베딩 768"),
        Field(name="category", dtype=Int64, description="카테고리 코드"),
        Field(name="published_ts", dtype=Float64, description="발행 시각(epoch 초)"),
    ],
    source=source("article_profile"),
)

reader_profile = FeatureView(
    name="reader_profile",
    entities=[reader],
    ttl=NO_TTL,   # 오래된 클릭은 타임 디케이 가중치로 이미 약해진다. 기간으로 자르지 않는다.
    schema=[
        Field(name="u_emb", dtype=Array(Float32),
              description="클릭 기사 임베딩의 타임 디케이 가중 평균(반감기 24h) 768"),
        Field(name="hist_clicks", dtype=Int64, description="지금까지 연 기사 수"),
        Field(name="last_click_ts", dtype=Float64, description="마지막 클릭 시각(epoch 초)"),
    ],
    source=source("reader_profile"),
)

reader_activity = FeatureView(
    name="reader_activity",
    entities=[reader],
    ttl=NO_TTL,
    schema=[Field(name="hist_exposures", dtype=Int64, description="지금까지 유효 노출 수")],
    source=source("reader_activity"),
)

reader_article = FeatureView(
    name="reader_article",
    entities=[reader, article],
    ttl=NO_TTL,
    schema=[
        Field(name="already_opened", dtype=Int64, description="이 기사를 전에 열었나"),
        Field(name="already_seen", dtype=Int64, description="이 기사를 전에 유효 노출로 봤나"),
    ],
    source=source("reader_article"),
)

# ── 요청 시각에만 정해지는 값: 학습 땐 entity_df 열, 서빙 땐 요청에서 받는다 ──
exposure_request = RequestSource(
    name="exposure_request",
    schema=[
        Field(name="request_ts", dtype=Float64, description="노출(학습)·요청(서빙) 시각 epoch 초"),
        Field(name="position", dtype=Int64, description="목록에서 몇 번째 카드인가"),
    ],
)


@on_demand_feature_view(
    sources=[reader_profile, article_profile, exposure_request],
    schema=[
        Field(name="article_age_h", dtype=Float64),
        Field(name="hour", dtype=Int64),
        Field(name="hours_since_last_click", dtype=Float64),
    ],
    mode="pandas",
)
def request_context(df: pd.DataFrame) -> pd.DataFrame:
    """요청 시각과 저장된 시각의 차이. 학습과 서빙이 이 함수 하나를 같이 쓴다."""
    t = df["request_ts"].astype("float64")
    last_click = pd.to_numeric(df["last_click_ts"], errors="coerce")
    out = pd.DataFrame()
    out["article_age_h"] = (t - df["published_ts"].astype("float64")) / 3600
    out["hour"] = (np.floor((t + KST_OFFSET_S) / 3600) % 24).astype("int64")
    out["hours_since_last_click"] = (t - last_click) / 3600
    return out


# ── 모델 하나가 쓰는 피처 목록 ──────────────────────────────────────────
ctr_lgbm_v1 = FeatureService(
    name="ctr_lgbm_v1",
    features=[
        request_context,
        article_profile[["category", "a_emb"]],
        reader_profile[["hist_clicks", "u_emb"]],
        reader_activity,
        reader_article,
    ],
    description="PCTR LightGBM: 일반 9 + u_emb 768 + a_emb 768",
)
