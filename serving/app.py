"""추론 서버. 뉴스 백엔드의 POST /v1/rank 요청에 PCTR 점수와 순위를 돌려준다."""
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from feast import FeatureStore
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .features import ARTICLE_ID, fetch_online, missing_flags, to_matrix
from .model import Model

ROOT = Path(__file__).resolve().parents[1]


class Context(BaseModel):
    request_time: datetime
    device: str | None = None

    @field_validator("request_time")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("request_time에 시간대가 필요하다")
        return value


class RankRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "request_id": "019a0000-0000-7000-8000-000000000001",
        "reader_id": "019a0000-0000-7000-8000-000000000002",
        "candidates": ["news-" + "0" * 31 + "1", "news-" + "0" * 31 + "2"],
        "context": {"request_time": "2026-11-25T10:00:00+09:00", "device": "mobile"},
        "top_k": 2,
    }})

    request_id: str
    reader_id: str
    candidates: list[str] = Field(min_length=1, max_length=200)
    context: Context
    top_k: int | None = None


class RankedItem(BaseModel):
    article_id: str
    pctr: float
    rank: int


class ReasonItem(BaseModel):
    article_id: str
    reason: str


class BadRequest(BaseModel):
    error: str
    field: str


class Unavailable(BaseModel):
    detail: str


class RankResponse(BaseModel):
    request_id: str
    model_version: str
    features_as_of: str
    ranked: list[RankedItem]
    defaulted: list[ReasonItem]
    unscored: list[ReasonItem]


def load_store() -> FeatureStore:
    repo = Path(os.environ.get("FEAST_REPO_PATH", ROOT / "features/feast/feature_repo"))
    fs_yaml = os.environ.get("FEAST_FS_YAML", "feature_store.gcp.yaml")
    return FeatureStore(repo_path=str(repo), fs_yaml_file=str(repo / fs_yaml))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.error = None
    app.state.store = None
    app.state.model = None
    try:
        app.state.store = load_store()
        model = Model()
        model.load()
        app.state.model = model
        app.state.ready = True
    except Exception as exc:  # readyz와 rank가 503으로 알린다
        app.state.error = str(exc)
    yield


app = FastAPI(title="news-rec inference", version="1.0.0", lifespan=lifespan)
UNAVAILABLE = {503: {"model": Unavailable, "description": "모델·피처 저장소를 쓸 수 없음"}}


def openapi() -> dict:
    """검증 오류를 400으로 돌려주므로 FastAPI가 자동으로 넣는 422를 명세에서 뺀다."""
    if not app.openapi_schema:
        schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
        for path in schema["paths"].values():
            for operation in path.values():
                operation["responses"].pop("422", None)
        for name in ("HTTPValidationError", "ValidationError"):
            schema["components"]["schemas"].pop(name, None)
        app.openapi_schema = schema
    return app.openapi_schema


app.openapi = openapi


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(part) for part in first.get("loc", []))
    return JSONResponse(status_code=400, content={"error": first.get("msg", "invalid"), "field": field})


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz", responses=UNAVAILABLE)
def readyz() -> dict:
    if not app.state.ready:
        raise HTTPException(503, detail=app.state.error or "not ready")
    return {"status": "ready", "model_version": app.state.model.version}


@app.get("/v1/model", responses=UNAVAILABLE)
def model_version() -> dict:
    if not app.state.ready:
        raise HTTPException(503, detail=app.state.error or "not ready")
    return {"model_version": app.state.model.version}


@app.post("/v1/rank", response_model=RankResponse, responses={
    400: {"model": BadRequest, "description": "필드가 없거나 형식이 틀림"}, **UNAVAILABLE,
})
def rank(request: RankRequest) -> RankResponse:
    if not app.state.ready:
        raise HTTPException(503, detail=app.state.error or "not ready")

    top_k = len(request.candidates) if request.top_k is None else request.top_k
    if top_k < 1 or top_k > len(request.candidates):
        return JSONResponse(status_code=400, content={"error": "invalid top_k", "field": "top_k"})

    known = [(article_id, position)
             for position, article_id in enumerate(request.candidates, start=1)
             if ARTICLE_ID.match(article_id)]
    unscored = [
        ReasonItem(article_id=article_id, reason="UNKNOWN_ARTICLE")
        for article_id in request.candidates if article_id not in {item[0] for item in known}
    ]
    if not known:
        return RankResponse(
            request_id=request.request_id, model_version=app.state.model.version,
            features_as_of=request.context.request_time.isoformat(),
            ranked=[], defaulted=[], unscored=unscored,
        )

    try:
        online = fetch_online(
            app.state.store, request.reader_id,
            [article_id for article_id, _ in known], request.context.request_time,
            [position for _, position in known],
        )
        matrix = to_matrix(online)
        pctrs = app.state.model.predict(matrix)
    except Exception as exc:
        raise HTTPException(503, detail=f"inference failed: {exc}") from exc

    reader_missing, article_missing = missing_flags(online)
    scored = sorted(
        [(article_id, float(pctr), missing)
         for (article_id, _), pctr, missing in zip(known, pctrs, article_missing)],
        key=lambda row: row[1], reverse=True,
    )[:top_k]
    ranked = [
        RankedItem(article_id=article_id, pctr=pctr, rank=index)
        for index, (article_id, pctr, _) in enumerate(scored, start=1)
    ]
    defaulted = []
    for article_id, _, article_is_missing in scored:
        if article_is_missing:
            defaulted.append(ReasonItem(article_id=article_id, reason="ARTICLE_FEATURES_NOT_FOUND"))
        elif reader_missing:
            defaulted.append(ReasonItem(article_id=article_id, reason="READER_FEATURES_NOT_FOUND"))

    return RankResponse(
        request_id=request.request_id,
        model_version=app.state.model.version,
        # TODO: 온라인 store의 피처 기준 시각을 읽을 수 있게 되면 바꾼다.
        features_as_of=request.context.request_time.isoformat(),
        ranked=ranked,
        defaulted=defaulted,
        unscored=unscored,
    )
