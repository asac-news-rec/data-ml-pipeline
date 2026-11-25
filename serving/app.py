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
    request_time: datetime = Field(description=(
        "WAS가 피드 요청을 받은 시각. 시간대를 꼭 붙인다(예: +09:00, Z). 기사 나이·시간대 피처의 기준이다. "
        "반환 기록의 served_at과 같은 값을 넣는다."))
    device: str | None = Field(default=None, description=(
        "독자 기기: mobile, desktop, tablet 또는 null. 지금 모델은 쓰지 않으므로 생략해도 결과가 같다."))

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

    request_id: str = Field(description=(
        "이번 피드 응답 번호. WAS가 만든 값을 그대로 돌려준다(로그 대조용). 반환 기록의 request_id와 같은 값을 넣는다."))
    reader_id: str = Field(description="익명 독자 ID(쿠키에서 검증한 reader_id). 독자 피처를 찾는 열쇠다.")
    candidates: list[str] = Field(min_length=1, max_length=200, description=(
        "점수를 매길 article_id 목록(news_articles.id, news- 뒤 16진수 32자). 1~200개, 권장 20~30개. "
        "현재 화면에 보여 줄 순서(최신순)대로 넣는다. 이 순서가 모델의 position(1부터) 피처가 된다. 중복 없이 넣는다."))
    context: Context = Field(description="요청 시점의 정보")
    top_k: int | None = Field(default=None, description=(
        "점수 상위 몇 개를 돌려받을지. 1 이상, candidates 개수 이하. 생략하거나 null이면 점수를 낸 후보를 모두 돌려준다. "
        "순서만 바꾸려면 생략한다."))


class RankedItem(BaseModel):
    article_id: str = Field(description="요청한 candidates 중 하나")
    pctr: float = Field(description="이 독자가 이 기사를 열 확률 예측값(0~1). 반환 기록의 score에 넣는다.")
    rank: int = Field(description="추천 순위. 1부터, pctr이 높은 순서")


class ReasonItem(BaseModel):
    article_id: str = Field(description="요청한 candidates 중 하나")
    reason: str = Field(description=(
        "READER_FEATURES_NOT_FOUND(독자 피처 없음), ARTICLE_FEATURES_NOT_FOUND(기사 임베딩 없음), "
        "UNKNOWN_ARTICLE(article_id 형식이 틀림)"))


class BadRequest(BaseModel):
    error: str = Field(description="틀린 이유")
    field: str = Field(description="틀린 필드(예: body.candidates)")


class Unavailable(BaseModel):
    detail: str = Field(description="준비되지 않은 이유. 화면에 보여 주지 말고 로그에만 남긴다.")


class RankResponse(BaseModel):
    request_id: str = Field(description="요청의 request_id 그대로")
    model_version: str = Field(description="점수를 낸 모델 버전. 반환 기록의 model_version에 그대로 넣는다.")
    features_as_of: str = Field(description=(
        "피처 기준 시각. 지금은 임시로 request_time을 그대로 돌려준다. 실제 피처 시각이 될 때까지 DB에 저장하지 않는다(NULL)."))
    ranked: list[RankedItem] = Field(description="점수를 낸 기사. 이 배열 순서대로 화면에 보여 준다.")
    defaulted: list[ReasonItem] = Field(description=(
        "피처 일부가 없이 점수를 낸 기사. ranked에도 들어 있다. 오류가 아니며 순서를 바꿀 필요가 없다(기록용)."))
    unscored: list[ReasonItem] = Field(description=(
        "점수를 내지 못한 기사. ranked에 없다. ranked 뒤에 원래 순서대로 붙인다(score는 NULL)."))


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


@app.get("/healthz", summary="프로세스 생존 확인(쿠버네티스용)")
def healthz() -> dict:
    """프로세스가 살아 있으면 200. 모델·Valkey 상태는 보지 않는다. 쿠버네티스 liveness가 부르고 WAS는 부르지 않는다."""
    return {"status": "ok"}


@app.get("/readyz", responses=UNAVAILABLE, summary="요청을 받을 준비 확인(쿠버네티스용)")
def readyz() -> dict:
    """시작할 때 모델과 Feast 정의를 읽었으면 200, 못 읽었으면 503. 쿠버네티스 readiness가 부르고 WAS는 부르지 않는다.

    Valkey 접속은 확인하지 않는다. Valkey 장애는 /v1/rank의 503으로 드러난다.
    """
    if not app.state.ready:
        raise HTTPException(503, detail=app.state.error or "not ready")
    return {"status": "ready", "model_version": app.state.model.version}


@app.get("/v1/model", responses=UNAVAILABLE, summary="올라간 모델 버전(운영 확인용)")
def model_version() -> dict:
    """지금 올라간 모델 버전. 사람이 확인할 때 쓰고 WAS는 부르지 않는다(/v1/rank 응답에 들어 있다)."""
    if not app.state.ready:
        raise HTTPException(503, detail=app.state.error or "not ready")
    return {"model_version": app.state.model.version}


@app.post("/v1/rank", response_model=RankResponse, summary="후보 기사 순위 매기기", responses={
    400: {"model": BadRequest, "description": "필드가 없거나 형식이 틀림"}, **UNAVAILABLE,
})
def rank(request: RankRequest) -> RankResponse:
    """독자 1명과 후보 기사를 받아 기사를 열 확률(pctr)이 높은 순서로 돌려준다.

    - 누가·언제: 뉴스 WAS가 피드 요청마다 후보(최신순 한 페이지)를 조회한 뒤 한 번 부른다. 기사마다 따로 부르지 않는다.
    - 200: ranked 순서대로 보여 주고 unscored를 뒤에 붙인다. 신규 독자도 200이다.
    - 400: 요청이 틀렸다(코드 버그). 최신순으로 응답하고 오류 로그를 남긴다.
    - 503·시간 초과·연결 실패: 최신순으로 응답한다. 반환 기록은 score=NULL, model_version="fallback".
    """
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
