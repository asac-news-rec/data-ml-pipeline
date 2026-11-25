"""추론 서버. 뉴스 백엔드의 POST /v1/rank 요청에 PCTR 점수와 순위를 돌려준다."""
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import Body, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from feast import FeatureStore
from pydantic import BaseModel, Field, field_validator

from . import examples
from .features import ARTICLE_ID, fetch_online, missing_flags, to_matrix
from .model import Model

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("uvicorn.error")
Reason = Literal["READER_FEATURES_NOT_FOUND", "ARTICLE_FEATURES_NOT_FOUND", "UNKNOWN_ARTICLE"]


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
    request_id: str = Field(description=(
        "이번 피드 응답 번호. WAS가 만든 값을 그대로 돌려준다(로그 대조용). 반환 기록의 request_id와 같은 값을 넣는다."))
    reader_id: str = Field(description="익명 독자 ID(쿠키에서 검증한 reader_id). 독자 피처를 찾는 열쇠다.")
    candidates: list[str] = Field(min_length=1, max_length=200, description=(
        "점수를 매길 article_id 목록(news_articles.id, news- 뒤 16진수 32자). 1~200개, 권장 20~30개. "
        "현재 화면에 보여 줄 순서(최신순)대로 넣는다. 이 순서가 모델의 position(1부터) 피처가 된다. 중복 없이 넣는다."))
    context: Context = Field(description="요청 시점의 정보")
    top_k: int | None = Field(default=None, ge=1, description=(
        "점수 상위 몇 개를 돌려받을지. 1 이상, candidates 개수 이하. 생략하거나 null이면 점수를 낸 후보를 모두 돌려준다. "
        "순서만 바꾸려면 생략한다. 잘린 후보는 ranked·defaulted·unscored 어디에도 없다."))

    @field_validator("candidates")
    @classmethod
    def no_duplicates(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("candidates에 중복이 있다")
        return value


class RankedItem(BaseModel):
    article_id: str = Field(description="요청한 candidates 중 하나")
    pctr: float = Field(ge=0, le=1, description="이 독자가 이 기사를 열 확률 예측값(0~1). 반환 기록의 score에 넣는다.")
    rank: int = Field(ge=1, description="추천 순위. 1부터, pctr이 높은 순서")


class ReasonItem(BaseModel):
    article_id: str = Field(description="요청한 candidates 중 하나")
    reason: Reason = Field(description=(
        "READER_FEATURES_NOT_FOUND(독자 피처 없음), ARTICLE_FEATURES_NOT_FOUND(기사 임베딩 없음), "
        "UNKNOWN_ARTICLE(article_id 형식이 틀림)"))


class BadRequest(BaseModel):
    error: str = Field(description="틀린 이유")
    field: str = Field(description="틀린 필드(예: body.candidates)")


class Unavailable(BaseModel):
    detail: Literal["not ready", "inference failed"] = Field(description="고정 문구. 자세한 원인은 추론 서버 로그에만 남는다.")


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
    app.state.store = None
    app.state.model = None
    try:
        app.state.store = load_store()
        model = Model()
        model.load()
        app.state.model = model
        app.state.ready = True
    except Exception:  # readyz와 rank가 503으로 알린다. 원인은 로그에만 남긴다.
        logger.exception("추론 서버 준비 실패")
    yield


DESCRIPTION = """뉴스 피드의 후보 기사에 점수를 매겨 순서를 돌려주는 서버.

| API | 누가 부르나 | 언제 |
|---|---|---|
| `POST /v1/rank` | 뉴스 WAS | 피드 요청마다 한 번 |
| `GET /healthz` | 쿠버네티스 liveness | 주기적으로 |
| `GET /readyz` | 쿠버네티스 readiness | 주기적으로 |
| `GET /v1/model` | 사람 | 올라간 모델을 확인할 때 |

WAS는 `POST /v1/rank`만 부른다.

- 순서: 최신순 후보 조회 → `/v1/rank` 한 번 호출 → `ranked` 순서대로 + `unscored`를 뒤에 붙여 응답 → 반환 기록 저장
- 반환 기록: `ranked[].pctr` → `score`, `model_version` → `model_version`. `features_as_of`는 임시값이라 NULL로 둔다.
- 실패(400·503·시간 초과·연결 실패): 최신순 그대로 응답하고 `score`는 NULL, `model_version`은 `fallback`
- 인증 없음. 클러스터 내부에서만 부른다.
"""
app = FastAPI(title="news-rec inference", version="1.0.0", description=DESCRIPTION, lifespan=lifespan)


def documented(model, description: str, values: dict) -> dict:
    return {"model": model, "description": description, "content": {"application/json": {"examples": values}}}


UNAVAILABLE = {503: documented(Unavailable, "모델·피처 저장소를 쓸 수 없음", examples.UNAVAILABLE)}


def openapi() -> dict:
    """검증 오류를 400으로 돌려주므로 FastAPI가 자동으로 넣는 422를 명세에서 뺀다."""
    if not app.openapi_schema:
        schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
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
        raise HTTPException(503, detail="not ready")
    return {"status": "ready", "model_version": app.state.model.version}


@app.get("/v1/model", responses=UNAVAILABLE, summary="올라간 모델 버전(운영 확인용)")
def model_version() -> dict:
    """지금 올라간 모델 버전. 사람이 확인할 때 쓰고 WAS는 부르지 않는다(/v1/rank 응답에 들어 있다)."""
    if not app.state.ready:
        raise HTTPException(503, detail="not ready")
    return {"model_version": app.state.model.version}


@app.post("/v1/rank", response_model=RankResponse, summary="후보 기사 순위 매기기", responses={
    200: documented(RankResponse, "추천 순서. 신규 독자도 200이다.", examples.RESPONSES),
    400: documented(BadRequest, "필드가 없거나 형식이 틀림", examples.BAD_REQUESTS),
    **UNAVAILABLE,
})
def rank(request: RankRequest = Body(openapi_examples=examples.REQUESTS)) -> RankResponse:
    """독자 1명과 후보 기사를 받아 기사를 열 확률(pctr)이 높은 순서로 돌려준다.

    - 누가·언제: 뉴스 WAS가 피드 요청마다 후보(최신순 한 페이지)를 조회한 뒤 한 번 부른다. 기사마다 따로 부르지 않는다.
    - 200: ranked 순서대로 보여 주고 unscored를 뒤에 붙인다. 신규 독자도 200이다.
    - 400: 요청이 틀렸다(코드 버그). 최신순으로 응답하고 오류 로그를 남긴다.
    - 503·시간 초과·연결 실패: 최신순으로 응답한다. 반환 기록은 score=NULL, model_version="fallback".
    """
    if not app.state.ready:
        raise HTTPException(503, detail="not ready")

    top_k = len(request.candidates) if request.top_k is None else request.top_k
    if top_k > len(request.candidates):
        return JSONResponse(status_code=400, content={
            "error": "top_k는 1 이상, candidates 개수 이하여야 한다", "field": "body.top_k"})

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
        logger.exception("추론 실패 request_id=%s", request.request_id)
        raise HTTPException(503, detail="inference failed") from exc

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
