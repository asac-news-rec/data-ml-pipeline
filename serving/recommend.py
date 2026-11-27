"""추천 피드: 후보 고르기(최신순) → 점수 → 정렬 목록 저장 → 커서로 나눠 주기.

후보 목록은 매시 DAG(`transform/export_candidates.py`)가 GCS에 쓴 JSON이다. 리트리벌 모델은 MVP 이후다.
정렬한 목록은 Valkey에 30분 둔다. 다음 페이지는 같은 목록에서 이어서 꺼내므로 중복·누락이 없다.
"""
import base64
import binascii
import json
import os
import secrets
import threading
import time
from pathlib import Path

CANDIDATE_LIMIT = int(os.environ.get("CANDIDATE_LIMIT", "200"))
LIST_TTL_S = int(os.environ.get("RANKED_LIST_TTL_S", "1800"))
REFRESH_S = int(os.environ.get("CANDIDATES_REFRESH_S", "300"))
KEY_PREFIX = "rec:list:"


class CursorError(ValueError):
    """커서가 틀렸거나 만료됐다. WAS는 커서 없이 첫 조회부터 다시 부른다."""


# ── 후보 목록 ───────────────────────────────────────────────────────────
def read_text(path: str) -> str:
    if path.startswith("gs://"):
        from .model import read_gcs

        return read_gcs(path)[0]
    return Path(path).read_text(encoding="utf-8")


def default_path() -> str:
    """CANDIDATES_PATH가 없으면 Feast registry와 같은 폴더의 candidates.json(DAG가 쓰는 곳)."""
    if os.environ.get("CANDIDATES_PATH"):
        return os.environ["CANDIDATES_PATH"]
    registry = os.environ.get("FEAST_REGISTRY_PATH", "")
    return registry.rsplit("/", 1)[0] + "/candidates.json" if "/" in registry else "candidates.json"


class CandidatePool:
    """GCS의 후보 JSON을 메모리에 두고 REFRESH_S초마다 다시 읽는다. 다시 읽기에 실패하면 이전 목록을 쓴다."""

    def __init__(self, path: str | None = None, reader=read_text, clock=time.monotonic):
        self.path = path or default_path()
        self.reader = reader
        self.clock = clock
        self.articles: list[dict] = []
        self.generated_at: str | None = None
        self.loaded_at: float | None = None
        self.lock = threading.Lock()

    def load(self) -> None:
        data = json.loads(self.reader(self.path))
        articles = sorted(data["articles"], key=lambda a: (a["published_at"], a["article_id"]), reverse=True)
        self.articles, self.generated_at, self.loaded_at = articles, data.get("generated_at"), self.clock()

    def refresh(self, logger) -> None:
        if self.loaded_at is not None and self.clock() - self.loaded_at < REFRESH_S:
            return
        with self.lock:
            if self.loaded_at is not None and self.clock() - self.loaded_at < REFRESH_S:
                return
            try:
                self.load()
            except Exception:
                if self.loaded_at is None:
                    raise
                self.loaded_at = self.clock()   # 실패해도 REFRESH_S 동안은 다시 시도하지 않는다
                logger.exception("후보 목록 다시 읽기 실패. 이전 목록을 쓴다")

    def pick(self, category: str | None, subcategory: str | None, limit: int = CANDIDATE_LIMIT) -> list[str]:
        """조건에 맞는 최신 기사 limit개. 발행 시각이 같으면 article_id 역순."""
        out = []
        for article in self.articles:
            if category is not None and article.get("category") != category:
                continue
            if subcategory is not None and article.get("subcategory") != subcategory:
                continue
            out.append(article["article_id"])
            if len(out) == limit:
                break
        return out


# ── 정렬 목록 저장 ──────────────────────────────────────────────────────
class MemoryLists:
    """Valkey가 없는 로컬 실행·테스트용."""

    def __init__(self, clock=time.monotonic):
        self.data: dict[str, tuple[float, str]] = {}
        self.clock = clock

    def set(self, key: str, value: str, ttl: int) -> None:
        self.data[key] = (self.clock() + ttl, value)

    def get(self, key: str) -> str | None:
        expires, value = self.data.get(key, (0, None))
        return value if self.clock() < expires else None


class ValkeyLists:
    def __init__(self, host: str, port: int = 6379):
        import redis

        self.client = redis.Redis(host=host, port=port, socket_timeout=0.5, socket_connect_timeout=0.5)

    def set(self, key: str, value: str, ttl: int) -> None:
        self.client.set(key, value, ex=ttl)

    def get(self, key: str) -> str | None:
        value = self.client.get(key)
        return None if value is None else value.decode()


def make_lists():
    host = os.environ.get("VALKEY_HOST")
    return ValkeyLists(host) if host else MemoryLists()


# ── 커서 ────────────────────────────────────────────────────────────────
def encode_cursor(list_id: str, offset: int) -> str:
    raw = json.dumps({"l": list_id, "o": offset}, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[str, int]:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        data = json.loads(raw)
        list_id, offset = data["l"], data["o"]
    except (binascii.Error, ValueError, KeyError, TypeError) as exc:
        raise CursorError("cursor 형식이 틀렸다") from exc
    if not isinstance(list_id, str) or not isinstance(offset, int) or offset < 0:
        raise CursorError("cursor 형식이 틀렸다")
    return list_id, offset


def save_ranking(lists, reader_id: str, category, subcategory, model_version: str,
                 ranked: list[tuple[str, float]]) -> str:
    list_id = secrets.token_urlsafe(12)
    lists.set(KEY_PREFIX + list_id, json.dumps({
        "reader_id": reader_id, "category": category, "subcategory": subcategory,
        "model_version": model_version, "items": ranked,
    }), LIST_TTL_S)
    return list_id


def load_ranking(lists, list_id: str, reader_id: str, category, subcategory) -> dict:
    raw = lists.get(KEY_PREFIX + list_id)
    if raw is None:
        raise CursorError("cursor가 만료됐다")
    saved = json.loads(raw)
    if (saved["reader_id"], saved["category"], saved["subcategory"]) != (reader_id, category, subcategory):
        raise CursorError("cursor가 이 reader_id·category·subcategory의 것이 아니다")
    return saved


def page(saved: dict, list_id: str, offset: int, size: int) -> tuple[list[tuple[int, str, float]], str | None]:
    """offset부터 size개와 다음 커서. 더 없으면 커서는 None."""
    items = saved["items"]
    if offset > len(items):
        raise CursorError("cursor 위치가 목록 밖이다")
    chunk = [(offset + i + 1, article_id, score) for i, (article_id, score) in enumerate(items[offset:offset + size])]
    end = offset + len(chunk)
    return chunk, (encode_cursor(list_id, end) if end < len(items) else None)
