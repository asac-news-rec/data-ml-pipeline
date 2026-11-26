"""추천 피드 /v1/recommend: 후보 고르기, 페이지 나누기, 커서 오류, 실패 응답."""
import json
import logging
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

try:
    import numpy as np
    from fastapi import HTTPException
    from pydantic import ValidationError
    from serving import app as A
    from serving import examples
    from serving import recommend as R
except ImportError as exc:
    raise unittest.SkipTest(f"serving 묶음이 없다: {exc}")

from transform.export_candidates import to_payload

# 45개: 정치 30(북한 10), 경제 15. published_at이 클수록 최신이다.
ARTICLES = [{"article_id": f"news-{i:032x}", "category": "정치" if i < 30 else "경제",
             "subcategory": "북한" if i < 10 else None, "published_at": f"2026-11-26T{i // 60:02d}:{i % 60:02d}:00+00:00"}
            for i in range(45)]
LOG = logging.getLogger("test")


def pool(articles=ARTICLES) -> R.CandidatePool:
    return R.CandidatePool("x", reader=lambda _: json.dumps({"generated_at": "t", "articles": articles}))


def scores_by_id(online_ids):
    """기사 번호가 클수록 점수가 높다(0.01 간격)."""
    return np.array([int(a[5:], 16) / 100 for a in online_ids])


class Pool(unittest.TestCase):
    def test_pick_latest_with_filters(self):
        p = pool()
        p.refresh(LOG)
        self.assertEqual(p.pick(None, None, 3), [ARTICLES[i]["article_id"] for i in (44, 43, 42)])
        self.assertEqual(len(p.pick("정치", None)), 30)
        self.assertEqual(p.pick("정치", "북한", 2), [ARTICLES[i]["article_id"] for i in (9, 8)])
        self.assertEqual(p.pick("없는분류", None), [])

    def test_refresh_keeps_old_list_on_failure(self):
        clock = [0.0]
        calls = []

        def reader(_):
            calls.append(1)
            if len(calls) > 1:
                raise OSError("gcs down")
            return json.dumps({"articles": ARTICLES})
        p = R.CandidatePool("x", reader=reader, clock=lambda: clock[0])
        p.refresh(LOG)
        clock[0] = R.REFRESH_S + 1
        with self.assertLogs("test", "ERROR"):
            p.refresh(LOG)
        self.assertEqual(len(p.pick(None, None)), 45)
        p.refresh(LOG)                       # 실패 뒤 REFRESH_S 동안은 다시 읽지 않는다
        self.assertEqual(len(calls), 2)

    def test_default_path_next_to_registry(self):
        env = {"FEAST_REGISTRY_PATH": "gs://b/feast/registry.pb"}
        with patch.dict("os.environ", env, clear=True):
            self.assertEqual(R.default_path(), "gs://b/feast/candidates.json")
        with patch.dict("os.environ", {**env, "CANDIDATES_PATH": "/tmp/c.json"}, clear=True):
            self.assertEqual(R.default_path(), "/tmp/c.json")

    def test_first_load_failure_raises(self):
        p = R.CandidatePool("x", reader=lambda _: (_ for _ in ()).throw(OSError("x")))
        with self.assertRaises(OSError):
            p.refresh(LOG)


class Cursor(unittest.TestCase):
    def test_round_trip_and_bad_values(self):
        self.assertEqual(R.decode_cursor(R.encode_cursor("abc", 20)), ("abc", 20))
        for bad in ("!!!", "e30", R.encode_cursor("abc", -1), "eyJsIjoxLCJvIjoyfQ"):
            with self.subTest(bad=bad), self.assertRaises(R.CursorError):
                R.decode_cursor(bad)

    def test_memory_lists_expire(self):
        clock = [0.0]
        lists = R.MemoryLists(clock=lambda: clock[0])
        lists.set("k", "v", 10)
        self.assertEqual(lists.get("k"), "v")
        clock[0] = 11
        self.assertIsNone(lists.get("k"))


class Recommend(unittest.TestCase):
    def setUp(self):
        A.app.state.ready = True
        A.app.state.store = MagicMock()
        A.app.state.model = MagicMock(version="pctr@3")
        A.app.state.pool = pool()
        A.app.state.lists = R.MemoryLists()
        self.fetched = []

        def fetch(store, reader_id, candidates, request_time):
            self.fetched.append((reader_id, candidates, request_time))
            return candidates
        self.patches = [patch.object(A, "fetch_online", side_effect=fetch),
                        patch.object(A, "to_matrix", side_effect=lambda ids: ids)]
        for p in self.patches:
            p.start()
        A.app.state.model.predict.side_effect = scores_by_id

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def call(self, **body):
        return A.recommend(A.RecommendRequest(reader_id="r1", **body))

    def test_pages_continue_without_duplicates(self):
        first = self.call(category="정치")
        self.assertEqual([i.position for i in first.items], list(range(1, 21)))
        self.assertEqual(first.items[0].article_id, ARTICLES[29]["article_id"])   # 점수 0.29가 1등
        self.assertEqual(first.model_version, "pctr@3")
        second = self.call(category="정치", cursor=first.next_cursor)
        self.assertEqual([i.position for i in second.items], list(range(21, 31)))
        self.assertIsNone(second.next_cursor)
        ids = [i.article_id for i in first.items + second.items]
        self.assertEqual(len(set(ids)), 30)
        self.assertEqual(len(self.fetched), 1)          # 다음 페이지는 점수를 다시 매기지 않는다
        self.assertEqual(self.fetched[0][2].tzinfo, timezone.utc)

    def test_size_20_returns_20_and_next_starts_after_last_returned(self):
        """후보 45개, size=20: 20 → 20 → 5개. 커서는 실제로 돌려준 마지막 기사 다음에서 이어진다."""
        pages, cursor = [], None
        while True:
            res = self.call(size=20, cursor=cursor)
            pages.append([i.position for i in res.items])
            cursor = res.next_cursor
            if cursor is None:
                break
        self.assertEqual([len(p) for p in pages], [20, 20, 5])
        self.assertEqual(sum(pages, []), list(range(1, 46)))   # 21~40번이 빠지지 않는다

    def test_size_and_exact_end(self):
        first = self.call(category="경제", size=15)
        self.assertEqual(len(first.items), 15)
        self.assertIsNone(first.next_cursor)

    def test_no_candidates_is_empty_200(self):
        out = self.call(category="없는분류")
        self.assertEqual((out.items, out.next_cursor), ([], None))
        self.assertEqual(self.fetched, [])

    def test_cursor_errors_are_400(self):
        first = self.call(category="정치")
        cases = {"다른 분류": dict(category="경제", cursor=first.next_cursor),
                 "깨진 값": dict(category="정치", cursor="!!"),
                 "없는 목록": dict(category="정치", cursor=R.encode_cursor("gone", 20)),
                 "목록 밖": dict(category="정치", cursor=R.encode_cursor(R.decode_cursor(first.next_cursor)[0], 99))}
        for name, body in cases.items():
            with self.subTest(name):
                response = self.call(**body)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(json.loads(response.body)["field"], "body.cursor")

    def test_failures_are_fixed_503(self):
        A.app.state.model.predict.side_effect = lambda ids: np.full(len(ids), np.nan)
        with self.assertLogs("uvicorn.error", "ERROR"), self.assertRaises(HTTPException) as caught:
            self.call()
        self.assertEqual((caught.exception.status_code, caught.exception.detail), (503, "inference failed"))
        A.app.state.ready = False
        with self.assertRaises(HTTPException) as caught:
            self.call()
        self.assertEqual(caught.exception.detail, "not ready")

    def test_request_validation(self):
        for body in ({"size": 0}, {"size": 51}, {"category": ""}, {"cursor": ""}):
            with self.subTest(body=body), self.assertRaises(ValidationError):
                A.RecommendRequest(reader_id="r1", **body)
        with self.assertRaises(ValidationError):
            A.RecommendRequest(reader_id="")

    def test_examples_match_schema(self):
        for item in examples.RECOMMEND_REQUESTS.values():
            A.RecommendRequest(**item["value"])
        for item in examples.RECOMMEND_RESPONSES.values():
            A.RecommendResponse(**item["value"])
        for item in examples.RECOMMEND_BAD_REQUESTS.values():
            A.BadRequest(**item["value"])


class Export(unittest.TestCase):
    def test_payload_is_utc(self):
        kst = datetime.fromisoformat("2026-11-26T09:00:00+09:00")
        out = to_payload([{"article_id": "news-1", "category": "정치", "subcategory": None, "published_at": kst}], kst)
        self.assertEqual(out["articles"][0]["published_at"], "2026-11-26T00:00:00+00:00")
        self.assertEqual(out["generated_at"], "2026-11-26T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
