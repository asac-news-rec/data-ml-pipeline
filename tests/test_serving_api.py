"""추론 서버 응답: 오류 문구, 중복 후보, top_k, OpenAPI 예시가 스키마에 맞는지."""
import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch

try:
    import numpy as np
    import pandas as pd
    from fastapi import HTTPException
    from pydantic import ValidationError
    from serving import app as A
    from serving import examples
except ImportError as exc:
    raise unittest.SkipTest(f"serving 묶음이 없다: {exc}")

IDS = [f"news-{i:032x}" for i in range(3)]


def request(**changes) -> A.RankRequest:
    body = {"request_id": "r", "reader_id": "u", "candidates": IDS,
            "context": {"request_time": "2026-11-25T10:00:00+09:00"}}
    return A.RankRequest(**{**body, **changes})


class RankApi(unittest.TestCase):
    def setUp(self):
        A.app.state.ready = True
        A.app.state.store = MagicMock()
        A.app.state.model = MagicMock(version="v")

    def test_duplicate_and_top_k_rejected(self):
        for changes in ({"candidates": [IDS[0], IDS[0]]}, {"top_k": 0}, {"candidates": []}):
            with self.assertRaises(ValidationError):
                request(**changes)
        response = A.rank(request(top_k=4))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.body)["field"], "body.top_k")

    def test_errors_do_not_leak_details(self):
        with patch.object(A, "fetch_online", side_effect=RuntimeError("valkey.secret:6379")), \
                self.assertLogs("uvicorn.error", "ERROR") as logs, self.assertRaises(HTTPException) as caught:
            A.rank(request())
        self.assertEqual((caught.exception.status_code, caught.exception.detail), (503, "inference failed"))
        self.assertIn("valkey.secret", "".join(logs.output))

    def test_startup_failure_is_fixed_message(self):
        async def start():
            async with A.lifespan(A.app):
                with self.assertRaises(HTTPException) as caught:
                    A.readyz()
                self.assertEqual(caught.exception.detail, "not ready")
        with patch.object(A, "load_store", side_effect=RuntimeError("gs://secret")), \
                self.assertLogs("uvicorn.error", "ERROR"):
            asyncio.run(start())

    def test_rank_order_and_defaulted(self):
        online = pd.DataFrame({"u_emb": [None] * 3, "a_emb": [[0.0] * 768, [0.0] * 768, None]})
        A.app.state.model.predict.return_value = np.array([0.1, 0.3, 0.2])
        with patch.object(A, "fetch_online", return_value=online), patch.object(A, "to_matrix"):
            out = A.rank(request(candidates=IDS + ["wrong"]))
        self.assertEqual([i.article_id for i in out.ranked], [IDS[1], IDS[2], IDS[0]])
        self.assertEqual({i.article_id: i.reason for i in out.defaulted}[IDS[2]], "ARTICLE_FEATURES_NOT_FOUND")
        self.assertEqual([i.reason for i in out.unscored], ["UNKNOWN_ARTICLE"])

    def test_bad_scores_are_fixed_503(self):
        bad = ([0.1, float("nan"), 0.2], [0.1, float("inf"), 0.2], [0.1, 1.2, 0.2], [-0.1, 0.1, 0.2],
               [0.1, 0.2], [0.1, 0.2, 0.3, 0.4], [[0.1], [0.2], [0.3]], None)
        for scores in bad:
            A.app.state.model.predict.return_value = scores if scores is None else np.array(scores)
            with self.subTest(scores=scores), patch.object(A, "fetch_online"), patch.object(A, "to_matrix"), \
                    self.assertLogs("uvicorn.error", "ERROR"), self.assertRaises(HTTPException) as caught:
                A.rank(request())
            self.assertEqual((caught.exception.status_code, caught.exception.detail), (503, "inference failed"))

    def test_boundary_scores_are_allowed(self):
        online = pd.DataFrame({"u_emb": [None] * 3, "a_emb": [None] * 3})
        A.app.state.model.predict.return_value = np.array([0.0, 1.0, 0.5])
        with patch.object(A, "fetch_online", return_value=online), patch.object(A, "to_matrix"):
            self.assertEqual([i.pctr for i in A.rank(request()).ranked], [1.0, 0.5, 0.0])

    def test_openapi_examples_match_schema(self):
        for item in examples.REQUESTS.values():
            A.RankRequest(**item["value"])
        for item in examples.RESPONSES.values():
            A.RankResponse(**item["value"])
        for item in examples.BAD_REQUESTS.values():
            A.BadRequest(**item["value"])
        for item in examples.UNAVAILABLE.values():
            A.Unavailable(**item["value"])


if __name__ == "__main__":
    unittest.main()
