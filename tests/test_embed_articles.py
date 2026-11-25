"""embed_articles의 입력 글·행 모양·새 기사 고르기 검증 (DB·BigQuery·모델 없이)."""
import importlib.util
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "embed_articles", Path(__file__).parents[1] / "ingest/embed_articles.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def article(i, **kw):
    row = {"id": f"a{i}", "title": "제목", "subtitle": None, "summary": "요약",
           "category": "정치", "subcategory": None, "tags": ["t1"], "publisher": "p1",
           "published_at": datetime(2026, 11, 24, i, tzinfo=timezone.utc)}
    row.update(kw)
    return row


class EmbedArticlesTests(unittest.TestCase):
    def test_passage_matches_local_embed_script(self):
        self.assertEqual(module.passage(article(1, subtitle="부제")), "passage: 제목 부제 요약")
        self.assertEqual(module.passage(article(1)), "passage: 제목 요약")

    def test_pick_new_skips_done_and_takes_latest(self):
        picked = module.pick_new([article(1), article(2), article(3)], {"a3"}, 1)
        self.assertEqual([a["id"] for a in picked], ["a2"])

    def test_meta_row_parses_jsonb_tags_text(self):
        row = module.meta_row(article(1, tags='["가", "나"]'), "2026-11-24T00:00:00+00:00")
        self.assertEqual(row["tags"], ["가", "나"])
        self.assertEqual(row["publisher_id"], "p1")
        self.assertIsNone(row["clickbait_score"])

    def test_embedding_rows_require_768(self):
        with self.assertRaises(ValueError):
            module.embedding_rows([article(1)], [[0.0] * 10], "m", "main", "t")
        rows = module.embedding_rows([article(1)], [[0.1] * 768], "m", "main", "t")
        self.assertEqual(len(rows[0]["embedding"]), 768)

    def test_embedder_loads_model_lazily(self):
        embed = module.e5_embedder("not-a-real-model", "main")   # 부르기 전에는 모델을 안 읽는다
        self.assertTrue(callable(embed))


if __name__ == "__main__":
    unittest.main()
