"""build_feast_sources 표 이름 치환 검증 (BigQuery 없이). SQL과 Python 결과 비교는 PR 본문의 시험 참고."""
import importlib.util
import re
import sys
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "build_feast_sources", Path(__file__).parents[1] / "transform/build_feast_sources.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class BuildFeastSourcesTests(unittest.TestCase):
    def test_defaults_write_feast_names_into_feat(self):
        names = module.table_names("p", {})
        self.assertEqual(names["clean_actions"], "p.clean.actions")
        self.assertEqual(names["article_embedding"], "p.feat.article_embedding")
        self.assertEqual(names["out_reader_profile"], "p.feat.reader_profile")

    def test_accepts_project_dot_dataset_like_feast(self):
        names = module.table_names("p", {"FEAST_SOURCE_DATASET": "other.dev_yoonmo"})
        self.assertEqual(names["out_article_profile"], "other.dev_yoonmo.article_profile")

    def test_every_placeholder_is_replaced(self):
        sql = module.render(module.table_names("p", {}))
        self.assertNotRegex(sql, r"`\{[a-z_]+\}`")
        for name in module.OUTPUTS:   # Feast definitions.py가 읽는 이름
            self.assertRegex(sql, rf"CREATE OR REPLACE TABLE `p\.feat\.{name}`")


if __name__ == "__main__":
    unittest.main()
