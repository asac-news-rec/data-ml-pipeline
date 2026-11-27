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

    def test_valid_from_empty_keeps_everything(self):
        for value in (None, "", "  "):
            self.assertEqual(module.valid_from(value).year, 1970)

    def test_valid_from_is_converted_to_utc(self):
        when = module.valid_from("2026-11-26T14:00:00+09:00")
        self.assertEqual(when.isoformat(), "2026-11-26T05:00:00+00:00")

    def test_valid_from_rejects_bad_or_naive_values(self):
        for value in ("어제", "2026-11-26 14:00", "2026-11-26T14:00:00"):
            with self.assertRaises(SystemExit):
                module.valid_from(value)

    def test_both_event_kinds_use_valid_from(self):
        """노출과 클릭 둘 다 거른다. 한쪽만 거르면 클릭은 있는데 노출이 없는 독자가 생긴다."""
        sql = module.render(module.table_names("p", {}))
        self.assertEqual(len(re.findall(r"event_time >= @data_valid_from", sql)), 2)


if __name__ == "__main__":
    unittest.main()
