"""clean_events 처리 범위와 SQL 표 이름 치환의 회귀 검증 (BigQuery 없이)."""
import importlib.util
import re
import sys
import unittest
from datetime import date, datetime
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "clean_events", Path(__file__).parents[1] / "transform/clean_events.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class WindowTests(unittest.TestCase):
    def test_same_range_as_load_raw(self):
        values = module.window(datetime.fromisoformat("2026-11-24T12:00:00+09:00"), 3)
        self.assertEqual(values["start"], "2026-11-24T00:00:00.000Z")
        self.assertEqual(values["end"], "2026-11-24T04:00:00.000Z")
        self.assertEqual(values["load_dates"], [date(2026, 11, 24)])
        self.assertEqual(values["min_date"], date(2026, 11, 22))

    def test_window_crossing_kst_midnight_reads_two_partitions(self):
        values = module.window(datetime.fromisoformat("2026-11-24T16:00:00+00:00"), 3)
        self.assertEqual(values["load_dates"], [date(2026, 11, 24), date(2026, 11, 25)])

    def test_naive_time_is_rejected(self):
        with self.assertRaises(ValueError):
            module.window(datetime(2026, 11, 24, 3), 3)

    def test_start_matches_gcs_interval_format(self):
        start = module.window(datetime.fromisoformat("2026-11-24T03:30:00+00:00"), 0)["start"]
        self.assertRegex(start, r"^\d{4}-\d{2}-\d{2}T\d{2}:00:00\.000Z$")


class RenderTests(unittest.TestCase):
    def test_all_tables_replaced_and_prefix_only_on_clean(self):
        sql = module.render("p", "dev", "dev", "clean_")
        self.assertNotRegex(sql, r"`\{[a-z_]+\}`")
        self.assertIn("`p.dev.served`", sql)
        self.assertIn("`p.dev.clean_served`", sql)
        self.assertIn("`p.dev.clean_rejected_actions`", sql)

    def test_every_clean_read_has_partition_filter(self):
        sql = module.render("p", "raw", "clean", "")
        for match in re.finditer(r"FROM `p\.clean\.(\w+)`([^;]*?)(?:WHERE|\))", sql):
            tail = sql[match.end():match.end() + 120]
            self.assertRegex(match.group(0) + tail, r"(served_date|event_date) >= @min_date",
                             f"partition filter 없음: {match.group(1)}")


if __name__ == "__main__":
    unittest.main()
