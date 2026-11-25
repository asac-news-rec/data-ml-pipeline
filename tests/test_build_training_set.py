"""학습 데이터: 시간 범위, 파티션 날짜, split, 쿼리의 시점 제한."""
import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import build_training_set as B


class Window(unittest.TestCase):
    def test_window(self):
        w = B.window(datetime(2026, 11, 25, 17, 30, tzinfo=timezone.utc), 14, 24, 3)
        self.assertEqual(w["stop"], datetime(2026, 11, 25, 14, 30, tzinfo=timezone.utc))
        self.assertEqual(w["valid_from"], datetime(2026, 11, 24, 14, 30, tzinfo=timezone.utc))
        self.assertEqual((w["first_date"], w["last_date"], w["action_last_date"]),
                         (date(2026, 11, 11), date(2026, 11, 25), date(2026, 11, 26)))

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            B.window(datetime(2026, 11, 25), 14, 24, 3)
        with self.assertRaises(ValueError):
            B.window(datetime(2026, 11, 25, tzinfo=timezone.utc), 1, 24, 3)

    def test_query_is_bounded(self):
        for text in ("served_date BETWEEN", "event_date BETWEEN", "event_time < @action_end",
                     "@include_simulator OR COALESCE(s.model_version, '') != 'simulator'"):
            self.assertIn(text, B.QUERY)
        self.assertNotIn("SELECT *", B.QUERY)

    def test_entity_frame(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))
        rows = pd.DataFrame({"event_timestamp": ["2026-11-25T00:00:00.500Z", "2026-11-25T01:00:00Z"], "label": [0, 1]})
        df = B.entity_frame(rows.assign(event_timestamp=pd.to_datetime(rows.event_timestamp, format="ISO8601")),
                            datetime(2026, 11, 25, 1, tzinfo=timezone.utc))
        self.assertEqual(df["split"].tolist(), ["train", "valid"])
        self.assertAlmostEqual(df["request_ts"][0], 1795564800.5)
        self.assertTrue(df["row_id"].is_unique)


if __name__ == "__main__":
    unittest.main()
