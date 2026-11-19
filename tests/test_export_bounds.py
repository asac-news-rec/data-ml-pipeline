"""Airflow 실행 구간 입력 계약의 회귀 검증."""
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
import unittest

from airflow.exceptions import AirflowFailException

spec = importlib.util.spec_from_file_location(
    "behavior_export_dag", Path(__file__).parents[1] / "dags/export_behavior_logs.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExportBoundsTests(unittest.TestCase):
    def test_manual_offset_normalizes_to_utc(self):
        start, end = module.export_bounds({
            "interval_start": "2026-11-19T09:00:00+09:00",
            "interval_end": "2026-11-19T09:05:00+09:00",
        }, {})
        self.assertEqual(start, datetime(2026, 11, 19, tzinfo=timezone.utc))
        self.assertEqual((end - start).total_seconds(), 300)

    def test_scheduled_bounds(self):
        start = datetime(2026, 11, 19, tzinfo=timezone.utc)
        end = datetime(2026, 11, 19, 0, 5, tzinfo=timezone.utc)
        self.assertEqual(module.export_bounds({}, {
            "data_interval_start": start, "data_interval_end": end,
        }), (start, end))

    def test_invalid_input_fails_without_retry(self):
        valid = "2026-11-19T00:00:00Z"
        cases = [
            {"interval_start": valid}, {"interval_end": valid},
            {"interval_start": "bad", "interval_end": valid},
            {"interval_start": None, "interval_end": valid},
            {"interval_start": "2026-11-18T00:00:00", "interval_end": valid},
            {"interval_start": valid, "interval_end": valid},
            {"interval_start": "2026-11-20T00:00:00Z", "interval_end": valid},
            {"interval_start": "2026-11-18T00:00:00.000001Z", "interval_end": valid},
            {},
        ]
        for conf in cases:
            with self.subTest(conf=conf), self.assertRaises(AirflowFailException):
                module.export_bounds(conf, {})

    def test_pruning_requires_upload_success(self):
        dag = module.export_behavior_logs
        self.assertEqual(dag.get_task("prune_postgresql").upstream_task_ids, {"upload"})
        self.assertEqual(dag.get_task("prune_postgresql").trigger_rule, "all_success")


if __name__ == "__main__":
    unittest.main()
