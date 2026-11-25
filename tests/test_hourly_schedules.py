"""매시 DAG가 같은 정시 구간(logical_date)으로 도는지 확인한다."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import airflow.timetables.interval as interval
    import pendulum
    from airflow.timetables.base import DataInterval, TimeRestriction
except ImportError as exc:
    raise unittest.SkipTest(f"airflow가 없다: {exc}")

DAGS = Path(__file__).resolve().parents[1] / "dags"
HOURLY = ["load_raw", "clean_events", "embed_articles", "build_feast_sources", "materialize_feast"]


def load(name):
    spec = importlib.util.spec_from_file_location(name, DAGS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dag


class HourlySchedules(unittest.TestCase):
    def next_interval(self, dag, last):
        now = pendulum.datetime(2026, 11, 25, 2, 30)
        limit = TimeRestriction(earliest=pendulum.datetime(2026, 11, 1), latest=None, catchup=False)
        with patch.object(interval, "utcnow", lambda: now):
            info = dag.timetable.next_dagrun_info(last_automated_data_interval=last, restriction=limit)
        return info.data_interval

    def test_first_run_is_previous_full_hour(self):
        for name in HOURLY:
            got = self.next_interval(load(name), None)
            self.assertEqual(got.start, pendulum.datetime(2026, 11, 25, 1), name)
            self.assertEqual(got.end, pendulum.datetime(2026, 11, 25, 2), name)

    def test_returns_to_full_hour_after_off_hour_run(self):
        last = DataInterval.exact(pendulum.datetime(2026, 11, 25, 2, 3, 1))
        for name in HOURLY:
            got = self.next_interval(load(name), last)
            self.assertEqual(got.start, pendulum.datetime(2026, 11, 25, 2), name)
            self.assertEqual(got.end, pendulum.datetime(2026, 11, 25, 3), name)


if __name__ == "__main__":
    unittest.main()
