"""load_raw 적재 규칙의 회귀 검증 (GCS·BigQuery는 가짜 client)."""
import gzip
import importlib.util
import json
import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "load_raw", Path(__file__).parents[1] / "ingest/load_raw.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

SERVED, ACTIONS = module.STREAMS
BUCKET = "logs"


def obj(stream, start, end="2026-11-24T03:05:00.000Z", job="j1"):
    return f"behavior/{stream}/interval_start={start}/interval_end={end}/part-{job}.jsonl.gz"


class FakeStorage:
    def __init__(self, objects):
        self.objects = objects  # 이름 → 내용(str)

    def list_blobs(self, bucket, prefix):
        return [SimpleNamespace(name=name) for name in self.objects if name.startswith(prefix)]

    def bucket(self, name):
        objects = self.objects
        return SimpleNamespace(blob=lambda key: SimpleNamespace(
            download_as_bytes=lambda: gzip.compress(objects[key].encode("utf-8"))))


class FakeBigQuery:
    def __init__(self, loaded=()):
        self.loaded = set(loaded)
        self.loads = []

    def query(self, sql, job_config):
        uris = next(p.values for p in job_config.query_parameters if p.name == "uris")
        rows = [SimpleNamespace(source_file=u) for u in uris if u in self.loaded]
        return SimpleNamespace(result=lambda: rows)

    def load_table_from_file(self, stream, table, job_config):
        rows = [json.loads(line) for line in stream.read().decode("utf-8").splitlines()]
        self.loads.append((table, rows, job_config))
        self.loaded |= {row["source_file"] for row in rows}
        return SimpleNamespace(result=lambda: None)


def run(bq, storage, stream=SERVED, hour="2026-11-24T03:00:00+00:00", lookback=1):
    hours = module.hour_prefixes(datetime.fromisoformat(hour), lookback)
    return module.load_stream(bq, storage, project="p", dataset="raw", bucket=BUCKET,
                              prefix="behavior", stream=stream, hours=hours,
                              now=datetime(2026, 11, 24, 4, tzinfo=timezone.utc))


SERVED_LINE = json.dumps({"impression_id": "imp-1", "request_id": "req-1", "reader_id": "r1",
                          "article_id": "a1", "position": 1,
                          "served_at": "2026-11-24T03:01:00.000Z", "new_field": "x"})


class HourPrefixTests(unittest.TestCase):
    def test_kst_hour_becomes_utc_prefixes_oldest_first(self):
        hours = module.hour_prefixes(datetime.fromisoformat("2026-11-24T12:00:00+09:00"), 2)
        self.assertEqual(hours, ["2026-11-24T01", "2026-11-24T02", "2026-11-24T03"])

    def test_naive_time_is_rejected(self):
        with self.assertRaises(ValueError):
            module.hour_prefixes(datetime(2026, 11, 24, 3), 0)

    def test_load_date_is_kst_date_of_interval_start(self):
        uri = "gs://logs/" + obj("served", "2026-11-24T15:00:00.000Z")
        self.assertEqual(module.load_date_of(uri), date(2026, 11, 25))


class LoadStreamTests(unittest.TestCase):
    def test_loads_new_file_and_keeps_whole_line(self):
        storage = FakeStorage({obj("served", "2026-11-24T03:00:00.000Z"): SERVED_LINE + "\n"})
        bq = FakeBigQuery()
        self.assertEqual(run(bq, storage), 1)
        table, rows, _ = bq.loads[0]
        self.assertEqual(table, "p.raw.served")
        self.assertEqual(rows[0]["impression_id"], "imp-1")
        self.assertEqual(rows[0]["payload"]["new_field"], "x")
        self.assertEqual(rows[0]["load_date"], "2026-11-24")

    def test_second_run_of_same_hour_adds_nothing(self):
        storage = FakeStorage({obj("served", "2026-11-24T03:00:00.000Z"): SERVED_LINE + "\n"})
        bq = FakeBigQuery()
        run(bq, storage)
        self.assertEqual(run(bq, storage), 0)
        self.assertEqual(len(bq.loads), 1)

    def test_late_file_in_lookback_hour_is_loaded_once(self):
        storage = FakeStorage({obj("served", "2026-11-24T03:00:00.000Z"): SERVED_LINE + "\n"})
        bq = FakeBigQuery()
        run(bq, storage)
        late = obj("served", "2026-11-24T02:55:00.000Z", "2026-11-24T03:00:00.000Z", "j2")
        storage.objects[late] = SERVED_LINE.replace("imp-1", "imp-0") + "\n"
        self.assertEqual(run(bq, storage), 1)
        self.assertEqual(bq.loads[1][1][0]["source_file"], f"gs://{BUCKET}/{late}")

    def test_other_hours_and_manifests_are_ignored(self):
        storage = FakeStorage({
            obj("served", "2026-11-24T05:00:00.000Z"): SERVED_LINE,
            "behavior/manifests/j1.json": "{}",
            "behavior/served/interval_start=2026-11-24T03:00:00.000Z/old.jsonl": SERVED_LINE,
        })
        self.assertEqual(run(FakeBigQuery(), storage), 0)

    def test_actions_keys_and_clustering(self):
        line = json.dumps({"event_id": "e1", "event_type": "article_open", "page_type": "list",
                           "event_time": "2026-11-24T03:02:00.000Z", "impression_id": "imp-1"})
        storage = FakeStorage({obj("actions", "2026-11-24T03:00:00.000Z"): line + "\n\n"})
        bq = FakeBigQuery()
        self.assertEqual(run(bq, storage, stream=ACTIONS), 1)
        _, rows, config = bq.loads[0]
        self.assertEqual((rows[0]["event_id"], rows[0]["event_type"]), ("e1", "article_open"))
        self.assertEqual(config.clustering_fields, ["event_type"])

    def test_broken_line_fails_whole_load(self):
        storage = FakeStorage({obj("served", "2026-11-24T03:00:00.000Z"): SERVED_LINE + "\n{bad\n"})
        bq = FakeBigQuery()
        with self.assertRaises(ValueError):
            run(bq, storage)
        self.assertEqual(bq.loads, [])


if __name__ == "__main__":
    unittest.main()
