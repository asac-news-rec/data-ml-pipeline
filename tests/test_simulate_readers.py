"""ingest/simulate_readers.py: 줄 형식이 로깅 서버 검사를 통과하는지, 시간·멱등이 맞는지."""
import gzip
import importlib.util
import json
import random
import re
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "simulate_readers", Path(__file__).resolve().parents[1] / "ingest" / "simulate_readers.py")
sim = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sim
spec.loader.exec_module(sim)

START = datetime(2026, 11, 25, 0, 0, tzinfo=timezone.utc)   # 한국 9시: 테크관심이 많이 온다
CATS = ["경제", "증권", "테크·과학", "스포츠", "연예", "정치", "사회", "문화", "건강"]
ARTICLES = [{"article_id": f"news-{i:04d}", "category": CATS[i % len(CATS)],
             "published_at": START - timedelta(minutes=10 * i)} for i in range(-3, 60)]
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
TIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z")


class FakeBlob:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def upload_from_string(self, data, content_type, if_generation_match):
        from google.api_core.exceptions import PreconditionFailed
        assert if_generation_match == 0
        if self.name in self.store:
            raise PreconditionFailed("exists")
        self.store[self.name] = data

    def exists(self):
        return self.name in self.store

    def download_as_bytes(self, raw_download=False):
        return self.store[self.name]


class FakeBucket:
    name = "logs"

    def __init__(self):
        self.objects = {}

    def blob(self, name):
        return FakeBlob(self.objects, name)


class SimulateReadersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hour = sim.simulate_hour(sim.make_readers(200), ARTICLES, START, visit_scale=3.0)

    def test_lines_have_logging_server_keys_in_order(self):
        served, actions = self.hour
        assert served and actions
        assert list(served[0]) == ["request_id", "impression_id", "reader_id", "article_id", "position",
                                   "served_at", "score", "model_version", "features_as_of", "received_at"]
        for row in actions:
            head = ["event_id", "event_type", "event_time", "impression_id", "page_type"]
            tail = ["received_at", "source", "schema_version", "measurement_version"]
            extra = ["duration_ms", "ended_at"] if row["event_type"] == "exposure" else []
            assert list(row) == head + extra + tail
            assert row["source"] == "simulator"

    def test_values_pass_logging_server_rules(self):
        served, actions = self.hour
        for row in served:
            assert 1 <= row["position"] <= sim.PAGE_SIZE
            assert TIME_RE.fullmatch(row["served_at"]) and TIME_RE.fullmatch(row["received_at"])
        for row in actions:
            assert UUID_RE.fullmatch(row["event_id"])
            assert TIME_RE.fullmatch(row["event_time"]) and TIME_RE.fullmatch(row["received_at"])
            kind, page = row["event_type"], row["page_type"]
            assert kind in ("exposure", "article_open", "article_close", "like")
            if kind == "exposure":
                assert page == "list" and row["duration_ms"] >= sim.MIN_EXPOSURE_MS
                assert row["ended_at"] >= row["event_time"]
            assert kind != "article_open" or page == "list"
            assert kind != "article_close" or page == "detail"

    def test_received_at_inside_interval_and_after_event(self):
        served, actions = self.hour
        s, e = sim.utc_ms(START), sim.utc_ms(START + sim.HOUR)
        for row in served:
            assert s <= row["served_at"] <= row["received_at"] < e
        for row in actions:
            assert s <= row["event_time"] <= row["received_at"] < e

    def test_actions_follow_their_impression(self):
        """모든 행동은 served에 있는 impression이고, open은 첫 노출이 끝난 뒤 100ms 이상 지나서 온다.

        clean_events.sql은 첫 노출 0.1초 안의 article_open을 fraud_click으로 버린다.
        """
        served, actions = self.hour
        by_imp = {r["impression_id"]: r for r in served}
        first_exposure = {}
        for row in sorted(actions, key=lambda r: r["event_time"]):
            assert row["impression_id"] in by_imp
            assert row["event_time"] >= by_imp[row["impression_id"]]["served_at"]
            if row["event_type"] == "exposure":
                first_exposure.setdefault(row["impression_id"], row)
            if row["event_type"] == "article_open":
                shown = first_exposure[row["impression_id"]]
                gap = (datetime.fromisoformat(row["event_time"].replace("Z", "+00:00"))
                       - datetime.fromisoformat(shown["event_time"].replace("Z", "+00:00")))
                assert gap >= timedelta(milliseconds=100)
        assert any(r["event_type"] == "article_open" for r in actions)

    def test_feed_only_shows_published_articles(self):
        served, _ = self.hour
        published = {a["article_id"]: a["published_at"] for a in ARTICLES}
        for row in served:
            at = datetime.fromisoformat(row["served_at"].replace("Z", "+00:00"))
            assert published[row["article_id"]] <= at

    def test_same_hour_gives_same_lines(self):
        first = sim.simulate_hour(sim.make_readers(50), ARTICLES, START, 3.0)
        second = sim.simulate_hour(sim.make_readers(50), ARTICLES, START, 3.0)
        assert first == second
        other = sim.simulate_hour(sim.make_readers(50), ARTICLES, START + sim.HOUR, 3.0)
        assert other != first

    def test_object_names_match_load_raw_paths(self):
        names = sim.object_names("behavior/", START)
        assert names["served"].startswith(
            "behavior/served/interval_start=2026-11-25T00:00:00.000Z/interval_end=2026-11-25T01:00:00.000Z/part-simulator-")
        assert names["actions"].endswith(".jsonl.gz")
        assert sim.object_names("", START)["served"].startswith("served/interval_start=")
        assert names == sim.object_names("behavior", START)

    def test_to_gzip_is_compact_jsonl(self):
        served, _ = self.hour
        lines = gzip.decompress(sim.to_gzip(served[:2])).decode().splitlines()
        assert json.loads(lines[0]) == served[0]
        assert ", " not in lines[0] and ": " not in lines[0]

    def test_write_skips_existing_object(self):
        try:
            import google.api_core  # noqa: F401
        except ImportError:
            self.skipTest("google-api-core가 없다")
        served, _ = self.hour
        bucket = FakeBucket()
        assert sim.write(bucket, "served/x.jsonl.gz", served) is True
        assert sim.write(bucket, "served/x.jsonl.gz", served) is False
        assert sim.write(bucket, "served/y.jsonl.gz", []) is False
        assert list(bucket.objects) == ["served/x.jsonl.gz"]

    def test_close_never_before_open_near_interval_end(self):
        """방문이 구간 끝 무렵이어도 닫은 시각이 연 시각보다 앞서지 않는다."""
        end = START + sim.HOUR
        readers = sim.make_readers(200)
        checked = 0
        for seed in range(3000):
            _, actions = sim.visit(random.Random(seed), readers[seed % 200], ARTICLES,
                                   end - timedelta(minutes=2, seconds=15), end)
            opened = {a["impression_id"]: a["event_time"] for a in actions if a["event_type"] == "article_open"}
            closed = {a["impression_id"]: a["event_time"] for a in actions if a["event_type"] == "article_close"}
            self.assertEqual(opened.keys(), closed.keys())
            for imp, at in opened.items():
                checked += 1
                self.assertLessEqual(at, closed[imp])
        self.assertGreater(checked, 0)

    def test_retry_with_changed_articles_does_not_write_actions(self):
        try:
            import google.api_core  # noqa: F401
        except ImportError:
            self.skipTest("google-api-core가 없다")
        served, actions = self.hour
        names = sim.object_names("", START)
        bucket = FakeBucket()
        bucket.objects[names["served"]] = sim.to_gzip(served[:-1])
        with self.assertRaises(RuntimeError):
            sim.write_hour(bucket, names, served, actions)
        self.assertNotIn(names["actions"], bucket.objects)

    def test_retry_with_same_served_writes_actions(self):
        try:
            import google.api_core  # noqa: F401
        except ImportError:
            self.skipTest("google-api-core가 없다")
        served, actions = self.hour
        names = sim.object_names("", START)
        bucket = FakeBucket()
        bucket.objects[names["served"]] = sim.to_gzip(served)
        self.assertEqual(sim.write_hour(bucket, names, served, actions),
                         {"served": False, "actions": True})


class SeparationTest(unittest.TestCase):
    """가상 로그가 실제 데이터와 섞이지 않게 하는 장치."""

    def test_reader_ids_are_uuid5_in_sim_namespace(self):
        """가상 독자 ID는 uuid5(SIM_NAMESPACE)라 로깅 서버가 발급하는 실제 독자 ID와 겹치지 않는다."""
        import uuid
        ids = [r["reader_id"] for r in sim.make_readers(200)]
        self.assertEqual(len(set(ids)), 200)
        for i, reader_id in enumerate(ids):
            self.assertEqual(uuid.UUID(reader_id).version, 5)
            self.assertEqual(reader_id, str(uuid.uuid5(sim.SIM_NAMESPACE, f"reader-{i}")))

    def test_feast_sources_are_per_reader(self):
        """Feast 원천 표의 집계는 모두 reader_id별이라 가상 독자가 실제 독자 피처 값을 바꾸지 않는다."""
        sql = (Path(__file__).resolve().parents[1] / "transform" / "sql" / "build_feast_sources.sql").read_text()
        # 행동 로그를 쓰지 않는 기사 임베딩 중복 제거(article_vec)만 article_id별이다
        sql = sql.replace("PARTITION BY article_id ORDER BY created_at DESC", "")
        windows = re.findall(r"PARTITION BY\s+(?:\w+\.)?(\w+)", sql)
        self.assertGreater(len(windows), 5)
        self.assertEqual(set(windows), {"reader_id"})

    def test_dag_is_off_by_default(self):
        """Variable이 없으면 쓰지 않는다. 켤 때만 simulate_readers_enabled=true."""
        src = (Path(__file__).resolve().parents[1] / "dags" / "simulate_readers.py").read_text()
        self.assertIn('Variable.get("simulate_readers_enabled", default_var="false")', src)


if __name__ == "__main__":
    unittest.main()
