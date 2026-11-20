"""실제 DAG의 날짜·실행 ID 렌더링을 외부 서비스 없이 검증합니다."""

from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

import pendulum
import yaml


ROOT = Path(__file__).resolve().parents[1]
DAG_MODULE = runpy.run_path(str(ROOT / "dags/generate_news.py"))
DAG = DAG_MODULE["dag"]
TASK = DAG.get_task("generate_and_store")


class NewsGenerationDagTests(unittest.TestCase):
    def render(self, **extra):
        context = {
            "dag": DAG,
            "dag_run": SimpleNamespace(run_after=pendulum.datetime(2026, 11, 20, 16, tz="UTC")),
            "run_id": "manual__same-run",
            **extra,
        }
        return TASK.render_template(TASK.arguments, context)

    def argument(self, arguments, name):
        return arguments[arguments.index(name) + 1]

    def test_scheduled_interval_takes_precedence(self):
        arguments = self.render(data_interval_end=pendulum.datetime(2026, 11, 19, 16, tz="UTC"))
        self.assertEqual(self.argument(arguments, "--date"), "2026-11-20")

    def test_manual_null_interval_uses_run_after(self):
        arguments = self.render(data_interval_end=None)
        self.assertEqual(self.argument(arguments, "--date"), "2026-11-21")

    def test_manual_missing_interval_uses_run_after(self):
        arguments = self.render()
        self.assertEqual(self.argument(arguments, "--date"), "2026-11-21")

    def test_manual_korean_midnight_boundary(self):
        for hour, minute, expected in [(14, 59, "2026-11-20"), (15, 0, "2026-11-21")]:
            with self.subTest(hour=hour, minute=minute):
                arguments = self.render(dag_run=SimpleNamespace(
                    run_after=pendulum.datetime(2026, 11, 20, hour, minute, tz="UTC")
                ))
                self.assertEqual(self.argument(arguments, "--date"), expected)

    def test_same_run_retry_or_clear_keeps_date_and_id(self):
        first = self.render()
        repeated = self.render()
        self.assertEqual(self.argument(first, "--date"), self.argument(repeated, "--date"))
        self.assertEqual(self.argument(first, "--run-id"), "generate_news:manual__same-run")
        self.assertEqual(self.argument(first, "--run-id"), self.argument(repeated, "--run-id"))
        another_run = self.render(run_id="manual__another-run")
        self.assertNotEqual(self.argument(first, "--run-id"), self.argument(another_run, "--run-id"))

    def test_image_owned_by_operator_and_template_path_exists(self):
        self.assertEqual(TASK.image, DAG_MODULE["GENERATOR_IMAGE"])
        template = Path(TASK.pod_template_file)
        self.assertTrue(template.is_file())
        pod = yaml.safe_load(template.read_text())
        base = next(c for c in pod["spec"]["containers"] if c["name"] == "base")
        self.assertNotIn("image", base)
        self.assertEqual(pod["metadata"]["namespace"], TASK.namespace)
        built = TASK.build_pod_request_obj(dry_run=True)
        self.assertEqual(built.spec.containers[0].image, TASK.image)
        self.assertEqual(built.spec.service_account_name, "airflow-task")
        self.assertEqual(built.spec.volumes[0].persistent_volume_claim.claim_name, "news-output")
        self.assertEqual(built.spec.containers[0].volume_mounts[0].mount_path, "/mnt/news-output")


if __name__ == "__main__":
    unittest.main()
