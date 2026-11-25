"""feast_materialize: feast_bq_temp_dataset 형식."""
import importlib.util
import unittest
from pathlib import Path

try:
    from airflow.exceptions import AirflowException
except ImportError as exc:
    raise unittest.SkipTest(f"airflow가 없다: {exc}")

spec = importlib.util.spec_from_file_location(
    "materialize_feast", Path(__file__).resolve().parents[1] / "dags" / "materialize_feast.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class DatasetName(unittest.TestCase):
    def test_accepts_both_forms(self):
        for value in ("dev_yoonmo", "proj-1.dev_yoonmo", " dev_yoonmo "):
            self.assertEqual(M.dataset_name(value, "proj-1"), "dev_yoonmo")

    def test_rejects_bad_values(self):
        for value in ("other.dev_yoonmo", "proj-1.", ".dev", "a.b.c", "", "dev yoonmo"):
            with self.assertRaises(AirflowException, msg=value):
                M.dataset_name(value, "proj-1")

    def test_pod_env_renders_dataset_only(self):
        pod = next(t for t in M.dag.tasks if "FEAST_BQ_TEMP_DATASET" in str(getattr(t, "env_vars", "")))
        env = pod.env_vars if isinstance(pod.env_vars, dict) else {e.name: e.value for e in pod.env_vars}
        macros = M.dag.user_defined_macros
        out = M.dag.get_template_env().from_string(env["FEAST_BQ_TEMP_DATASET"]).render(
            var={"value": {"feast_bq_temp_dataset": "proj-1.dev_yoonmo", "gcp_project_id": "proj-1"}}, **macros)
        self.assertEqual(out, "dev_yoonmo")


if __name__ == "__main__":
    unittest.main()
