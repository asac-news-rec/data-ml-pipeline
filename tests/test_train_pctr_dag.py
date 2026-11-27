"""train_pctr: champion 버전과 추론 서버 annotation을 비교해 필요할 때만 재시작한다."""
import importlib.util
import io
import json
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    from airflow.exceptions import AirflowException, AirflowSkipException
except ImportError as exc:
    raise unittest.SkipTest(f"airflow가 없다: {exc}")

spec = importlib.util.spec_from_file_location(
    "train_pctr", Path(__file__).resolve().parents[1] / "dags" / "train_pctr.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def deployment(annotation=None, model_uri="models:/pctr@champion", done=True):
    env = [SimpleNamespace(name="MODEL_URI", value=model_uri)] if model_uri else []
    annotations = {T.VERSION_ANNOTATION: annotation} if annotation else {}
    n = 1 if done else 0
    return SimpleNamespace(
        metadata=SimpleNamespace(generation=2),
        spec=SimpleNamespace(replicas=1, template=SimpleNamespace(
            metadata=SimpleNamespace(annotations=annotations),
            spec=SimpleNamespace(containers=[SimpleNamespace(env=env)]))),
        status=SimpleNamespace(observed_generation=2, updated_replicas=1, replicas=1, available_replicas=n))


class Restart(unittest.TestCase):
    def run_restart(self, champion, reads):
        apps = self.apps = MagicMock()
        apps.read_namespaced_deployment.side_effect = reads
        variables = {"mlflow_tracking_uri": "http://mlflow"}
        with patch.object(T, "champion_version", return_value=champion), \
             patch.object(T.Variable, "get", side_effect=lambda k, default_var=None: variables.get(k, default_var)), \
             patch("kubernetes.config.load_incluster_config"), \
             patch("kubernetes.client.AppsV1Api", return_value=apps), \
             patch.object(T.time, "sleep"):
            T.restart_inference_server()
        return apps

    def test_no_champion_skips(self):
        with self.assertRaises(AirflowSkipException):
            self.run_restart(None, [deployment()])

    def test_without_model_uri_skips_without_patch(self):
        with self.assertRaises(AirflowSkipException):
            self.run_restart("5", [deployment(model_uri=None)])
        self.apps.patch_namespaced_deployment.assert_not_called()

    def test_same_version_and_ready_skips(self):
        with self.assertRaises(AirflowSkipException):
            self.run_restart("5", [deployment(annotation="5")])
        self.apps.patch_namespaced_deployment.assert_not_called()

    def test_new_champion_patches_both_annotations_and_waits(self):
        apps = self.run_restart("6", [deployment(annotation="5"), deployment(annotation="6", done=False),
                                      deployment(annotation="6")])
        body = apps.patch_namespaced_deployment.call_args.args[2]
        annotations = body["spec"]["template"]["metadata"]["annotations"]
        self.assertEqual(annotations[T.VERSION_ANNOTATION], "6")
        self.assertIn("kubectl.kubernetes.io/restartedAt", annotations)

    def test_missed_restart_is_recovered_next_run(self):
        """alias를 옮긴 직후 Pod가 죽어 재시작이 빠졌다: 다음 실행에서 서빙 v5 ≠ champion v6이라 재시작한다."""
        apps = self.run_restart("6", [deployment(annotation="5"), deployment(annotation="6")])
        apps.patch_namespaced_deployment.assert_called_once()

    def test_unfinished_rollout_waits_without_new_patch(self):
        """앞 실행의 롤아웃이 안 끝났다: 다시 patch하지 않고 기다린다. 시간 안에 안 끝나면 실패한다."""
        stuck = deployment(annotation="6", done=False)
        with patch.object(T, "ROLLOUT_TIMEOUT_S", 0), self.assertRaises(AirflowException):
            self.run_restart("6", [stuck, stuck])
        self.apps.patch_namespaced_deployment.assert_not_called()


class Dag(unittest.TestCase):
    def test_restart_runs_after_any_train_result(self):
        self.assertEqual(T.dag.get_task("restart_inference_server").trigger_rule, "all_done")

    def test_train_pod_waits_for_worker_node(self):
        train = T.dag.get_task("train_pctr")
        self.assertEqual(train.startup_timeout_seconds, 600)
        self.assertEqual(train.execution_timeout.total_seconds(), 3600)

    def test_failed_training_fails_the_run(self):
        """restart(all_done)만 leaf면 학습이 실패해도 run이 success다. train_result leaf가 실패를 남긴다."""
        result = T.dag.get_task("train_result")
        self.assertEqual(result.upstream_task_ids, {"train_pctr"})
        self.assertEqual(result.trigger_rule, "all_success")   # train 실패 → upstream_failed, skip → skip
        self.assertEqual({t.task_id for t in T.dag.leaves}, {"restart_inference_server", "train_result"})


def http_error(code, error_code=None):
    body = json.dumps({"error_code": error_code}).encode() if error_code else b"<html>"
    return urllib.error.HTTPError("http://mlflow/x", code, "err", {}, io.BytesIO(body))


class ChampionVersion(unittest.TestCase):
    def call(self, outcome):
        with patch.object(T.urllib.request, "urlopen") as urlopen:
            if isinstance(outcome, Exception):
                urlopen.side_effect = outcome
            else:
                urlopen.return_value.__enter__.return_value = io.BytesIO(json.dumps(outcome).encode())
            return T.champion_version("http://mlflow/")

    def test_200_returns_version(self):
        self.assertEqual(self.call({"model_version": {"version": "7"}}), "7")

    def test_404_no_model_is_none(self):
        self.assertIsNone(self.call(http_error(404, "RESOURCE_DOES_NOT_EXIST")))

    def test_400_no_alias_is_none(self):
        """모델은 등록됐는데 alias만 없다(첫 학습이 min_auc에 못 미침): champion 없음으로 skip한다."""
        self.assertIsNone(self.call(http_error(400, "INVALID_PARAMETER_VALUE")))

    def test_other_400_and_500_raise(self):
        for err in (http_error(400, "BAD_REQUEST"), http_error(500, "INTERNAL_ERROR"), http_error(502)):
            with self.assertRaises(urllib.error.HTTPError):
                self.call(err)
