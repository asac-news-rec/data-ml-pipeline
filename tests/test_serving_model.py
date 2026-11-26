"""추론 서버 모델 로드: gs:// 경로와 feature 이름 검사."""
import os
import unittest
from unittest.mock import MagicMock, patch

try:
    import lightgbm  # noqa: F401
    from serving import model as M
    from serving.features import feature_names
except ImportError as exc:
    raise unittest.SkipTest(f"serving 묶음이 없다: {exc}")


class ModelLoad(unittest.TestCase):
    def load(self, names):
        booster = MagicMock()
        booster.feature_name.return_value = names
        with patch.dict(os.environ, {"MODEL_PATH": "gs://b/models/pctr/lgbm.txt", "MODEL_URI": ""}), \
                patch.object(M, "read_gcs", return_value=("tree", 7)) as read, \
                patch("lightgbm.Booster", return_value=booster):
            model = M.Model()
            model.load()
        read.assert_called_once_with("gs://b/models/pctr/lgbm.txt")
        return model

    def test_gcs_version_is_generation(self):
        self.assertEqual(self.load(feature_names()).version, "lgbm@gcs-7")

    def test_wrong_feature_order_fails(self):
        with self.assertRaises(RuntimeError):
            self.load(list(reversed(feature_names())))

    def load_mlflow(self, loaded, names):
        import sys
        import types
        loaded_booster = loaded if isinstance(loaded, lightgbm.Booster) else loaded.booster_
        loaded_booster.feature_name.return_value = names
        mlflow = types.ModuleType("mlflow")
        mlflow.set_tracking_uri = lambda uri: None
        mlflow.lightgbm = types.ModuleType("mlflow.lightgbm")
        mlflow.lightgbm.load_model = lambda uri: loaded
        mlflow.MlflowClient = MagicMock()
        mlflow.MlflowClient.return_value.get_model_version_by_alias.return_value.version = "7"
        env = {"MODEL_URI": "models:/pctr@champion", "MLFLOW_TRACKING_URI": "http://mlflow"}
        with patch.dict(os.environ, env), patch.dict(sys.modules, {"mlflow": mlflow, "mlflow.lightgbm": mlflow.lightgbm}):
            model = M.Model()
            model.load()
        return model

    def test_mlflow_wrong_feature_order_fails(self):
        with self.assertRaises(RuntimeError):
            self.load_mlflow(MagicMock(spec=lightgbm.Booster), list(reversed(feature_names())))

    def test_mlflow_sklearn_model_uses_booster(self):
        loaded = MagicMock()
        model = self.load_mlflow(loaded, feature_names())
        self.assertIs(model.model, loaded.booster_)
        self.assertEqual(model.version, "pctr@7")


if __name__ == "__main__":
    unittest.main()
