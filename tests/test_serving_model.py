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


if __name__ == "__main__":
    unittest.main()
