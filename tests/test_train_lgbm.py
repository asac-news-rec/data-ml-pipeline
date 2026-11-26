"""학습: 시간 분할, 1:1 sampling, calibration, champion 교체 기준."""
import sys
import unittest
from pathlib import Path

try:
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    import sklearn  # noqa: F401
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
    import train_lgbm as T
except ImportError as exc:
    raise unittest.SkipTest(f"train 묶음이 없다: {exc}")


class Train(unittest.TestCase):
    def test_split_is_by_time_and_keeps_ties_together(self):
        when = pd.to_datetime([f"2026-11-01T00:{m:02d}:00Z" for m in [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9]])
        df = pd.DataFrame({"split": ["train"] * 10 + ["valid"], "event_timestamp": when})
        fit, calib, valid = T.split_calibration(df, 0.1)
        self.assertEqual((fit.sum(), calib.sum(), valid.sum()), (8, 2, 1))
        self.assertLess(when[fit].max(), when[calib].min())

    def test_negatives_are_one_to_one(self):
        y = np.array([1, 0, 0, 0, 1, 0, 0])
        chosen = T.sample_negatives(y, np.arange(6), seed=1)
        self.assertEqual((y[chosen] == 1).sum(), 2)
        self.assertEqual((y[chosen] == 0).sum(), 2)
        self.assertNotIn(6, chosen)

    def test_calibration_restores_base_rate_and_order(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20000, 3))
        y = (rng.random(20000) < 1 / (1 + np.exp(-(x[:, 0] - 3)))).astype(int)
        chosen = T.sample_negatives(y, np.arange(10000), seed=1)
        model = lgb.train({"objective": "binary", "verbose": -1, "seed": 1},
                          lgb.Dataset(x[chosen], label=y[chosen]), num_boost_round=20)
        test = x[10000:]
        final, a, b = T.calibrate(model, test, y[10000:])
        raw = model.predict(test, raw_score=True)
        np.testing.assert_allclose(final.predict(test), 1 / (1 + np.exp(-(a * raw + b))), atol=1e-6)
        self.assertGreater(model.predict(test).mean(), 2 * y[10000:].mean())
        self.assertAlmostEqual(final.predict(test).mean(), y[10000:].mean(), places=3)

    def test_promotion(self):
        good, bad = {"valid_auc": 0.8}, {"valid_auc": 0.7}
        self.assertTrue(T.should_promote(bad, None, 0, 7)[0])
        self.assertTrue(T.should_promote(good, bad, 0, 7)[0])
        self.assertFalse(T.should_promote(bad, good, 6.9, 7)[0])
        self.assertTrue(T.should_promote(bad, good, 7, 7, max_auc_drop=0.15)[0])

    def test_stale_champion_is_not_replaced_by_much_worse_model(self):
        champion, slight, broken = {"valid_auc": 0.72}, {"valid_auc": 0.71}, {"valid_auc": 0.60}
        self.assertTrue(T.should_promote(slight, champion, 8, 7)[0])      # 하락 0.01 <= 0.02
        self.assertFalse(T.should_promote(broken, champion, 8, 7)[0])     # 하락 0.12: 기한이 지나도 유지
        self.assertFalse(T.should_promote(broken, champion, 30, 7)[0])

    def test_min_auc_blocks_every_case(self):
        random_like = {"valid_auc": 0.52}
        self.assertFalse(T.should_promote(random_like, None, 0, 7)[0])           # 첫 모델도 막는다
        self.assertFalse(T.should_promote(random_like, {"valid_auc": 0.50}, 0, 7)[0])
        self.assertFalse(T.should_promote(random_like, {"valid_auc": 0.53}, 30, 7)[0])

    def test_promotion_boundaries(self):
        self.assertTrue(T.should_promote({"valid_auc": 0.55}, None, 0, 7)[0])               # 0.55 정확히는 통과
        self.assertFalse(T.should_promote({"valid_auc": 0.7}, {"valid_auc": 0.7}, 0, 7)[0])  # 같은 AUC면 유지
        self.assertFalse(T.should_promote({"valid_auc": float("nan")}, None, 0, 7)[0])      # NaN도 막는다

    def test_matrix_is_shared_with_serving(self):
        """학습과 추론 서버가 같은 입력 행렬 함수를 쓴다(복사본이 생기면 실패)."""
        from serving import matrix
        self.assertIs(T.to_matrix, matrix.to_matrix)
        self.assertIs(T.feature_names, matrix.feature_names)
        try:
            from serving import features
        except ImportError:
            return
        self.assertIs(features.to_matrix, matrix.to_matrix)
        self.assertIs(features.feature_names, matrix.feature_names)


if __name__ == "__main__":
    unittest.main()


class FakeMlflowException(Exception):
    def __init__(self, error_code):
        super().__init__(error_code)
        self.error_code = error_code


class LogMlflow(unittest.TestCase):
    """MlflowClient를 mock으로 바꿔 champion 조회 실패·alias 이동을 확인한다."""

    def run_log(self, champion=None, error=None, champion_auc=0.70, new_auc=0.75,
                champion_features=None, real_data=True):
        from unittest import mock

        client = mock.MagicMock()
        if error:
            client.get_model_version_by_alias.side_effect = FakeMlflowException(error)
        else:
            client.get_model_version_by_alias.return_value = champion
        loaded = mock.MagicMock(spec=["predict", "feature_name"])
        loaded.feature_name.return_value = champion_features or T.feature_names()
        mlflow = mock.MagicMock()
        mlflow.MlflowClient.return_value = client
        mlflow.lightgbm.load_model.return_value = loaded
        mlflow.lightgbm.log_model.return_value.registered_model_version = 5
        exceptions = mock.MagicMock(MlflowException=FakeMlflowException)
        modules = {"mlflow": mlflow, "mlflow.lightgbm": mlflow.lightgbm, "mlflow.exceptions": exceptions}
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "http://mlflow"}), \
                mock.patch.object(T, "evaluate", return_value={"valid_auc": champion_auc}):
            promoted = T.log_mlflow(None, {}, {"valid_auc": new_auc}, None, None, 7, 0.55, 0.02,
                                    real_data=real_data)
        return promoted, client, mlflow

    def champion(self):
        import time
        from unittest import mock
        return mock.MagicMock(version="3", creation_timestamp=time.time() * 1000)

    def test_missing_champion_promotes(self):
        promoted, client, _ = self.run_log(error="RESOURCE_DOES_NOT_EXIST")
        self.assertTrue(promoted)
        client.set_registered_model_alias.assert_called_once_with("pctr", "champion", "5")

    def test_server_error_is_not_treated_as_missing_champion(self):
        with self.assertRaises(FakeMlflowException):
            self.run_log(error="INTERNAL_ERROR")

    def test_alias_stays_when_not_promoted(self):
        promoted, client, mlflow = self.run_log(champion=self.champion(), champion_auc=0.80, new_auc=0.75)
        self.assertFalse(promoted)
        client.set_registered_model_alias.assert_not_called()
        mlflow.lightgbm.log_model.assert_called_once()   # 등록은 한다

    def test_changed_features_skip_comparison(self):
        promoted, client, _ = self.run_log(champion=self.champion(), champion_auc=0.99,
                                           champion_features=["old_feature"])
        self.assertTrue(promoted)   # champion 점수(0.99)를 쓰지 않고 새 모델로 본다
        client.set_registered_model_alias.assert_called_once()

    def test_without_simulated_column_only_registers(self):
        promoted, client, mlflow = self.run_log(error="RESOURCE_DOES_NOT_EXIST", real_data=False)
        self.assertFalse(promoted)
        client.set_registered_model_alias.assert_not_called()
        mlflow.lightgbm.log_model.assert_called_once()


class MatrixRows(unittest.TestCase):
    def test_row_slices_match_full_matrix(self):
        from serving.matrix import HEAD, DIM
        rng = np.random.default_rng(0)
        df = pd.DataFrame({c: rng.random(6) for c in HEAD})
        df["u_emb"] = [rng.random(DIM).astype(np.float32) if i % 2 else None for i in range(6)]
        df["a_emb"] = [rng.random(DIM).astype(np.float32) for _ in range(6)]
        rows = np.array([1, 4, 5])
        np.testing.assert_array_equal(T.to_matrix(df.iloc[rows]), T.to_matrix(df)[rows])
