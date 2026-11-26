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
