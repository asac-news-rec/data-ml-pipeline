"""LightGBM 모델 로드. MLflow models:/pctr@champion, GCS 파일(gs://), 로컬 파일 중 하나."""
import os
from pathlib import Path

import numpy as np

from .features import feature_names


def read_gcs(uri: str) -> tuple[str, int]:
    """gs://버킷/객체의 내용과 generation을 한 번의 읽기로 얻는다."""
    from google.cloud import storage

    bucket, _, name = uri.removeprefix("gs://").partition("/")
    blob = storage.Client().bucket(bucket).blob(name)
    text = blob.download_as_text()
    return text, blob.generation


class Model:
    def __init__(self) -> None:
        self.uri = os.environ.get("MODEL_URI", "").strip()
        self.source = os.environ.get("MODEL_PATH", "models/lgbm.txt").strip()
        self.model = None
        self.version = "unknown"

    def load(self) -> None:
        import lightgbm as lgb

        if self.uri:
            import mlflow
            import mlflow.lightgbm
            from mlflow import MlflowClient

            tracking = os.environ.get("MLFLOW_TRACKING_URI")
            if not tracking:
                raise RuntimeError("MODEL_URI를 쓰려면 MLFLOW_TRACKING_URI가 필요하다")
            mlflow.set_tracking_uri(tracking)
            loaded = mlflow.lightgbm.load_model(self.uri)
            # sklearn 방식으로 등록한 모델은 predict가 확률이 아니라 0/1이라 안의 Booster를 쓴다.
            self.model = loaded if isinstance(loaded, lgb.Booster) else loaded.booster_
            self.version = self._mlflow_version(MlflowClient())
        elif self.source.startswith("gs://"):
            # MLflow 서버가 뜨기 전의 임시 경로. 버전은 GCS 객체 generation이다(덮어쓰면 바뀐다).
            text, generation = read_gcs(self.source)
            self.model = lgb.Booster(model_str=text)
            self.version = f"{Path(self.source).stem}@gcs-{generation}"
        else:
            path = Path(self.source)
            if not path.exists():
                raise FileNotFoundError(f"모델 파일이 없다: {path}")
            self.model = lgb.Booster(model_file=str(path))
            self.version = f"local@{int(path.stat().st_mtime)}"

        # 열 순서가 다르면 점수가 조용히 틀린다. 시작할 때 막는다.
        if self.model.feature_name() != feature_names():
            raise RuntimeError("모델 feature 이름·순서가 serving/features.py와 다르다")

    def _mlflow_version(self, client) -> str:
        text = self.uri.removeprefix("models:/")
        if "@" in text:
            name, alias = text.split("@", 1)
            mv = client.get_model_version_by_alias(name, alias)
            return f"{name}@{mv.version}"
        return self.uri

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return self.model.predict(matrix)


def check_scores(scores, count: int) -> np.ndarray:
    """예측이 후보마다 하나씩 있고 모두 0~1인지 본다. 아니면 ValueError(/v1/rank는 503으로 바꾼다)."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (count,):
        raise ValueError(f"예측 모양이 {scores.shape}이다. 후보는 {count}개다")
    # NaN은 비교가 항상 거짓이라 여기서 같이 걸린다.
    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("예측에 NaN이거나 0~1 밖인 값이 있다")
    return scores
