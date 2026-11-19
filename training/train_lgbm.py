"""[파이프라인 구간: 학습 데이터 → 모델 학습·평가] Feast training set으로 PCTR baseline LightGBM을 학습한다.

입력:  features/feast/data/training_set.parquet (label, split 포함)
출력:  models/lgbm.txt (gitignore)
실행:  uv run --frozen --extra train python training/train_lgbm.py

MLflow는 선택이다. MLFLOW_TRACKING_URI가 있으면 실험 기록과 모델 등록을 한다.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HEAD = ["position", "article_age_h", "category", "hour", "hist_clicks", "hist_exposures",
        "hours_since_last_click", "already_opened", "already_seen"]
FILL_ZERO = ["hist_clicks", "hist_exposures", "already_opened", "already_seen"]
DIM = 768


def to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Feast 결과 → 일반 9 + u_emb 768 + a_emb 768 (cold u_emb = NaN)."""
    head = df[HEAD].astype("float64").copy()
    head[FILL_ZERO] = head[FILL_ZERO].fillna(0)

    def stack(col: str) -> np.ndarray:
        nan = np.full(DIM, np.nan, dtype=np.float32)
        return np.vstack([
            np.asarray(value, dtype=np.float32) if value is not None and len(value) else nan
            for value in df[col]
        ])

    return np.hstack([head.to_numpy(np.float32), stack("u_emb"), stack("a_emb")])


def feature_names() -> list[str]:
    return HEAD + [f"u_emb_{i}" for i in range(DIM)] + [f"a_emb_{i}" for i in range(DIM)]


def log_mlflow(model, params: dict, metrics: dict) -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return
    import mlflow
    import mlflow.lightgbm
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "pctr"))
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        name = os.environ.get("MODEL_NAME", "pctr")
        info = mlflow.lightgbm.log_model(model, "model", registered_model_name=name)
        version = getattr(info, "registered_model_version", None)
        if version:
            MlflowClient().set_registered_model_alias(name, "champion", str(version))
        print(f"MLflow 등록: {name} version={version}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-set", type=Path,
                        default=ROOT / "features/feast/data/training_set.parquet")
    parser.add_argument("--model-out", type=Path, default=ROOT / "models/lgbm.txt")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--feature-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.training_set)
    train = df["split"] == "train"
    valid = df["split"] == "valid"
    if not train.any() or not valid.any():
        raise SystemExit(f"split에 train/valid가 없다: {df['split'].value_counts().to_dict()}")

    x = to_matrix(df)
    y = df["label"].to_numpy(np.int32)
    print(f"rows {len(df):,} (train {int(train.sum()):,} / valid {int(valid.sum()):,}), "
          f"features {x.shape[1]:,}, label 1 {y.mean():.3%}")

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("lightgbm이 없다. `uv sync --frozen --extra train`로 설치한다.") from exc

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 100,
        "lambda_l2": 1.0,
        "seed": args.seed,
        "num_threads": 0,
        "verbose": -1,
    }
    names = feature_names()
    dtrain = lgb.Dataset(x[train], label=y[train], feature_name=names)
    dvalid = lgb.Dataset(x[valid], label=y[valid], feature_name=names, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=args.rounds,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(args.early_stopping, verbose=False),
                   lgb.log_evaluation(50)],
    )

    auc = float(model.best_score["valid_0"]["auc"])
    logloss = float(model.best_score["valid_0"]["binary_logloss"])
    print(f"best_iteration {model.best_iteration}, valid AUC {auc:.4f}, logloss {logloss:.4f}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.model_out))
    print(f"저장: {args.model_out}")
    log_mlflow(model, params, {"valid_auc": auc, "valid_logloss": logloss})


if __name__ == "__main__":
    main()
