"""[파이프라인 구간: 학습 데이터 → 모델 학습·평가] Feast training set으로 PCTR baseline LightGBM을 학습한다.

입력:  features/feast/data/training_set.parquet (label, split 포함)
출력:  models/lgbm.txt (gitignore)
실행:  uv run --frozen --extra train python training/train_lgbm.py

순서: train을 시간순으로 fit/calib으로 나눈다 → fit의 negative를 1:1로 줄여 학습한다
→ 줄이지 않은 calib으로 확률을 보정한다 → 줄이지 않은 valid로 평가한다.
MLflow는 선택이다. MLFLOW_TRACKING_URI가 있으면 기록·등록하고, 이전 champion보다 낫거나
champion이 오래됐을 때만 champion alias를 옮긴다.
종료 코드: 0 = 끝(champion을 바꿨거나 MLflow 없음), 98 = champion 유지, 99 = 클릭 부족.
98·99는 Airflow가 skipped로 보고 추론 서버 재시작을 건너뛴다.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HEAD = ["position", "article_age_h", "category", "hour", "hist_clicks", "hist_exposures",
        "hours_since_last_click", "already_opened", "already_seen"]
FILL_ZERO = ["hist_clicks", "hist_exposures", "already_opened", "already_seen"]
DIM = 768
NOT_PROMOTED, NOT_ENOUGH_CLICKS = 98, 99   # dags/train_pctr.py의 skip_on_exit_code와 같아야 한다


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or len(value) == 0


def to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Feast 결과 → 일반 9 + u_emb 768 + a_emb 768 (cold u_emb = NaN)."""
    head = df[HEAD].astype("float64").copy()
    head[FILL_ZERO] = head[FILL_ZERO].fillna(0)

    def stack(col: str) -> np.ndarray:
        nan = np.full(DIM, np.nan, dtype=np.float32)
        return np.vstack([
            nan if is_missing(value) else np.asarray(value, dtype=np.float32)
            for value in df[col]
        ])

    return np.hstack([head.to_numpy(np.float32), stack("u_emb"), stack("a_emb")])


def feature_names() -> list[str]:
    return HEAD + [f"u_emb_{i}" for i in range(DIM)] + [f"a_emb_{i}" for i in range(DIM)]


def split_calibration(df: pd.DataFrame, fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """train의 가장 최근 fraction을 calib으로 뗀다. 같은 시각의 행은 한쪽에만 둔다."""
    train = (df["split"] == "train").to_numpy()
    when = pd.to_datetime(df["event_timestamp"], utc=True)
    cut = when[train].sort_values().iloc[int(train.sum() * (1 - fraction))]
    calib = train & (when >= cut).to_numpy()
    return train & ~calib, calib, (df["split"] == "valid").to_numpy()


def sample_negatives(y: np.ndarray, index: np.ndarray, seed: int) -> np.ndarray:
    """index 안에서 positive는 모두, negative는 같은 수만큼 뽑는다."""
    pos, neg = index[y[index] == 1], index[y[index] == 0]
    picked = np.random.default_rng(seed).choice(neg, size=min(len(pos), len(neg)), replace=False)
    return np.sort(np.concatenate([pos, picked]))


def calibrate(model, x: np.ndarray, y: np.ndarray):
    """Platt scaling p = sigmoid(a * raw + b)를 모델 안에 넣은 Booster와 (a, b)를 돌려준다.

    raw는 모든 트리의 leaf 값 합이다. leaf 값에 a를 곱하고 첫 트리의 모든 leaf에 b를 더하면
    predict()가 바로 보정된 확률을 낸다. 추론 서버 코드를 바꾸지 않아도 된다.
    """
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression

    best = lgb.Booster(model_str=model.model_to_string(num_iteration=model.best_iteration))
    raw = best.predict(x, raw_score=True).reshape(-1, 1)
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(raw, y)
    a, b = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    if a <= 0:
        raise SystemExit(f"calibration 기울기가 {a:.4f}다. 순위가 뒤집히므로 멈춘다.")
    for info in best.dump_model()["tree_info"]:
        for leaf in range(info["num_leaves"]):
            value = a * best.get_leaf_output(info["tree_index"], leaf)
            best.set_leaf_output(info["tree_index"], leaf, value + (b if info["tree_index"] == 0 else 0.0))
    return lgb.Booster(model_str=best.model_to_string()), a, b


def evaluate(model, x: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score

    p = model.predict(x)
    return {"valid_auc": float(roc_auc_score(y, p)),
            "valid_logloss": float(log_loss(y, p, labels=[0, 1])),
            "valid_mean_pctr": float(p.mean()), "valid_ctr": float(y.mean())}


def should_promote(new: dict, old: dict | None, age_days: float, max_age_days: float) -> tuple[bool, str]:
    if old is None:
        return True, "비교할 champion이 없다"
    if new["valid_auc"] > old["valid_auc"]:
        return True, f"AUC {old['valid_auc']:.4f} → {new['valid_auc']:.4f}"
    if age_days >= max_age_days:
        return True, f"champion이 {age_days:.1f}일 됐다(한도 {max_age_days}일)"
    return False, f"AUC {new['valid_auc']:.4f} <= champion {old['valid_auc']:.4f}, {age_days:.1f}일"


def log_mlflow(model, params: dict, metrics: dict, x_valid, y_valid, max_age_days: float) -> bool:
    """기록·등록하고 추론 서버가 새 모델을 읽어야 하면 True."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("MLFLOW_TRACKING_URI가 없어 파일만 저장했다")
        return True
    import mlflow
    import mlflow.lightgbm
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "pctr"))
    name, client = os.environ.get("MODEL_NAME", "pctr"), MlflowClient()
    old, age_days = None, 0.0
    try:
        champion = client.get_model_version_by_alias(name, "champion")
    except MlflowException as exc:
        # 서버 장애를 "champion 없음"으로 보면 비교 없이 교체된다. 없다는 응답만 넘긴다.
        if exc.error_code not in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"):
            raise
        champion = None
    if champion is not None:
        loaded = mlflow.lightgbm.load_model(f"models:/{name}/{champion.version}")
        old = evaluate(loaded, x_valid, y_valid)
        age_days = (time.time() - champion.creation_timestamp / 1000) / 86400
    promote, reason = should_promote(metrics, old, age_days, max_age_days)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if old:
            mlflow.log_metrics({f"champion_{k}": v for k, v in old.items()})
        mlflow.set_tags({"promoted": promote, "promote_reason": reason})
        info = mlflow.lightgbm.log_model(model, name="model", registered_model_name=name)
    version = str(info.registered_model_version)
    if promote:
        client.set_registered_model_alias(name, "champion", version)
    print(f"MLflow 등록: {name} version={version}, champion 교체={promote} ({reason})")
    return promote


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
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-fraction", type=float, default=0.1)
    parser.add_argument("--min-clicks", type=int, default=20, help="fit·calib·valid 각각 필요한 최소 클릭 수")
    parser.add_argument("--champion-max-age-days", type=float, default=7.0,
                        help="champion이 이보다 오래되면 점수가 낮아도 바꾼다")
    args = parser.parse_args()

    df = pd.read_parquet(args.training_set)
    if not {"train", "valid"} <= set(df["split"]):
        raise SystemExit(f"split에 train/valid가 없다: {df['split'].value_counts().to_dict()}")

    y = df["label"].to_numpy(np.int32)
    fit, calib, valid = split_calibration(df, args.calib_fraction)
    counts = {k: (int(m.sum()), int(y[m].sum())) for k, m in
              [("fit", fit), ("calib", calib), ("valid", valid)]}
    print("행 수, 클릭 수: " + ", ".join(f"{k} {n:,}/{c:,}" for k, (n, c) in counts.items()))
    if any(c < args.min_clicks or c == n for n, c in counts.values()):
        print(f"클릭이 부족하다(각 {args.min_clicks}건 필요). 학습하지 않는다.")
        sys.exit(NOT_ENOUGH_CLICKS)

    x = to_matrix(df)
    chosen = sample_negatives(y, np.flatnonzero(fit), args.seed)
    print(f"negative sampling 뒤 fit {len(chosen):,}행 (클릭 {int(y[chosen].sum()):,})")

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("lightgbm이 없다. `uv sync --frozen --extra train`로 설치한다.") from exc

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": args.min_data_in_leaf,
        "lambda_l2": 1.0,
        "seed": args.seed,
        "num_threads": 0,
        "verbose": -1,
    }
    names = feature_names()
    dtrain = lgb.Dataset(x[chosen], label=y[chosen], feature_name=names)
    dcalib = lgb.Dataset(x[calib], label=y[calib], feature_name=names, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=args.rounds,
        valid_sets=[dcalib],
        callbacks=[lgb.early_stopping(args.early_stopping, verbose=False),
                   lgb.log_evaluation(50)],
    )

    final, a, b = calibrate(model, x[calib], y[calib])
    metrics = evaluate(final, x[valid], y[valid])
    print(f"best_iteration {model.best_iteration}, calibration a={a:.4f} b={b:.4f}, "
          + ", ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(args.model_out))
    print(f"저장: {args.model_out}")
    params |= {"calib_a": a, "calib_b": b, "best_iteration": model.best_iteration,
               "negative_ratio": 1, "fit_rows": len(chosen), "calib_rows": counts["calib"][0],
               "valid_rows": counts["valid"][0],
               "simulated_rows": int(df["simulated"].sum()) if "simulated" in df else -1}
    if not log_mlflow(final, params, metrics, x[valid], y[valid], args.champion_max_age_days):
        sys.exit(NOT_PROMOTED)


if __name__ == "__main__":
    main()
