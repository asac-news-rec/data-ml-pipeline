"""[파이프라인 구간: 학습 데이터 → 모델 학습·평가] Feast training set으로 PCTR baseline LightGBM을 학습한다.

입력:  features/feast/data/training_set.parquet (label, split 포함)
출력:  models/lgbm.txt (gitignore)
실행:  uv run --frozen --extra train python training/train_lgbm.py

순서: train을 시간순으로 fit/calib으로 나눈다 → fit의 negative를 1:1로 줄여 학습한다
→ 줄이지 않은 calib으로 확률을 보정한다 → 줄이지 않은 valid로 평가한다.
MLflow는 선택이다. MLFLOW_TRACKING_URI가 있으면 기록·등록하고, 이전 champion보다 낫거나
champion이 오래됐는데 AUC 하락이 허용 폭 안일 때만 champion alias를 옮긴다.
새 모델 AUC가 --min-auc보다 낮으면 어떤 경우에도 옮기지 않는다.
training set에 simulated 열이 없으면(build_training_set.py가 만든 것이 아니면, 예: 로컬 가짜 로그)
등록만 하고 champion은 옮기지 않는다.
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
sys.path.insert(0, str(ROOT))   # 학습 이미지는 저장소 전체를 /app에 둔다
from serving.matrix import feature_names, to_matrix  # noqa: E402  (추론 서버와 같은 입력 행렬)
NOT_PROMOTED, NOT_ENOUGH_CLICKS = 98, 99   # dags/train_pctr.py의 skip_on_exit_code와 같아야 한다


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


def should_promote(new: dict, old: dict | None, age_days: float, max_age_days: float,
                   min_auc: float = 0.55, max_auc_drop: float = 0.02) -> tuple[bool, str]:
    """champion을 새 모델로 바꿀지 정한다.

    어떤 경우든 새 모델 valid AUC가 min_auc 미만이면 바꾸지 않는다(상류 피처가 깨진 모델 차단).
    champion이 max_age_days보다 오래됐어도 champion보다 max_auc_drop 넘게 낮으면 바꾸지 않는다.
    """
    auc = new["valid_auc"]
    if not auc >= min_auc:   # NaN도 막는다
        return False, f"AUC {auc:.4f} < 최소 {min_auc}"
    if old is None:
        return True, "비교할 champion이 없다"
    if auc > old["valid_auc"]:
        return True, f"AUC {old['valid_auc']:.4f} → {auc:.4f}"
    if age_days >= max_age_days:
        if auc >= old["valid_auc"] - max_auc_drop:
            return True, f"champion이 {age_days:.1f}일 됐고 AUC 하락 {old['valid_auc'] - auc:.4f} <= {max_auc_drop}"
        return False, (f"champion이 {age_days:.1f}일 됐지만 AUC {old['valid_auc']:.4f} → {auc:.4f}, "
                       f"하락 폭이 {max_auc_drop}보다 크다")
    return False, f"AUC {auc:.4f} <= champion {old['valid_auc']:.4f}, {age_days:.1f}일"


def as_booster(loaded):
    """sklearn 방식으로 등록한 모델은 predict가 0/1이라 안의 Booster를 쓴다(serving/model.py와 같다)."""
    return getattr(loaded, "booster_", loaded)


def log_mlflow(model, params: dict, metrics: dict, x_valid, y_valid, max_age_days: float,
               min_auc: float, max_auc_drop: float, real_data: bool = True) -> bool:
    """기록·등록하고 추론 서버가 새 모델을 읽어야 하면 True.

    real_data가 False면(실제 로그로 만든 training set이 아니면) 등록만 하고 champion은 옮기지 않는다.
    """
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
    old, age_days, note, schema_changed = None, 0.0, "", False
    try:
        champion = client.get_model_version_by_alias(name, "champion")
    except MlflowException as exc:
        # 서버 장애를 "champion 없음"으로 보면 비교 없이 교체된다. 없다는 응답만 넘긴다.
        if exc.error_code not in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"):
            raise
        champion = None
    if champion is not None:
        loaded = as_booster(mlflow.lightgbm.load_model(f"models:/{name}/{champion.version}"))
        if loaded.feature_name() != feature_names():
            # 피처 열이 바뀌면 지금 떠 있는 추론 서버 이미지가 새 모델을 못 읽을 수 있다.
            # 등록만 하고 champion은 옮기지 않는다. 새 서빙 이미지를 배포한 뒤 사람이 alias를 옮긴다.
            schema_changed = True
            note = f"champion v{champion.version}과 피처 열이 다르다. "
        else:
            old = evaluate(loaded, x_valid, y_valid)
            age_days = (time.time() - champion.creation_timestamp / 1000) / 86400
    promote, reason = should_promote(metrics, old, age_days, max_age_days, min_auc, max_auc_drop)
    if schema_changed:
        promote, reason = False, "새 서빙 이미지 배포 뒤 champion alias를 직접 옮긴다"
    reason = note + reason
    if not real_data:
        promote, reason = False, "training set에 simulated 열이 없다(실제 로그 학습이 아니다). " + reason

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
                        help="champion이 이보다 오래되면 AUC가 조금 낮아도(--max-auc-drop 안) 바꾼다")
    parser.add_argument("--min-auc", type=float, default=0.55,
                        help="새 모델 valid AUC가 이보다 낮으면 어떤 경우에도 champion으로 올리지 않는다")
    parser.add_argument("--max-auc-drop", type=float, default=0.02,
                        help="champion이 오래됐을 때 허용하는 champion 대비 AUC 하락 폭")
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

    chosen = sample_negatives(y, np.flatnonzero(fit), args.seed)
    print(f"negative sampling 뒤 fit {len(chosen):,}행 (클릭 {int(y[chosen].sum()):,})")
    # 1,545열 행렬은 쓰는 행(sampling된 fit + calib + valid)만 만든다. 버릴 negative까지 만들면 메모리가 몇 배다.
    calib_idx, valid_idx = np.flatnonzero(calib), np.flatnonzero(valid)
    x_fit, x_calib, x_valid = (to_matrix(df.iloc[i]) for i in (chosen, calib_idx, valid_idx))
    y_fit, y_calib, y_valid = y[chosen], y[calib_idx], y[valid_idx]

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
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "0")),   # 0 = 노드 전체 코어
        "verbose": -1,
    }
    names = feature_names()
    dtrain = lgb.Dataset(x_fit, label=y_fit, feature_name=names)
    dcalib = lgb.Dataset(x_calib, label=y_calib, feature_name=names, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=args.rounds,
        valid_sets=[dcalib],
        callbacks=[lgb.early_stopping(args.early_stopping, verbose=False),
                   lgb.log_evaluation(50)],
    )

    final, a, b = calibrate(model, x_calib, y_calib)
    metrics = evaluate(final, x_valid, y_valid)
    print(f"best_iteration {model.best_iteration}, calibration a={a:.4f} b={b:.4f}, "
          + ", ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(args.model_out))
    print(f"저장: {args.model_out}")
    params |= {"calib_a": a, "calib_b": b, "best_iteration": model.best_iteration,
               "negative_ratio": 1, "fit_rows": len(chosen), "calib_rows": counts["calib"][0],
               "valid_rows": counts["valid"][0],
               "simulated_rows": int(df["simulated"].sum()) if "simulated" in df else -1}
    if not log_mlflow(final, params, metrics, x_valid, y_valid, args.champion_max_age_days,
                      args.min_auc, args.max_auc_drop, real_data="simulated" in df):
        sys.exit(NOT_PROMOTED)


if __name__ == "__main__":
    main()
