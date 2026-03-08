import sys
from pathlib import Path

# Ensure project root is on sys.path so "ecs171_project.*" imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Any, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# IMPORTANT: move dataclass to a real module so pickle won't bind to __main__
from ecs171_project.preprocess_artifacts import PreprocessArtifacts

# 0) Paths / Constants
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = BASE_DIR / "heart_statlog_cleveland_hungary_final.csv"
PROCESSED_CSV = BASE_DIR / "heart_disease_processed.csv"
OUT_DIR = BASE_DIR / "ecs171_project" / "outputs_model3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.2

CONTINUOUS_COLS = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
CATEGORICAL_COLS = [
    "sex", "chest pain type", "fasting blood sugar",
    "resting ecg", "exercise angina", "ST slope"
]
ZERO_AS_MISSING_COLS = ["cholesterol", "resting bp s", "ST slope"]
ONEHOT_COLS = ["chest pain type", "ST slope", "resting ecg"]


# 1) Helpers
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def write_text(path: str, s: str):
    Path(path).write_text(s, encoding="utf-8")


def assert_cols_exist(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing columns: {missing}\nExisting: {df.columns.tolist()}")


def dump_json(path: str, obj: Any):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_pickle(path: str, obj: Any):
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# 2) Preprocess Artifacts (so Flask can reproduce preprocessing)
def _basic_clean_and_impute(df: pd.DataFrame, fit: bool, artifacts: Optional[PreprocessArtifacts]) -> Dict[str, Any]:
    """
    Apply:
      - drop duplicates
      - zeros -> NaN for specific cols
      - impute continuous by median
      - impute categorical by mode
      - cast categoricals + target to int
    Returns dict with:
      df_clean, medians, modes
    """
    df = df.drop_duplicates().reset_index(drop=True)

    # zeros -> NaN
    for c in ZERO_AS_MISSING_COLS:
        if c in df.columns:
            df.loc[df[c] == 0, c] = np.nan

    medians = {}
    modes = {}

    if fit:
        # compute and apply medians
        for c in CONTINUOUS_COLS:
            med = float(df[c].median())
            medians[c] = med
            df[c] = df[c].fillna(med)

        # compute and apply modes
        for c in CATEGORICAL_COLS:
            mode = df[c].mode(dropna=True)
            if len(mode) == 0:
                raise ValueError(f"Cannot compute mode for {c}")
            mv = int(mode.iloc[0])
            modes[c] = mv
            df[c] = df[c].fillna(mv)
    else:
        if artifacts is None:
            raise ValueError("artifacts required for transform (fit=False)")
        medians = artifacts.medians
        modes = artifacts.modes
        for c in CONTINUOUS_COLS:
            df[c] = df[c].fillna(medians[c])
        for c in CATEGORICAL_COLS:
            df[c] = df[c].fillna(modes[c])

    # cast categorical + target to int when present
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(int)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    return {"df": df, "medians": medians, "modes": modes}


def _feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - age_group
      - heart_rate_reserve
      - chol_risk, bp_risk, oldpeak_abnormal
    """
    df_original = df.copy()

    bins = [0, 40, 55, 70, 100]
    df["age_group"] = pd.cut(
        df_original["age"],
        bins=bins,
        labels=[0, 1, 2, 3]
    ).astype(int)

    df["heart_rate_reserve"] = (220 - df_original["age"]) - df_original["max heart rate"]
    df["chol_risk"] = (df_original["cholesterol"] > 200).astype(int)
    df["bp_risk"] = (df_original["resting bp s"] > 140).astype(int)
    df["oldpeak_abnormal"] = (df_original["oldpeak"] > 0).astype(int)

    return df


def preprocess_fit(raw_csv: str, out_processed_csv: str, out_dir: str) -> (pd.DataFrame, PreprocessArtifacts):
    """
    Fit preprocessing on full dataset,
    save processed CSV, and return df_processed + artifacts.
    """
    df = pd.read_csv(raw_csv)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    uniq = set(df[TARGET_COL].unique())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"Target not binary {{0,1}}. Unique={uniq}")

    assert_cols_exist(df, CONTINUOUS_COLS, "continuous")
    assert_cols_exist(df, CATEGORICAL_COLS, "categorical")
    assert_cols_exist(df, ONEHOT_COLS, "onehot")

    cleaned = _basic_clean_and_impute(df, fit=True, artifacts=None)
    df = cleaned["df"]

    # feature engineering
    df = _feature_engineer(df)

    # one-hot
    df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=True)

    # scaling
    scale_cols = CONTINUOUS_COLS + ["heart_rate_reserve"]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    df.to_csv(out_processed_csv, index=False)

    X = df.drop(TARGET_COL, axis=1)
    artifacts = PreprocessArtifacts(
        medians=cleaned["medians"],
        modes=cleaned["modes"],
        scaler=scaler,
        scale_cols=scale_cols,
        feature_columns=X.columns.tolist(),
        dummy_columns=df.columns.tolist(),
        age_bins=[0, 40, 55, 70, 100],
    )

    ensure_dir(out_dir)
    dump_pickle(Path(out_dir) / "preprocess.pkl", artifacts)
    dump_json(Path(out_dir) / "feature_columns.json", {"feature_columns": artifacts.feature_columns})

    return df, artifacts


def preprocess_transform(df_raw: pd.DataFrame, artifacts: PreprocessArtifacts, has_target: bool) -> pd.DataFrame:
    """
    Transform NEW raw data using fitted artifacts, producing a processed dataframe
    aligned to training feature_columns order.
    This is what Flask should use for user inputs.
    """
    need_cols = CONTINUOUS_COLS + CATEGORICAL_COLS
    if has_target:
        need_cols = need_cols + [TARGET_COL]
    assert_cols_exist(df_raw, need_cols, "raw_input_required")

    cleaned = _basic_clean_and_impute(df_raw.copy(), fit=False, artifacts=artifacts)
    df = cleaned["df"]

    df = _feature_engineer(df)
    df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=True)

    if has_target and TARGET_COL in df.columns:
        y = df[TARGET_COL].astype(int)
        df = df.drop(TARGET_COL, axis=1)
    else:
        y = None

    for col in artifacts.feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[artifacts.feature_columns]
    df[artifacts.scale_cols] = artifacts.scaler.transform(df[artifacts.scale_cols])

    if y is not None:
        df[TARGET_COL] = y.values

    return df


def save_roc_curve(y_true: pd.Series, y_prob: np.ndarray, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc_score(y_true, y_prob):.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# 3) Model 3: Random Forest
def run_model3(df_processed: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    out_dir = Path(out_dir)

    X = df_processed.drop(TARGET_COL, axis=1)
    y = df_processed[TARGET_COL].astype(int)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # baseline RF
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    # metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_prob)

    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics_txt = "\n".join([
        "=== Model 3: Random Forest (Baseline) ===",
        f"Test Accuracy : {acc:.4f}",
        f"Test Precision: {prec:.4f}",
        f"Test Recall   : {rec:.4f}",
        f"Test F1       : {f1:.4f}",
        f"Test AUC      : {auc_score:.4f}",
        "",
        "--- classification_report ---",
        report
    ])
    write_text(out_dir / "metrics.txt", metrics_txt)
    write_text(out_dir / "confusion_matrix.txt", np.array2string(cm))
    write_text(out_dir / "auc.txt", f"AUC: {auc_score:.4f}\n")

    # save ROC curve figure
    save_roc_curve(y_test, y_prob, out_dir / "roc_curve.png")

    # save baseline model too (optional)
    dump_pickle(out_dir / "baseline_model.pkl", rf)

    # CV (F1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="f1")
    cv_txt = "\n".join([
        "=== 5-Fold Stratified CV (Baseline RF) ===",
        f"F1 mean: {cv_scores.mean():.4f}",
        f"F1 std : {cv_scores.std():.4f}",
        f"All scores: {np.round(cv_scores, 4).tolist()}",
    ])
    write_text(out_dir / "cv_results.txt", cv_txt)

    # GridSearch tuning (F1)
    param_grid = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", "log2", None],
    }
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X, y)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = float(grid.best_score_)

    dump_json(
        out_dir / "best_params.json",
        {"best_params": best_params, "best_cv_f1": best_score}
    )

    # save the actual trained model for Flask
    dump_pickle(out_dir / "model.pkl", best_model)

    # feature importance
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.to_csv(out_dir / "feature_importance.csv", header=["importance"])

    top15 = importances.head(15)
    top15_txt = "=== Top 15 Feature Importances (Best RF) ===\n" + "\n".join(
        [f"{k}: {v:.6f}" for k, v in top15.items()]
    )
    write_text(out_dir / "feature_importance_top15.txt", top15_txt)


# 4) Main
def main():
    print("RAW_CSV:", RAW_CSV)
    print("PROCESSED_CSV:", PROCESSED_CSV)
    print("OUT_DIR:", OUT_DIR)

    print("\n[1/2] Building processed dataset + saving preprocess artifacts...")
    df_processed, _art = preprocess_fit(str(RAW_CSV), str(PROCESSED_CSV), str(OUT_DIR))
    print("Processed shape:", df_processed.shape)
    print("Saved:", PROCESSED_CSV)
    print("Saved preprocess artifacts:", OUT_DIR / "preprocess.pkl")

    print("\n[2/2] Running Model 3 (Random Forest) + saving model.pkl ...")
    run_model3(df_processed, str(OUT_DIR))

    print("\nALL DONE.")
    print("Outputs saved to:", OUT_DIR)
    print(" - metrics.txt")
    print(" - confusion_matrix.txt")
    print(" - auc.txt")
    print(" - roc_curve.png")
    print(" - cv_results.txt")
    print(" - best_params.json")
    print(" - feature_importance.csv / feature_importance_top15.txt")
    print(" - baseline_model.pkl (optional)")
    print(" - model.pkl (BEST MODEL for Flask)")
    print(" - preprocess.pkl (needed for Flask preprocessing)")
    print(" - feature_columns.json")


if __name__ == "__main__":
    main()
