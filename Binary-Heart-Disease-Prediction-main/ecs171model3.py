import os
import json
import pickle
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

# =============================================================================
# 0) Paths / Constants (keep your original layout)
# =============================================================================
RAW_CSV = r"C:\Users\victo\OneDrive\Desktop\CS\ecs171\ecs171_project\heart_statlog_cleveland_hungary_final.csv"
PROJECT_DIR = os.path.dirname(RAW_CSV)
PROCESSED_CSV = os.path.join(PROJECT_DIR, "heart_disease_processed.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs_model3")

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


# =============================================================================
# 1) Helpers
# =============================================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_text(path: str, s: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def assert_cols_exist(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing columns: {missing}\nExisting: {df.columns.tolist()}")

def dump_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def dump_pickle(path: str, obj: Any):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# =============================================================================
# 2) Preprocess Artifacts (so Flask can reproduce preprocessing)
# =============================================================================
@dataclass
class PreprocessArtifacts:
    medians: Dict[str, float]
    modes: Dict[str, int]
    scaler: StandardScaler
    scale_cols: List[str]
    feature_columns: List[str]          # final X columns order after get_dummies + scaling
    dummy_columns: List[str]            # all columns after get_dummies (including non-dummies)

    # For age_group bins (fixed)
    age_bins: List[int]


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

    # fixed bins
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
    Fit preprocessing on full dataset (like your original),
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

    # save processed CSV (like your original)
    df.to_csv(out_processed_csv, index=False)

    # build artifacts (for Flask)
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
    dump_pickle(os.path.join(out_dir, "preprocess.pkl"), artifacts)
    dump_json(os.path.join(out_dir, "feature_columns.json"), {"feature_columns": artifacts.feature_columns})

    return df, artifacts


def preprocess_transform(df_raw: pd.DataFrame, artifacts: PreprocessArtifacts, has_target: bool) -> pd.DataFrame:
    """
    Transform NEW raw data using fitted artifacts, producing a processed dataframe
    aligned to training feature_columns order.
    This is what your Flask should use for user inputs.
    """
    # Validate required cols exist
    need_cols = CONTINUOUS_COLS + CATEGORICAL_COLS
    if has_target:
        need_cols = need_cols + [TARGET_COL]
    assert_cols_exist(df_raw, need_cols, "raw_input_required")

    cleaned = _basic_clean_and_impute(df_raw.copy(), fit=False, artifacts=artifacts)
    df = cleaned["df"]

    df = _feature_engineer(df)

    df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=True)

    # Ensure all training columns exist (add missing with 0)
    # We'll align on full df columns (including target if present)
    # First separate X/y
    if has_target and TARGET_COL in df.columns:
        y = df[TARGET_COL].astype(int)
        df = df.drop(TARGET_COL, axis=1)
    else:
        y = None

    # Add missing columns
    for col in artifacts.feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Drop extra columns not seen in training
    df = df[artifacts.feature_columns]

    # Scale columns
    df[artifacts.scale_cols] = artifacts.scaler.transform(df[artifacts.scale_cols])

    if y is not None:
        df[TARGET_COL] = y.values

    return df


# =============================================================================
# 3) Model 3: Random Forest
# =============================================================================
def run_model3(df_processed: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)

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

    # metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics_txt = "\n".join([
        "=== Model 3: Random Forest (Baseline) ===",
        f"Test Accuracy : {acc:.4f}",
        f"Test Precision: {prec:.4f}",
        f"Test Recall   : {rec:.4f}",
        f"Test F1       : {f1:.4f}",
        "",
        "--- classification_report ---",
        report
    ])
    write_text(os.path.join(out_dir, "metrics.txt"), metrics_txt)
    write_text(os.path.join(out_dir, "confusion_matrix.txt"), np.array2string(cm))

    # save baseline model too (optional)
    dump_pickle(os.path.join(out_dir, "baseline_model.pkl"), rf)

    # CV (F1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="f1")
    cv_txt = "\n".join([
        "=== 5-Fold Stratified CV (Baseline RF) ===",
        f"F1 mean: {cv_scores.mean():.4f}",
        f"F1 std : {cv_scores.std():.4f}",
        f"All scores: {np.round(cv_scores, 4).tolist()}",
    ])
    write_text(os.path.join(out_dir, "cv_results.txt"), cv_txt)

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
        os.path.join(out_dir, "best_params.json"),
        {"best_params": best_params, "best_cv_f1": best_score}
    )

    # ✅ Save the actual trained model for Flask
    dump_pickle(os.path.join(out_dir, "model.pkl"), best_model)

    # feature importance
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.to_csv(os.path.join(out_dir, "feature_importance.csv"), header=["importance"])

    top15 = importances.head(15)
    top15_txt = "=== Top 15 Feature Importances (Best RF) ===\n" + "\n".join(
        [f"{k}: {v:.6f}" for k, v in top15.items()]
    )
    write_text(os.path.join(out_dir, "feature_importance_top15.txt"), top15_txt)


# =============================================================================
# 4) Main
# =============================================================================
def main():
    print("RAW_CSV:", RAW_CSV)
    print("PROCESSED_CSV:", PROCESSED_CSV)
    print("OUT_DIR:", OUT_DIR)

    print("\n[1/2] Building processed dataset + saving preprocess artifacts...")
    df_processed, _art = preprocess_fit(RAW_CSV, PROCESSED_CSV, OUT_DIR)
    print("Processed shape:", df_processed.shape)
    print("Saved:", PROCESSED_CSV)
    print("Saved preprocess artifacts:", os.path.join(OUT_DIR, "preprocess.pkl"))

    print("\n[2/2] Running Model 3 (Random Forest) + saving model.pkl ...")
    run_model3(df_processed, OUT_DIR)

    print("\n✅ ALL DONE.")
    print("Outputs saved to:", OUT_DIR)
    print(" - metrics.txt")
    print(" - confusion_matrix.txt")
    print(" - cv_results.txt")
    print(" - best_params.json")
    print(" - feature_importance.csv / feature_importance_top15.txt")
    print(" - baseline_model.pkl (optional)")
    print(" - model.pkl (BEST MODEL for Flask)")
    print(" - preprocess.pkl (needed for Flask preprocessing)")
    print(" - feature_columns.json")


if __name__ == "__main__":
    main()
