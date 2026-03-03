import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 1) 让Python能import到项目根目录的 ecs171model3.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../web
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   # .../project root
sys.path.insert(0, PROJECT_ROOT)

import ecs171model3  # 里面有 preprocess_transform

# NEW: 导入 PreprocessArtifacts，让 pickle 能找到正确模块路径
from ecs171_project.preprocess_artifacts import PreprocessArtifacts  # noqa: F401

# 加载 pkl
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs_model3")
model = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"), "rb"))
preprocess = pickle.load(open(os.path.join(MODEL_DIR, "preprocess.pkl"), "rb"))

# ---------- 预加载前端需要的数据 ----------

def _load_feature_importance():
    fi = pd.read_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index_col=0)
    top = fi.head(10)
    return {"labels": top.index.tolist(), "values": [round(v, 6) for v in top["importance"]]}

def _load_model_metrics():
    path = os.path.join(MODEL_DIR, "metrics.txt")
    metrics = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "Accuracy" in line and ":" in line:
                metrics["accuracy"] = float(line.split(":")[-1].strip())
            elif "Precision" in line and ":" in line:
                metrics["precision"] = float(line.split(":")[-1].strip())
            elif "Recall" in line and ":" in line and "macro" not in line:
                metrics["recall"] = float(line.split(":")[-1].strip())
            elif "F1" in line and ":" in line:
                metrics["f1"] = float(line.split(":")[-1].strip())
    return metrics

def _load_data_distributions():
    csv_path = os.path.join(PROJECT_ROOT, "heart_statlog_cleveland_hungary_final.csv")
    df = pd.read_csv(csv_path)

    continuous_cols = ["age", "cholesterol", "resting bp s", "max heart rate", "oldpeak"]
    dist = {}
    for col in continuous_cols:
        disease = df[df["target"] == 1][col].dropna().tolist()
        no_disease = df[df["target"] == 0][col].dropna().tolist()
        dist[col] = {"disease": disease, "no_disease": no_disease}

    # target counts
    tc = df["target"].value_counts()
    dist["target_counts"] = {
        "no_disease": int(tc.get(0, 0)),
        "disease": int(tc.get(1, 0)),
    }

    # sex × target 分组
    sex_target = df.groupby(["sex", "target"]).size().reset_index(name="count")
    dist["sex_target"] = sex_target.to_dict(orient="records")

    return dist

FEATURE_IMPORTANCE = _load_feature_importance()
MODEL_METRICS = _load_model_metrics()
DATA_DISTRIBUTIONS = _load_data_distributions()


# ---------- 路由 ----------

@app.route("/")
def home():
    return render_template(
        "index.html",
        feature_importance=json.dumps(FEATURE_IMPORTANCE),
        model_metrics=json.dumps(MODEL_METRICS),
        data_distributions=json.dumps(DATA_DISTRIBUTIONS),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)

    input_dict = {
        "age": float(data["age"]),
        "sex": int(data["sex"]),
        "chest pain type": int(data["chest_pain"]),
        "resting bp s": float(data["rest_bp"]),
        "cholesterol": float(data["chol"]),
        "fasting blood sugar": int(data["fbs"]),
        "resting ecg": int(data["rest_ecg"]),
        "max heart rate": float(data["max_hr"]),
        "exercise angina": int(data["ex_ang"]),
        "oldpeak": float(data["oldpeak"]),
        "ST slope": int(data["st_slope"]),
    }

    df_input = pd.DataFrame([input_dict])
    df_processed = ecs171model3.preprocess_transform(df_input, preprocess, has_target=False)

    pred = int(model.predict(df_processed)[0])
    proba = model.predict_proba(df_processed)[0]  # [p_class0, p_class1]
    risk_pct = round(float(proba[1]) * 100, 1)

    # 各特征对预测的贡献度（用 feature importance × 特征值方向 近似）
    fi = pd.Series(model.feature_importances_, index=df_processed.columns)
    processed_vals = df_processed.iloc[0]
    contributions = []
    for feat in fi.sort_values(ascending=False).head(8).index:
        imp = fi[feat]
        val = processed_vals[feat]
        # 正值倾向 disease, 负值倾向 healthy（简化近似）
        direction = 1 if val > 0 else -1
        contrib = round(float(imp * direction * 100), 1)
        contributions.append({"feature": feat, "contribution": contrib})

    # 识别风险因素
    risk_factors = []
    age = input_dict["age"]
    if age >= 55:
        risk_factors.append({"factor": "Age", "value": f"{int(age)} years", "note": "Age above 55 increases cardiac risk."})
    if input_dict["cholesterol"] > 240:
        risk_factors.append({"factor": "Cholesterol", "value": f"{int(input_dict['cholesterol'])} mg/dl", "note": "Above 240 mg/dl is considered high."})
    if input_dict["resting bp s"] > 140:
        risk_factors.append({"factor": "Resting BP", "value": f"{int(input_dict['resting bp s'])} mmHg", "note": "Above 140 mmHg suggests hypertension."})
    if input_dict["exercise angina"] == 1:
        risk_factors.append({"factor": "Exercise Angina", "value": "Yes", "note": "Chest pain during exercise is a warning sign."})
    if input_dict["oldpeak"] > 2.0:
        risk_factors.append({"factor": "Oldpeak", "value": str(input_dict["oldpeak"]), "note": "Significant ST depression detected."})
    if input_dict["max heart rate"] < 120:
        risk_factors.append({"factor": "Max Heart Rate", "value": str(int(input_dict["max heart rate"])), "note": "Low max heart rate may indicate poor cardiac fitness."})

    label = "Heart Disease Detected" if pred == 1 else "No Heart Disease"

    return jsonify({
        "result": pred,
        "risk_pct": risk_pct,
        "label": label,
        "contributions": contributions,
        "risk_factors": risk_factors,
        "input_summary": {
            "Age": int(age),
            "Sex": "Male" if input_dict["sex"] == 1 else "Female",
            "BP": int(input_dict["resting bp s"]),
            "Chol": int(input_dict["cholesterol"]),
            "MaxHR": int(input_dict["max heart rate"]),
        },
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
