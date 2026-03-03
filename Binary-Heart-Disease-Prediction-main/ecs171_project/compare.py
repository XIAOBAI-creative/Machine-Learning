from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ====== 路径 ======
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_CSV = BASE_DIR / "heart_disease_processed.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "target"

# ====== 读取数据 ======
df = pd.read_csv(PROCESSED_CSV)

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# ====== 统一分割 ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

results = []

# ========================
# Model 1: Logistic
# ========================
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)

results.append([
    "Logistic Regression",
    accuracy_score(y_test, pred1),
    precision_score(y_test, pred1),
    recall_score(y_test, pred1),
    f1_score(y_test, pred1)
])

# ========================
# Model 2: SVM (RBF)
# ========================
model2 = SVC(kernel="rbf")
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)

results.append([
    "SVM (RBF)",
    accuracy_score(y_test, pred2),
    precision_score(y_test, pred2),
    recall_score(y_test, pred2),
    f1_score(y_test, pred2)
])

# ========================
# Model 3: Random Forest
# ========================
model3 = RandomForestClassifier(
    n_estimators=400,
    random_state=RANDOM_STATE
)
model3.fit(X_train, y_train)
pred3 = model3.predict(X_test)

results.append([
    "Random Forest",
    accuracy_score(y_test, pred3),
    precision_score(y_test, pred3),
    recall_score(y_test, pred3),
    f1_score(y_test, pred3)
])

# ====== 输出表格 ======
df_results = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1"]
)

print("\n=== Model Comparison ===")
print(df_results)

df_results.to_csv("model_comparison.csv", index=False)
