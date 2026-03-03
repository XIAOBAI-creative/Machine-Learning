import pandas as pd
import numpy as np

# Cleaning
# Load data
path = "heart_statlog_cleveland_hungary_final.csv"
df = pd.read_csv(path)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# Basic quality checks
print("\nMissing values per column:\n", df.isna().sum())
dup_count = df.duplicated().sum()
print("\nDuplicated rows:", dup_count)

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)
print("\nAfter dropping duplicates:", df.shape)

# Validate target
target_col = "target"
print("\nTarget value counts:\n", df[target_col].value_counts())
assert set(df[target_col].unique()).issubset({0, 1}), "Target is not binary {0,1}!"

# Handle implicit missing values
zero_as_missing_cols = ["cholesterol", "resting bp s", "ST slope"]

for c in zero_as_missing_cols:
    if c in df.columns:
        zero_count = (df[c] == 0).sum()
        print(f"\n{c}: zero count before = {zero_count}")
        df.loc[df[c] == 0, c] = np.nan

print("\nMissing values after converting zeros to NaN:\n", df.isna().sum())

# Imputation
# Decide which columns are continuous vs categorical
continuous_cols = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
categorical_cols = ["sex", "chest pain type", "fasting blood sugar",
                    "resting ecg", "exercise angina", "ST slope"]

# Median imputation for continuous
for c in continuous_cols:
    if c in df.columns:
        med = df[c].median()
        df[c] = df[c].fillna(med)

# Mode imputation for categorical/discrete
for c in categorical_cols:
    if c in df.columns:
        mode = df[c].mode(dropna=True)[0]
        df[c] = df[c].fillna(mode)

# Ensure categorical cols are int-coded
for c in categorical_cols + [target_col]:
    if c in df.columns:
        df[c] = df[c].astype(int)

print("\nFinal missing values per column:\n", df.isna().sum())
print("\nFinal shape:", df.shape)
print("\nData types:\n", df.dtypes)


# EDA
import os
import matplotlib.pyplot as plt

target_col = "target"

OUT_DIR = "eda_figures"

def save_fig(filename: str):
    path = os.path.join(OUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", path)

# Target class distribution
counts = df[target_col].value_counts().sort_index()

plt.figure(figsize=(5, 4))
plt.bar(counts.index.astype(str), counts.values)
plt.title("Target Class Distribution")
plt.xlabel("target (0=no disease, 1=disease)")
plt.ylabel("count")
save_fig("fig1_target_distribution.png")

# Continuous feature histograms
continuous_cols = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
continuous_cols = [c for c in continuous_cols if c in df.columns]

n = len(continuous_cols)
cols = 3
rows = int(np.ceil(n / cols))

plt.figure(figsize=(12, 4 * rows))
for i, c in enumerate(continuous_cols, 1):
    ax = plt.subplot(rows, cols, i)
    ax.hist(df[c], bins=30)
    ax.set_title(c)
    ax.set_xlabel(c)
    ax.set_ylabel("count")

plt.suptitle("Continuous Feature Distributions", y=1.02)
save_fig("fig2_continuous_histograms.png")

# oldpeak by target
feature = "oldpeak"
if feature in df.columns:
    data0 = df[df[target_col] == 0][feature]
    data1 = df[df[target_col] == 1][feature]

    plt.figure(figsize=(6, 4))
    plt.boxplot([data0, data1], labels=["target=0", "target=1"])
    plt.title("oldpeak by Target Class")
    plt.ylabel(feature)
    save_fig("fig3_oldpeak_by_target.png")
else:
    print("Warning: oldpeak not found, skip FIG 3")

# max heart rate by target (boxplot)
feature = "max heart rate"
if feature in df.columns:
    data0 = df[df[target_col] == 0][feature]
    data1 = df[df[target_col] == 1][feature]

    plt.figure(figsize=(6, 4))
    plt.boxplot([data0, data1], labels=["target=0", "target=1"])
    plt.title("max heart rate by Target Class")
    plt.ylabel(feature)
    save_fig("fig4_maxheartrate_by_target.png")

# Disease rate by exercise angina
cat = "exercise angina"
if cat in df.columns:
    rates = df.groupby(cat)[target_col].mean().sort_index()
    counts = df[cat].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    plt.bar(rates.index.astype(str), rates.values)
    plt.title("Disease Rate by Exercise Angina")
    plt.xlabel(cat + " (0=no, 1=yes)")
    plt.ylabel("P(target=1)")

    for i, (k, v) in enumerate(rates.items()):
        plt.text(i, v, f"n={counts.loc[k]}", ha="center", va="bottom", fontsize=9)

    save_fig("fig5_exerciseangina_disease_rate.png")

print("\nDone. Figures saved in:", OUT_DIR)
