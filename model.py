import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, classification_report,
                             precision_recall_curve, average_precision_score)


# ==========================================
# 1. Custom Feature Transformer
# ==========================================
class FraudFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that:
    - Extracts 'Hour', 'Amount_log', and 'is_fraud_hour'.
    - Drops raw 'Amount'.
    - Scales ['Time', 'Amount_log'] with StandardScaler.
    - Adds an unsupervised 'anomaly_score' via IsolationForest (fitted on normal samples).
    """
    def __init__(self, contamination=0.002, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.iso = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.scale_cols = ["Time", "Amount_log"]

    def _engineer(self, X):
        d = X.copy()
        d["Hour"] = ((d["Time"] // 3600) % 24).astype(int)
        d["Amount_log"] = np.log1p(d["Amount"])
        d["is_fraud_hour"] = d["Hour"].isin([2, 11, 18]).astype(int)
        return d.drop(columns=["Amount"])

    def fit(self, X, y=None):
        X_eng = self._engineer(X)
        self.scaler.fit(X_eng[self.scale_cols])

        X_scaled = X_eng.copy()
        X_scaled[self.scale_cols] = self.scaler.transform(X_scaled[self.scale_cols])

        # Fit Isolation Forest on non-fraud (normal) samples to capture normal density
        if y is not None:
            normal_mask = (y == 0)
            self.iso.fit(X_scaled[normal_mask])
        else:
            self.iso.fit(X_scaled)
        return self

    def transform(self, X):
        X_eng = self._engineer(X)
        X_eng[self.scale_cols] = self.scaler.transform(X_eng[self.scale_cols])
        # Negate score_samples so higher value = more anomalous
        X_eng["anomaly_score"] = -self.iso.score_samples(X_eng)
        return X_eng


# ==========================================
# 2. Data Loading & Transformation
# ==========================================
df_train = pd.read_csv("E:/AI Track/Machine Learning Projects/2 Credit Card Fraud Detection/data/split/train.csv")
df_val = pd.read_csv("E:/AI Track/Machine Learning Projects/2 Credit Card Fraud Detection/data/split/val.csv")
df_test = pd.read_csv("E:/AI Track/Machine Learning Projects/2 Credit Card Fraud Detection/data/split/test.csv")

x_train = df_train.drop(columns="Class")
y_train = df_train["Class"]

x_val = df_val.drop(columns="Class")
y_val = df_val["Class"]

x_test = df_test.drop(columns="Class")
y_test = df_test["Class"]

transformer = FraudFeatureTransformer(contamination=0.002, random_state=42)
X_train_fe = transformer.fit_transform(x_train, y_train)
x_val_fe = transformer.transform(x_val)
x_test_fe = transformer.transform(x_test)


# ==========================================
# 3. Threshold Selection Helper
# ==========================================
def best_f1_threshold(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1_all = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    return thr[f1_all.argmax()]


# ==========================================
# 4. Model Training & Evaluation
# ==========================================
models = {
    "LogReg": LogisticRegression(max_iter=1000, random_state=42),
    "RF": RandomForestClassifier(random_state=42, n_jobs=-1),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

p_val = {}   # val probabilities per model
p_test = {}  # test probabilities per model

for m_name, base_model in models.items():
    # Threshold = best F1 chosen on train Out-Of-Fold predictions
    oof = cross_val_predict(clone(base_model), X_train_fe, y_train, cv=cv, method="predict_proba")[:, 1]
    t = best_f1_threshold(y_train, oof)

    model = clone(base_model).fit(X_train_fe, y_train)
    proba_train = model.predict_proba(X_train_fe)[:, 1]
    proba_val = model.predict_proba(x_val_fe)[:, 1]
    proba_test = model.predict_proba(x_test_fe)[:, 1]

    p_val[m_name] = proba_val
    p_test[m_name] = proba_test

    pred_train = (proba_train >= t).astype(int)
    pred_val = (proba_val >= t).astype(int)
    pred_test = (proba_test >= t).astype(int)

    print(f"\n==================== {m_name} (Threshold: {t:.3f}) ====================")
    print(f"TRAIN Fraud F1: {f1_score(y_train, pred_train):.4f}")
    print(f"VAL   Fraud F1: {f1_score(y_val, pred_val):.4f} | PR-AUC: {average_precision_score(y_val, proba_val):.4f}")
    print(f"TEST  Fraud F1: {f1_score(y_test, pred_test):.4f} | PR-AUC: {average_precision_score(y_test, proba_test):.4f}")
    print("\n--- Validation Classification Report ---")
    print(classification_report(y_val, pred_val, digits=4))
    print("--- Test Classification Report ---")
    print(classification_report(y_test, pred_test, digits=4))


# ==========================================
# 5. Precision / Recall / F1 vs Threshold Plots
# ==========================================
fig, axes = plt.subplots(1, len(p_val), figsize=(6 * len(p_val), 4.5), sharey=True)

for ax, (name, proba) in zip(axes, p_test.items()):
    prec, rec, thr = precision_recall_curve(y_test, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    ap = average_precision_score(y_test, proba)

    ax.plot(thr, prec[:-1], label="Precision", color="tab:blue")
    ax.plot(thr, rec[:-1], label="Recall", color="tab:orange")
    ax.plot(thr, f1, label="F1", color="tab:green", linestyle="--")

    ax.axvline(0.5, color="gray", linestyle=":", label="0.5")
    ax.axvline(thr[f1.argmax()], color="tab:red", linestyle=":", label=f"best on test ({thr[f1.argmax()]:.2f})")

    ax.set_title(f"{name} (Test Set) | PR-AUC = {ap:.4f}")
    ax.set_xlabel("Threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Score")
plt.tight_layout()
plt.show()
