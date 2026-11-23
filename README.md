# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning pipeline for detecting credit card fraud on highly imbalanced transaction data, featuring custom scikit-learn transformers, unsupervised anomaly scoring via Isolation Forest, out-of-fold optimal threshold tuning, and comprehensive evaluation on an unseen test set.

---

## 📌 Project Overview

Credit card fraud represents a severe challenge for financial institutions due to extreme class imbalance: fraudulent transactions represent less than **0.17%** of all activity. Standard accuracy metrics fail in this domain, requiring optimization for **Precision-Recall Area Under the Curve (PR-AUC)** and **F1-Score**.

### Dataset Specifications
- **Origin**: Anonymized European cardholder transactions over a 48-hour period (2 days).
- **Features**:
  - `V1` – `V28`: Anonymized principal components resulting from PCA transformation.
  - `Time`: Elapsed seconds since the first recorded transaction.
  - `Amount`: Transaction amount in currency.
  - `Class`: Ground-truth target label (`1` = Fraud, `0` = Legitimate).
- **Dataset Partitioning (Stratified 60% / 20% / 20%)**:
  - **Train**: 170,884 samples (~290 frauds)
  - **Validation**: 56,960 samples (90 frauds)
  - **Test**: 56,960 samples (97 frauds)

---

## 🔍 Exploratory Data Analysis & Experimental Learnings

Detailed exploratory findings documented in [`EDA.ipynb`](./EDA.ipynb) and experimental logs in [`steps.txt`](./steps.txt):

1. **Zero-Amount Transactions**:
   - Analysis identified 1,067 transactions with `Amount == 0.0`.
   - *Decision*: Kept in the dataset. In financial networks, $0.00 transactions are card verification/authorization checks frequently utilized by automated card-testing fraud rings. Removing them hurts downstream recall.
2. **Amount Transformation**:
   - The transaction amount is heavily right-skewed. Applied `np.log1p(Amount)` to stabilize variance and prevent extreme outliers from distorting linear models and tree split points.
3. **Outlier Filtering vs. Fraud Detection**:
   - Experiments filtering extreme amount quantiles (e.g., upper 0.75 or 0.85 quantiles) removed up to **48 genuine fraud cases**. In fraud detection, statistical outliers often represent the very malicious patterns being targeted; therefore, outlier pruning was abandoned.
4. **Resampling Techniques (SMOTE / Undersampling)**:
   - Synthetic Minority Over-sampling (SMOTE) and random sampling proved counterproductive. In high-dimensional PCA space, SMOTE generates synthetic points along line segments between minority examples, placing artificial frauds inside dense legitimate clusters. This caused a surge in False Positives and heavily degraded Precision.
5. **Temporal Distribution**:
   - Transactions span 48 hours. Converting `Time` to `Hour = ((Time // 3600) % 24)` exposed cyclical transaction rhythms and fraud concentration clusters during off-peak hours (e.g., hours 2, 11, 18).

---

## ⚙️ Architecture & Pipeline Implementation

### 1. Custom Transformer: `FraudFeatureTransformer`
Implemented in [`model.py`](./model.py) conforming to the `scikit-learn` `BaseEstimator` and `TransformerMixin` API:
- **Feature Extraction**:
  - `Hour`: Extracted from cumulative seconds modulo 24.
  - `Amount_log`: Log-transformed amount (`log1p`).
  - `is_fraud_hour`: Indicator flag for high-risk temporal windows (`Hour in [2, 11, 18]`).
  - Dropped raw `Amount` to eliminate collinearity with `Amount_log`.
- **Scaling**:
  - `StandardScaler` fitted on `['Time', 'Amount_log']` on training data and applied to validation/test sets.
- **Unsupervised Anomaly Feature (`anomaly_score`)**:
  - An `IsolationForest` (`contamination=0.002`) is fitted exclusively on **legitimate transactions** (`Class == 0`) in the training set to map baseline spending geometry.
  - Computes `-iso.score_samples(X)`, producing a continuous metric of multidimensional geometric rarity across all 30 features simultaneously.

### 2. Out-Of-Fold (OOF) Optimal Threshold Tuning
- Default `0.5` classification thresholds assume balanced classes and equal error costs.
- Using 5-Fold Stratified Cross-Validation on the training set, Out-Of-Fold predicted probabilities were generated:
  $$\text{oof} = \text{cross\_val\_predict}(\text{model}, X_{\text{train}}, y_{\text{train}}, \text{cv}=5, \text{method}="predict\_proba")$$
- The decision boundary $t$ is calculated via `precision_recall_curve` to maximize F1-score:
  $$t = \arg\max_{t} F_1(t)$$
- **Zero Data Leakage**: Threshold $t$ was derived solely from training OOF folds and applied statically to the validation and test sets.

---

## 📊 Final Performance Results

### Metric Summary Across Datasets

| Model | Evaluated Split | Optimal Threshold $t$ | Precision | Recall | **Fraud F1-Score** | **PR-AUC** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Validation | 0.420 | 0.9351 | 0.8000 | **0.8623** | **0.8549** |
| **Random Forest** | **Test Set (Unseen)** | **0.420** | **0.8617** | **0.8351** | **0.8482** | **0.8506** |
| **Logistic Regression** | Validation | 0.316 | 0.8767 | 0.7111 | **0.7853** | **0.7711** |
| **Logistic Regression** | **Test Set (Unseen)** | **0.316** | **0.7912** | **0.7423** | **0.7660** | **0.7326** |

---

### Test Set Detailed Classification Reports (56,960 Transactions, 97 Frauds)

#### 🌲 Random Forest Classifier (Final Champion Model)
```text
              precision    recall  f1-score   support

           0     0.9997    0.9998    0.9997     56863
           1     0.8617    0.8351    0.8482        97

    accuracy                         0.9995     56960
   macro avg     0.9307    0.9174    0.9240     56960
weighted avg     0.9995    0.9995    0.9995     56960
```
- **Frauds Caught**: **81 out of 97** (83.51% Recall).
- **False Positives**: Only **13 false alarms** across 56,863 legitimate transactions (False Positive Rate = 0.022%).

#### 📈 Logistic Regression
```text
              precision    recall  f1-score   support

           0     0.9996    0.9997    0.9996     56863
           1     0.7912    0.7423    0.7660        97

    accuracy                         0.9992     56960
   macro avg     0.8954    0.8710    0.8828     56960
weighted avg     0.9992    0.9992    0.9992     56960
```

---

## 🔬 Key Technical Insights & Explorations

1. **Probability Calibration (Platt Scaling & Isotonic Regression)**:
   - Evaluated `CalibratedClassifierCV` on Random Forest.
   - *Finding*: Because calibration is a monotonic transformation, it reshapes the raw score distribution but preserves rank ordering. Since the pipeline already searches the full precision-recall space for the optimal threshold $t$, monotonic calibration shifts the cutoff threshold ($0.34 \to 0.45$) while yielding identical classification boundaries.
2. **Gradient Boosting (LightGBM / XGBoost)**:
   - Evaluated LightGBM on the 600:1 imbalanced dataset.
   - Standard gradient boosting minimizes global log-loss, which causes the overwhelming negative gradient to suppress minority trees unless heavily penalized via `scale_pos_weight` or custom focal loss. Random Forest with feature subspace sampling (`max_features="sqrt"`) proved more resilient to single-tree noise.
3. **Generalization to Test Data**:
   - The Random Forest maintained **0.8482 F1** on the unseen test set, confirming that combining Isolation Forest anomaly features with threshold optimization successfully avoids overfitting.

---

## 🚀 How to Run the Project

### 1. Environment Setup
The project uses Python with `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and `lightgbm`:

```powershell
# Activate the existing virtual environment
& "e:\AI Track\Machine Learning Projects\2 Credit Card Fraud Detection\data\venv\Scripts\Activate.ps1"
```

### 2. Execute Training & Evaluation
Run the complete pipeline script:
```powershell
python "e:\AI Track\Machine Learning Projects\2 Credit Card Fraud Detection\data\Project\model.py"
```

This will:
1. Load train, validation, and test splits from `../split/`.
2. Fit the `FraudFeatureTransformer` and extract features and anomaly scores.
3. Perform 5-fold Stratified Cross-Validation OOF threshold selection.
4. Output classification reports for Train, Validation, and Test sets.
5. Display Precision, Recall, and F1 vs. Threshold curves for the test set.

---

## 📁 Repository Structure

```text
2 Credit Card Fraud Detection/
├── data/
│   ├── split/
│   │   ├── train.csv         # 60% training split
│   │   ├── val.csv           # 20% validation split
│   │   ├── test.csv          # 20% held-out test split
│   │   └── trainval.csv      # Merged train + validation split
│   ├── Project/
│   │   ├── model.py          # Complete pipeline & evaluation code
│   │   ├── EDA.ipynb         # Exploratory Data Analysis & visual inspections
│   │   ├── steps.txt         # Iterative experimentation log
│   │   ├── Figure_1.png      # Saved evaluation plots
│   │   └── README.md         # Full project documentation & benchmark results
│   └── venv/                 # Python virtual environment
```
