# DA5401 A6 — Imputation via Regression for Missing Data

This repository contains the complete workflow for **DA5401 Assignment 6**, focused on introducing missingness and comparing **simple**, **linear-regression**, and **non‑linear regression** imputation strategies by evaluating downstream **credit‑default** classification performance.

## 📌 Objective
1. Introduce **Missing At Random (MAR)** values (≈5–10%) in a few numeric columns.
2. Construct clean datasets using three imputation strategies (**Median**, **Linear Regression**, **Non‑Linear Regression**) and a **Listwise Deletion** baseline.
3. Train a **Logistic Regression** classifier on each dataset and compare performance (Accuracy, Precision, Recall, F1).
4. Provide a short analysis + recommendation on which strategy is best and why.

---

## 📂 Dataset
- **Source:** UCI Credit Card Default Clients (30,000 × 25).
- **Target label:** `default.payment.next.month` (0/1).
- **File used in Colab:** `UCI_Credit_Card.csv` (uploaded via Files pane).

### Columns we perturbed with MAR
- `AGE`, `BILL_AMT1`, `PAY_AMT1` (≈7% each)
- Target column is **never** perturbed.

### Column imputed via regression (B & C)
- **Chosen column:** `AGE`  
  - **B:** Linear Regression  
  - **C:** Non‑Linear Regression (KNN Regressor, k=5)

> **Professor’s clarification respected:** For B & C, **only one** column is imputed by regression; **other NaNs remain unchanged**. Using one perturbed column is considered an **individual imputation experiment**.

---

## 🧪 Part A — Data Preprocessing & Imputation

### A1. Introduce MAR
Randomly blank ≈7% values in `AGE`, `BILL_AMT1`, `PAY_AMT1`.

### A2. Strategy 1 — Median Imputation → **Dataset A**
- Median‑impute **only the columns that contain NaNs** (never the label).
- **Why median over mean?** Robust to outliers and skew (typical of bill/payment amounts).

### A3. Strategy 2 — Regression Imputation (Linear) → **Dataset B**
- Impute **only `AGE`** using **Linear Regression** trained on other features.
- **Assumption:** **MAR** — missingness depends only on observed variables, not on the unobserved value itself.

### A4. Strategy 3 — Regression Imputation (Non‑Linear) → **Dataset C**
- Impute **only `AGE`** using **KNN Regressor (k=5)**.
- **Assumption:** Same **MAR**; non‑linear model can capture curved/threshold relations if present.

### A5. Dataset D — Listwise Deletion
- Drop any row with **any** missing value.

---

## ⚙️ Part B — Model Training & Assessment

### B1. Train/Test Split
- Stratified 80/20 split for each dataset (A, B, C, D).

### B2. Standardization
- Same pipeline for all models: `SimpleImputer(median) → StandardScaler → LogisticRegression`  
  (This is **train‑time preprocessing only**, not a change to B/C rules.)

### B3. Classifier
- `LogisticRegression(max_iter=500, random_state=42)`

---

## 📊 Part C — Comparative Analysis

### Summary Performance (your run)
| Model | Precision (Macro) | Recall (Macro) | F1 (Macro) | Accuracy |
|:--|:--:|:--:|:--:|:--:|
| A — Median Imputation | **0.7546** | 0.6053 | 0.6225 | 0.8083 |
| B — Linear Regression Imputation | 0.7531 | 0.6044 | 0.6213 | 0.8078 |
| C — Non‑Linear Regression Imputation | 0.7550 | 0.6056 | 0.6230 | 0.8085 |
| D — Listwise Deletion | 0.7784 | 0.6081 | **0.6268** | **0.8139** |

### Discussion
- **Listwise Deletion vs Imputation:** D shows slightly higher scores by **dropping incomplete rows**, which reduces noise but **shrinks** the dataset and risks **bias** if missingness is not MCAR. Imputation (A–C) **preserves data**, keeping models more representative and robust.
- **Linear vs Non‑Linear Imputation:** C ≈ B ≈ A. This suggests `AGE` has either **weak** or roughly **linear** relationships w.r.t. other features/label. Non‑linear benefits are minimal here.
- **Recommendation:** Prefer **Regression Imputation** over Deletion. Given similar performance and better interpretability, choose **Linear Regression Imputation** as the practical default for this dataset.

> **One‑line takeaway:** *Keep the data, impute a single column with a simple, explainable model under MAR → stable metrics and better generalization.*

---

## ▶️ How to Run (Colab)

1. Upload `UCI_Credit_Card.csv` to the Colab Files pane.  
2. Run the notebook cells in order:
   - **Part A**: MAR injection → Datasets A/B/C/D
   - **Part B**: Train/evaluate classifiers
   - **Part C**: Summarize results and discussion
3. (Optional) Change the chosen imputation column or the non‑linear model.

---

## 🗂️ Artifacts
- `DA5401_A6_template.ipynb` — ready‑to‑run notebook with all tasks.
- Generated datasets (optional):
  - `dataset_A_median.csv`
  - `dataset_B_linear_onecol.csv`
  - `dataset_C_nonlinear_onecol.csv`

---

## ✍️ Notes & Assumptions
- MAR fraction used ≈ **7%** per selected column.
- Target column is never altered.
- B & C **only** impute the **chosen column**; others stay as‑is (per clarification).
- Train‑time imputer/scaler are applied **uniformly** to ensure a fair comparison.

---

## 🙏 Acknowledgments
- UCI Machine Learning Repository — *Default of Credit Card Clients Data Set*.
- Course staff for the clarification regarding B & C single‑column regression imputation.