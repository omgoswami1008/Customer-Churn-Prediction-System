# 📘 Customer Churn Prediction System — Project Notes

> Everything you need to understand, run and present this project.

---

## 1. What does this project do?

It predicts whether a **telecom customer will churn** (cancel their service).
You give the model a customer's profile — contract, tenure, services, charges,
payment method — and it returns:

- a **churn probability** (e.g. 0–100%)
- a **risk band** (Low / Medium / High)
- **targeted retention recommendations**

The project is a **complete end-to-end ML pipeline**, built like production
software: modular, reproducible, tested, and served through a polished
Streamlit dashboard.

---

## 2. Tech stack

| Layer | Tools |
|---|---|
| Data handling | Python, Pandas, NumPy |
| Machine learning | Scikit-learn (Random Forest + GridSearchCV) |
| Visualisation | Matplotlib, Seaborn |
| Serving | Streamlit (5-page dashboard) |
| Persistence | Joblib (model, scaler, config) |

---

## 3. Dataset (IBM Telco Churn)

- **7,043 customers, 21 columns**
- **Target:** `Churn` = Yes / No
- **Imbalanced:** 26.5% churn (5174 No / 1869 Yes)
- Features: gender, senior citizen, partner, dependents, tenure, phone &
  internet services, add-ons (security, backup, protection, support,
  streaming), contract, paperless billing, payment method, monthly & total
  charges.

| Feature group | Columns |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod |
| Services | PhoneService, MultipleLines, InternetService |
| Add-ons | OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Charges | MonthlyCharges, TotalCharges |
| Target | Churn |

---

## 4. Project structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/telecom_churn.csv          # raw dataset
│   └── processed/                     # pipeline outputs
│       ├── clean_data.csv             # cleaned 7021×20
│       ├── features.csv               # encoded 7021×28 (+Churn, tenure_group)
│       └── data_splits.joblib         # exact train/test split
│
├── src/                               # 6 modular pipeline stages
│   ├── data_loader.py                 # load & validate data
│   ├── preprocessing.py               # cleaning (dedup, dtypes, imputation)
│   ├── feature_engineering.py         # derived features + encoding
│   ├── train_model.py                 # CV tuning + training
│   ├── evaluate.py                    # metrics + PDF report
│   └── predict.py                     # single/batch live prediction
│
├── models/                            # trained artifacts (committed for deploy)
│   ├── random_forest.pkl              # model + feature schema
│   ├── scaler.pkl                     # fitted StandardScaler
│   └── feature_config.joblib          # encoding schema
│
├── reports/
│   ├── figures/                       # confusion_matrix, roc, pr, importance
│   └── model_report.pdf               # 5-page evaluation report
│
├── notebooks/
│   └── 01_EDA.ipynb                   # (placeholder for EDA)
│
├── .streamlit/config.toml             # app theme + server config
├── app.py                             # Streamlit dashboard (5 pages)
├── requirements.txt                   # pinned dependencies
├── README.md                          # GitHub-facing README
└── PROJECT_NOTES.md                   # this file
```

---

## 5. The ML pipeline (data flow)

```
telecom_churn.csv (raw, 7043×21)
        │
        ▼  python -m src.preprocessing
clean_data.csv (7021×20)          ← drops customerID, 22 dup rows, fixes types
        │
        ▼  python -m src.feature_engineering
features.csv (7021×28)            ← 28 encoded features + target
feature_config.joblib             ← schema for re-encoding new input
        │
        ▼  python -m src.train_model
random_forest.pkl + scaler.pkl    ← tuned model (GridSearchCV, 5-fold CV)
data_splits.joblib                ← fixed train/test holdout
        │
        ▼  python -m src.evaluate
reports/figures/*.png             ← confusion matrix, ROC, PR, importance
reports/model_report.pdf
        │
        ▼  python -m src.predict  (used by app.py)
churn probability + risk band for any new customer
```

### Module-by-module

1. **`data_loader.py`** — Paths resolve from the project root, so everything
   works from CLI, notebook or the app. Validates the file exists and is
   non-empty; raises clear errors otherwise.

2. **`preprocessing.py`** — Drops `customerID`, strips whitespace, coerces
   numerics (handles the classic *TotalCharges-as-text* quirk), maps
   `SeniorCitizen` 0/1 → No/Yes, drops duplicates, imputes missing
   `TotalCharges` with `tenure × MonthlyCharges`, validates the target.
   **Idempotent** — safe to re-run.

3. **`feature_engineering.py`** — Maps `No internet/phone service` → `No`;
   creates `num_addon_services`, `avg_monthly_charges`, `tenure_group`;
   encodes 13 binary columns → 1/0 and 3 multi-class columns → one-hot.
   Saves `features.csv` + `feature_config.joblib`.

4. **`train_model.py`** — Stratified 80/20 split (churn ratio preserved),
   `StandardScaler` fit on **train only** (no leakage), Random Forest tuned
   with `GridSearchCV` (27 combos × 5 folds) scored on **ROC-AUC** with
   `class_weight='balanced'`. Saves model, scaler and the exact split.

5. **`evaluate.py`** — Computes metrics at 0.5 and at the **Youden-optimal
   threshold**; plots confusion matrix, ROC, PR curves and top-15 feature
   importances; bundles everything into `model_report.pdf`.

6. **`predict.py`** — Loads artifacts once (cached). Re-applies the exact
   training transforms to a single customer (via `build_feature_row`) so
   predictions match training-time behaviour. Risk bands: <0.30 Low,
   0.30–0.60 Medium, ≥0.60 High.

---

## 6. Model performance (holdout = 1,405 customers)

| Metric | Default (0.5) | Optimal (0.41) |
|---|---|---|
| Accuracy | 0.774 | 0.745 |
| Precision | 0.557 | 0.512 |
| Recall | 0.729 | **0.815** |
| F1-score | 0.631 | 0.629 |
| ROC-AUC | **0.842** | 0.842 |
| PR-AUC | 0.658 | 0.658 |

**Top churn drivers (feature importance):** tenure, TotalCharges,
MonthlyCharges, contract length, payment method.

---

## 7. How to run (Windows / macOS / Linux)

### Prerequisites
- Python 3.10+ (project tested on 3.13)
- `pip` available

### Setup

```bash
# 1. Clone or open the project
cd customer-churn-prediction

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the full pipeline (optional — artifacts are already committed)

```bash
python -m src.data_loader
python -m src.preprocessing
python -m src.feature_engineering
python -m src.train_model      # takes a few minutes (grid search)
python -m src.evaluate
```

### Launch the dashboard

```bash
streamlit run app.py
# → open http://localhost:8501
```

> Because the trained model, scaler and data are already committed to the
> repo, you can skip straight to `streamlit run app.py`.

---

## 8. How to use the app (5 pages)

1. **🏠 Home** — overview, KPIs, and two quick buttons:
   - 🔴 *High-risk example* → opens the predictor pre-filled (86% risk)
   - 🟢 *Low-risk example* → opens the predictor pre-filled (14% risk)
2. **🎯 Churn Predictor** — fill the form → **Predict churn risk**:
   - probability gauge + risk badge (green/amber/red)
   - decision at 0.5 threshold
   - **retention recommendations** generated from the answers
3. **📊 Data Explorer** — KPI cards + churn by contract / payment method,
   tenure & charges distributions, sample data table.
4. **📈 Model Report** — holdout metrics, confusion matrix, ROC / PR curves,
   feature importance, PDF download.
5. **ℹ️ About** — pipeline commands, stack, structure, key results.

---

## 9. Deployment (free — Streamlit Community Cloud)

1. **Push to GitHub** (artifacts are committed on purpose so the app works):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: customer churn prediction system"
   git remote add origin https://github.com/<you>/customer-churn-prediction.git
   git push -u origin main
   ```
2. Go to **share.streamlit.io** → sign in with GitHub → **New app** →
   repo / branch `main` / file `app.py` → **Deploy**.
3. Your live URL looks like: `https://<your-repo>.streamlit.app`.

---

## 10. How to present it (GitHub + LinkedIn)

- **README.md** already has the structure, metrics and a LinkedIn post draft.
- Screenshot these in your portfolio:
  - Home page KPI dashboard
  - Predictor with a high-risk result (badge + recommendations)
  - ROC curve + feature importance from the Model Report page
  - `reports/model_report.pdf`

---

## 11. Troubleshooting / FAQ

| Problem | Fix |
|---|---|
| `streamlit: command not found` | `python -m streamlit run app.py` or activate venv |
| Predictor shows an error about artifacts | Ensure `models/*.pkl` exist: run `python -m src.train_model` |
| Model report figures missing | Run `python -m src.evaluate` |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |
| Slow first load | Grid search retrains; artifacts are already committed so normally not needed |

---

## 12. Possible improvements (roadmap)

- Add XGBoost / Logistic Regression baselines for comparison
- SHAP explanations per customer
- Batch CSV upload for scoring many customers
- Build the EDA notebook `notebooks/01_EDA.ipynb`
- Add unit tests with pytest
