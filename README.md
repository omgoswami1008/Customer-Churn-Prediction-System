<div align="center">

# 📡 Customer Churn Prediction System

**A production-style end-to-end ML project that predicts telecom customer churn — from raw data to a deployed interactive dashboard.**

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.3-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Key result: Holdout ROC-AUC 0.842 · Recall 0.81 at optimal threshold**

</div>

---

## 🚀 Live demo

> https://customer-churn-prediction-system-8vchmzxpkknmvvb7eisggf.streamlit.app/

## 📌 Table of contents

- [Overview](#overview)
- [Business problem](#business-problem)
- [Tech stack](#tech-stack)
- [Pipeline](#pipeline)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Model performance](#model-performance)
- [Deployment](#deployment)
- [Sharing on LinkedIn](#sharing-on-linkedin)
- [Roadmap](#roadmap)

---

## Overview

This project solves a classic **customer retention** problem: given a telecom
customer's profile (contract, tenure, services, charges, payment method...),
predict whether they will **churn** (leave the company). Churn detection lets
retention teams act *before* revenue is lost.

It is built as a **modular, reproducible pipeline** — each stage is a reusable
Python module — and ships with a polished **Streamlit dashboard** for live
scoring and exploration.

## Business problem

| | |
|---|---|
| **Question** | Will this customer churn? |
| **Target** | `Churn` — Yes / No |
| **Dataset** | IBM Telco Churn — 7,043 customers, 21 attributes |
| **Class balance** | 26.5% churn (imbalanced) |
| **Why it matters** | Acquiring a customer costs **5–7× more** than retaining one |

The model returns a **churn probability**, a **risk band** (Low / Medium / High)
and **targeted retention recommendations** for each customer.

## Tech stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `Streamlit` · `Joblib`

## Pipeline

```
data_loader.py → preprocessing.py → feature_engineering.py
              → train_model.py → evaluate.py → predict.py → app.py
```

1. **Load & validate** raw data (`src/data_loader.py`)
2. **Clean** — drop IDs, fix dtypes, handle missing values, deduplicate (`src/preprocessing.py`)
3. **Engineer features** — simplify service columns, add `num_addon_services`, `avg_monthly_charges`, `tenure_group`, one-hot encode (`src/feature_engineering.py`)
4. **Train** — stratified split, `StandardScaler`, Random Forest tuned with 5-fold cross-validation on ROC-AUC (`src/train_model.py`)
5. **Evaluate** — metrics, confusion matrix, ROC / PR curves, feature importance → PDF report (`src/evaluate.py`)
6. **Predict** — single & batch scoring reusing the exact training transforms (`src/predict.py`)
7. **Serve** — Streamlit dashboard (`app.py`)

## Project structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/telecom_churn.csv          # raw dataset (7,043 rows)
│   └── processed/                     # clean_data.csv, features.csv, splits
│
├── notebooks/
│   └── 01_EDA.ipynb                   # exploratory data analysis
│
├── src/
│   ├── data_loader.py                 # load + validate datasets
│   ├── preprocessing.py               # cleaning pipeline
│   ├── feature_engineering.py         # derived features + encoding
│   ├── train_model.py                 # CV tuning + training
│   ├── evaluate.py                    # metrics + PDF report
│   └── predict.py                     # single / batch prediction
│
├── models/                            # random_forest.pkl, scaler.pkl, config
├── reports/                           # figures + model_report.pdf
├── .streamlit/config.toml             # app theme & server settings
├── app.py                             # Streamlit dashboard (5 pages)
├── requirements.txt
└── README.md
```

## Getting started

### 1. Clone & setup

```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python -m src.data_loader         # validate the raw dataset
python -m src.preprocessing       # → data/processed/clean_data.csv
python -m src.feature_engineering # → data/processed/features.csv + config
python -m src.train_model         # → models/random_forest.pkl + scaler.pkl
python -m src.evaluate            # → reports/figures + model_report.pdf
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) 🎉

## Model performance

Evaluation on a **1,405-customer holdout** (never seen during training):

| Metric | Default (0.5) | Optimal (0.41) |
|---|---|---|
| Accuracy | 0.774 | 0.745 |
| Precision | 0.557 | 0.512 |
| Recall | 0.729 | **0.815** |
| F1-score | 0.631 | 0.629 |
| **ROC-AUC** | **0.842** | 0.842 |
| PR-AUC | 0.658 | 0.658 |






</div>
