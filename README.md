# Explainable Fraud Investigation Platform

> An end-to-end machine learning project for detecting and investigating fraudulent financial transactions using Explainable Artificial Intelligence (XAI).

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue)

---

# Overview

Financial institutions process millions of transactions every day, making fraud detection one of the most important applications of machine learning.

The goal of this project is to develop an **Explainable Fraud Investigation Platform** capable of identifying potentially fraudulent financial transactions while providing transparent explanations that support fraud analysts during the investigation process.

Unlike traditional black-box fraud detection systems, this project emphasizes **model interpretability**, enabling investigators to understand **why** a transaction has been classified as suspicious.

---

# Project Objectives

- Perform comprehensive Exploratory Data Analysis (EDA)
- Understand fraud patterns within financial transaction data
- Build multiple supervised machine learning models
- Compare model performance using appropriate evaluation metrics
- Explain model predictions using Explainable AI (SHAP)
- Generate business insights for fraud investigation teams
- Build a reusable fraud analytics project suitable for an AI portfolio

---

# Current Project Status

This project is currently under active development.

| Phase | Status |
|---------|:------:|
| Business Understanding | ✅ Completed |
| Exploratory Data Analysis | 🚧 In Progress |
| Data Preprocessing | ⏳ Planned |
| Feature Engineering | ⏳ Planned |
| Model Development | ⏳ Planned |
| Model Evaluation | ⏳ Planned |
| Explainable AI (SHAP) | ⏳ Planned |
| Final Report | ⏳ Planned |

---

# Dataset

This project uses the **IEEE-CIS Fraud Detection Dataset**, published by Kaggle.

The dataset contains:

- Financial transaction information
- Card information
- Device information
- Identity attributes
- Email domains
- Time-related features

Target variable:

```
isFraud
```

where

- **0** = Legitimate Transaction
- **1** = Fraudulent Transaction

---

# Dataset Setup

The dataset is **not included** in this repository because of its size and Kaggle licensing restrictions.

Download it from:

https://www.kaggle.com/competitions/ieee-fraud-detection/data

Place the following files inside:

```
data/raw/
```

Required files:

```
train_transaction.csv
train_identity.csv
test_transaction.csv
test_identity.csv
sample_submission.csv
```

---

# Repository Structure

Current repository:

```
explainable-fraud-investigation-platform/

│
├── README.md
├── notebooks/
│
└── .gitignore
```

Planned structure:

```
explainable-fraud-investigation-platform/

├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Model_Development.ipynb
│   ├── 04_Model_Evaluation.ipynb
│   └── 05_Explainability_SHAP.ipynb
│
├── figures/
├── models/
├── reports/
└── src/
```

---

# Machine Learning Workflow

The project follows a standard machine learning lifecycle.

### Phase 1 — Exploratory Data Analysis

- Business Understanding
- Dataset Overview
- Target Analysis
- Numerical Feature Analysis
- Categorical Feature Analysis
- Missing Value Analysis
- Correlation Analysis
- Business Insights

### Phase 2 — Data Preparation

- Missing Value Treatment
- Encoding
- Feature Engineering
- Feature Selection
- Train/Test Split

### Phase 3 — Model Development

Models to be evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Phase 4 — Model Evaluation

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Precision-Recall Curve
- Confusion Matrix

### Phase 5 — Explainable AI

Explainability techniques:

- SHAP
- Feature Importance
- Global Explanations
- Local Explanations

---

# Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Jupyter Notebook
- Git
- GitHub

---

# Expected Business Value

The completed platform will demonstrate how machine learning can assist fraud investigation teams by:

- Prioritizing suspicious transactions
- Reducing manual investigation effort
- Improving fraud detection effectiveness
- Providing transparent model explanations
- Supporting evidence-based decision making

---

# Future Enhancements

Potential improvements include:

- Real-time fraud scoring
- Graph-based fraud detection
- Deep Learning models
- AutoML experimentation
- FastAPI deployment
- Streamlit dashboard
- Azure cloud deployment

---

# Installation

Clone the repository

```bash
git clone https://github.com/antonsoss/explainable-fraud-investigation-platform.git
```

Navigate into the project

```bash
cd explainable-fraud-investigation-platform
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter

```bash
jupyter lab
```

---

# Author

**Antonio Sosa**

Master of Interdisciplinary Artificial Intelligence  
University of Ottawa

Software Developer | Machine Learning | Artificial Intelligence

🌐 https://www.antoniososa.ca

GitHub: https://github.com/antonsoss

LinkedIn: https://www.linkedin.com/in/antonio-sosa