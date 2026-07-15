# Explainable Fraud Investigation Platform

An MIA 5100 machine learning project for detecting fraudulent transactions and producing transparent, investigator-oriented explanations.

## Overview

The project uses the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) to compare classification models under severe class imbalance. The workflow emphasizes leakage prevention, precision-recall evaluation, decision-threshold selection, and explainability.

## Current results

Preprocessing produced 324 model features and stratified training, validation, and test sets. XGBoost achieved the strongest validation average precision and was selected as the champion model.

| Test metric | Result |
|---|---:|
| Average precision | 0.7119 |
| ROC-AUC | 0.9511 |
| Precision | 0.5512 |
| Recall | 0.7160 |
| F2 score | 0.6756 |
| Alert rate | 4.54% |

These test results use the decision threshold selected on validation data by maximizing F2.

## Workflow

| Notebook | Status |
|---|:---:|
| `01_EDA_Business_Understanding.ipynb` | Complete |
| `02_Data_Preprocessing_Feature_Engineering.ipynb` | Complete |
| `03_Model_Development_Comparison.ipynb` | Complete |
| Model explainability and fraud investigation | Next |

The model comparison includes a dummy baseline, logistic regression, decision tree, random forest, histogram gradient boosting, multi-layer perceptron, and XGBoost. Average precision is the primary selection metric because fraud is rare; ROC-AUC, precision, recall, F1, F2, balanced accuracy, and alert volume provide supporting evidence.

## Repository structure

```text
├── data/                  # Raw and processed data (not tracked)
├── models/
│   ├── preprocessing/    # Fitted preprocessing artifacts
│   └── trained/          # Trained models and metadata
├── notebooks/            # EDA, preprocessing, and modeling workflow
├── results/              # Model metrics and predictions
├── src/                  # Reusable project code
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/antonsoss/explainable-fraud-investigation-platform.git
cd explainable-fraud-investigation-platform
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On macOS, XGBoost also requires the OpenMP runtime:

```bash
brew install libomp
```

Download the competition data from Kaggle and place the CSV files in `data/raw/`. Then run the notebooks in numerical order.

## Next phase

The next notebook will apply SHAP for global and transaction-level explanations, analyze model errors, and develop evidence-grounded summaries to support fraud investigation.
