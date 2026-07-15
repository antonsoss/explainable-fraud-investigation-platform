# Explainable Fraud Investigation Platform

An MIA 5100 machine learning project for detecting fraudulent transactions and supporting evidence-based human investigation.

## Overview

The project uses the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) to compare fraud classifiers under severe class imbalance. The workflow emphasizes leakage prevention, precision-recall evaluation, threshold selection, model-native interpretation, similar-case retrieval, and human review.

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
| `04_Fraud_Investigation_AI_Assistant.ipynb` | Complete |
| Fraud investigation platform (Streamlit) | Next |

Notebook 04 uses native XGBoost feature importance, error cohorts, TF-IDF, and cosine similarity to retrieve comparable historical cases. It produces offline deterministic investigation summaries and includes an optional grounded LLM extension.

## Repository structure

```text
├── app/                   # Streamlit application
├── data/                  # Raw and processed data (not tracked)
├── models/
│   ├── preprocessing/    # Fitted preprocessing artifacts
│   ├── trained/          # Trained fraud models
│   └── investigation/    # TF-IDF retrieval artifacts
├── notebooks/            # EDA, preprocessing, modeling, and investigation
├── reports/              # Investigation and final reports
├── results/              # Model and investigation outputs
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

On macOS, XGBoost also requires:

```bash
brew install libomp
```

Download the competition data from Kaggle, place the CSV files in `data/raw/`, and run the notebooks in numerical order.

The core investigation notebook does not require an API key. To enable its optional OpenAI-generated summaries, set `OPENAI_API_KEY` securely and change `RUN_OPENAI_LLM` to `True`. Never commit API keys.

## Next phase

The final phase will integrate the model KPIs, transaction profiles, similar historical cases, and grounded investigation summaries into a Streamlit application.
