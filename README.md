# Explainable Fraud Investigation Platform

An MIA 5100 machine learning project for detecting fraudulent transactions and supporting evidence-based human investigation.

## Overview

The project uses the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) to compare classifiers under severe class imbalance. The workflow emphasizes chronological evaluation, training-only preprocessing, precision-recall analysis, operational threshold selection, model-native global interpretation, labeled reference-case retrieval, and human review.

Notebooks 01–03 form the course-aligned machine learning core. Notebook 04 and the planned Streamlit application are applied investigation extensions.

## Current results

Transactions are ordered by `TransactionDT`: the earliest 70% form training, the next 15% validation, and the latest 15% the final test period. Preprocessing produces 359 model features. An XGBoost model tuned with expanding-window cross-validation is the champion.

| Later-period test metric | Result |
|---|---:|
| Average precision | 0.5213 |
| ROC-AUC | 0.8996 |
| Precision | 0.3258 |
| Recall | 0.6160 |
| F2 score | 0.5228 |
| Alert rate | 6.58% |

The threshold was selected on validation data by maximizing F2. Model outputs are uncalibrated fraud risk scores, not fraud probabilities.

The investigation layer evaluates TF-IDF retrieval on 500 balanced held-out queries. Fraud queries retrieve fraud-labeled reference cases at 4.59 times the reference prevalence. Retrieval remains descriptive evidence and does not explain or prove an individual outcome.

## Workflow

| Notebook | Status |
|---|:---:|
| `01_EDA_Business_Understanding.ipynb` | Complete |
| `02_Data_Preprocessing_Feature_Engineering.ipynb` | Complete |
| `03_Model_Development_Comparison.ipynb` | Complete |
| `04_Fraud_Investigation_AI_Assistant.ipynb` | Complete |
| Fraud investigation platform (Streamlit) | Next |

## Repository structure

```text
├── app/                    # Streamlit application
├── data/                   # Raw and processed data (not tracked)
├── models/
│   ├── preprocessing/     # Frozen preprocessing artifacts and schema metadata
│   ├── trained/           # Champion model, metadata, and manifest
│   └── investigation/     # TF-IDF retrieval artifacts
├── notebooks/             # EDA, preprocessing, modeling, and investigation
├── reports/               # Generated investigation report
├── results/               # Model, cohort, retrieval, and assistant outputs
├── src/                   # Reusable raw-to-score pipeline
├── tests/                 # Pipeline integration tests
├── requirements.txt       # Core environment
├── requirements-llm.txt   # Optional model-generated summaries
├── requirements-app.txt   # Future Streamlit application
└── requirements-lock.txt  # Verified direct dependency versions
```

## Setup

```bash
git clone https://github.com/antonsoss/explainable-fraud-investigation-platform.git
cd explainable-fraud-investigation-platform
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the exact verified direct dependency versions, install `requirements-lock.txt`. On macOS, XGBoost also requires:

```bash
brew install libomp
```

Download the competition files from Kaggle, place them in `data/raw/`, and run the notebooks in numerical order.

## Reusable scoring contract

`src/fraud_pipeline.py` provides one interface that converts merged raw transaction and identity records into the frozen 359-feature schema and returns risk scores and alerts.

```python
from src.fraud_pipeline import FraudScoringPipeline

pipeline = FraudScoringPipeline.from_project_root(".")
results = pipeline.predict(raw_transactions)
```

Only load project-generated joblib files; joblib uses pickle internally and is unsafe for untrusted artifacts.

Run the integration tests with:

```bash
python -m unittest discover -s tests -v
```

## Optional investigation summaries

The core investigation workflow and deterministic summaries require no API key. To enable optional OpenAI-generated summaries:

```bash
python -m pip install -r requirements-llm.txt
export OPENAI_API_KEY="your-key"
```

Then set `RUN_OPENAI_LLM = True` in Notebook 04. No transaction evidence is sent unless that switch is enabled. Never commit API keys.

## Next phase

The final phase will integrate performance and alert-volume KPIs, transaction profiles, labeled reference cases, deterministic or optional generated summaries, cohort limitations, and human-review controls into a Streamlit application.
