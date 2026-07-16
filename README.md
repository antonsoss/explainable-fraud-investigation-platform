# Explainable Fraud Investigation Platform

An MIA 5100 machine learning project for detecting fraudulent transactions and supporting evidence-based human investigation.

## Overview

The project uses the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) to compare classifiers under severe class imbalance. The workflow emphasizes chronological evaluation, training-only preprocessing, precision-recall analysis, operational threshold selection, model-native global interpretation, labeled reference-case retrieval, and human review.

Notebooks 01–03 form the MIA 5100 machine learning core. Notebook 04 and the implemented application layer are applied investigation extensions. Notebook 05 adds post-hoc SHAP and LIME analysis aligned with the XAI material from MIA 5126; it does not alter the frozen model or threshold. The final platform uses a standalone FastAPI backend and a Streamlit frontend connected only through HTTP/JSON.

## Current results

Transactions are ordered by `TransactionDT`. Training contains 413,378 transactions from elapsed days 1.0–120.8, validation contains 88,581 transactions from days 120.8–152.2, and the final test contains 88,581 transactions from days 152.2–183.0. Their fraud rates are 3.52%, 3.43%, and 3.48%, respectively. Preprocessing produces 359 model features, and XGBoost tuned with expanding-window cross-validation is the champion.

| Later-period test metric | Result |
|---|---:|
| Average precision | 0.5213 |
| ROC-AUC | 0.8996 |
| Precision | 0.3258 |
| Recall | 0.6160 |
| F2 score | 0.5228 |
| Alert rate | 6.58% |

The validation average precision was 0.5904. Its decline to 0.5213 on the later test period indicates temporal generalization loss and is more conservative than the previous random-split estimate. The 0.2757 threshold was selected on validation data by maximizing F2. It generated 5,829 test alerts: 1,899 true positives and 3,930 false positives, while 1,184 fraud cases were missed. Model outputs are uncalibrated fraud risk scores, not fraud probabilities.

The investigation layer evaluates TF-IDF retrieval on 500 balanced held-out queries. Fraud queries retrieved 15.8% fraud-labeled neighbors versus 3.4% reference prevalence, a 4.59× lift. Retrieval remains descriptive evidence and does not explain or prove an individual outcome.

For the XAI extension, Tree SHAP was computed with a fixed 100-row training background and a fixed 1,000-row test reporting sample. `TransactionDT` ranked first by mean absolute SHAP, and the SHAP/native-gain rank correlation was 0.621. SHAP reconstructed model scores within a maximum global error of 3.83×10⁻⁷. LIME was evaluated on the same four representative cases, but its mean local fidelity was only R²=0.188 and its median top-10 feature overlap with SHAP was 11.1%; the platform therefore presents LIME as secondary evidence only. These explanations describe model behavior, not causal reasons for fraud.

## Workflow

| Notebook | Status |
|---|:---:|
| `01_EDA_Business_Understanding.ipynb` | Complete |
| `02_Data_Preprocessing_Feature_Engineering.ipynb` | Complete |
| `03_Model_Development_Comparison.ipynb` | Complete |
| `04_Fraud_Investigation_AI_Assistant.ipynb` | Complete |
| `05_XAI_Model_Explanations.ipynb` | Complete |
| Fraud investigation API (FastAPI) | Complete |
| Fraud investigation frontend (Streamlit) | Complete |

## Repository structure

```text
├── api/                    # Standalone FastAPI backend and services
├── data/                   # Raw and processed data (not tracked)
├── models/
│   ├── preprocessing/     # Frozen preprocessing artifacts and schema metadata
│   ├── trained/           # Champion model, metadata, and manifest
│   └── investigation/     # TF-IDF retrieval artifacts
├── notebooks/             # EDA, preprocessing, modeling, investigation, and XAI
├── reports/               # Generated investigation and XAI reports
├── results/               # Model, retrieval, assistant, and XAI outputs
├── frontend/              # Standalone Streamlit frontend and API client
├── src/                   # Reusable raw-to-score pipeline
├── tests/                 # Pipeline integration and API contract tests
├── requirements.txt       # Core environment
├── requirements-api.txt   # Backend environment
├── requirements-frontend.txt # Frontend environment
├── requirements-llm.txt   # Optional model-generated summaries
├── requirements-xai.txt   # Notebook-only SHAP and LIME extension
├── requirements-app.txt   # Complete local application environment
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

Download the competition files from Kaggle and place them in `data/raw/`. Run Notebooks 01–04 in numerical order. To reproduce Notebook 05 and its exported explanation artifacts, install its separate environment additions and execute it last:

```bash
python -m pip install -r requirements-xai.txt
jupyter notebook notebooks/05_XAI_Model_Explanations.ipynb
```

The API and frontend consume the precomputed XAI files and do not import SHAP or LIME at runtime.

## Run the investigation platform

Install the complete local application environment:

```bash
python -m pip install -r requirements-app.txt
```

Start the FastAPI backend from the repository root:

```bash
uvicorn api.main:app --reload
```

The API documentation is available at `http://127.0.0.1:8000/docs`. In a second terminal, start the standalone frontend:

```bash
streamlit run frontend/streamlit_app.py
```

The frontend uses `http://127.0.0.1:8000/api/v1` by default. Set `FRAUD_API_URL` to point it at another API deployment. The backend can use `FRAUD_PROJECT_ROOT` when artifacts are mounted outside the repository and `FRAUD_ALLOWED_ORIGINS` to configure comma-separated browser origins.

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Check artifact readiness |
| `GET` | `/api/v1/model` | Read the frozen model contract |
| `GET` | `/api/v1/metrics` | Read persisted evaluation results |
| `GET` | `/api/v1/xai` | Read global XAI results and reliability diagnostics |
| `POST` | `/api/v1/predict` | Score merged raw transaction records |
| `GET` | `/api/v1/investigations` | List sanitized demonstration cases |
| `POST` | `/api/v1/similar-transactions` | Retrieve labeled validation references |
| `POST` | `/api/v1/investigate` | Return evidence and a grounded summary |

The frontend does not import model code or load joblib artifacts. The API owns preprocessing, scoring, retrieval, explanation, and assistant evidence. Demonstration investigations intentionally withhold the query transaction's individual ground-truth label from both the assistant and explanation responses.

For the **Score merged raw transactions** page, upload `examples/sample_merged_transactions.csv`. It contains three synthetic records with the complete 433-column merged raw schema and no ground-truth label. Regenerate it after a schema change with:

```bash
python scripts/create_example_score_csv.py
```

## Reusable scoring contract

`src/fraud_pipeline.py` provides one interface that converts merged raw transaction and identity records into the frozen 359-feature schema and returns risk scores and alerts.

```python
from src.fraud_pipeline import FraudScoringPipeline

pipeline = FraudScoringPipeline.from_project_root(".")
results = pipeline.predict(raw_transactions)
```

Only load project-generated joblib files; joblib uses pickle internally and is unsafe for untrusted artifacts.

Run the pipeline and API contract tests with:

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

## Application scope

The final platform integrates saved performance and alert-volume KPIs, global SHAP analysis, transaction-level SHAP and LIME explanations, transaction profiles, labeled reference cases, deterministic summaries, cohort limitations, and human-review controls. It consumes frozen notebook artifacts and does not retrain the model or use the final test set for additional selection. SHAP is the primary local decomposition because additivity was verified; low-fidelity LIME results are visibly marked as secondary. Neither method establishes causality or supports autonomous enforcement.

## AI assistance disclosure

*I used AI as an engineering productivity tool for brainstorming, troubleshooting, and documentation, while remaining responsible for all technical decisions, implementation, testing, validation, and conclusions.*
