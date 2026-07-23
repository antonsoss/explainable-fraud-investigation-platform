# Explainable Fraud Detection and Investigation Platform

An MIA 5100 machine learning project for detecting fraudulent transactions and supporting evidence-based human investigation.

## Overview

The project uses the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection) to compare classifiers under severe class imbalance. The workflow emphasizes chronological evaluation, training-only preprocessing, precision-recall analysis, operational threshold selection, post-hoc explanation, and human review.

### Course alignment

| Project component | Topics implemented | Related lecture material |
|---|---|---|
| `01_eda_and_business_understanding.ipynb` | Problem framing, exploratory analysis, data quality, class imbalance, and business context | **MIA 5100 Week 1:** Introduction to Machine Learning & Applications; **Week 2:** Machine Learning Workflow |
| `02_data_preparation.ipynb` | Data wrangling, missing-value treatment, categorical encoding, scaling, feature engineering, feature selection, and chronological splitting | **MIA 5100 Week 2:** Machine Learning Workflow; **Week 3:** Feature Engineering |
| `03_model_selection_and_evaluation.ipynb` | Parametric and non-parametric classifiers, ensemble models, neural networks, expanding-window cross-validation, hyperparameter tuning, threshold selection, and performance evaluation | **MIA 5100 Week 4:** Parametric & Non-parametric Methods; **Week 6:** Model Evaluation & Performance Improvement; **Week 7:** Introduction to Deep Learning |
| `04_post_hoc_explainability.ipynb` | Global and local explanations, SHAP, LIME, and explanation reliability checks applied after model selection | **MIA 5126 Data-Centric AI lecture:** Explainable AI (XAI), including global/local and post-hoc explanations |
| FastAPI backend and Streamlit frontend | Frozen-artifact loading, transaction scoring, model metrics, XAI evidence, HTTP/JSON integration, and an investigator-facing interface | **MIA 5100 Week 10:** Model Deployment |

## Current results

Transactions are ordered by `TransactionDT`. Training contains 413,378 transactions from elapsed days 1.0–120.8, validation contains 88,581 transactions from days 120.8–152.2, and the final test contains 88,581 transactions from days 152.2–183.0. Their fraud rates are 3.52%, 3.43%, and 3.48%, respectively. Data wrangling, preprocessing, and feature engineering produce 359 model features, and XGBoost tuned with expanding-window cross-validation is the champion.

| Later-period test metric | Result |
|---|---:|
| Average precision | 0.5213 |
| ROC-AUC | 0.8996 |
| Precision | 0.3258 |
| Recall | 0.6160 |
| F2 score | 0.5228 |
| Alert rate | 6.58% |

The validation average precision was 0.5904. Its decline to 0.5213 on the later test period indicates temporal generalization loss and is more conservative than the previous random-split estimate. The 0.2757 threshold was selected on validation data by maximizing F2. It generated 5,829 test alerts: 1,899 true positives and 3,930 false positives, while 1,184 fraud cases were missed. Model outputs are uncalibrated fraud risk scores, not fraud probabilities.

For the XAI extension, Tree SHAP was computed with a fixed 100-row training background and a fixed 1,000-row test reporting sample. `TransactionDT` ranked first by mean absolute SHAP, and the SHAP/native-gain rank correlation was 0.621. SHAP reconstructed model scores within a maximum global error of 3.83×10⁻⁷. LIME was evaluated on four representative test outcomes selected from the saved model predictions, but its mean local fidelity was only R²=0.188 and its median top-10 feature overlap with SHAP was 11.1%; the platform therefore presents LIME as secondary evidence only. These explanations describe model behavior, not causal reasons for fraud.

## Workflow

| Notebook | Status |
|---|:---:|
| `01_eda_and_business_understanding.ipynb` | Complete |
| `02_data_preparation.ipynb` | Complete |
| `03_model_selection_and_evaluation.ipynb` | Complete |
| `04_post_hoc_explainability.ipynb` | Complete |
| Fraud investigation API (FastAPI) | Complete |
| Fraud investigation frontend (Streamlit) | Complete |

## Repository structure

```text
├── api/                    # Standalone FastAPI backend and services
├── data/                   # Raw and processed data (not tracked)
├── models/
│   ├── preprocessing/     # Frozen preprocessing artifacts and schema metadata
│   └── trained/           # Champion model, metadata, and manifest
├── notebooks/             # Exploratory analysis, data preparation, modeling, and XAI
├── reports/               # Generated XAI report
├── results/               # Model-comparison and XAI outputs
├── frontend/              # Standalone Streamlit frontend and API client
├── src/                   # Reusable raw-to-score pipeline
├── tests/                 # Pipeline integration and API contract tests
└── pyproject.toml         # Project metadata, packaging, and dependencies
```

## Setup

```bash
git clone https://github.com/antonsoss/explainable-fraud-investigation-platform.git
cd explainable-fraud-investigation-platform
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[api,frontend,xai,notebook]"
```

The editable install keeps the Python packages connected to the repository-local
model, preprocessing, evaluation, and XAI artifacts used by the application.

On macOS, XGBoost also requires:

```bash
brew install libomp
```

Download the [IEEE-CIS Fraud Detection competition files](https://www.kaggle.com/competitions/ieee-fraud-detection) and place them in `data/raw/`. Run the notebooks in numerical order. `04_post_hoc_explainability.ipynb` produces the SHAP and LIME artifacts consumed by the application.

```bash
jupyter notebook notebooks/04_post_hoc_explainability.ipynb
```

The API and frontend consume the precomputed XAI files and do not import SHAP or LIME at runtime.

## Run the investigation platform

Start the FastAPI backend from the repository root:

```bash
uvicorn api.main:app --reload
```

The API documentation is available at `http://127.0.0.1:8000/docs`. In a second terminal, start the standalone frontend:

```bash
streamlit run frontend/streamlit_app.py
```

The frontend uses `http://127.0.0.1:8000/api/v1` by default. Set `FRAUD_API_URL` to point it at another API deployment. The backend can use `FRAUD_PROJECT_ROOT` when artifacts are mounted outside the repository and `FRAUD_ALLOWED_ORIGINS` to configure comma-separated browser origins.

## Deploy on Render

The repository includes a [`render.yaml`](render.yaml) Blueprint that creates two
Docker-based web services on Render.

**Live deployment:** [explainable-fraud-app.onrender.com](https://explainable-fraud-app.onrender.com)

| Service | Purpose | Health check |
|---|---|---|
| `explainable-fraud-api` | FastAPI preprocessing, scoring, metrics, and XAI artifact service | `/api/v1/health` |
| `explainable-fraud-app` | Streamlit investigator interface | `/_stcore/health` |

`Dockerfile.api` packages the API, scoring pipeline, and frozen artifacts, while
`Dockerfile.frontend` packages the Streamlit interface and downloadable example.
Both images use Python 3.14.3 and install only their service-specific dependency
groups. The Blueprint also copies the backend's Render-generated public URL into the frontend's
`FRAUD_API_URL`; `FraudApiClient` accepts either the service origin or the full
`/api/v1` URL.

To deploy:

1. Commit and push the repository to GitHub.
2. In Render, select **New → Blueprint**.
3. Connect `antonsoss/explainable-fraud-investigation-platform`.
4. Confirm the two services defined in `render.yaml` and apply the Blueprint.
5. Open the [deployed Streamlit application](https://explainable-fraud-app.onrender.com)
   after both health checks pass.

The Blueprint uses free instances to avoid an automatic charge. Render may spin
down a free service after 15 minutes without inbound traffic, so the first visit
can take approximately one minute to wake each service. If the API exceeds the
free instance's memory limit, upgrade only `explainable-fraud-api`.

To build and run the same containers locally:

```bash
docker build -f Dockerfile.api -t explainable-fraud-api .
docker run --rm -p 8000:10000 explainable-fraud-api
```

In a second terminal, point the frontend at the local API:

```bash
docker build -f Dockerfile.frontend -t explainable-fraud-app .
docker run --rm -p 8501:10000 \
  -e FRAUD_API_URL=http://host.docker.internal:8000 \
  explainable-fraud-app
```

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Check artifact readiness |
| `GET` | `/api/v1/model` | Read the frozen model contract |
| `GET` | `/api/v1/metrics` | Read persisted evaluation results |
| `GET` | `/api/v1/xai` | Read global XAI results and reliability diagnostics |
| `GET` | `/api/v1/xai/{transaction_id}` | Read a representative local SHAP/LIME explanation |
| `POST` | `/api/v1/predict` | Score merged raw transaction records |

The frontend does not import model code or load joblib artifacts. The API owns preprocessing, scoring, saved evaluation metrics, and XAI evidence.

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

## Application scope

The platform integrates saved performance and alert-volume KPIs, global SHAP analysis, representative transaction-level SHAP and LIME explanations, raw-transaction scoring, and human-review boundaries. It consumes frozen notebook artifacts and does not retrain the model or use the final test set for additional selection. SHAP is the primary local decomposition because additivity was verified; low-fidelity LIME results are visibly marked as secondary. Neither method establishes causality or supports autonomous enforcement.

## AI assistance disclosure

*I used AI as an engineering productivity tool for brainstorming, troubleshooting, and documentation, while remaining responsible for all technical decisions, implementation, testing, validation, and conclusions.*
