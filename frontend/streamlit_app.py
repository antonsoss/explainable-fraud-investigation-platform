"""Streamlit presentation layer for the fraud investigation API."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.api_client import FraudApiClient, FraudApiError


DEFAULT_API_URL = os.getenv(
    "FRAUD_API_URL", "http://127.0.0.1:8000/api/v1"
)

st.set_page_config(
    page_title="Fraud Investigation Platform",
    page_icon="🔎",
    layout="wide",
)


def format_percent(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}%}"


def dataframe_from_records(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


@st.cache_data(ttl=30, show_spinner=False)
def load_health(api_url: str) -> dict:
    return FraudApiClient(api_url).health()


@st.cache_data(ttl=300, show_spinner=False)
def load_metrics(api_url: str) -> dict:
    return FraudApiClient(api_url).metrics()


@st.cache_data(ttl=300, show_spinner=False)
def load_demo_cases(api_url: str) -> list[dict]:
    return FraudApiClient(api_url).demo_cases()


@st.cache_data(ttl=300, show_spinner=False)
def load_investigation(api_url: str, transaction_id: int) -> dict:
    return FraudApiClient(api_url).investigate(transaction_id)


def render_overview(api_url: str) -> None:
    st.title("Fraud investigation overview")
    st.caption(
        "Frozen later-period results from the tuned XGBoost model. "
        "The application does not retrain or retune the model."
    )

    metrics = load_metrics(api_url)
    model = metrics["model"]
    test = model["test_metrics"]

    st.subheader("Model and operating point")
    st.markdown(f"**Champion model:** {model['model_name']}")
    model_columns = st.columns(3)
    model_columns[0].metric("Test average precision", f"{test['Average precision']:.4f}")
    model_columns[1].metric("Test ROC-AUC", f"{test['ROC-AUC']:.4f}")
    model_columns[2].metric("Frozen threshold", f"{model['decision_threshold']:.4f}")

    performance_columns = st.columns(4)
    performance_columns[0].metric("Precision", format_percent(test["Precision"]))
    performance_columns[1].metric("Recall", format_percent(test["Recall"]))
    performance_columns[2].metric("F2", f"{test['F2']:.4f}")
    performance_columns[3].metric("Alert rate", format_percent(test["Alert rate"]))

    st.info(
        f"The model generated {int(test['Alerts']):,} alerts: "
        f"{int(test['TP']):,} true positives and {int(test['FP']):,} false positives. "
        "Risk scores rank cases; they are not calibrated fraud probabilities."
    )

    st.subheader("Test outcomes")
    outcomes = dataframe_from_records(metrics["test_outcomes"])
    outcomes["Share"] = outcomes["Share"].map(lambda value: f"{value:.2%}")
    st.dataframe(outcomes, hide_index=True, width="stretch")

    st.subheader("Cohort monitoring")
    cohort_choice = st.radio(
        "Cohort view", ["Product", "Device", "Amount"], horizontal=True
    )
    cohort_key = cohort_choice.lower()
    cohort = dataframe_from_records(metrics["cohorts"][cohort_key])
    st.dataframe(cohort, hide_index=True, width="stretch")

    if cohort_choice == "Product" and not cohort.empty:
        chart = cohort.set_index("ProductCD")[["Precision", "Recall"]]
        st.bar_chart(chart)
    elif cohort_choice == "Device" and not cohort.empty:
        chart = cohort.set_index("DeviceType")[["Precision", "Recall"]]
        st.bar_chart(chart)

    retrieval = metrics["retrieval"]
    st.subheader("Similar-case retrieval diagnostic")
    retrieval_columns = st.columns(3)
    retrieval_columns[0].metric(
        "Reference fraud rate", format_percent(retrieval["Reference fraud rate"])
    )
    retrieval_columns[1].metric(
        "Fraud-query neighbor rate",
        format_percent(retrieval["Fraud-query neighbor fraud rate"]),
    )
    retrieval_columns[2].metric(
        "Lift vs. reference", f"{retrieval['Fraud-query lift vs reference']:.2f}×"
    )
    st.caption(
        "This retrospective retrieval result is descriptive evidence and does not "
        "convert neighbor outcomes into a fraud probability."
    )


def render_investigation(api_url: str) -> None:
    st.title("Transaction investigation")
    st.caption(
        "Review a curated retrospective transaction exported by Notebook 04. "
        "Individual ground truth is intentionally withheld from the assistant view."
    )

    cases = load_demo_cases(api_url)
    if not cases:
        st.warning("No demonstration transactions are available.")
        return

    case_by_id = {int(case["transaction_id"]): case for case in cases}
    transaction_id = st.selectbox(
        "Transaction",
        options=list(case_by_id),
        format_func=lambda value: (
            f"{value} — ${case_by_id[value]['transaction_amount']:,.2f} — "
            f"Product {case_by_id[value]['product_code']}"
        ),
    )

    investigation = load_investigation(api_url, int(transaction_id))
    transaction = investigation["transaction"]
    signal = investigation["model_signal"]

    status = "Alert generated" if signal["alert_generated"] else "No alert"
    columns = st.columns(4)
    columns[0].metric("Risk score", format_percent(signal["fraud_risk_score"], 2))
    columns[1].metric("Model decision", status)
    columns[2].metric("Amount", f"${transaction['transaction_amount']:,.2f}")
    columns[3].metric("Amount percentile", format_percent(transaction["amount_percentile"]))

    st.warning(signal["score_warning"])

    profile, evidence = st.columns([1, 2])
    with profile:
        st.subheader("Transaction profile")
        profile_rows = [
            {"Field": "Transaction ID", "Value": transaction["transaction_id"]},
            {"Field": "Product", "Value": transaction["product_code"]},
            {"Field": "Card network", "Value": transaction["card_network"]},
            {"Field": "Card type", "Value": transaction["card_type"]},
            {"Field": "Device", "Value": transaction["device_type"]},
            {
                "Field": "Purchase email domain",
                "Value": transaction["purchase_email_domain"],
            },
            {
                "Field": "Elapsed dataset day",
                "Value": f"{transaction['elapsed_dataset_day']:.1f}",
            },
        ]
        profile_frame = pd.DataFrame.from_records(profile_rows)
        profile_frame["Value"] = profile_frame["Value"].map(str)
        st.dataframe(profile_frame, hide_index=True, width="stretch")

    with evidence:
        st.subheader("Similar labeled reference cases")
        neighbors = dataframe_from_records(investigation["similar_reference_cases"])
        display_columns = [
            "rank",
            "transaction_id",
            "cosine_similarity",
            "reference_fraud_label",
            "reference_model_risk_score",
            "transaction_amount",
            "product_code",
            "device_type",
        ]
        st.dataframe(
            neighbors[display_columns], hide_index=True, width="stretch"
        )
        st.caption(
            "Reference labels are historical context. Similarity does not prove that "
            "the query transaction has the same outcome."
        )

    st.subheader("Investigation assistant")
    assistant = investigation["assistant_summary"]
    st.caption(
        f"Provider: {assistant['provider']} · Automated grounding checks: "
        f"{'passed' if assistant['automated_checks_pass'] else 'failed'}"
    )
    st.markdown(assistant["markdown"])
    st.error(investigation["required_boundary"])


def render_scoring(api_url: str) -> None:
    st.title("Score merged raw transactions")
    st.caption(
        "Upload up to 100 merged IEEE-CIS transaction and identity records. "
        "The CSV must contain the same raw schema used by Notebook 02."
    )
    uploaded = st.file_uploader("Merged transaction CSV", type=["csv"])
    if uploaded is None:
        st.info(
            "This page sends records to the API. The Streamlit frontend never loads "
            "the preprocessing pipeline or model artifacts."
        )
        return

    frame = pd.read_csv(uploaded)
    st.write(f"Rows: {len(frame):,} · Columns: {len(frame.columns):,}")
    st.dataframe(frame.head(10), hide_index=True, width="stretch")

    if len(frame) > 100:
        st.error("The API accepts at most 100 transactions per request.")
        return

    if st.button("Score transactions", type="primary"):
        records = json.loads(frame.to_json(orient="records"))
        with st.spinner("Scoring with the frozen model..."):
            result = FraudApiClient(api_url).predict(records)
        predictions = dataframe_from_records(result["predictions"])
        st.success(f"Scored {len(predictions):,} transaction(s).")
        st.dataframe(predictions, hide_index=True, width="stretch")
        st.caption(result["score_semantics"])


def render_methodology(api_url: str) -> None:
    st.title("Methodology and responsible use")
    model = FraudApiClient(api_url).model()

    st.subheader("System boundary")
    st.code(
        "Notebooks → frozen artifacts → FastAPI backend → HTTP/JSON → Streamlit frontend",
        language=None,
    )
    st.write(
        "Notebooks 01–03 provide the course-aligned machine learning workflow. "
        "Notebook 04 exports retrieval evidence and grounded assistant summaries. "
        "The API owns all artifacts; the frontend is presentation-only."
    )

    st.subheader("Frozen model contract")
    st.json({
        "model": model["model_name"],
        "features": model["feature_count"],
        "split policy": model["split_policy"],
        "threshold": model["decision_threshold"],
        "threshold rule": model["threshold_rule"],
        "score semantics": model["score_semantics"],
    })

    st.subheader("Limitations")
    st.markdown(
        """
- The data has an unspecified time origin and anonymized fields.
- Later-period results show temporal generalization loss.
- Risk scores are uncalibrated ranking scores, not fraud probabilities.
- Global model importance does not explain an individual prediction.
- Similar-case retrieval depends on selected tokens and is contextual only.
- Missing attributes and cohort-level recall gaps require monitoring.
- The assistant cannot establish fraud or make an autonomous enforcement decision.
- A human investigator must verify evidence and follow approved policy.
"""
    )


api_url = st.sidebar.text_input("Fraud API URL", value=DEFAULT_API_URL)
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Investigation", "Score transactions", "Methodology"],
)

try:
    health = load_health(api_url)
except FraudApiError as error:
    st.sidebar.error("API unavailable")
    st.error(str(error))
    st.code("uvicorn api.main:app --reload", language="bash")
    st.stop()

if health["artifacts_ready"]:
    st.sidebar.success("API ready")
else:
    st.sidebar.error("API artifacts missing")
    st.sidebar.write(health["missing_artifacts"])

try:
    if page == "Overview":
        render_overview(api_url)
    elif page == "Investigation":
        render_investigation(api_url)
    elif page == "Score transactions":
        render_scoring(api_url)
    else:
        render_methodology(api_url)
except FraudApiError as error:
    st.error(str(error))
