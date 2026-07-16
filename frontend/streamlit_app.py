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
def load_xai(api_url: str) -> dict:
    return FraudApiClient(api_url).xai()


@st.cache_data(ttl=3600, show_spinner=False)
def load_artifact(api_url: str, path: str) -> bytes:
    return FraudApiClient(api_url).artifact_bytes(path)


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


def render_explanations(api_url: str) -> None:
    st.title("Model explanations")
    st.caption(
        "Post-hoc analysis of the frozen tuned XGBoost model. Notebook 05 used the "
        "final test set for reporting only; it did not change the model or threshold."
    )

    xai = load_xai(api_url)
    shap_summary = xai["shap"]
    lime_summary = xai["lime"]

    st.subheader("Explanation diagnostics")
    primary_diagnostics = st.columns(2)
    primary_diagnostics[0].markdown(
        f"**Top global feature**  \n`{shap_summary['top_global_feature']}`"
    )
    primary_diagnostics[1].markdown(
        "**SHAP vs. gain rank correlation**  \n"
        f"`{shap_summary['native_gain_spearman_rank_correlation']:.3f}`"
    )
    reliability_diagnostics = st.columns(2)
    reliability_diagnostics[0].markdown(
        "**Maximum SHAP reconstruction error**  \n"
        f"`{shap_summary['maximum_global_reconstruction_error']:.2e}`"
    )
    reliability_diagnostics[1].markdown(
        "**Mean LIME local fidelity R²**  \n"
        f"`{lime_summary['mean_local_fidelity_r2']:.3f}`"
    )

    st.warning(
        "TransactionDT is elapsed time from an unspecified origin. Its prominence may "
        "reflect temporal drift or transaction ordering; it is not a calendar effect "
        "or a causal reason for fraud."
    )

    global_bar, beeswarm = st.columns(2)
    with global_bar:
        st.image(
            load_artifact(api_url, xai["figures"]["global_bar"]),
            caption="Global mean absolute SHAP magnitude",
            width="stretch",
        )
    with beeswarm:
        st.image(
            load_artifact(api_url, xai["figures"]["global_beeswarm"]),
            caption="Feature values and their signed score contributions",
            width="stretch",
        )

    st.subheader("Leading global features")
    importance = dataframe_from_records(xai["top_global_features"])
    importance = importance.rename(columns={
        "feature": "Feature",
        "mean_absolute_shap": "Mean |SHAP|",
        "mean_signed_shap": "Mean signed SHAP",
        "positive_contribution_share": "Positive share",
        "shap_rank": "SHAP rank",
        "gain_rank": "Gain rank",
        "feature_family": "Feature family",
    })
    st.dataframe(
        importance[[
            "SHAP rank",
            "Feature",
            "Feature family",
            "Mean |SHAP|",
            "Mean signed SHAP",
            "Positive share",
            "Gain rank",
        ]],
        hide_index=True,
        width="stretch",
    )

    comparison, dependence = st.columns(2)
    with comparison:
        st.subheader("SHAP and native gain")
        st.image(
            load_artifact(api_url, xai["figures"]["native_comparison"]),
            caption="Different importance definitions need not produce identical ranks",
            width="stretch",
        )
    with dependence:
        st.subheader("Feature contribution pattern")
        feature = st.selectbox(
            "Dependence feature", options=list(xai["figures"]["dependence"])
        )
        st.image(
            load_artifact(api_url, xai["figures"]["dependence"][feature]),
            caption=f"Processed {feature} value versus its SHAP contribution",
            width="stretch",
        )

    st.subheader("LIME reliability check")
    st.error(
        "LIME is secondary supporting evidence in this project. Its mean local "
        f"fidelity is R²={lime_summary['mean_local_fidelity_r2']:.3f}, so its sparse "
        "surrogate weights should not replace the verified SHAP decomposition."
    )
    fidelity = dataframe_from_records(xai["lime_case_fidelity"])
    fidelity = fidelity.rename(columns={
        "transaction_id": "Transaction ID",
        "lime_local_fidelity_r2": "Local fidelity R²",
        "lime_local_prediction": "LIME prediction",
        "model_risk_score": "Model risk score",
        "absolute_local_prediction_error": "Absolute error",
    })
    st.dataframe(fidelity, hide_index=True, width="stretch")
    st.caption(lime_summary["warning"])
    st.error(xai["required_boundary"])


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
    signal_columns = st.columns(2)
    signal_columns[0].markdown(
        "**Risk score**  \n"
        f"`{format_percent(signal['fraud_risk_score'], 2)}`"
    )
    signal_columns[1].markdown(f"**Model decision**  \n`{status}`")
    transaction_columns = st.columns(2)
    transaction_columns[0].markdown(
        f"**Amount**  \n`${transaction['transaction_amount']:,.2f}`"
    )
    transaction_columns[1].markdown(
        "**Amount percentile**  \n"
        f"`{format_percent(transaction['amount_percentile'])}`"
    )

    st.warning(signal["score_warning"])

    profile = st.container()
    evidence = st.container()
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

    st.subheader("Post-hoc model explanation")
    explanation = investigation["model_explanation"]
    explanation_metrics = st.columns(3)
    explanation_metrics[0].metric(
        "SHAP base risk score", format_percent(explanation["base_risk_score"], 2)
    )
    explanation_metrics[1].metric(
        "Model risk score", format_percent(explanation["model_risk_score"], 2)
    )
    explanation_metrics[2].metric(
        "Frozen threshold", format_percent(explanation["decision_threshold"], 2)
    )

    shap_tab, lime_tab = st.tabs(["SHAP — primary", "LIME — secondary"])
    with shap_tab:
        shap_evidence = explanation["shap"]
        st.image(
            load_artifact(api_url, shap_evidence["figure"]),
            caption="Local score decomposition for this transaction",
            width="stretch",
        )
        contributions = dataframe_from_records(shap_evidence["contributions"])
        contributions = contributions.rename(columns={
            "rank": "Rank",
            "feature": "Feature",
            "processed_feature_value": "Processed value",
            "shap_contribution": "SHAP contribution",
            "direction": "Direction",
        })
        st.dataframe(
            contributions[[
                "Rank",
                "Feature",
                "Processed value",
                "SHAP contribution",
                "Direction",
            ]],
            hide_index=True,
            width="stretch",
        )
        st.info(shap_evidence["interpretation"])

    with lime_tab:
        lime_evidence = explanation["lime"]
        st.error(
            f"Secondary evidence only · local fidelity R²="
            f"{lime_evidence['local_fidelity_r2']:.3f} · "
            f"absolute prediction error={lime_evidence['absolute_prediction_error']:.3f}."
        )
        st.image(
            load_artifact(api_url, lime_evidence["figure"]),
            caption="Sparse LIME surrogate around this transaction",
            width="stretch",
        )
        contributions = dataframe_from_records(lime_evidence["contributions"])
        contributions = contributions.rename(columns={
            "rank": "Rank",
            "feature": "Feature",
            "condition": "Local condition",
            "lime_weight": "LIME weight",
            "direction": "Direction",
        })
        st.dataframe(
            contributions[[
                "Rank",
                "Feature",
                "Local condition",
                "LIME weight",
                "Direction",
            ]],
            hide_index=True,
            width="stretch",
        )
        st.caption(lime_evidence["warning"])

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
        "Notebook 05 adds the MIA 5126 post-hoc SHAP and LIME extension without "
        "changing the model. The API owns all artifacts; the frontend is "
        "presentation-only."
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
- SHAP and LIME describe learned model behavior; neither establishes causality.
- TransactionDT importance may reflect temporal drift or transaction ordering.
- LIME has low local fidelity for these cases and is secondary evidence only.
- Similar-case retrieval depends on selected tokens and is contextual only.
- Missing attributes and cohort-level recall gaps require monitoring.
- The assistant cannot establish fraud or make an autonomous enforcement decision.
- A human investigator must verify evidence and follow approved policy.
"""
    )


api_url = st.sidebar.text_input("Fraud API URL", value=DEFAULT_API_URL)
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model explanations",
        "Investigation",
        "Score transactions",
        "Methodology",
    ],
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
    elif page == "Model explanations":
        render_explanations(api_url)
    elif page == "Investigation":
        render_investigation(api_url)
    elif page == "Score transactions":
        render_scoring(api_url)
    else:
        render_methodology(api_url)
except FraudApiError as error:
    st.error(str(error))
