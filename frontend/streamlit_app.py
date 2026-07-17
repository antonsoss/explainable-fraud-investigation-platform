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
EXAMPLE_CSV_PATH = PROJECT_ROOT / "examples" / "sample_merged_transactions.csv"

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
def load_xai(api_url: str) -> dict:
    return FraudApiClient(api_url).xai()


@st.cache_data(ttl=300, show_spinner=False)
def load_local_xai(api_url: str, transaction_id: int) -> dict:
    return FraudApiClient(api_url).local_xai(transaction_id)


@st.cache_data(ttl=3600, show_spinner=False)
def load_artifact(api_url: str, path: str) -> bytes:
    return FraudApiClient(api_url).artifact_bytes(path)


def render_overview(api_url: str) -> None:
    st.title("Overview")
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

    st.subheader("Validation model comparison")
    comparison = dataframe_from_records(metrics["validation_model_comparison"])
    comparison = comparison.sort_values("Average precision", ascending=False)
    st.dataframe(
        comparison[[
            "Model",
            "Average precision",
            "ROC-AUC",
            "Precision",
            "Recall",
            "F2",
            "Alert rate",
        ]],
        hide_index=True,
        width="stretch",
    )
    st.bar_chart(comparison.set_index("Model")["Average precision"])


def render_explanations(api_url: str) -> None:
    st.title("Explainable AI (XAI)")
    st.caption(
        "SHAP and LIME explain how the frozen XGBoost model produces fraud-risk "
        "scores globally and for representative transactions. These explanations "
        "describe model behavior; they do not establish causality."
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

    st.subheader("Top Global Features by Mean Absolute SHAP")
    st.caption(
        "Features are ranked by their average absolute contribution to the model's "
        "fraud-risk score across the fixed 1,000-transaction reporting sample."
    )
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

    st.subheader("Individual Explanations for Selected Test Transactions")
    st.caption(
        "Four transactions were selected: one near the median model score in each "
        "confusion-matrix group (true positive, false positive, false negative, and "
        "true negative). Select a transaction to compare its SHAP and LIME evidence; "
        "the app does not display its ground-truth outcome."
    )
    transaction_ids = fidelity["Transaction ID"].astype("int64").tolist()
    transaction_id = st.selectbox(
        "Representative transaction", options=transaction_ids
    )
    local = load_local_xai(api_url, int(transaction_id))

    local_metrics = st.columns(3)
    local_metrics[0].metric(
        "SHAP base risk score", f"{local['base_risk_score']:.4f}"
    )
    local_metrics[1].metric(
        "Model risk score", f"{local['model_risk_score']:.4f}"
    )
    local_metrics[2].metric(
        "Frozen threshold", f"{local['decision_threshold']:.4f}"
    )

    shap_tab, lime_tab = st.tabs(["SHAP — primary", "LIME — secondary"])
    with shap_tab:
        shap_evidence = local["shap"]
        st.image(
            load_artifact(api_url, shap_evidence["figure"]),
            caption="Local decomposition of the frozen model score",
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
        lime_evidence = local["lime"]
        st.error(
            "Secondary evidence only · "
            f"local fidelity R²={lime_evidence['local_fidelity_r2']:.3f} · "
            f"absolute prediction error={lime_evidence['absolute_prediction_error']:.3f}."
        )
        st.image(
            load_artifact(api_url, lime_evidence["figure"]),
            caption=(
                "Top feature weights from LIME's simplified approximation of the "
                "XGBoost score for this transaction"
            ),
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

    st.error(xai["required_boundary"])


def render_scoring(api_url: str) -> None:
    st.title("Fraud Risk Scoring")
    st.caption(
        "Upload up to 100 merged transaction and identity records from the "
        "[IEEE-CIS Fraud Detection dataset]"
        "(https://www.kaggle.com/competitions/ieee-fraud-detection). "
        "The CSV must contain the same merged raw schema used during Data "
        "Wrangling, Preprocessing & Feature Engineering."
    )
    if EXAMPLE_CSV_PATH.exists():
        st.download_button(
            "Download example CSV",
            data=EXAMPLE_CSV_PATH.read_bytes(),
            file_name=EXAMPLE_CSV_PATH.name,
            mime="text/csv",
            help="Download a correctly structured sample file for fraud scoring.",
        )
    else:
        st.warning("The example scoring CSV is currently unavailable.")

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


def render_about(api_url: str) -> None:
    st.title("About")
    st.markdown("## Explainable Fraud Detection and Investigation Platform")
    identity_columns = st.columns(3)
    identity_columns[0].markdown("**Author**  \nAntonio Sosa")
    identity_columns[1].markdown("**University**  \nUniversity of Ottawa")
    identity_columns[2].markdown(
        "**Course**  \nMIA 5100 — Foundations and Applications of Machine Learning"
    )

    st.markdown(
        "This project uses the [IEEE-CIS Fraud Detection dataset]"
        "(https://www.kaggle.com/competitions/ieee-fraud-detection) to develop and "
        "compare machine-learning models for fraud detection. The selected XGBoost "
        "model scores transaction risk, while SHAP and LIME help explain its "
        "behavior. The platform is designed to support, not replace, human review."
    )
    model = FraudApiClient(api_url).model()

    st.subheader("System Architecture")
    architecture_diagram = r"""
        digraph FraudPlatform {
            rankdir=TB;
            graph [
                bgcolor="transparent",
                pad="0.25",
                nodesep="0.65",
                ranksep="0.55",
                fontname="Arial"
            ];
            node [
                shape=box,
                style="rounded,filled",
                color="#4B5563",
                fillcolor="#262730",
                fontcolor="white",
                fontname="Arial",
                fontsize=11,
                margin="0.18,0.12"
            ];
            edge [
                color="#9CA3AF",
                fontcolor="#D1D5DB",
                fontname="Arial",
                fontsize=9,
                arrowsize=0.7
            ];

            subgraph cluster_offline {
                label="OFFLINE MODEL DEVELOPMENT";
                color="#6B7280";
                fontcolor="#9CA3AF";
                style="rounded,dashed";

                raw [
                    label="IEEE-CIS raw dataset",
                    URL="https://www.kaggle.com/competitions/ieee-fraud-detection",
                    target="_blank",
                    tooltip="Open the IEEE-CIS Fraud Detection dataset on Kaggle",
                    shape=note,
                    style="filled",
                    color="#3B82F6",
                    fillcolor="#1F4E79",
                    penwidth=3
                ];
                eda [
                    label="01_Exploratory_Data_Analysis_\nBusiness_Understanding.ipynb",
                    shape=note,
                    style="filled",
                    color="#F37626",
                    fillcolor="#30323D",
                    penwidth=3
                ];
                preparation [
                    label="02_Data_Wrangling_Preprocessing_\nFeature_Engineering.ipynb",
                    shape=note,
                    style="filled",
                    color="#F37626",
                    fillcolor="#30323D",
                    penwidth=3
                ];
                modeling [
                    label="03_ML_Model_Selection_\nTuning_Evaluation.ipynb",
                    shape=note,
                    style="filled",
                    color="#F37626",
                    fillcolor="#30323D",
                    penwidth=3
                ];
                xai [
                    label="04_Post_Hoc_XAI_SHAP_LIME.ipynb",
                    shape=note,
                    style="filled",
                    color="#F37626",
                    fillcolor="#30323D",
                    penwidth=3
                ];
                preprocessing_artifacts [
                    label="PREPROCESSING ARTIFACTS\nmedian_imputer.pkl • mode_imputer.pkl\nfrequency_maps.pkl • onehot_encoder.pkl\nrobust_scaler.pkl • selected_features.pkl\n+ schema and feature-list .pkl files",
                    fillcolor="#4C1D95"
                ];
                model_artifacts [
                    label="CHAMPION MODEL + METRICS\nchampion_manifest.json\nxgboost_tuned.joblib\nxgboost_tuned_metadata.json\nvalidation_model_comparison.csv",
                    fillcolor="#5B2C6F"
                ];
                xai_artifacts [
                    label="XAI ARTIFACTS\nxai_metadata.json • shap_global_importance.csv\nlocal_shap_values.parquet • local_lime_values.parquet\nlime_case_fidelity.csv • shap_lime_comparison.csv\nSHAP/LIME figure PNGs",
                    fillcolor="#6B21A8"
                ];

                raw -> eda;
                eda -> preparation;
                preparation -> modeling;
                modeling -> xai;
                preparation -> preprocessing_artifacts [label="fitted preprocessing"];
                modeling -> model_artifacts [label="champion model + metrics"];
                xai -> xai_artifacts [label="SHAP + LIME"];

                {
                    rank=same;
                    preprocessing_artifacts;
                    model_artifacts;
                    xai_artifacts;
                }
            }

            subgraph cluster_runtime {
                label="RUN-TIME APPLICATION";
                color="#6B7280";
                fontcolor="#9CA3AF";
                style="rounded,dashed";

                upload [
                    label="CSV\nCSV of raw transaction(s)\nfor fraud scoring",
                    shape=note,
                    style="filled",
                    color="#16A34A",
                    fillcolor="#14532D",
                    penwidth=3
                ];
                api [
                    label="FastAPI backend\nPreprocessing • scoring • metrics • XAI",
                    fillcolor="#7F1D1D"
                ];
                app [
                    label="Streamlit frontend\nOverview • risk scoring • XAI • About",
                    fillcolor="#1F4E79"
                ];
                reviewer [
                    label="Human reviewer\nInterprets scores and evidence",
                    fillcolor="#14532D"
                ];

                upload -> app [label="score new data"];
                api -> app [
                    dir=both,
                    label="HTTP/JSON\nrequests + responses"
                ];
                app -> reviewer [label="decision support"];
            }

            preprocessing_artifacts -> api [label="preprocessing"];
            model_artifacts -> api [label="scoring + metrics"];
            xai_artifacts -> api [label="explanations"];
        }
    """
    st.graphviz_chart(
        architecture_diagram,
        width="stretch",
        height=900,
    )
    st.caption(
        "Orange-bordered boxes represent offline analytical stages. Model development and "
        "XAI run offline. The deployed application never retrains the model: "
        "FastAPI loads the frozen preprocessing, champion-model, and XAI artifact "
        "categories, while Streamlit communicates with it only "
        "through HTTP/JSON. The CSV upload is optional and is used only to score new "
        "transactions; Overview and XAI use saved artifacts."
    )

    st.subheader("Deployed trained ML model")
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
- Missing attributes and temporal performance drift require monitoring.
- Model scores and post-hoc explanations cannot establish fraud or make an autonomous enforcement decision.
- A human reviewer must verify evidence and follow approved policy.
"""
    )

    st.subheader("AI assistance disclosure")
    st.info(
        "I used AI as an engineering productivity tool for brainstorming, "
        "troubleshooting, and documentation, while remaining responsible for all "
        "technical decisions, implementation, testing, validation, and conclusions."
    )


st.sidebar.markdown("### System status")
with st.sidebar.expander("API connection", expanded=False):
    api_url = st.text_input("Fraud API URL", value=DEFAULT_API_URL)

try:
    health = load_health(api_url)
except FraudApiError as error:
    st.sidebar.error("API unavailable")
    st.error(str(error))
    st.code("uvicorn api.main:app --reload", language="bash")
    st.stop()

if health["artifacts_ready"]:
    st.sidebar.markdown("🟢 **API ready**")
else:
    st.sidebar.markdown("🔴 **API artifacts missing**")
    with st.sidebar.expander("Missing artifacts"):
        st.write(health["missing_artifacts"])


def overview_page() -> None:
    render_overview(api_url)


def scoring_page() -> None:
    render_scoring(api_url)


def xai_page() -> None:
    render_explanations(api_url)


def about_page() -> None:
    render_about(api_url)


pages = [
    st.Page(
        overview_page,
        title="Overview",
        icon=":material/dashboard:",
        url_path="overview",
        default=True,
    ),
    st.Page(
        scoring_page,
        title="Fraud Risk Scoring",
        icon=":material/security:",
        url_path="fraud-risk-scoring",
    ),
    st.Page(
        xai_page,
        title="XAI",
        icon=":material/psychology:",
        url_path="xai",
    ),
    st.Page(
        about_page,
        title="About",
        icon=":material/info:",
        url_path="about",
    ),
]

try:
    st.navigation(pages, position="sidebar", expanded=True).run()
except FraudApiError as error:
    st.error(str(error))
