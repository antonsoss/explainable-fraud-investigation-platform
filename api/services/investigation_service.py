"""Similar-case retrieval and grounded investigation-summary service."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import linear_kernel

from api.utils import dataframe_records, json_safe


class InvestigationService:
    """Serve the frozen Notebook 04 investigation artifacts without label leakage."""

    def __init__(self, project_root: Path):
        self.model_dir = project_root / "models" / "investigation"
        self.results_dir = project_root / "results" / "investigation"

    @cached_property
    def selected_cases(self) -> pd.DataFrame:
        return pd.read_parquet(
            self.results_dir / "selected_investigation_cases.parquet"
        )

    @cached_property
    def summaries(self) -> pd.DataFrame:
        return pd.read_parquet(self.results_dir / "assistant_summaries.parquet")

    @cached_property
    def reference_cases(self) -> pd.DataFrame:
        return pd.read_parquet(
            self.model_dir / "validation_case_library.parquet"
        )

    @cached_property
    def vectorizer(self):
        return joblib.load(self.model_dir / "tfidf_vectorizer.joblib")

    @cached_property
    def reference_matrix(self):
        return load_npz(self.model_dir / "validation_case_matrix.npz")

    def _case(self, transaction_id: int) -> pd.Series:
        match = self.selected_cases.loc[
            self.selected_cases["TransactionID"] == int(transaction_id)
        ]
        if match.empty:
            raise KeyError(
                f"Demo transaction {transaction_id} was not exported by Notebook 04."
            )
        return match.iloc[0]

    def list_demo_cases(self) -> list[dict[str, Any]]:
        """List sanitized retrospective examples without individual ground truth."""
        columns = [
            "TransactionID",
            "fraud_risk_score",
            "predicted_fraud",
            "TransactionAmt",
            "ProductCD",
            "card4",
            "card6",
            "DeviceType",
            "P_emaildomain",
            "TransactionDay",
        ]
        cases = self.selected_cases[columns].rename(columns={
            "TransactionID": "transaction_id",
            "predicted_fraud": "alert_generated",
            "TransactionAmt": "transaction_amount",
            "ProductCD": "product_code",
            "DeviceType": "device_type",
            "P_emaildomain": "purchase_email_domain",
            "TransactionDay": "elapsed_dataset_day",
        })
        cases["alert_generated"] = cases["alert_generated"].astype(bool)
        return dataframe_records(cases)

    def similar_cases(self, transaction_id: int, top_n: int = 5) -> list[dict[str, Any]]:
        case = self._case(transaction_id)
        query_vector = self.vectorizer.transform([case["case_text"]])
        similarities = linear_kernel(query_vector, self.reference_matrix).ravel()
        top_n = min(top_n, len(similarities))

        if top_n == len(similarities):
            top_indices = np.arange(len(similarities))
        else:
            top_indices = np.argpartition(similarities, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        reference = self.reference_cases.iloc[top_indices]
        neighbors: list[dict[str, Any]] = []
        for rank, (similarity, (_, row)) in enumerate(
            zip(similarities[top_indices], reference.iterrows()), start=1
        ):
            neighbors.append({
                "rank": rank,
                "transaction_id": int(row["TransactionID"]),
                "cosine_similarity": float(similarity),
                "reference_fraud_label": int(row["isFraud"]),
                "reference_model_risk_score": float(row["fraud_risk_score"]),
                "reference_alert_generated": bool(row["predicted_fraud"]),
                "transaction_amount": float(row["TransactionAmt"]),
                "product_code": json_safe(row["ProductCD"]),
                "card_network": json_safe(row["card4"]),
                "card_type": json_safe(row["card6"]),
                "device_type": json_safe(row["DeviceType"]),
                "purchase_email_domain": json_safe(row["P_emaildomain"]),
            })
        return neighbors

    def investigate(self, transaction_id: int) -> dict[str, Any]:
        case = self._case(transaction_id)
        summary_match = self.summaries.loc[
            self.summaries["TransactionID"] == int(transaction_id)
        ]
        if summary_match.empty:
            raise KeyError(f"No grounded summary exists for transaction {transaction_id}.")
        summary = summary_match.iloc[0]

        transaction = {
            "transaction_id": int(case["TransactionID"]),
            "transaction_amount": float(case["TransactionAmt"]),
            "amount_percentile": float(case["AmountPercentile"]),
            "product_code": json_safe(case["ProductCD"]),
            "card_network": json_safe(case["card4"]),
            "card_type": json_safe(case["card6"]),
            "device_type": json_safe(case["DeviceType"]),
            "purchase_email_domain": json_safe(case["P_emaildomain"]),
            "elapsed_dataset_day": float(case["TransactionDay"]),
        }
        return {
            "purpose": "Human fraud-investigation decision support",
            "transaction": transaction,
            "model_signal": {
                "fraud_risk_score": float(case["fraud_risk_score"]),
                "alert_generated": bool(case["predicted_fraud"]),
                "score_warning": "The risk score is not a calibrated fraud probability.",
                "interpretation_warning": (
                    "Global model importance does not explain this individual score."
                ),
            },
            "similar_reference_cases": self.similar_cases(transaction_id, top_n=5),
            "assistant_summary": {
                "provider": str(summary["Provider"]),
                "markdown": str(summary["Summary"]),
                "automated_checks_pass": bool(summary["All automated checks pass"]),
            },
            "required_boundary": (
                "Decision support only. A human investigator must verify the evidence "
                "and make any decision."
            ),
        }

