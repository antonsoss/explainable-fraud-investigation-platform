"""Serve frozen post-hoc explanation artifacts exported by Notebook 04."""

from __future__ import annotations

from functools import cached_property
import json
from pathlib import Path
from typing import Any

import pandas as pd

from api.utils import dataframe_records


class XAIService:
    """Read precomputed SHAP and LIME evidence without loading either library."""

    def __init__(self, project_root: Path):
        self.results_dir = project_root / "results" / "xai"

    @cached_property
    def metadata(self) -> dict[str, Any]:
        with (self.results_dir / "xai_metadata.json").open(encoding="utf-8") as file:
            return json.load(file)

    @cached_property
    def global_importance(self) -> pd.DataFrame:
        return pd.read_csv(self.results_dir / "shap_global_importance.csv")

    @cached_property
    def local_shap(self) -> pd.DataFrame:
        return pd.read_parquet(self.results_dir / "local_shap_values.parquet")

    @cached_property
    def local_lime(self) -> pd.DataFrame:
        return pd.read_parquet(self.results_dir / "local_lime_values.parquet")

    @cached_property
    def lime_fidelity(self) -> pd.DataFrame:
        return pd.read_csv(self.results_dir / "lime_case_fidelity.csv")

    @cached_property
    def comparison(self) -> pd.DataFrame:
        return pd.read_csv(self.results_dir / "shap_lime_comparison.csv")

    @staticmethod
    def _figure_url(filename: str) -> str:
        """Return a stable API-hosted URL for a trusted exported figure name."""
        return f"/artifacts/xai/{Path(filename).name}"

    def _figure_urls(self) -> dict[str, Any]:
        figures = self.metadata["figures"]
        return {
            "global_bar": self._figure_url(figures["global_bar"]),
            "global_beeswarm": self._figure_url(figures["global_beeswarm"]),
            "native_comparison": self._figure_url(figures["native_comparison"]),
            "dependence": {
                feature: self._figure_url(filename)
                for feature, filename in figures["dependence"].items()
            },
        }

    def global_summary(self, top_n: int = 20) -> dict[str, Any]:
        """Return global SHAP evidence and LIME reliability diagnostics."""
        top_n = max(1, min(int(top_n), len(self.global_importance)))
        top_features = self.global_importance.head(top_n).rename(columns={
            "Feature": "feature",
            "Mean absolute SHAP": "mean_absolute_shap",
            "Mean signed SHAP": "mean_signed_shap",
            "Positive contribution share": "positive_contribution_share",
            "SHAP rank": "shap_rank",
            "Normalized mean absolute SHAP": "normalized_mean_absolute_shap",
            "Normalized Gain": "normalized_gain",
            "Gain Rank": "gain_rank",
            "Feature family": "feature_family",
        })

        fidelity = self.lime_fidelity.rename(columns={
            "TransactionID": "transaction_id",
            "LIME local fidelity R2": "lime_local_fidelity_r2",
            "LIME local prediction": "lime_local_prediction",
            "Model risk score": "model_risk_score",
            "Absolute local prediction error": "absolute_local_prediction_error",
        })
        comparison = self.comparison.drop(columns=["Outcome"], errors="ignore").rename(
            columns={
                "TransactionID": "transaction_id",
                "Top-k": "top_k",
                "Overlapping features": "overlapping_features",
                "Top-feature Jaccard": "top_feature_jaccard",
                "Direction agreement on overlap": "direction_agreement_on_overlap",
                "LIME local fidelity R2": "lime_local_fidelity_r2",
                "LIME absolute prediction error": "lime_absolute_prediction_error",
            }
        )

        return {
            "model_name": self.metadata["model_name"],
            "decision_threshold": self.metadata["decision_threshold"],
            "score_semantics": self.metadata["score_semantics"],
            "feature_count": self.metadata["feature_count"],
            "test_reporting_only": self.metadata["test_reporting_only"],
            "model_or_threshold_changed": self.metadata["model_or_threshold_changed"],
            "shap": self.metadata["shap"],
            "lime": self.metadata["lime"],
            "top_global_features": dataframe_records(top_features),
            "lime_case_fidelity": dataframe_records(fidelity),
            "shap_lime_comparison": dataframe_records(comparison),
            "figures": self._figure_urls(),
            "required_boundary": self.metadata["required_boundary"],
        }

    def local_explanation(
        self, transaction_id: int, top_n: int = 10
    ) -> dict[str, Any]:
        """Return label-free local explanations for one representative test case."""
        transaction_id = int(transaction_id)
        shap_rows = self.local_shap.loc[
            self.local_shap["TransactionID"] == transaction_id
        ].sort_values("Rank")
        lime_rows = self.local_lime.loc[
            self.local_lime["TransactionID"] == transaction_id
        ].sort_values("Rank")
        if shap_rows.empty or lime_rows.empty:
            raise KeyError(
                f"No Notebook 04 explanation exists for transaction {transaction_id}."
            )

        top_n = max(1, min(int(top_n), len(shap_rows)))
        shap_display = shap_rows.head(top_n).rename(columns={
            "Feature": "feature",
            "Processed feature value": "processed_feature_value",
            "SHAP contribution": "shap_contribution",
            "Absolute SHAP contribution": "absolute_shap_contribution",
            "Direction": "direction",
            "Rank": "rank",
        })[[
            "feature",
            "processed_feature_value",
            "shap_contribution",
            "absolute_shap_contribution",
            "direction",
            "rank",
        ]]
        lime_display = lime_rows.head(top_n).rename(columns={
            "Feature": "feature",
            "Condition": "condition",
            "Processed feature value": "processed_feature_value",
            "LIME weight": "lime_weight",
            "Absolute LIME weight": "absolute_lime_weight",
            "Direction": "direction",
            "Rank": "rank",
        })[[
            "feature",
            "condition",
            "processed_feature_value",
            "lime_weight",
            "absolute_lime_weight",
            "direction",
            "rank",
        ]]

        first_shap = shap_rows.iloc[0]
        first_lime = lime_rows.iloc[0]
        figures = self.metadata["figures"]
        transaction_key = str(transaction_id)
        return {
            "transaction_id": transaction_id,
            "base_risk_score": float(first_shap["Base risk score"]),
            "model_risk_score": float(first_shap["Model risk score"]),
            "decision_threshold": float(first_shap["Decision threshold"]),
            "shap": {
                "method": "Tree SHAP with an interventional training background",
                "contributions": dataframe_records(shap_display),
                "maximum_local_reconstruction_error": self.metadata["shap"][
                    "maximum_local_reconstruction_error"
                ],
                "figure": self._figure_url(figures["waterfall"][transaction_key]),
                "interpretation": (
                    "Contributions explain this frozen model score in processed-feature "
                    "space; they do not establish a causal reason for fraud."
                ),
            },
            "lime": {
                "method": "LIME local surrogate",
                "contributions": dataframe_records(lime_display),
                "local_fidelity_r2": float(first_lime["LIME local fidelity R2"]),
                "local_prediction": float(first_lime["LIME local prediction"]),
                "absolute_prediction_error": abs(
                    float(first_lime["LIME local prediction"])
                    - float(first_lime["Model risk score"])
                ),
                "figure": self._figure_url(figures["lime"][transaction_key]),
                "warning": self.metadata["lime"]["warning"],
                "role": "Secondary supporting evidence because local fidelity is low.",
            },
            "required_boundary": self.metadata["required_boundary"],
        }
