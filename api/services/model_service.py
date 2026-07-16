"""Frozen-model loading and scoring service."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from api.utils import dataframe_records
from src.fraud_pipeline import FraudScoringPipeline


class ModelService:
    """Own the raw-to-score pipeline and keep model loading out of the frontend."""

    def __init__(self, project_root: Path, max_records: int = 100):
        self.project_root = project_root
        self.max_records = max_records

    @cached_property
    def pipeline(self) -> FraudScoringPipeline:
        return FraudScoringPipeline.from_project_root(self.project_root)

    def model_info(self) -> dict[str, Any]:
        metadata = self.pipeline.metadata
        return {
            "model_name": metadata["model_name"],
            "decision_threshold": float(metadata["decision_threshold"]),
            "threshold_rule": metadata["threshold_rule"],
            "score_semantics": metadata["score_semantics"],
            "feature_count": int(metadata["feature_count"]),
            "split_policy": metadata["data_split"]["split_policy"],
            "test_metrics": metadata["test_metrics"],
        }

    def predict_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if len(records) > self.max_records:
            raise ValueError(
                f"At most {self.max_records} transactions may be scored per request."
            )

        raw = pd.DataFrame.from_records(records)
        predictions = self.pipeline.predict(raw).rename(columns={
            "TransactionID": "transaction_id",
            "predicted_fraud": "alert_generated",
        })
        predictions["transaction_id"] = predictions["transaction_id"].astype("int64")
        predictions["alert_generated"] = predictions["alert_generated"].astype(bool)

        return {
            "model_name": self.pipeline.metadata["model_name"],
            "score_semantics": self.pipeline.metadata["score_semantics"],
            "predictions": dataframe_records(predictions),
        }

