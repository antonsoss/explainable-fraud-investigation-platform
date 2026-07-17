"""Read-only service for saved evaluation and monitoring results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from api.utils import dataframe_records, json_safe


class MetricsService:
    """Expose only persisted results; the API never recomputes or retunes metrics."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.model_dir = project_root / "models" / "trained"
        self.model_results = project_root / "results" / "model_comparison"

    def _metadata(self) -> dict[str, Any]:
        manifest = json.loads(
            (self.model_dir / "champion_manifest.json").read_text(encoding="utf-8")
        )
        return json.loads(
            (self.model_dir / manifest["metadata_file"]).read_text(encoding="utf-8")
        )

    def model_info(self) -> dict[str, Any]:
        metadata = self._metadata()
        tuning = metadata.get("tuning", {})
        return {
            "model_name": metadata["model_name"],
            "decision_threshold": float(metadata["decision_threshold"]),
            "threshold_rule": metadata["threshold_rule"],
            "primary_selection_metric": metadata["primary_selection_metric"],
            "score_semantics": metadata["score_semantics"],
            "feature_count": int(metadata["feature_count"]),
            "split_policy": metadata["data_split"]["split_policy"],
            "best_cv_average_precision": json_safe(
                tuning.get("best_cv_average_precision")
            ),
            "best_parameters": tuning.get("best_parameters", {}),
            "test_metrics": {
                key: json_safe(value)
                for key, value in metadata["test_metrics"].items()
            },
        }

    def metrics(self) -> dict[str, Any]:
        validation_comparison = pd.read_csv(
            self.model_results / "validation_model_comparison.csv"
        )
        model = self.model_info()
        test = model["test_metrics"]
        total = sum(int(test[key]) for key in ["TN", "FP", "FN", "TP"])
        outcomes = pd.DataFrame(
            [
                {"Outcome": "True positive", "Transactions": int(test["TP"])},
                {"Outcome": "False positive", "Transactions": int(test["FP"])},
                {"Outcome": "False negative", "Transactions": int(test["FN"])},
                {"Outcome": "True negative", "Transactions": int(test["TN"])},
            ]
        )
        outcomes["Share"] = outcomes["Transactions"] / total

        return {
            "model": model,
            "validation_model_comparison": dataframe_records(validation_comparison),
            "test_outcomes": dataframe_records(outcomes),
        }
