"""Reusable raw-transaction preprocessing and fraud scoring.

Only load joblib artifacts produced by this project. Joblib files use pickle
internally and must not be loaded from untrusted sources.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


SECONDS_PER_DAY = 24 * 60 * 60
SECONDS_PER_HOUR = 60 * 60


def engineer_transaction_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create the deterministic row-level features.

    These match Notebook 02: Data Wrangling, Preprocessing & Feature Engineering.
    """
    required = {
        "TransactionAmt", "TransactionDT", "P_emaildomain", "R_emaildomain",
        "card1", "card2", "DeviceType", "DeviceInfo",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Raw input is missing required columns: {missing}")

    frame = raw.copy()
    engineered = pd.DataFrame(index=frame.index)
    engineered["TransactionAmt_Log"] = np.log1p(frame["TransactionAmt"])
    engineered["TransactionAmt_Rounded"] = frame["TransactionAmt"].round()
    engineered["TransactionDay"] = (
        frame["TransactionDT"] // SECONDS_PER_DAY
    ).astype("int32")
    engineered["TransactionHour"] = (
        (frame["TransactionDT"] % SECONDS_PER_DAY) // SECONDS_PER_HOUR
    ).astype("int8")
    engineered["EmailMatch"] = (
        frame["P_emaildomain"].notna()
        & frame["R_emaildomain"].notna()
        & (frame["P_emaildomain"] == frame["R_emaildomain"])
    ).astype("int8")
    engineered["P_EmailProvider"] = (
        frame["P_emaildomain"].fillna("unknown").str.split(".").str[0]
    )
    engineered["R_EmailProvider"] = (
        frame["R_emaildomain"].fillna("unknown").str.split(".").str[0]
    )
    engineered["CardID"] = (
        frame["card1"].astype("string").fillna("unknown")
        + "_"
        + frame["card2"].astype("string").fillna("unknown")
    )
    engineered["IsMobile"] = (frame["DeviceType"] == "mobile").astype("int8")
    engineered["UnknownDevice"] = frame["DeviceInfo"].isna().astype("int8")
    return pd.concat([frame, engineered], axis=1)


class FraudPreprocessor:
    """Apply the frozen raw-record transformations.

    These come from Notebook 02: Data Wrangling, Preprocessing & Feature Engineering.
    """

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.high_missing_features = self._load("high_missing_features.pkl")
        self.numeric_columns = self._load("imputed_numeric_features.pkl")
        self.categorical_columns = self._load("imputed_categorical_features.pkl")
        self.median_imputer = self._load("median_imputer.pkl")
        self.mode_imputer = self._load("mode_imputer.pkl")
        self.low_information_features = self._load("low_information_features.pkl")
        self.high_cardinality_features = self._load("high_cardinality_features.pkl")
        self.low_cardinality_features = self._load("low_cardinality_features.pkl")
        self.frequency_maps = self._load("frequency_maps.pkl")
        self.encoder = self._load("onehot_encoder.pkl")
        self.encoded_feature_names = list(self._load("encoded_feature_names.pkl"))
        self.scaled_features = self._load("scaled_feature_names.pkl")
        self.scaler = self._load("robust_scaler.pkl")
        self.correlated_features = self._load("correlated_features_removed.pkl")
        self.selected_features = list(self._load("selected_features.pkl"))

    def _load(self, filename: str):
        path = self.artifact_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing preprocessing artifact: {path}")
        return joblib.load(path)

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: list[str], stage: str) -> None:
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise ValueError(f"Missing columns before {stage}: {missing}")

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Transform merged raw transaction/identity rows into model features."""
        frame = engineer_transaction_features(raw)
        frame = frame.drop(columns=["isFraud", "TransactionID"], errors="ignore")
        frame = frame.drop(columns=self.high_missing_features, errors="ignore")

        self._require_columns(frame, self.numeric_columns, "numeric imputation")
        self._require_columns(frame, self.categorical_columns, "categorical imputation")
        frame.loc[:, self.numeric_columns] = self.median_imputer.transform(
            frame[self.numeric_columns]
        )

        # A CSV upload can contain a categorical feature that is blank in every
        # submitted row. Pandas then infers float64, which cannot accept the
        # string modes produced by the frozen categorical imputer.
        for column in self.categorical_columns:
            frame[column] = frame[column].astype("object")
        frame.loc[:, self.categorical_columns] = self.mode_imputer.transform(
            frame[self.categorical_columns]
        )
        frame = frame.drop(columns=self.low_information_features, errors="ignore")

        self._require_columns(frame, self.high_cardinality_features, "frequency encoding")
        for column in self.high_cardinality_features:
            frame[f"{column}_Frequency"] = (
                frame[column]
                .map(self.frequency_maps[column])
                .fillna(0)
                .astype(np.float32)
            )
        frame = frame.drop(columns=self.high_cardinality_features)

        self._require_columns(frame, self.low_cardinality_features, "one-hot encoding")
        encoded = pd.DataFrame(
            self.encoder.transform(frame[self.low_cardinality_features]),
            index=frame.index,
            columns=self.encoded_feature_names,
        )
        frame = pd.concat(
            [frame.drop(columns=self.low_cardinality_features), encoded], axis=1
        )

        self._require_columns(frame, self.scaled_features, "robust scaling")
        frame[self.scaled_features] = frame[self.scaled_features].astype(np.float32)
        frame[self.scaled_features] = self.scaler.transform(
            frame[self.scaled_features]
        )
        frame = frame.drop(columns=self.correlated_features, errors="ignore")

        self._require_columns(frame, self.selected_features, "feature alignment")
        transformed = frame.loc[:, self.selected_features].astype(np.float32)
        if transformed.isna().to_numpy().any():
            raise ValueError("Preprocessing produced missing values.")
        if not np.isfinite(transformed.to_numpy()).all():
            raise ValueError("Preprocessing produced non-finite values.")
        return transformed


@dataclass
class FraudScoringPipeline:
    """Single callable interface for preprocessing and frozen-model scoring."""

    preprocessor: FraudPreprocessor
    model: object
    metadata: dict

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "FraudScoringPipeline":
        root = Path(project_root)
        model_dir = root / "models" / "trained"
        manifest_path = model_dir / "champion_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Run Notebook 03 to create champion_manifest.json.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = joblib.load(model_dir / manifest["model_file"])
        metadata = json.loads(
            (model_dir / manifest["metadata_file"]).read_text(encoding="utf-8")
        )
        preprocessor = FraudPreprocessor(root / "models" / "preprocessing")
        return cls(preprocessor=preprocessor, model=model, metadata=metadata)

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return uncalibrated fraud risk scores and thresholded alerts."""
        features = self.preprocessor.transform(raw)
        risk_scores = self.model.predict_proba(features)[:, 1]
        threshold = float(self.metadata["decision_threshold"])
        transaction_ids = (
            raw["TransactionID"].to_numpy()
            if "TransactionID" in raw
            else np.arange(len(raw))
        )
        return pd.DataFrame({
            "TransactionID": transaction_ids,
            "fraud_risk_score": risk_scores.astype(np.float32),
            "decision_threshold": threshold,
            "predicted_fraud": (risk_scores >= threshold).astype("int8"),
        })
