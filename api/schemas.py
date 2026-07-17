"""Validated request and response contracts for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(StrictModel):
    status: str
    service: str
    version: str
    artifacts_ready: bool
    missing_artifacts: list[str]


class RawTransactionBatch(StrictModel):
    transactions: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="Merged IEEE-CIS transaction and identity records.",
    )

    @field_validator("transactions")
    @classmethod
    def records_must_be_objects(
        cls, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if any(not record for record in records):
            raise ValueError("Each transaction must contain at least one field.")
        return records


class PredictionResult(StrictModel):
    transaction_id: int
    fraud_risk_score: float
    decision_threshold: float
    alert_generated: bool


class PredictionResponse(StrictModel):
    model_name: str
    score_semantics: str
    predictions: list[PredictionResult]
