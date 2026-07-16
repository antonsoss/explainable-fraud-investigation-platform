"""Application configuration derived from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Runtime settings shared by the API services."""

    project_root: Path
    max_prediction_records: int = 100

    @classmethod
    def from_environment(cls) -> "Settings":
        configured_root = os.getenv("FRAUD_PROJECT_ROOT")
        project_root = (
            Path(configured_root).expanduser().resolve()
            if configured_root
            else DEFAULT_PROJECT_ROOT
        )
        return cls(project_root=project_root)

    @property
    def allowed_origins(self) -> list[str]:
        configured = os.getenv(
            "FRAUD_ALLOWED_ORIGINS",
            "http://localhost:8501,http://127.0.0.1:8501",
        )
        return [origin.strip() for origin in configured.split(",") if origin.strip()]

