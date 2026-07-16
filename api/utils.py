"""Serialization helpers for values loaded from pandas artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    """Convert pandas and NumPy scalar values into JSON-safe Python values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize a dataframe without emitting NaN or NumPy scalar values."""
    return [
        {str(column): json_safe(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]

