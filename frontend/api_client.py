"""Small HTTP client used by the Streamlit frontend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests


class FraudApiError(RuntimeError):
    """Raised when the backend is unavailable or rejects a request."""


@dataclass
class FraudApiClient:
    base_url: str
    timeout_seconds: int = 60
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        api_suffix = "/api/v1"
        self.server_url = (
            self.base_url[: -len(api_suffix)]
            if self.base_url.endswith(api_suffix)
            else self.base_url
        )

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                timeout=self.timeout_seconds,
                **kwargs,
            )
        except requests.RequestException as error:
            raise FraudApiError(f"Could not reach the fraud API: {error}") from error

        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise FraudApiError(f"API request failed ({response.status_code}): {detail}")

        try:
            return response.json()
        except ValueError as error:
            raise FraudApiError("The fraud API returned invalid JSON.") from error

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def model(self) -> dict[str, Any]:
        return self._request("GET", "/model")

    def metrics(self) -> dict[str, Any]:
        return self._request("GET", "/metrics")

    def xai(self) -> dict[str, Any]:
        return self._request("GET", "/xai")

    def local_xai(self, transaction_id: int, top_n: int = 10) -> dict[str, Any]:
        return self._request(
            "GET", f"/xai/{transaction_id}", params={"top_n": top_n}
        )

    def artifact_bytes(self, path: str) -> bytes:
        """Download a trusted figure path returned by the API."""
        if not path.startswith("/artifacts/xai/"):
            raise FraudApiError("The API returned an unexpected artifact path.")
        try:
            response = self.session.get(
                f"{self.server_url}{path}", timeout=self.timeout_seconds
            )
        except requests.RequestException as error:
            raise FraudApiError(f"Could not reach the fraud API: {error}") from error
        if not response.ok:
            raise FraudApiError(
                f"Artifact request failed ({response.status_code}): {response.text}"
            )
        return response.content

    def predict(self, transactions: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request(
            "POST", "/predict", json={"transactions": transactions}
        )
