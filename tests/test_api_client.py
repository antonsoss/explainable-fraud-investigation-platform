"""Unit tests for the Streamlit-to-FastAPI URL contract."""

import unittest

from frontend.api_client import FraudApiClient


class FraudApiClientUrlTest(unittest.TestCase):
    def test_service_origin_gets_versioned_api_prefix(self):
        client = FraudApiClient("https://explainable-fraud-api.onrender.com/")

        self.assertEqual(
            client.base_url,
            "https://explainable-fraud-api.onrender.com/api/v1",
        )
        self.assertEqual(
            client.server_url,
            "https://explainable-fraud-api.onrender.com",
        )

    def test_versioned_api_url_is_preserved(self):
        client = FraudApiClient("http://127.0.0.1:8000/api/v1")

        self.assertEqual(client.base_url, "http://127.0.0.1:8000/api/v1")
        self.assertEqual(client.server_url, "http://127.0.0.1:8000")


if __name__ == "__main__":
    unittest.main()
