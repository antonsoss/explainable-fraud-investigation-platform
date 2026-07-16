"""Contract tests for the FastAPI backend."""

from pathlib import Path
import json
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi.testclient import TestClient

    from api.main import app
except ImportError as error:  # pragma: no cover - dependency guard
    TestClient = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


@unittest.skipIf(TestClient is None, f"API dependencies are not installed: {IMPORT_ERROR}")
class FraudApiContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_reports_frozen_artifacts_ready(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["artifacts_ready"])
        self.assertEqual(payload["missing_artifacts"], [])

    def test_model_contract_uses_saved_threshold(self):
        response = self.client.get("/api/v1/model")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_name"], "XGBoost (tuned)")
        self.assertEqual(payload["feature_count"], 359)
        self.assertAlmostEqual(payload["decision_threshold"], 0.2756975889)
        self.assertIn("not a fraud probability", payload["score_semantics"])

    def test_metrics_match_executed_test_results(self):
        response = self.client.get("/api/v1/metrics")
        self.assertEqual(response.status_code, 200)
        metrics = response.json()["model"]["test_metrics"]
        self.assertAlmostEqual(metrics["Average precision"], 0.5212717480)
        self.assertEqual(metrics["Alerts"], 5829)
        self.assertEqual(metrics["TP"], 1899)
        self.assertEqual(metrics["FP"], 3930)

    def test_investigation_view_withholds_query_ground_truth(self):
        cases_response = self.client.get("/api/v1/investigations")
        self.assertEqual(cases_response.status_code, 200)
        cases = cases_response.json()["transactions"]
        self.assertEqual(len(cases), 4)
        self.assertNotIn("isFraud", cases[0])
        self.assertNotIn("Outcome", cases[0])

        response = self.client.post(
            "/api/v1/investigate",
            json={"transaction_id": cases[0]["transaction_id"]},
        )
        self.assertEqual(response.status_code, 200)
        investigation = response.json()
        self.assertNotIn("isFraud", investigation["transaction"])
        self.assertNotIn("Outcome", investigation["transaction"])
        self.assertEqual(len(investigation["similar_reference_cases"]), 5)
        self.assertTrue(
            investigation["assistant_summary"]["automated_checks_pass"]
        )

    def test_unknown_investigation_transaction_returns_404(self):
        response = self.client.post(
            "/api/v1/investigate", json={"transaction_id": 1}
        )
        self.assertEqual(response.status_code, 404)

    def test_malformed_scoring_request_returns_clear_validation_error(self):
        response = self.client.post(
            "/api/v1/predict",
            json={"transactions": [{"TransactionID": 1}]},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("TransactionAmt", response.json()["detail"])

    def test_merged_raw_transaction_can_be_scored_over_http(self):
        raw_dir = PROJECT_ROOT / "data" / "raw"
        transaction_file = raw_dir / "train_transaction.csv"
        identity_file = raw_dir / "train_identity.csv"
        if not transaction_file.exists() or not identity_file.exists():
            self.skipTest("Local IEEE-CIS raw data is required for HTTP scoring.")

        transaction = pd.read_csv(transaction_file, nrows=1)
        transaction_id = int(transaction.loc[0, "TransactionID"])
        identity_columns = pd.read_csv(identity_file, nrows=0).columns.tolist()
        identity = pd.DataFrame(columns=identity_columns)
        for chunk in pd.read_csv(identity_file, chunksize=50_000):
            match = chunk.loc[chunk["TransactionID"] == transaction_id]
            if not match.empty:
                identity = match
                break

        raw = transaction.merge(
            identity, on="TransactionID", how="left", validate="one_to_one"
        )
        records = json.loads(raw.to_json(orient="records"))
        response = self.client.post(
            "/api/v1/predict", json={"transactions": records}
        )

        self.assertEqual(response.status_code, 200, response.text)
        prediction = response.json()["predictions"][0]
        self.assertEqual(prediction["transaction_id"], transaction_id)
        self.assertGreaterEqual(prediction["fraud_risk_score"], 0.0)
        self.assertLessEqual(prediction["fraud_risk_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
