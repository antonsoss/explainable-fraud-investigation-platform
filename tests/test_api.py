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

    from api.main import app, required_artifacts
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
        self.assertEqual(payload["version"], "1.1.0")
        self.assertTrue(payload["artifacts_ready"])
        self.assertEqual(payload["missing_artifacts"], [])

    def test_health_inventory_covers_every_runtime_artifact_category(self):
        inventory = {
            str(path.relative_to(PROJECT_ROOT))
            for path in required_artifacts(PROJECT_ROOT)
        }
        preprocessing_files = {
            "high_missing_features.pkl",
            "imputed_numeric_features.pkl",
            "imputed_categorical_features.pkl",
            "median_imputer.pkl",
            "mode_imputer.pkl",
            "low_information_features.pkl",
            "high_cardinality_features.pkl",
            "low_cardinality_features.pkl",
            "frequency_maps.pkl",
            "onehot_encoder.pkl",
            "encoded_feature_names.pkl",
            "scaled_feature_names.pkl",
            "robust_scaler.pkl",
            "correlated_features_removed.pkl",
            "selected_features.pkl",
        }
        expected_preprocessing = {
            f"models/preprocessing/{filename}"
            for filename in preprocessing_files
        }
        expected_model_and_metrics = {
            "models/trained/champion_manifest.json",
            "models/trained/xgboost_tuned.joblib",
            "models/trained/xgboost_tuned_metadata.json",
            "results/model_comparison/validation_model_comparison.csv",
        }
        expected_xai_data = {
            "results/xai/xai_metadata.json",
            "results/xai/shap_global_importance.csv",
            "results/xai/local_shap_values.parquet",
            "results/xai/local_lime_values.parquet",
            "results/xai/lime_case_fidelity.csv",
            "results/xai/shap_lime_comparison.csv",
        }
        xai_figure_files = {
            "shap_global_bar.png",
            "shap_global_beeswarm.png",
            "shap_vs_native_importance.png",
            "shap_dependence_TransactionDT.png",
            "shap_dependence_C13.png",
            "shap_dependence_C1.png",
            "shap_waterfall_3519397.png",
            "shap_waterfall_3524909.png",
            "shap_waterfall_3541077.png",
            "shap_waterfall_3551357.png",
            "lime_3519397.png",
            "lime_3524909.png",
            "lime_3541077.png",
            "lime_3551357.png",
        }
        expected_xai_figures = {
            f"results/xai/figures/{filename}"
            for filename in xai_figure_files
        }
        expected_inventory = (
            expected_preprocessing
            | expected_model_and_metrics
            | expected_xai_data
            | expected_xai_figures
        )

        self.assertEqual(inventory, expected_inventory)

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

    def test_xai_contract_matches_executed_notebook_results(self):
        response = self.client.get("/api/v1/xai")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["shap"]["top_global_feature"], "TransactionDT")
        self.assertEqual(payload["shap"]["global_reporting_rows"], 1000)
        self.assertLess(
            payload["shap"]["maximum_global_reconstruction_error"], 1e-5
        )
        self.assertAlmostEqual(
            payload["lime"]["mean_local_fidelity_r2"], 0.1884388063
        )
        self.assertFalse(payload["model_or_threshold_changed"])
        self.assertEqual(payload["top_global_features"][0]["feature"], "TransactionDT")

    def test_xai_figure_is_served_by_the_backend(self):
        response = self.client.get("/artifacts/xai/shap_global_beeswarm.png")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertGreater(len(response.content), 1000)

    def test_local_xai_is_served_without_ground_truth(self):
        summary = self.client.get("/api/v1/xai").json()
        transaction_id = summary["lime_case_fidelity"][0]["transaction_id"]
        response = self.client.get(f"/api/v1/xai/{transaction_id}")

        self.assertEqual(response.status_code, 200)
        explanation = response.json()
        self.assertEqual(explanation["transaction_id"], transaction_id)
        self.assertEqual(len(explanation["shap"]["contributions"]), 10)
        self.assertEqual(len(explanation["lime"]["contributions"]), 10)
        self.assertLess(
            explanation["shap"]["maximum_local_reconstruction_error"], 1e-5
        )
        serialized = json.dumps(explanation)
        self.assertNotIn('"Outcome"', serialized)
        self.assertNotIn('"isFraud"', serialized)

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
            self.skipTest("Local competition data files are required for HTTP scoring.")

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
