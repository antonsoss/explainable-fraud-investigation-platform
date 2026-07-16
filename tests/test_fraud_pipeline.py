"""Integration tests for the reusable preprocessing and scoring contract."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_pipeline import FraudScoringPipeline  # noqa: E402


class FraudPipelineIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw_dir = PROJECT_ROOT / "data" / "raw"
        processed_dir = PROJECT_ROOT / "data" / "processed"
        required = [
            raw_dir / "train_transaction.csv",
            raw_dir / "train_identity.csv",
            processed_dir / "X_train.parquet",
            processed_dir / "id_train.parquet",
            PROJECT_ROOT / "models" / "trained" / "champion_manifest.json",
        ]
        if any(not path.exists() for path in required):
            raise unittest.SkipTest("Local data and regenerated model artifacts are required.")

        cls.pipeline = FraudScoringPipeline.from_project_root(PROJECT_ROOT)
        sample_size = 32
        cls.raw = pd.read_csv(raw_dir / "train_transaction.csv", nrows=sample_size)
        target_ids = set(cls.raw["TransactionID"])

        identity_parts = []
        identity_columns = pd.read_csv(
            raw_dir / "train_identity.csv", nrows=0
        ).columns.tolist()
        for chunk in pd.read_csv(raw_dir / "train_identity.csv", chunksize=50_000):
            match = chunk[chunk["TransactionID"].isin(target_ids)]
            if not match.empty:
                identity_parts.append(match)
        identity = (
            pd.concat(identity_parts, ignore_index=True)
            if identity_parts
            else pd.DataFrame(columns=identity_columns)
        )
        cls.raw = cls.raw.merge(
            identity, on="TransactionID", how="left", validate="one_to_one"
        )
        cls.expected = pd.read_parquet(processed_dir / "X_train.parquet").head(sample_size)
        cls.expected_ids = pd.read_parquet(
            processed_dir / "id_train.parquet"
        )["TransactionID"].head(sample_size)

    def test_raw_transform_matches_notebook_output(self):
        self.assertListEqual(
            self.raw["TransactionID"].tolist(), self.expected_ids.tolist()
        )
        actual = self.pipeline.preprocessor.transform(self.raw).reset_index(drop=True)
        self.assertListEqual(actual.columns.tolist(), self.expected.columns.tolist())
        np.testing.assert_allclose(
            actual.to_numpy(), self.expected.to_numpy(), rtol=1e-5, atol=1e-5
        )

    def test_unseen_category_is_handled(self):
        raw = self.raw.head(1).copy()
        raw["ProductCD"] = "UNSEEN_PRODUCT"
        transformed = self.pipeline.preprocessor.transform(raw)
        self.assertEqual(transformed.shape[1], len(self.pipeline.preprocessor.selected_features))

    def test_missing_required_feature_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "TransactionAmt"):
            self.pipeline.preprocessor.transform(self.raw.drop(columns="TransactionAmt"))

    def test_scoring_contract(self):
        result = self.pipeline.predict(self.raw.head(4))
        self.assertListEqual(
            result.columns.tolist(),
            ["TransactionID", "fraud_risk_score", "decision_threshold", "predicted_fraud"],
        )
        self.assertTrue(result["fraud_risk_score"].between(0, 1).all())
        self.assertTrue(set(result["predicted_fraud"]).issubset({0, 1}))

    def test_sparse_csv_with_all_missing_categorical_columns_scores(self):
        example_file = PROJECT_ROOT / "examples" / "sample_merged_transactions.csv"
        if not example_file.exists():
            self.skipTest("Generate the synthetic scoring example first.")

        raw = pd.read_csv(example_file)
        result = self.pipeline.predict(raw)

        self.assertEqual(len(result), 3)
        self.assertListEqual(
            result["TransactionID"].tolist(), [9_000_001, 9_000_002, 9_000_003]
        )
        self.assertTrue(result["fraud_risk_score"].between(0, 1).all())
        self.assertSetEqual(set(result["predicted_fraud"]), {0, 1})


if __name__ == "__main__":
    unittest.main()
