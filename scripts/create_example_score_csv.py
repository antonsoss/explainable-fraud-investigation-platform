"""Create synthetic merged IEEE-CIS records for the Streamlit scoring page."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = PROJECT_ROOT / "examples" / "sample_merged_transactions.csv"


def merged_raw_columns() -> list[str]:
    """Return the transaction/identity schema without the retrospective label."""
    transaction_file = RAW_DIR / "train_transaction.csv"
    identity_file = RAW_DIR / "train_identity.csv"
    if not transaction_file.exists() or not identity_file.exists():
        raise FileNotFoundError(
            "Place train_transaction.csv and train_identity.csv in data/raw first."
        )

    transaction_columns = pd.read_csv(transaction_file, nrows=0).columns.tolist()
    identity_columns = pd.read_csv(identity_file, nrows=0).columns.tolist()
    transaction_columns.remove("isFraud")
    return transaction_columns + [
        column for column in identity_columns if column != "TransactionID"
    ]


def synthetic_records() -> list[dict]:
    """Return varied, fictional inputs using valid-looking dataset categories."""
    return [
        {
            "TransactionID": 9_000_001,
            "TransactionDT": 15_000_000,
            "TransactionAmt": 49.00,
            "ProductCD": "W",
            "card1": 7239,
            "card2": 452,
            "card3": 150,
            "card4": "mastercard",
            "card5": 117,
            "card6": "debit",
            "addr1": 264,
            "addr2": 87,
            "dist1": 4,
            "P_emaildomain": "gmail.com",
        },
        {
            "TransactionID": 9_000_002,
            "TransactionDT": 15_100_000,
            "TransactionAmt": 1265.50,
            "ProductCD": "W",
            "card1": 18227,
            "card2": 583,
            "card3": 150,
            "card4": "visa",
            "card5": 226,
            "card6": "credit",
            "addr1": 472,
            "addr2": 87,
            "P_emaildomain": "yahoo.com",
        },
        {
            "TransactionID": 9_000_003,
            "TransactionDT": 15_200_000,
            "TransactionAmt": 262.397,
            "ProductCD": "C",
            "card1": 16132,
            "card2": 111,
            "card3": 150,
            "card4": "visa",
            "card5": 226,
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "gmail.com",
            "DeviceType": "desktop",
            "DeviceInfo": "Windows",
            "id_31": "chrome 66.0",
        },
    ]


def main() -> None:
    columns = merged_raw_columns()
    records = synthetic_records()
    frame = pd.DataFrame(records).reindex(columns=columns)

    # Make the third fictional record a deliberately sparse zero-count/zero-V
    # edge case so the example produces both sides of the frozen threshold.
    zero_pattern_columns = [
        column for column in columns if re.fullmatch(r"[CV]\d+", column)
    ]
    frame.loc[2, zero_pattern_columns] = 0.0

    assert "isFraud" not in frame.columns
    assert frame["TransactionID"].is_unique
    assert len(frame.columns) == len(set(frame.columns))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_FILE, index=False)
    print(
        f"Created {OUTPUT_FILE} with {len(frame)} synthetic rows "
        f"and {len(frame.columns)} merged raw features."
    )


if __name__ == "__main__":
    main()
