# Setup Feast Feature Store with Sample Fraud Detection Data
from datetime import datetime
from pathlib import Path


try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing required dependency. Install it with: python -m pip install pandas pyarrow"
    ) from exc


def main():
    repo_dir = Path(__file__).resolve().parent
    feature_repo_dir = repo_dir / "feature_repo"
    output_path = feature_repo_dir / "fraud_transactions.parquet"

    # Sample Fraud Detection Feature Data
    data = {
        "customer_id": [1001, 1002, 1003, 1004, 1005],
        "transaction_amount": [1200, 50, 3000, 400, 5000],
        "account_balance": [15000, 3000, 25000, 7000, 50000],
        "is_high_risk": [1, 0, 1, 0, 1],
        "event_timestamp": [
            datetime.now(),
            datetime.now(),
            datetime.now(),
            datetime.now(),
            datetime.now(),
        ],
    }

    df = pd.DataFrame(data)

    print("\n-----------------------------------------------------------")
    print("Fraud Detection Feature Data:")
    print("-------------------------------------------------------------\n")
    print(df)

    feature_repo_dir.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_path)

    print("\n-----------------------------------------------------------")
    print("Parquet file 'fraud_transactions.parquet' created successfully in 'feature_repo' directory.")
    print("-------------------------------------------------------------\n")
    print(f"Saved File: {output_path}")


if __name__ == "__main__":
    main()