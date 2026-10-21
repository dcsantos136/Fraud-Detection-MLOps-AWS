import json
import time
from pathlib import Path

import joblib
import pandas as pd

from onnx_utils import export_model_to_onnx, load_onnx_session, predict_onnx

FEATURE_STORE_DIR = Path(__file__).resolve().parent


def load_sample_data():
    data_path = FEATURE_STORE_DIR / "streaming" / "streaming_features.json"
    with open(data_path, "r") as f:
        features = json.load(f)

    rows = []
    for customer_id, customer in features.items():
        rows.append({
            "customer_id": int(customer_id),
            "transaction_amount": customer["average_amount"],
            "account_balance": customer["total_amount"],
            "is_high_risk": 1 if customer["average_amount"] > 2000 else 0,
        })

    return pd.DataFrame(rows)


def benchmark_model(model_name, model, feature_cols):
    feature_df = load_sample_data()[feature_cols]
    onnx_path = FEATURE_STORE_DIR / f"{model_name}.onnx"

    export_model_to_onnx(model, feature_cols, onnx_path)
    session = load_onnx_session(onnx_path)

    start = time.perf_counter()
    sklearn_preds = model.predict(feature_df)
    sklearn_time = time.perf_counter() - start

    start = time.perf_counter()
    onnx_preds = predict_onnx(session, feature_df)
    onnx_time = time.perf_counter() - start

    accuracy = (sklearn_preds == onnx_preds).mean()

    print(f"Model: {model_name}")
    print(f"  sklearn time: {sklearn_time:.4f}s")
    print(f"  onnx time:    {onnx_time:.4f}s")
    print(f"  agreement:    {accuracy:.4%}")


if __name__ == "__main__":
    model_a = joblib.load(FEATURE_STORE_DIR / "model_a.pkl")
    model_b = joblib.load(FEATURE_STORE_DIR / "model_b.pkl")

    feature_cols = ["transaction_amount", "account_balance", "is_high_risk"]
    benchmark_model("model_a", model_a, feature_cols)
    benchmark_model("model_b", model_b, feature_cols)
