import json
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from ab_testing import choose_model_for_customer
except ImportError:
    from feature_store.ab_testing import choose_model_for_customer

from onnx_utils import load_onnx_session, predict_onnx

MODEL_PATHS = {
    "model_a": BASE_DIR / "model_a.pkl",
    "model_b": BASE_DIR / "model_b.pkl",
    "fraud_model": BASE_DIR / "fraud_model.pkl",
}
ONNX_PATHS = {
    "model_a": BASE_DIR / "model_a.onnx",
    "model_b": BASE_DIR / "model_b.onnx",
    "fraud_model": BASE_DIR / "fraud_model.onnx",
}
FEATURE_PATH = BASE_DIR / "streaming" / "streaming_features.json"


def load_model_for_route(customer_id, force_model=None):
    model_name = choose_model_for_customer(customer_id, force_model=force_model)
    candidate = MODEL_PATHS.get(model_name)

    if candidate and candidate.exists():
        return model_name, joblib.load(candidate)

    legacy = MODEL_PATHS["fraud_model"]
    if legacy.exists():
        return "fraud_model", joblib.load(legacy)

    raise FileNotFoundError(
        "No trained model found. Train the models first with train_model.py or save fraud_model.pkl."
    )


def load_onnx_model_for_route(customer_id, force_model=None):
    model_name = choose_model_for_customer(customer_id, force_model=force_model)
    onnx_path = ONNX_PATHS.get(model_name)

    if onnx_path and onnx_path.exists():
        return model_name, load_onnx_session(onnx_path)

    legacy_path = ONNX_PATHS["fraud_model"]
    if legacy_path.exists():
        return "fraud_model", load_onnx_session(legacy_path)

    return None, None


def predict_fraud(customer_id, force_model=None):
    customer_id = str(customer_id)
    with open(FEATURE_PATH, "r") as f:
        features = json.load(f)

    if customer_id not in features:
        return {"error": f"Customer {customer_id} not found"}

    customer = features[customer_id]
    customer_row = {
        "transaction_amount": customer["average_amount"],
        "account_balance": customer["total_amount"],
    }

    model_name, session = load_onnx_model_for_route(customer_id, force_model=force_model)
    if session is not None:
        input_df = pd.DataFrame([customer_row])
        prediction = int(predict_onnx(session, input_df)[0])
    else:
        model_name, model = load_model_for_route(customer_id, force_model=force_model)
        expected_cols = list(getattr(model, "feature_names_in_", ["transaction_amount", "account_balance"]))
        input_df = pd.DataFrame([customer_row], columns=expected_cols)
        prediction = int(model.predict(input_df)[0])

    return {
        "customer_id": customer_id,
        "model_used": model_name,
        "transaction_count": customer["transaction_count"],
        "average_amount": customer["average_amount"],
        "total_amount": customer["total_amount"],
        "prediction": "High Risk" if prediction == 1 else "Low Risk",
        "inference_backend": "onnx" if session is not None else "sklearn",
    }
