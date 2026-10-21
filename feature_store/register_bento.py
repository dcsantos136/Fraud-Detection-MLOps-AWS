import joblib
import bentoml
from pathlib import Path

MODEL_FILE = Path(__file__).parent / "fraud_model.pkl"
if not MODEL_FILE.exists():
    raise SystemExit(f"Model file not found: {MODEL_FILE}\nRun train_model.py first.")

model = joblib.load(MODEL_FILE)
# Save to BentoML model store
tag = bentoml.sklearn.save_model("fraud_model", model)
print(f"Registered model in BentoML model store with tag: {tag}")
