import pandas as pd
import bentoml
from bentoml.io import JSON

MODEL_TAG = "fraud_model:latest"


@bentoml.service(name="fraud_detection_service")
class FraudDetectionService:
    model_runner = bentoml.sklearn.get(MODEL_TAG).to_runner()

    @bentoml.api(input=JSON(), output=JSON())
    def predict(self, payload):
        if isinstance(payload, dict):
            rows = [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError("Request body must be a JSON object or list of objects.")

        df = pd.DataFrame(rows)
        df = df.select_dtypes(include=["number"]).copy()
        id_cols = [c for c in df.columns if c.lower().endswith("id") or "id" in c.lower()]
        df = df.drop(columns=[c for c in id_cols if c in df.columns], errors="ignore")

        if df.empty:
            raise ValueError("No numeric feature columns were found in the request payload.")

        preds = self.model_runner.predict.run(df)
        return {"predictions": pd.Series(preds).tolist()}
