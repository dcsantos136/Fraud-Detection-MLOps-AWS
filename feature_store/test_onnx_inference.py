import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

from onnx_utils import export_model_to_onnx, load_onnx_session, predict_onnx


class OnnxInferenceTests(unittest.TestCase):
    def test_onnx_export_and_prediction(self):
        model = LogisticRegression(random_state=42)
        X = pd.DataFrame(
            {
                "transaction_amount": [100, 150, 200, 250],
                "account_balance": [500, 600, 700, 800],
                "is_high_risk": [0, 0, 1, 1],
            }
        )
        y = [0, 0, 1, 1]
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            export_model_to_onnx(model, list(X.columns), onnx_path)
            self.assertTrue(os.path.exists(onnx_path))

            session = load_onnx_session(onnx_path)
            predictions = predict_onnx(session, X)
            self.assertEqual(len(predictions), len(X))
            self.assertTrue(all(value in (0, 1) for value in predictions))


if __name__ == "__main__":
    unittest.main()
