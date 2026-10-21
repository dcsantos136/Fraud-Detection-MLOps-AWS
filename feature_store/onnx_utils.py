from pathlib import Path

import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_model_to_onnx(model, feature_columns, output_path):
    """Export a scikit-learn model to ONNX and return the saved file path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    initial_type = [("features", FloatTensorType([None, len(feature_columns)]))]
    converted_model = convert_sklearn(model, initial_types=initial_type)

    with open(output_path, "wb") as fp:
        fp.write(converted_model.SerializeToString())

    return output_path


def load_onnx_session(model_path, providers=None):
    if providers is None:
        providers = ["CPUExecutionProvider"]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)


def predict_onnx(session, feature_df):
    input_name = session.get_inputs()[0].name
    feature_array = feature_df.astype("float32").to_numpy()
    outputs = session.run(None, {input_name: feature_array})

    output = outputs[0]
    if output.ndim > 1 and output.shape[1] > 1:
        return output.argmax(axis=1)

    return output.astype(int)
