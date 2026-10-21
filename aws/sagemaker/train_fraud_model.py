import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def pick_target_column(df):
    for candidate in ("is_fraud", "is_high_risk", "label", "fraud"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Dataset must contain one of: is_fraud, is_high_risk, label, fraud")


def main():
    train_path = os.environ.get("SM_CHANNEL_TRAIN")
    if not train_path:
        train_path = "/opt/ml/input/data/train.csv"

    df = pd.read_csv(train_path)
    target_col = pick_target_column(df)

    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].select_dtypes(include=["number"]).copy()
    id_cols = [c for c in X.columns if c.lower().endswith("id") or "id" in c.lower()]
    X = X.drop(columns=[c for c in id_cols if c in X.columns], errors="ignore")
    y = df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    metrics = {
        "accuracy": accuracy_score(y_valid, preds),
        "f1": f1_score(y_valid, preds, zero_division=0),
    }

    output_dir = Path("/opt/ml/model")
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "fraud_model.joblib")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(json.dumps({"training_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
