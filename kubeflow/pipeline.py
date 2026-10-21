from pathlib import Path

from kfp import dsl
from kfp.dsl import InputPath, OutputPath


@dsl.component(base_image="python:3.11-slim", packages_to_install=["pandas", "numpy", "scikit-learn", "joblib"])
def load_raw_data(raw_data_path: OutputPath("CSV")) -> None:
    import pandas as pd

    fraud_data = {
        "transaction_amount": [1200, 50, 3000, 400, 5000, 150, 2500, 90, 7000, 100, 4300, 220, 1500, 8000, 60],
        "transaction_type": ["Online", "POS", "Online", "ATM", "Online", "POS", "ATM", "POS", "Online", "ATM", "Online", "POS", "ATM", "Online", "POS"],
        "customer_age": [25, 45, 35, 50, 29, 40, 33, 38, 31, 47, None, 36, 44, 28, 52],
        "account_balance": [15000, 3000, 25000, 4000, 50000, 3500, 20000, None, 60000, 4500, 30000, 5000, 18000, 70000, 25000],
        "merchant_category": ["Electronics", "Groceries", "Travel", "Cash Withdrawal", "Luxury", "Groceries", "Travel", "Food", "Luxury", "Cash Withdrawal", "Electronics", "Food", "Travel", "Luxury", "Groceries"],
        "is_fraud": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    }

    df = pd.DataFrame(fraud_data)
    output = Path(raw_data_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Raw dataset saved to: {output}")


@dsl.component(base_image="python:3.11-slim", packages_to_install=["pandas"])
def validate_data(input_csv: InputPath("CSV"), validation_report_path: OutputPath("JSON")) -> None:
    import json
    import pandas as pd

    df = pd.read_csv(input_csv)
    checks = {
        "rows": int(len(df)),
        "transaction_amount_min": float(df["transaction_amount"].min()),
        "transaction_amount_max": float(df["transaction_amount"].max()),
        "customer_age_min": float(df["customer_age"].min()),
        "customer_age_max": float(df["customer_age"].max()),
        "valid_types": sorted(df["transaction_type"].dropna().unique().tolist()),
        "null_transaction_type": bool(df["transaction_type"].isnull().any()),
        "is_valid": bool(
            (df["transaction_amount"] >= 0).all()
            & (df["customer_age"].between(18, 100, inclusive="both")).all()
            & (df["transaction_type"].notna().all())
            & (df["transaction_type"].isin(["Online", "POS", "ATM"]).all())
        ),
    }

    output = Path(validation_report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=2)
    print(f"Validation report saved to: {output}")


@dsl.component(base_image="python:3.11-slim", packages_to_install=["pandas", "numpy", "scikit-learn"])
def preprocess_features(input_csv: InputPath("CSV"), processed_csv: OutputPath("CSV")) -> None:
    import pandas as pd
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_csv(input_csv)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    numerical_features = ["transaction_amount", "customer_age", "account_balance"]
    categorical_features = ["transaction_type", "merchant_category"]

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    transformed = preprocessor.fit_transform(X)
    transformed_df = pd.DataFrame(
        transformed.toarray() if hasattr(transformed, "toarray") else transformed
    )
    transformed_df["is_fraud"] = y.values

    output = Path(processed_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    transformed_df.to_csv(output, index=False)
    print(f"Processed features saved to: {output}")


@dsl.component(base_image="python:3.11-slim", packages_to_install=["pandas", "scikit-learn", "joblib", "mlflow"])
def train_model(processed_csv: InputPath("CSV"), model_path: OutputPath("MODEL")) -> None:
    import joblib
    import os
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import mlflow

    df = pd.read_csv(processed_csv)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    print(f"Model saved to: {output}")

    # MLflow tracking: use MLFLOW_TRACKING_URI env var if set for central tracking
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # fallback to local file store inside component
            mlflow.set_tracking_uri("file:///tmp/mlruns")

        mlflow.set_experiment("fraud_detection")
        with mlflow.start_run():
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("trained_samples", int(len(X)))
            mlflow.log_artifact(str(output))
            print("Logged model and params to MLflow")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")


@dsl.component(base_image="python:3.11-slim", packages_to_install=["pandas", "scikit-learn", "joblib", "mlflow"])
def evaluate_model(processed_csv: InputPath("CSV"), model_path: InputPath("MODEL"), metrics_path: OutputPath("JSON")) -> None:
    import json
    from pathlib import Path

    import joblib
    import pandas as pd
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    import os
    import mlflow

    df = pd.read_csv(processed_csv)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)

    output = Path(metrics_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Metrics saved to: {output}")

    # Log metrics to MLflow (will follow MLFLOW_TRACKING_URI if set)
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("file:///tmp/mlruns")
        mlflow.set_experiment("fraud_detection")
        with mlflow.start_run():
            # log summary metrics from classification report (accuracy as example)
            if "accuracy" in report:
                mlflow.log_metric("accuracy", report["accuracy"])
            # log entire report as artifact
            mlflow.log_artifact(str(output))
            print("Logged evaluation metrics to MLflow")
    except Exception as e:
        print(f"MLflow metric logging skipped: {e}")


@dsl.pipeline(name="fraud-detection-kubeflow-pipeline")
def fraud_detection_pipeline():
    raw_data_task = load_raw_data()
    validation_task = validate_data(input_csv=raw_data_task.outputs["raw_data_path"])
    processed_task = preprocess_features(input_csv=raw_data_task.outputs["raw_data_path"])
    model_task = train_model(processed_csv=processed_task.outputs["processed_csv"])
    evaluate_model(
        processed_csv=processed_task.outputs["processed_csv"],
        model_path=model_task.outputs["model_path"],
    )


if __name__ == "__main__":
    from kfp import compiler

    package_path = Path(__file__).with_name("fraud_detection_pipeline.yaml")
    compiler.Compiler().compile(pipeline_func=fraud_detection_pipeline, package_path=str(package_path))
    print(f"Compiled Kubeflow pipeline YAML: {package_path}")
