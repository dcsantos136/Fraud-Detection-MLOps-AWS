# Train two fraud detection models for a simple A/B test setup.
# model_a is the champion/random forest model and model_b is the challenger/logistic model.

import os
from pathlib import Path

import joblib
import mlflow
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from onnx_utils import export_model_to_onnx

FEATURE_STORE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]

print("\n-----------------------------------------------------")
print("Loading features from feature store")
print("-----------------------------------------------------\n")

try:
    df = pd.read_parquet(FEATURE_STORE_DIR / "feature_repo" / "fraud_transactions.parquet")
except FileNotFoundError:
    data_path = REPO_ROOT / "Fraud Detection" / "feature pipeline" / "data" / "fraud_data.csv"
    df = pd.read_csv(data_path)

print(df.head())

# Feature selection and target variable
# Accept a few common target column names if available.
target_col = None
for candidate in ("is_fraud", "is_high_risk", "label", "fraud"):
    if candidate in df.columns:
        target_col = candidate
        break

if target_col is None:
    raise ValueError("The dataset must contain a target column (is_fraud / is_high_risk / label / fraud).")

feature_cols = [col for col in df.columns if col != target_col]
X = df[feature_cols]
y = df[target_col]

# Keep only numeric features for sklearn; drop datetime/object cols and obvious IDs
X = X.select_dtypes(include=["number"]).copy()
id_cols = [c for c in X.columns if c.lower().endswith("id") or "id" in c.lower()]
X = X.drop(columns=[c for c in id_cols if c in X.columns], errors="ignore")

try:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
except ValueError:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=None,
    )


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "random_state": 42,
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_valid)[:, 1]
    return roc_auc_score(y_valid, proba)


print("\n-----------------------------------------------------")
print("Running Optuna Hyperparameter Tuning for model_a")
print("-----------------------------------------------------\n")

class_counts = y.value_counts()
if len(y) < 20 or class_counts.min() < 2:
    print("Dataset too small or imbalanced for hyperparameter search; using defaults.")
    model_a_params = {
        "n_estimators": 100,
        "max_depth": 7,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": None,
    }
else:
    study = optuna.create_study(direction="maximize", study_name="fraud_detection_model_a")
    study.optimize(objective, n_trials=20)
    model_a_params = study.best_params
    print("Best model_a trial score:", study.best_value)
    print("Best model_a parameters:", model_a_params)

model_a = RandomForestClassifier(**model_a_params, random_state=42)
model_a.fit(X_train, y_train)

model_b = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)
model_b.fit(X_train, y_train)

joblib.dump(model_a, FEATURE_STORE_DIR / "model_a.pkl")
joblib.dump(model_b, FEATURE_STORE_DIR / "model_b.pkl")
joblib.dump(model_a, FEATURE_STORE_DIR / "fraud_model.pkl")

onnx_model_a_path = export_model_to_onnx(model_a, list(X.columns), FEATURE_STORE_DIR / "model_a.onnx")
onnx_model_b_path = export_model_to_onnx(model_b, list(X.columns), FEATURE_STORE_DIR / "model_b.onnx")

print("Models saved successfully.")
print(f"Files: model_a.pkl, model_b.pkl, fraud_model.pkl, {onnx_model_a_path}, {onnx_model_b_path}")

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
else:
    mlflow.set_tracking_uri(f"file://{REPO_ROOT / 'mlruns'}")

mlflow.set_experiment("fraud_detection_ab")

for model_name, model, params in [
    ("model_a", model_a, model_a_params),
    ("model_b", model_b, {"C": 1.0, "max_iter": 2000, "class_weight": "balanced"}),
]:
    with mlflow.start_run(run_name=f"{model_name}_ab_test"):
        for key, value in params.items():
            mlflow.log_param(key, value)

        if hasattr(model, "predict_proba"):
            valid_proba = model.predict_proba(X_valid)[:, 1]
        else:
            valid_proba = model.decisi   on_function(X_valid)
        valid_auc = roc_auc_score(y_valid, valid_proba)
        mlflow.log_metric("validation_roc_auc", float(valid_auc))
        mlflow.log_artifact(str(FEATURE_STORE_DIR / f"{model_name}.pkl"))
        print(f"Logged {model_name} validation AUC ({valid_auc}) to MLflow")

try:
    import bentoml

    try:
        bentoml.sklearn.save_model("fraud_model", model_a)
        bentoml.sklearn.save_model("fraud_model_challenger", model_b)
        print("Registered model_a and model_b with BentoML.")
    except Exception as e:
        print("BentoML model registration failed:", e)
except Exception:
    pass

