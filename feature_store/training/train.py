import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

print("\n----------------------------------")
print("Loading Data and Training Model")
print("----------------------------------\n")

df = pd.read_parquet("fraud_transactions.parquet")

print(df)

X = df[["transaction_amount", "account_balance", "is_high_risk"]]
y = [1, 0, 1, 0, 1]

print("\n----------------------------------")
print("Training Random Forest Classifier")
print("----------------------------------\n")

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

joblib.dump(model, "fraud_model.pkl")

print("\nModel Saved Successfully")