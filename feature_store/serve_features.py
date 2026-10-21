# Serve features for real-time fraud detection

from feast import FeatureStore
import pandas as pd
import joblib


print("\n-----------------------------------------------------")
print("Connecting to Feature Store")
print("-----------------------------------------------------\n")

store = FeatureStore(repo_path="feature_repo")


# Online feature retrieval for a specific entity
print("\n-----------------------------------------------------")
print("Retrieving Online Features ")
print("-----------------------------------------------------\n")

online_features = store.get_online_features(
    features=[ "fraud_features:transaction_amount", "fraud_features:account_balance", "fraud_features:is_high_risk" ],
    entity_rows=[{"customer_id": 1001}]).to_dict()

print(online_features)
 

# Load Model
print("\n-----------------------------------------------------")
print("Loading Model for Inference")
print("-----------------------------------------------------\n")

model = joblib.load("fraud_model.pkl")


# Prepare input data for prediction
input_df = pd.DataFrame({ "transaction_amount": online_features["transaction_amount"], 
                         "account_balance": online_features["account_balance"],
                         "is_high_risk": online_features["is_high_risk"]})

print(input_df)


# Predict fraud using the model and the retrieved features
prediction = model.predict(input_df)

print("\n-----------------------------------------------------")
print("Fraud Prediction Result")
print("-----------------------------------------------------\n")

print("Prediction:", prediction[0])

if prediction[0] == 1:
    print("High Risk Transaction")
else:
    print("Low Risk Transaction")