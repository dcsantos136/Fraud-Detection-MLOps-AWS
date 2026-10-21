# Feature Engineering Pipeline for Fraud Detection

# Import Libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import ( StandardScaler, OneHotEncoder)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


# Fraud Detection Dataset

fraud_data = {
    "transaction_amount": [ 1200, 50, 3000, 400, 5000, 150, 2500, 90, 7000, 100, 4300, 220, 1500, 8000, 60],
    "transaction_type": [ "Online", "POS", "Online", "ATM", "Online", "POS", "ATM", "POS", "Online", "ATM",
        "Online", "POS", "ATM", "Online", "POS" ],
    "customer_age": [ 25, 45, 35, 50, 29, 40, 33, 38, 31, 47,  np.nan, 36, 44, 28, 52 ],
    "account_balance": [ 15000, 3000,  25000, 4000, 50000, 3500,  20000, np.nan, 60000, 4500,  30000, 5000, 18000, 70000, 25000],
    "merchant_category": [ "Electronics", "Groceries", "Travel", "Cash Withdrawal", "Luxury", "Groceries",
        "Travel", "Food", "Luxury",  "Cash Withdrawal", "Electronics", "Food",  "Travel",  "Luxury", "Groceries"],
    "is_fraud": [ 1,  0, 1, 0, 1,  0, 1,  0, 1,  0, 1, 0, 0,  1, 0 ]
}

df = pd.DataFrame(fraud_data)

# Save raw data to CSV
df.to_csv( "data/fraud_data.csv", index=False)

print("\n================================================")
print("Raw Fraud Detection Dataset")
print("==================================================\n")
print(df.head())

# Split features and target variable
X = df.drop( "is_fraud", axis=1)
y = df["is_fraud"]

# Feature Engineering Pipeline Steps
numerical_features = [ "transaction_amount", "customer_age", "account_balance" ]
categorical_features = [ "transaction_type", "merchant_category" ]


# Numerical Pipeline
numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])

# Categorical Pipeline
categorical_pipeline = Pipeline(steps=[( "imputer", SimpleImputer(strategy="most_frequent")), 
                                       ( "encoder", OneHotEncoder(handle_unknown="ignore"))])

#  Combine Pipelines into ColumnTransformer
preprocessor = ColumnTransformer(transformers=[("num", numerical_pipeline, numerical_features ),
                                               ("cat", categorical_pipeline, categorical_features)])


# Combine Preprocessor with Classifier in a Pipeline
model_pipeline = Pipeline(steps=[( "preprocessing", preprocessor),
                                 ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))])


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


# Train the Pipeline
print("\n=========================================================================")
print("Training the Feature Engineering Pipeline with Random Forest Classifier")
print("===========================================================================\n")

model_pipeline.fit( X_train, y_train)
print("Pipeline Training Completed Successfully!")

# Transform Features
processed_features = model_pipeline.named_steps[ "preprocessing"].transform(X)


# Convert Processed Features to DataFrame for Better Visualization
processed_df = pd.DataFrame(
    processed_features.toarray()
    if hasattr(processed_features, "toarray")
    else processed_features)


# Save Processed Features to CSV
processed_df.to_csv( "data/processed_features.csv", index=False )

print("\n===================================================")
print("Processed Features After Feature Engineering")
print("===================================================\n")

print(processed_df.head())


#  Make Predictions
predictions = model_pipeline.predict(X_test)

print("\n===================================================")
print("Model Predictions on Test Set")
print("===================================================\n")

print(predictions)


# Evaluate Model
print("\n===================================================")
print("Classification Report for Test Set Predictions")
print("===================================================\n")

print(classification_report( y_test, predictions))


# Save the Entire Pipeline
joblib.dump(model_pipeline, "data/fraud_feature_pipeline.pkl")

print("\n=============================================================")
print("Pipeline Saved Successfully as 'fraud_feature_pipeline.pkl'")
print("=============================================================\n")

print("Saved File: data/fraud_feature_pipeline.pkl")

# Load the Pipeline for Future Use
loaded_pipeline = joblib.load("data/fraud_feature_pipeline.pkl")

print("\n=============================================================")
print("Pipeline Loaded Successfully from 'fraud_feature_pipeline.pkl'")
print("=============================================================\n")


# Test the Loaded Pipeline with a New Transaction
new_transaction = pd.DataFrame({ "transaction_amount": [4200], "transaction_type": ["Online"], "customer_age": [31], 
                                "account_balance": [22000], "merchant_category": ["Electronics"]})

prediction = loaded_pipeline.predict(new_transaction)

print("\n=========================================================================")
print("New Transaction Prediction")
print("===========================================================================\n")

if prediction[0] == 1:
    print("Fraudulent Transaction Detected")
else:
    print("Legitimate Transaction")



