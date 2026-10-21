# Validation Pipeline for Fraud Detection Dataset

# Install Great Expectations for data validation
# pip install great_expectations==0.17.23


# Importing necessary libraries
import pandas as pd
import json
from great_expectations.dataset import PandasDataset

# Load the dataset
df = pd.read_csv("data/fraud_data.csv")

print("\n----------------------------------------------------")
print("RAW FRAUD DETECTION DATA")
print("----------------------------------------------------\n")

print(df.head())


# Create great expectations dataset
gx_df = PandasDataset(df)

# Create expectation suite
expectation_suite = { "transaction_amount": { "min": 0, "max": 100000}, "customer_age": {"min": 18, "max": 100},
                     "allowed_transaction_types": [ "Online",  "POS", "ATM" ]}

with open("expectations/expectation_suite.json", "w") as f:
    json.dump(expectation_suite, f, indent=4)

print("\n----------------------------------------------------")
print("EXPECTATION SUITE SAVED")
print("----------------------------------------------------\n")



# Validate the dataset against the expectation suite
print("\n----------------------------------------------------")
print("VALIDATING TRANSACTION AMOUNT")
print("----------------------------------------------------\n")

result_1 = gx_df.expect_column_values_to_be_between( "transaction_amount", min_value=0, max_value=100000)
print(result_1)


# Validate customer age
print("\n----------------------------------------------------")
print("VALIDATING CUSTOMER AGE")
print("----------------------------------------------------\n")

result_2 = gx_df.expect_column_values_to_be_between( "customer_age", min_value=18, max_value=100)
print(result_2)


# Null Value Check for Transaction Type 
print("\n----------------------------------------------------")
print("CHECKING NULL VALUES")
print("----------------------------------------------------\n")

result_3 = gx_df.expect_column_values_to_not_be_null("transaction_type")
print(result_3)


# Validate transaction types
print("\n----------------------------------------------------")
print("VALIDATING TRANSACTION TYPES")
print("----------------------------------------------------\n")

result_4 = gx_df.expect_column_values_to_be_in_set("transaction_type", ["Online", "POS", "ATM"])
print(result_4)


# Validate account balance 
print("\n----------------------------------------------------")
print("VALIDATING ACCOUNT BALANCE")
print("----------------------------------------------------\n")

result_5 = gx_df.expect_column_values_to_be_between( "account_balance", min_value=0, max_value=1000000)
print(result_5)


# Validation summary
print("\n----------------------------------------------------")
print("VALIDATION SUMMARY")
print("----------------------------------------------------\n")

validation_results = { "Transaction Amount Validation": result_1.success, "Customer Age Validation": result_2.success,
                       "Transaction Type Null Check": result_3.success, "Transaction Type Validation": result_4.success,
                       "Account Balance Validation": result_5.success}

for check, status in validation_results.items():
    if status:
        print(f"[PASSED] {check}")
    else:
        print(f"[FAILED] {check}")


# Save validation results to a text file
with open("expectations/validation_results.txt", "w") as f:
    for check, status in validation_results.items():
        if status:
            f.write(f"[PASSED] {check}\n")
        else:
            f.write(f"[FAILED] {check}\n")

print("\n----------------------------------------------------")
print("VALIDATION RESULTS SAVED")
print("----------------------------------------------------\n")
 

# Detect invalid records based on the expectation suite
invalid_rows = df[ (df["transaction_amount"] < 0) | (df["customer_age"] > 100) | (df["account_balance"] < 0) | 
                  (df["transaction_type"].isnull())]

print("\n----------------------------------------------------")
print("INVALID RECORDS")
print("----------------------------------------------------\n")

print(invalid_rows)

# Save invalid records to a CSV file for review
invalid_rows.to_csv( "expectations/failed_records.csv", index=False)

print("\n----------------------------------------------------")
print("FAILED RECORDS SAVED")
print("----------------------------------------------------\n")


# Create Clean text file with only valid records for downstream processing
clean_df = df[ (df["transaction_amount"] >= 0) & (df["customer_age"].fillna(0) <= 100) 
              & (df["account_balance"] >= 0)  &  (df["transaction_type"].notnull())]

print("\n----------------------------------------------------")
print("CLEANED DATA")
print("----------------------------------------------------\n")

print(clean_df.head())

