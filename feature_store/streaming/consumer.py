import json
import time
from streaming_features import update_customer_features


STREAM_FILE = "transactions_stream.jsonl"


print("\n---------------------------------------")
print("CONSUMING TRANSACTIONS")
print("---------------------------------------\n")


processed_lines = 0

while True:
    try:
        with open(STREAM_FILE, "r") as f:
            lines = f.readlines()
        new_lines = lines[processed_lines:]

        for line in new_lines:
            transaction = json.loads(line)
            customer_id = transaction["customer_id"]
            amount = transaction["transaction_amount"]
            updated_features = (update_customer_features(customer_id,amount))
            print("\nCustomer:", customer_id)
            print("Updated Features:", updated_features)

        processed_lines = len(lines)

    except FileNotFoundError:
        pass

    time.sleep(1)