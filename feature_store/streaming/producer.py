import json
import random
import time


STREAM_FILE = "transactions_stream.jsonl"
customers = [1001, 1002, 1003, 1004, 1005]


print("\n---------------------------------------")
print("Starting Transaction Producer")
print("---------------------------------------\n")

while True:

    transaction = {"customer_id": random.choice(customers),
        "transaction_amount": random.randint(100, 5000)}

    with open(STREAM_FILE, "a") as f:
        f.write(json.dumps(transaction) + "\n" )

    print("New Transaction:", transaction)

    time.sleep(2)