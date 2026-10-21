import json
import os


FEATURE_FILE = "streaming_features.json"


def load_features():

    if os.path.exists(FEATURE_FILE):
        try:
            with open(FEATURE_FILE, "r") as f:
                return json.load(f)

        except json.JSONDecodeError:
            return {}
    return {}

def save_features(features):
    with open(FEATURE_FILE, "w") as f:
        json.dump(features, f, indent=4)


def update_customer_features(customer_id, amount):
    features = load_features()
    customer_id = str(customer_id)

    if customer_id not in features:
        features[customer_id] = { "transaction_count": 0, "total_amount": 0, "average_amount": 0}

    customer = features[customer_id]
    customer["transaction_count"] += 1
    customer["total_amount"] += amount

    customer["average_amount"] = round( customer["total_amount"] / customer["transaction_count"], 2)

    save_features(features)

    return customer