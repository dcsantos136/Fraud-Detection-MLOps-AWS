import hashlib

MODEL_OPTIONS = ("model_a", "model_b")


def choose_model_for_customer(customer_id, traffic_split=0.2, force_model=None):
    """Route a customer to either the control or challenger model.

    By default, 80% of requests go to model_a and 20% to model_b.
    The route is deterministic per customer_id so the same customer is
    consistently assigned to the same model for a given split policy.
    """
    if force_model is not None:
        if force_model not in MODEL_OPTIONS:
            raise ValueError(f"force_model must be one of {MODEL_OPTIONS}")
        return force_model

    if not 0 <= traffic_split <= 1:
        raise ValueError("traffic_split must be between 0 and 1")

    digest = hashlib.md5(str(customer_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    split_pct = int(traffic_split * 100)

    return "model_b" if bucket < split_pct else "model_a"
