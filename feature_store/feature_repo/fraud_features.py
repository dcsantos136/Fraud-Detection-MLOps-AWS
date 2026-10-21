from datetime import timedelta
from feast import Entity
from feast import FeatureView
from feast import Field
from feast import FileSource
from feast.types import Float32, Int64
from feast.value_type import ValueType


# Entity    
customer = Entity(name="customer_id", value_type=ValueType.INT64, join_keys=["customer_id"],)

# Data Source
fraud_source = FileSource(path="fraud_transactions.parquet", timestamp_field="event_timestamp",)


# Feature View
fraud_feature_view = FeatureView(name="fraud_features", entities=[customer], ttl=timedelta(days=1),
    schema=[Field( name="transaction_amount", dtype=Float32,), Field(name="account_balance", dtype=Float32,),
            Field(name="is_high_risk",dtype=Int64,),], source=fraud_source, online=True,)
