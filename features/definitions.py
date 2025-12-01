from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

customer = Entity(name="customer_id")

churn_source = FileSource(
    path="data/churn_features.parquet",
    timestamp_field="event_timestamp",
)

churn_features_view = FeatureView(
    name="churn_features",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[
        Field(name="tenure_months", dtype=Int64),
        Field(name="monthly_charges", dtype=Float32),
        Field(name="total_charges", dtype=Float32),
    ],
    online=False,  # offline only
    source=churn_source,
)


