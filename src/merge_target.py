import pandas as pd


customer_features = pd.read_csv(
    "data/processed/customer_features.csv"
)

target_df = pd.read_csv(
    "data/processed/rfm_target.csv"
)

final_df = customer_features.merge(
    target_df[
        ["CustomerId", "is_high_risk"]
    ],
    on="CustomerId",
    how="left"
)

final_df.to_csv(
    "data/processed/model_dataset.csv",
    index=False
)

print(final_df.shape)
print(final_df["is_high_risk"].value_counts())