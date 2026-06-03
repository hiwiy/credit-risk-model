"""
Data processing and feature engineering pipeline for the Bati Bank
Credit Risk Probability Model.

This module transforms raw transaction-level data into a customer-level,
model-ready dataset.
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


REQUIRED_COLUMNS = [
    "TransactionId",
    "CustomerId",
    "Amount",
    "Value",
    "TransactionStartTime",
    "ProductCategory",
    "ChannelId",
    "ProviderId",
    "PricingStrategy",
]


def validate_input_data(df: pd.DataFrame) -> None:
    """
    Validate that the raw dataset contains required columns.
    """

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from TransactionStartTime.
    """

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"],
        errors="coerce"
    )

    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year

    return df


def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer-level aggregate transaction features.
    """

    df = df.copy()

    customer_features = df.groupby("CustomerId").agg(
        total_transaction_amount=("Amount", "sum"),
        avg_transaction_amount=("Amount", "mean"),
        std_transaction_amount=("Amount", "std"),
        min_transaction_amount=("Amount", "min"),
        max_transaction_amount=("Amount", "max"),
        total_transaction_value=("Value", "sum"),
        avg_transaction_value=("Value", "mean"),
        transaction_count=("TransactionId", "count"),
        avg_transaction_hour=("transaction_hour", "mean"),
        avg_transaction_day=("transaction_day", "mean"),
        most_common_product_category=(
            "ProductCategory",
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ),
        most_common_channel=(
            "ChannelId",
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ),
        most_common_provider=(
            "ProviderId",
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ),
        most_common_pricing_strategy=(
            "PricingStrategy",
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ),
    ).reset_index()

    customer_features["std_transaction_amount"] = (
        customer_features["std_transaction_amount"].fillna(0)
    )

    return customer_features


def build_preprocessing_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline for numerical and categorical columns.
    """

    numeric_features = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_features = [
        col for col in numeric_features
        if col != "CustomerId"
    ]

    categorical_features = [
        col for col in categorical_features
        if col != "CustomerId"
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def process_data(df: pd.DataFrame):
    """
    Full feature engineering workflow.

    Returns:
        customer_features: customer-level dataframe before encoding
        processed_array: transformed model-ready array
        preprocessor: fitted preprocessing pipeline
    """

    validate_input_data(df)

    df = extract_time_features(df)

    customer_features = create_aggregate_features(df)

    preprocessor = build_preprocessing_pipeline(customer_features)

    feature_data = customer_features.drop(columns=["CustomerId"])

    processed_array = preprocessor.fit_transform(feature_data)

    return customer_features, processed_array, preprocessor


def save_processed_data(
    input_path: str = "data/raw/data.csv",
    output_path: str = "data/processed/customer_features.csv"
) -> pd.DataFrame:
    """
    Load raw data, engineer customer-level features, and save output.
    """

    df = pd.read_csv(input_path)

    customer_features, _, _ = process_data(df)

    customer_features.to_csv(output_path, index=False)

    return customer_features


if __name__ == "__main__":
    processed_df = save_processed_data()
    print("Processed customer-level dataset saved successfully.")
    print(f"Shape: {processed_df.shape}")