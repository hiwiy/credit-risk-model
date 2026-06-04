import pandas as pd

from src.data_processing import (
    validate_input_data,
    extract_time_features,
    create_aggregate_features,
)


def sample_raw_data():
    return pd.DataFrame(
        {
            "TransactionId": ["T1", "T2", "T3"],
            "CustomerId": ["C1", "C1", "C2"],
            "Amount": [100.0, 200.0, 50.0],
            "Value": [100.0, 200.0, 50.0],
            "TransactionStartTime": [
                "2023-01-01 10:00:00",
                "2023-01-02 12:00:00",
                "2023-01-03 14:00:00",
            ],
            "ProductCategory": ["airtime", "airtime", "utility_bill"],
            "ChannelId": ["web", "web", "android"],
            "ProviderId": ["provider1", "provider1", "provider2"],
            "PricingStrategy": ["strategy1", "strategy1", "strategy2"],
        }
    )


def test_extract_time_features_creates_expected_columns():
    df = sample_raw_data()

    result = extract_time_features(df)

    expected_columns = [
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_create_aggregate_features_returns_customer_level_rows():
    df = sample_raw_data()
    df = extract_time_features(df)

    result = create_aggregate_features(df)

    assert result.shape[0] == 2
    assert "total_transaction_amount" in result.columns
    assert "transaction_count" in result.columns


def test_validate_input_data_raises_error_for_missing_column():
    df = sample_raw_data().drop(columns=["Amount"])

    try:
        validate_input_data(df)
        assert False
    except ValueError as error:
        assert "Missing required columns" in str(error)