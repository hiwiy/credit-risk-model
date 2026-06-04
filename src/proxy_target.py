"""
Proxy target engineering using RFM analysis and KMeans clustering.

Creates:
    is_high_risk

based on customer engagement behavior.
"""

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, Monetary metrics.
    """

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    snapshot_date = (
        df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    snapshot_date - x.max()
                ).days
            ),
            Frequency=("TransactionId", "count"),
            Monetary=("Value", "sum")
        )
        .reset_index()
    )

    return rfm


def cluster_customers(
    rfm_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Cluster customers into 3 RFM groups.
    """

    rfm_df = rfm_df.copy()

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        rfm_df[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=RANDOM_STATE,
        n_init=10
    )

    rfm_df["cluster"] = kmeans.fit_predict(
        scaled_features
    )

    return rfm_df


def identify_high_risk_cluster(
    clustered_df: pd.DataFrame
):
    """
    Identify least engaged customer cluster.

    High Recency
    Low Frequency
    Low Monetary
    """

    cluster_summary = (
        clustered_df
        .groupby("cluster")
        .agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean")
        )
    )

    cluster_summary["risk_score"] = (
        cluster_summary["Recency"]
        - cluster_summary["Frequency"]
        - cluster_summary["Monetary"] / 1000
    )

    high_risk_cluster = (
        cluster_summary["risk_score"]
        .idxmax()
    )

    return high_risk_cluster, cluster_summary


def create_proxy_target(
    clustered_df: pd.DataFrame,
    high_risk_cluster: int
) -> pd.DataFrame:
    """
    Create binary target variable.
    """

    clustered_df = clustered_df.copy()

    clustered_df["is_high_risk"] = (
        clustered_df["cluster"]
        == high_risk_cluster
    ).astype(int)

    return clustered_df


def build_proxy_target(
    input_path="data/raw/data.csv",
    output_path="data/processed/rfm_target.csv"
):
    """
    Full RFM target generation workflow.
    """

    df = pd.read_csv(input_path)

    rfm = calculate_rfm(df)

    clustered = cluster_customers(rfm)

    high_risk_cluster, summary = (
        identify_high_risk_cluster(clustered)
    )

    final_df = create_proxy_target(
        clustered,
        high_risk_cluster
    )

    final_df.to_csv(
        output_path,
        index=False
    )

    print("\nCluster Summary")
    print(summary)

    print(
        f"\nHigh Risk Cluster: {high_risk_cluster}"
    )

    print(
        "\nTarget Distribution"
    )

    print(
        final_df["is_high_risk"]
        .value_counts()
    )

    return final_df


if __name__ == "__main__":
    build_proxy_target()