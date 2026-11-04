
import pandas as pd


def create_features(df):
    """Create additional customer features for churn prediction."""

    df = df.copy()

    # Group customers according to tenure
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New", "Medium", "Long", "Loyal"]
    )

    # Calculate average monthly spending
    df["AverageMonthlySpend"] = (
        df["TotalCharges"] /
        df["tenure"].replace(0, 1)
    )

    # Count the services used by each customer
    service_columns = [
        "PhoneService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["TotalServices"] = (
        (df[service_columns] == "Yes").sum(axis=1)
    )

    return df
