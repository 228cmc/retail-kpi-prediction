import os
import pandas as pd

class FeatureEngineer:

    def __init__(self, cleaned_data_path, output_path):

        self.cleaned_data_path = cleaned_data_path
        self.output_path = output_path

    def aggregate_to_quarterly(self, orders, transactions):


        """
        Aggregates daily data to quarterly data.
        """
        # Convert dates to periods for aggregation
        orders["quarter"] = orders["date"].dt.to_period("Q")
        transactions["quarter"] = transactions["date"].dt.to_period("Q")

        # Aggregate orders: Sum daily orders per quarter
        orders_agg = orders.groupby("quarter")["order_number"].count().reset_index()
        orders_agg.rename(columns={"order_number": "total_orders"}, inplace=True)

        # Aggregate transactions: Average indices per quarter
        transactions_agg = transactions.groupby("quarter").agg({
            "total_spend_index": "mean",
            "weekly_active_users_index": "mean"
        }).reset_index()

        # Merge aggregated datasets
        quarterly_data = pd.merge(orders_agg, transactions_agg, on="quarter", how="left")
        return quarterly_data


    def normalize_features(self, quarterly_data):

        """
        Normalizes features for better comparability.
        """
        # Add a spend-per-user feature
        quarterly_data["spend_per_user"] = (
            quarterly_data["total_spend_index"] / quarterly_data["weekly_active_users_index"]
        )

        return quarterly_data

    def process(self):
        # Load cleaned data

        orders = pd.read_csv(os.path.join(self.cleaned_data_path, "orders_cleaned.csv"))
        transactions = pd.read_csv(os.path.join(self.cleaned_data_path, "transactions_cleaned.csv"))

        # Convert date columns to datetime

        orders["date"] = pd.to_datetime(orders["date"])
        transactions["date"] = pd.to_datetime(transactions["date"])

        # Aggregate and create features
        quarterly_data = self.aggregate_to_quarterly(orders, transactions)
        quarterly_data = self.normalize_features(quarterly_data)

        # Save engineered features
        os.makedirs(self.output_path, exist_ok=True)
        quarterly_data.to_csv(os.path.join(self.output_path, "features.csv"), index=False)
        print("Feature engineering it's done, the  data  was saved to 'features.csv'.")

if __name__ == "__main__":
    cleaned_path = "../data/processed/"
    output_path = "../data/processed/"

    engineer = FeatureEngineer(cleaned_path, output_path)
    engineer.process()
