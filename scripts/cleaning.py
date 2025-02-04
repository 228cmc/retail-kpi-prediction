import pandas as pd
import os

class DataProcessor:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    def clean_order_numbers(self, df_orders):
        """
        Cleans the order_numbers dataset by removing incorrect values.
        """
        df_orders = df_orders.copy()
        df_orders['date'] = pd.to_datetime(df_orders['date'])
        df_orders = df_orders.sort_values(by='date')
        df_orders['order_diff'] = df_orders['order_number'].diff()
        df_orders = df_orders[df_orders['order_diff'] >= 0]
        df_orders = df_orders.drop(columns=['order_diff'])
        return df_orders

    def clean_transaction_data(self, df_transactions):
        """
        Cleans the transaction_data dataset.
        """
        df_transactions = df_transactions.copy()
        df_transactions['date'] = pd.to_datetime(df_transactions['date'])
        return df_transactions

    def clean_reported_data(self, df_reported):
        """
        Cleans the reported_data dataset.
        """
        df_reported = df_reported.copy()

        df_reported['start_date'] = pd.to_datetime(df_reported['start_date'])
        df_reported['end_date'] = pd.to_datetime(df_reported['end_date'])

        df_reported['period'] = df_reported['period'].str.replace(" ", "", regex=False)
        
        return df_reported


    def process(self):
        # Load datasets
        df_orders = pd.read_excel(self.raw_data_path, sheet_name="order_numbers")

        df_transactions = pd.read_excel(self.raw_data_path, sheet_name="transaction_data")
        df_reported = pd.read_excel(self.raw_data_path, sheet_name="reported_data")

        # Clean data
        df_orders = self.clean_order_numbers(df_orders)
        df_transactions = self.clean_transaction_data(df_transactions)
        df_reported = self.clean_reported_data(df_reported)

        # Save cleaned data
        os.makedirs(self.processed_data_path, exist_ok=True)
        df_orders.to_csv(os.path.join(self.processed_data_path, "orders_cleaned.csv"), index=False)
        df_transactions.to_csv(os.path.join(self.processed_data_path, "transactions_cleaned.csv"), index=False)
        df_reported.to_csv(os.path.join(self.processed_data_path, "reported_cleaned.csv"), index=False)
        print("Data cleaned and saved successfully.")

if __name__ == "__main__":
    raw_path = "../data/raw/data_task.xlsx"
    processed_path = "../data/processed/"
    processor = DataProcessor(raw_path, processed_path)
    processor.process()
