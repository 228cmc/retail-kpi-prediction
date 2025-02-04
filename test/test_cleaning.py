import os
import pandas as pd

class TestCleaning:

    """Class to test the data cleaning process."""

    def __init__(self, processed_path):

        self.processed_path = processed_path

    def test_orders_cleaned(self):

        """Test for orders_cleaned.csv."""

        file_path = os.path.join(self.processed_path, "orders_cleaned.csv")
        df = pd.read_csv(file_path)
        assert "order_number" in df.columns, "Column 'order_number' is missing in orders_cleaned.csv."
        assert not df.isnull().values.any(), "orders_cleaned.csv contains null values."

    def test_transactions_cleaned(self):

        """Test for transactions_cleaned.csv."""

        file_path = os.path.join(self.processed_path, "transactions_cleaned.csv")
        df = pd.read_csv(file_path)
        assert "total_spend_index" in df.columns, "Column 'total_spend_index' is missing in transactions_cleaned.csv."
        assert not df.isnull().values.any(), "transactions_cleaned.csv contains null values."

    def test_reported_cleaned(self):

        """Test for reported_cleaned.csv."""
        file_path = os.path.join(self.processed_path, "reported_cleaned.csv")
        df = pd.read_csv(file_path)
        assert "revenue_index" in df.columns, "Column 'revenue_index' is missing in reported_cleaned.csv."
        assert not df.isnull().values.any(), "reported_cleaned.csv contains null values."


if __name__ == "__main__":

    processed_path = "../data/processed/"

    tester = TestCleaning(processed_path)
    tester.test_orders_cleaned()
    tester.test_transactions_cleaned()
    tester.test_reported_cleaned()
    
    print("All cleaning tests passed!")
