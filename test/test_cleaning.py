import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.abspath(os.path.join(BASE_DIR, "../data/processed/"))

def test_orders_cleaned():
    """ test for orders_cleaned.csv """
    file_path = os.path.join(PROCESSED_PATH, "orders_cleaned.csv")
    assert os.path.exists(file_path), f"{file_path} does not exist"

    df = pd.read_csv(file_path)
    assert "order_number" in df.columns, "column 'order_number' is missing in orders_cleaned.csv"

    assert not df.isnull().values.any(), "orders_cleaned.csv contains null values."


def test_transactions_cleaned():
    """test for transactions_cleaned.csv.
    """
    file_path = os.path.join(PROCESSED_PATH, "transactions_cleaned.csv")
    assert os.path.exists(file_path), f"{file_path} does not exist."

    df = pd.read_csv(file_path)
    assert "total_spend_index" in df.columns, "column 'total_spend_index' is missing in transactions_cleaned.csv"
    assert not df.isnull().values.any(), "transactions_cleaned.csv contains null values."


def test_reported_cleaned():
    """Test for reported_cleaned.csv
    """
    file_path = os.path.join(PROCESSED_PATH, "reported_cleaned.csv")
    assert os.path.exists(file_path), f"{file_path} does not exist."

    df = pd.read_csv(file_path)
    assert "revenue_index" in df.columns, "Column 'revenue_index' is missing in reported_cleaned.csv."
    assert not df.isnull().values.any(), "reported_cleaned.csv contains null values"
