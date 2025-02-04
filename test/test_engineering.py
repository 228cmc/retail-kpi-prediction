import os
import pandas as pd

# Generar ruta absoluta para la carpeta `data/processed`
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.abspath(os.path.join(BASE_DIR, "../data/processed/"))

def test_features_file():
    """Test for features.csv."""
    file_path = os.path.join(PROCESSED_PATH, "features.csv")
    assert os.path.exists(file_path), f"{file_path} does not exist."

    df = pd.read_csv(file_path)

    required_columns = ["quarter", "total_orders", "total_spend_index", 
                        "weekly_active_users_index", "spend_per_user"]

    for column in required_columns:
        assert column in df.columns, f"Column '{column}' is missing in features.csv."

    assert not df.isnull().values.any(), "features.csv contains null values."
