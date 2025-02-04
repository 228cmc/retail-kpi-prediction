import os
import pandas as pd

class TestEngineering:
    """Class to test the feature engineering process."""

    def __init__(self, processed_path):
        self.processed_path = processed_path

    def test_features_file(self):
        """Test for features.csv."""
        file_path = os.path.join(self.processed_path, "features.csv")
        assert os.path.exists(file_path), f"File 'features.csv' not found in {self.processed_path}."

        df = pd.read_csv(file_path)
        
        # Columnas que deben existir en features.csv
        required_columns = ["quarter", "total_orders", "total_spend_index", 
                            "weekly_active_users_index", "spend_per_user"]
        
        # Verificar que todas las columnas requeridas existan
        for column in required_columns:
            assert column in df.columns, f"Column '{column}' is missing in features.csv."
        
        # Verificar que no existan valores nulos
        assert not df.isnull().values.any(), "features.csv contains null values."

        print("All feature engineering tests passed!")

if __name__ == "__main__":
    # Ruta de la carpeta processed donde se guardan los datos procesados
    processed_path = "../data/processed/"
    tester = TestEngineering(processed_path)
    tester.test_features_file()
