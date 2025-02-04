import os
import joblib

# Generar ruta absoluta para la carpeta `models`
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../models/"))

def test_model_file():
    """Test for the trained model file."""
    file_path = os.path.join(MODEL_PATH, "revenue_model.pkl")
    assert os.path.exists(file_path), "The trained model file does not exist."

    model = joblib.load(file_path)
    assert model is not None, "Failed to load the trained model."
