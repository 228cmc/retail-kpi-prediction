import os
import joblib

class TestModelTraining:

    """Class to test the model training process."""

    def __init__(self, model_path):

        self.model_path = model_path#

    def test_model_file(self):
        """Test for the trained model file."""
        
        file_path = os.path.join(self.model_path, "revenue_model.pkl")
        assert os.path.exists(file_path), "The trained model file does not exist."
        
        # Try to load the model
        model = joblib.load(file_path)
        assert model is not None, "Failed to load the trained model."


if __name__ == "__main__":

    model_path = "../models/"
    tester = TestModelTraining(model_path)
    tester.test_model_file()
    print("All training model tests passed!")
