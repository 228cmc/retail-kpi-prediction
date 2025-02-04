import os
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

class ModelEvaluator:
    def __init__(self, model_path, features_path, revenue_path, output_path):
        """
        Initializes the ModelEvaluator class
        Args:
        - model_path (str): Path to the saved model (pickle file).
        - features_path (str): Path to the processed features dataset

        - revenue_path (str): Path to the revenue dataset (target values)
        - output_path (str): Directory to save evaluation results

        """
        self.model_path = model_path
        self.features_path = features_path
        self.revenue_path = revenue_path
        self.output_path = output_path

    def load_test_data(self):
        """
        Loads and merges the test dataset (features + revenue).

        Returns:
        - pd.DataFrame: The test dataset
        """
        # Load features dataset
        features = pd.read_csv(self.features_path)

        # Load revenue dataset
        revenue = pd.read_csv(self.revenue_path)

        # Ensure period format matches between features and revenue
        features["quarter"] = features["quarter"].str.replace(" ", "", regex=False)

        # Merge on period (quarter)
        test_data = pd.merge(features, revenue, left_on="quarter", right_on="period", how="inner")

        # Drop unnecessary columns
        test_data = test_data.drop(columns=["quarter", "start_date", "end_date"], errors="ignore")

        return test_data

    def load_model(self):
        """
        Loads the trained model

        Returns:
        - Trained model object
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"The model file {self.model_path} does not exist.")
        
        model = joblib.load(self.model_path)
        return model

    def evaluate(self, model, test_data):
        """
        Evaluates the model on the test dataset.
        Args:
        - model: The trained model.
        - test_data (pd.DataFrame): The test dataset
        Returns:
        - dict: Dictionary of evaluation metrics
        """
        # Features used during training

        training_features = ["total_orders", "total_spend_index", "weekly_active_users_index", "spend_per_user"]
        
        # Ensure the test data contains only these features
        missing_features = [f for f in training_features if f not in test_data.columns]
        if missing_features:
            raise ValueError(f"Missing features in the test dataset: {missing_features}")

        X_test = test_data[training_features]
        y_test = test_data["revenue_index"]

        # Predictions

        y_pred = model.predict(X_test)

        # Metrics
        metrics = {
            "R2 Score": r2_score(y_test, y_pred),
            "Mean Absolute Error": mean_absolute_error(y_test, y_pred),

            "Mean Squared Error": mean_squared_error(y_test, y_pred),

            "Mean Absolute Percentage Error": mean_absolute_percentage_error(y_test, y_pred)
        }

        return metrics

    def save_evaluation(self, metrics):
        """
        Saves the evaluation results to a file

        Args:
        - metrics (dict): Evaluation metrics
        """
        os.makedirs(self.output_path, exist_ok=True)
        output_file = os.path.join(self.output_path, "evaluation_results.txt")
        
        with open(output_file, "w") as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        print(f"Evaluation results saved to {output_file}")

    def process(self):
        """
        Orchestrates the evaluation process.
        """
        print("Loading test data...")
        test_data = self.load_test_data()

        print("Loading trained model...")
        model = self.load_model()

        print("Evaluating model")

        metrics = self.evaluate(model, test_data)

        print("Saving evaluation results")
        self.save_evaluation(metrics)

        print("Evaluation completed")
        return metrics


if __name__ == "__main__":

    # Define paths
    model_path = "../models/revenue_model.pkl"
    features_path = "../data/processed/features.csv"  

    revenue_path = "../data/processed/reported_cleaned.csv"
    output_path = "../reports/"

    # Create an instance of ModelEvaluator and run the process
    evaluator = ModelEvaluator(model_path, features_path, revenue_path, output_path)
    results = evaluator.process()

    # Print metrics to console
    print("\nEvaluation Metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
