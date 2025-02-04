import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error

class ModelTrainer:
    def __init__(self, features_path, revenue_path, model_output_path):
        """
        Initializes the ModelTrainer class with paths for:
        - features_path (features CSV)
        - revenue_path (revenue CSV)
        - model_output_path (directory to save the trained model)
        """
        self.features_path = features_path
        self.revenue_path = revenue_path
        self.model_output_path = model_output_path

    def load_data(self):
        """
        Loads data from the features and revenue CSV files, cleans
        the 'quarter' column in features, and merges the two datasets.
        Returns the merged DataFrame.
        """
        # Load the datasets
        features = pd.read_csv(self.features_path)
        revenue = pd.read_csv(self.revenue_path)

        # Remove any spaces in the 'quarter' column in features 
        # so that it matches the 'period' column in the revenue dataset
        features["quarter"] = features["quarter"].str.replace(" ", "", regex=False)

        # Debugging: Check unique values after the adjustment
        print("Unique values in features['quarter'] after adjustment:", features["quarter"].unique())
        print("Unique values in revenue['period']:", revenue["period"].unique())

        # Merge datasets on quarter/period
        df = pd.merge(features, revenue, left_on="quarter", right_on="period", how="inner")

        # Debugging: Check rows, columns, and specific quarter data after merge
        print("Number of rows after merge:", df.shape[0])
        print("Columns in merged DataFrame:", df.columns)
        print("Rows where period == '2022Q4' after merge:\n", df[df["period"] == "2022Q4"])

        # Drop unnecessary columns (besides 'period') if they exist
        df = df.drop(columns=["quarter", "start_date", "end_date"], errors="ignore")

        return df

    def train_model(self, df):
        """
        Trains a Linear Regression model to predict the 'revenue_index' from the merged DataFrame.
        The test set is filtered for 2022Q4, and the rest is used for training.
        """
        # Verify the 'period' column exists
        if "period" not in df.columns:
            raise ValueError("The 'period' column does not exist in the merged DataFrame.")

        # Debugging: Display the total number of rows in the dataset
        print(f"Total rows in the dataset: {df.shape[0]}")

        # Separate the dataset into training (all except 2022Q4) and testing (just 2022Q4)
        test = df[df["period"] == "2022Q4"]
        train = df[df["period"] != "2022Q4"]

        # Debugging: Check the sizes of training and testing sets
        print("Number of rows in training set (excluding 2022Q4):", train.shape[0])
        print("Number of rows in testing set (2022Q4):", test.shape[0])

        # Verify that both sets are non-empty
        if train.empty:
            raise ValueError("Training set is empty.")
        if test.empty:
            raise ValueError("Test set is empty.")

        # Define X (features) and y (target) for training
        X_train = train.drop(columns=["revenue_index", "period"], errors="ignore")
        y_train = train["revenue_index"]

        X_test = test.drop(columns=["revenue_index", "period"], errors="ignore")
        y_test = test["revenue_index"]

        # Train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on both training and test sets
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Evaluate the model
        r2_train = r2_score(y_train, y_pred_train)
        error_test = mean_absolute_percentage_error(y_test, y_pred_test)

        print(f"Model trained: R² (train) = {r2_train:.4f}, MAPE (test) = {error_test:.4%}")

        # Create output directory if it doesn't exist, and save the model
        os.makedirs(self.model_output_path, exist_ok=True)
        joblib.dump(model, os.path.join(self.model_output_path, "revenue_model.pkl"))

    def process(self):
        """
        Orchestrates the data loading, model training, 
        and model saving process.
        """
        df = self.load_data()
        self.train_model(df)

if __name__ == "__main__":
    # Define paths for the features, revenue data, and the model output directory
    features_path = "../data/processed/features.csv"
    revenue_path = "../data/processed/reported_cleaned.csv"
    model_output_path = "../models/"

    # Create an instance of ModelTrainer and run the process
    trainer = ModelTrainer(features_path, revenue_path, model_output_path)
    trainer.process()
