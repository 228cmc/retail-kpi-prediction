import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error


class ModelTrainer:
    def __init__(self, features_path, revenue_path, model_output_path):
        self.features_path = features_path
        self.revenue_path = revenue_path
        self.model_output_path = model_output_path

    def load_data(self):
        """Loads and merges feature and revenue datasets."""
        # Cargar datasets
        features = pd.read_csv(self.features_path)
        revenue = pd.read_csv(self.revenue_path)

        # Ajustar formatos: Eliminar espacios en 'quarter' para que coincida con 'period'
        features["quarter"] = features["quarter"].str.replace(" ", "", regex=False)

        # Depuración: Verificar valores únicos después de la unificación
        print("Valores únicos en features['quarter'] (después de ajuste):", features["quarter"].unique())
        print("Valores únicos en revenue['period'] (después de ajuste):", revenue["period"].unique())

        # Merge datasets on quarter/period
        df = pd.merge(features, revenue, left_on="quarter", right_on="period", how="inner")

        # Depuración: Verificar filas y columnas después del merge
        print("Filas después del merge:", df.shape[0])
        print("Columnas del DataFrame fusionado:", df.columns)

        # Depuración: Verificar filas específicas para '2022Q4'
        print("Filas con 'period' == '2022Q4' después del merge:", df[df["period"] == "2022Q4"])

        # Eliminar columnas innecesarias excepto 'period'
        df = df.drop(columns=["quarter", "start_date", "end_date"], errors="ignore")

        return df




    def train_model(self, df):
        """Trains a regression model to predict revenue index."""

        if "period" not in df.columns:
            raise ValueError("La columna 'period' no existe en el DataFrame fusionado.")

        # Dividir en conjuntos de entrenamiento y prueba
        print(f"Filas disponibles en el dataset total: {df.shape[0]}")

        # Filtrar datos de prueba (2022Q4) y entrenamiento (todo lo demás)
        test = df[df["period"] == "2022Q4"]
        train = df[df["period"] != "2022Q4"]

        # Depuración: Verificar tamaño de los conjuntos
        print("Filas en train (sin 2022Q4):", train.shape[0])
        print("Filas en test (2022Q4):", test.shape[0])

        # Verificar que haya datos en ambos conjuntos
        if train.empty:
            raise ValueError("El conjunto de entrenamiento está vacío.")
        if test.empty:
            raise ValueError("El conjunto de prueba está vacío.")

        # Definir X (características) e y (target)
        X_train = train.drop(columns=["revenue_index", "period"], errors="ignore")
        y_train = train["revenue_index"]

        X_test = test.drop(columns=["revenue_index", "period"], errors="ignore")
        y_test = test["revenue_index"]

        # Entrenar el modelo
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Evaluar el modelo
        r2_train = r2_score(y_train, y_pred_train)
        error_test = mean_absolute_percentage_error(y_test, y_pred_test)

        print(f"Model trained: R² (train) = {r2_train:.4f}, MAPE (test) = {error_test:.4%}")

        # Guardar el modelo
        os.makedirs(self.model_output_path, exist_ok=True)
        joblib.dump(model, os.path.join(self.model_output_path, "revenue_model.pkl"))


    def process(self):
        
        """Orchestrates the data loading, model training, and saving."""
        df = self.load_data()
        self.train_model(df)


if __name__ == "__main__":
    features_path = "../data/processed/features.csv"
    revenue_path = "../data/processed/reported_cleaned.csv"
    model_output_path = "../models/"

    trainer = ModelTrainer(features_path, revenue_path, model_output_path)
    trainer.process()
