

# **Retail KPI Prediction**

This project predicts the revenue of an online retailer based on order and transaction data. The workflow includes data cleaning, feature engineering, model training, evaluation, and generating results in an organized pipeline.



## **Project Structure**

```
Retail-KPI-Prediction/
│
├── data/
│   ├── processed/                 datasets
│   │   ├── features.csv
│   │   ├── orders_cleaned.csv
│   │   ├── reported_cleaned.csv
│   │   └── transactions_cleaned.csv
│   ├── raw/                      # Original raw dataset
│   │   └── data_task.xlsx
│
├── models/                       # Trained models
│
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_processing.ipynb
│   ├── 03_modelling.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_results_reporting.ipynb
│
├── reports/               # Project reports (if applicable)
│
├── scripts/                # Core scripts for the pipeline
│   ├── cleaning.py
│   ├── engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
│
├── test/                         # Unit tests for the pipeline
│   ├── test_cleaning.py
│   ├── test_engineering.py
│   └── test_train_model.py
│
├── requirements.txt                # Python dependencies
├── README.md                     # Project documentation
└── .gitignore
```



## **Installation**

To run the project, follow these steps:


1. Set up a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```



## **Workflow**

Follow this sequence to run the project:

1. **Data Cleaning**:  
   Run `cleaning.py` to clean and prepare raw data:
   ```bash
   python scripts/cleaning.py
   ```

2. **Feature Engineering**:  
   Generate features from cleaned data using `engineering.py`:
   ```bash
   python scripts/engineering.py
   ```

3. **Model Training**:  
   Train the model using `train_model.py`:
   ```bash
   python scripts/train_model.py
   ```

4. **Model Evaluation**:  
   Evaluate the trained model with `evaluate_model.py`:
   ```bash
   python scripts/evaluate_model.py
   ```



## **Notebooks**

The **notebooks** folder includes detailed step-by-step documentation of the workflow:

1. `01_data_exploration.ipynb`: Explore the dataset.  
2. `02_data_processing.ipynb`: Clean and process the data.  
3. `03_modelling.ipynb`: Train models.  
4. `04_model_evaluation.ipynb`: Evaluate model performance.  
5. `05_results_reporting.ipynb`: Present results and insights.



## **Testing**

unit tests are available in the `test/` folder. Run the tests using:
```bash
pytest test/
```



## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.

