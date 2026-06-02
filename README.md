 # Telecom Customer Churn Prediction Model

## 📌 Project Overview
This project applies machine learning to solve a critical revenue and customer retention challenge: **predicting customer churn**. Using historical telecom data, this predictive model identifies which subscribers are at a high risk of canceling their services, allowing Customer Success and Sales operations teams to intervene proactively.

## 🛠️ Tech Stack & Tools
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn
* **Environment:** Google Colab / Jupyter Notebook
* **Algorithm:** Random Forest Classifier

## 📊 The Dataset
The project utilizes the **Telco Customer Churn dataset** containing 7,032 verified customer records across 20 distinct operational attributes, including:
* **Account Metrics:** Tenure, contract type, payment method, monthly charges, and total charges.
* **Service Usage:** Internet type (DSL/Fiber optic), tech support access, online security, and streaming data.
* **Target Variable:** `Churn` (Whether the customer canceled their subscription).

## ⚙️ Machine Learning Pipeline
1. **Data Cleaning & Wrangling:** Evaluated structural integrity, forced text-based financial columns (`TotalCharges`) into numeric floats, and purged incomplete rows.
2. **Feature Engineering & Encoding:** Pruned non-predictive variables (`customerID`) and utilized One-Hot Encoding to convert categorical data into model-ready binary vectors.
3. **Validation Strategy:** Implemented an 80/20 train-test split to ensure rigorous validation on completely unseen customer profiles.
4. **Model Architecture:** Trained a Random Forest Classifier to evaluate data patterns across a network of randomized decision trees.

## 📈 Key Results & Business Impact
* **Model Accuracy:** Achieved a baseline classification accuracy of **78.45%** on the test dataset.
* **Operational Value:** Provides data-driven risk indicators that allow retention teams to optimize customer lifetime value (LTV) and reduce revenue churn.

## 🚀 How to Run This Project
1. Clone this repository to your local machine or open it directly in Google Colab.
2. Ensure you have `pandas` and `scikit-learn` installed.
3. Run the notebook cells sequentially to execute the full data cleaning, preprocessing, and model training pipeline.
