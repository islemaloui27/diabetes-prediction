# Diabetes Prediction

Machine learning project predicting diabetes from patient health indicators (`PFA.ipynb`).

## Workflow
- **Exploration** – data overview, unique values, missing-value analysis, distributions
- **Cleaning** – fixing malformed BMI values, handling missing data
- **Correlation analysis** – heatmaps on numeric and one-hot-encoded features
- **Preprocessing** – scikit-learn `Pipeline` / `ColumnTransformer` (imputation, scaling, encoding)
- **Class balancing** – SMOTE on the training set
- **Modeling** – Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Naive Bayes and XGBoost, compared on accuracy and F1

## Dataset
diabetes_prediction_dataset.csv (included): 100,000 rows, a deliberately noisy version of the Kaggle Diabetes prediction dataset with inconsistent labels, malformed BMI values and missing data. The notebook reads it from /content/ (Google Colab); change the path to run it locally.

## Tech stack
Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · matplotlib · seaborn
