# Heart Disease Prediction using Machine Learning

## Overview

This project aims to develop predictive models for heart disease using machine learning techniques. The dataset used is the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) from Kaggle. The project involves data preprocessing, one-hot encoding, and model implementation using Decision Tree, Random Forest, and XGBoost classifiers.

## Libraries Used

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

## Project Steps

### 1. Data Loading and Preprocessing

1. **Load the Dataset**
   - The dataset is loaded from a CSV file using Pandas.

2. **One-Hot Encoding**
   - Categorical features are one-hot encoded to prepare them for machine learning models. The following categorical features are encoded:
     - Sex
     - ChestPainType
     - RestingECG
     - ExerciseAngina
     - ST_Slope

3. **Feature Selection**
   - The target variable is `HeartDisease`.
   - All other variables are used as input features for the models.

### 2. Data Splitting

- Split the dataset into training (80%) and validation (20%) sets using `train_test_split` from Scikit-learn.

### 3. Model Building and Evaluation

#### 3.1 Decision Tree Classifier

- **Hyperparameters Tuned:**
  - `min_samples_split`
  - `max_depth`

- **Metrics Evaluated:**
  - Training accuracy
  - Validation accuracy

- **Optimal Parameters:**
  - `min_samples_split = 50`
  - `max_depth = 3`

#### 3.2 Random Forest Classifier

- **Hyperparameters Tuned:**
  - `min_samples_split`
  - `max_depth`
  - `n_estimators`

- **Metrics Evaluated:**
  - Training accuracy
  - Validation accuracy

- **Optimal Parameters:**
  - `min_samples_split = 10`
  - `max_depth = 16`
  - `n_estimators = 100`

## Results

- **Decision Tree Classifier:**
  - Training Accuracy: `0.8583`
  - Validation Accuracy: `0.8641`

- **Random Forest Classifier:**
  - Training Accuracy: `0.9332`
  - Validation Accuracy: `0.8804`

## Conclusion

This project demonstrates the application of machine learning models to predict heart disease. The performance of Decision Tree and Random Forest classifiers was evaluated, and optimal hyperparameters were identified to improve model accuracy and reduce overfitting.
