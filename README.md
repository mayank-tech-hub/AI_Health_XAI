# AI-Based Early Disease Prediction with Explainable AI (XAI)

## Project Overview

This project presents an AI-based heart disease prediction system using Machine Learning and Explainable AI (XAI). The system predicts whether heart disease is present and provides probability-based results along with model explanations.

> **Disclaimer:** This project is developed for educational and research purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

## Problem Statement

Heart disease is a major health concern. Machine Learning can help identify patterns associated with heart disease, but model predictions should also be understandable. This project combines Machine Learning with Explainable AI to make predictions more interpretable.

## Objectives

- Predict heart disease presence.
- Compare multiple Machine Learning algorithms.
- Perform hyperparameter tuning.
- Select the best-performing model.
- Explain predictions using SHAP and LIME.
- Build an interactive Streamlit application.

## Dataset

The project uses the UCI Heart Disease Dataset.

- Total records: 303
- Input features: 13
- Problem type: Binary classification

### Target Classes

| Value | Meaning |
|---|---|
| 0 | No Heart Disease |
| 1 | Heart Disease Present |

### Input Features

| Feature | Description |
|---|---|
| age | Age of patient |
| sex | Sex |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression |
| slope | ST segment slope |
| ca | Number of major vessels |
| thal | Thalassemia |

## Data Preprocessing

- Missing-value imputation
- Numerical feature scaling
- Categorical feature encoding
- One-hot encoding
- Scikit-learn Pipeline
- ColumnTransformer

Original features: 13

Transformed features: 28

## Machine Learning Models

The following models were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. Support Vector Machine (SVM)

## Model Evaluation

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Final Tuned Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 88.52% | 83.87% | 92.86% | 88.14% | **96.65%** |
| XGBoost | 88.52% | 86.21% | 89.29% | 87.72% | 95.35% |
| Random Forest | 85.25% | 80.65% | 89.29% | 84.75% | 94.81% |
| SVM | 83.61% | 80.00% | 85.71% | 82.76% | 94.26% |

## Final Model

The final selected model is **Tuned Logistic Regression**.

Test ROC-AUC: **0.9665**

Test Recall: **92.86%**

The model was selected based on its strong ROC-AUC, high recall, and suitability for explainable predictions.

## Explainable AI

### SHAP

SHAP (SHapley Additive exPlanations) is used to understand how individual features influence model predictions.

The application displays the top contributing features and indicates whether each feature increases or decreases the estimated disease probability.

### LIME

LIME (Local Interpretable Model-agnostic Explanations) was also explored to understand individual predictions.

## Streamlit Application

An interactive Streamlit application allows users to enter patient information and receive:

- Heart disease prediction
- Disease probability
- No-disease probability
- SHAP-based explanation
- Top contributing features

### Application Flow

Patient Input -> Data Validation -> Preprocessing -> Logistic Regression -> Prediction + Probability -> SHAP Explanation

## Project Structure

AI_Health_XAI/

    app/
        app.py

    models/
        heart_disease_logistic_pipeline.pkl
        heart_disease_preprocessor.pkl
        model_info.pkl
        shap_background.pkl
        feature_names.pkl
        clean_feature_names.pkl

    notebooks/
        AI_Based_Early_Disease_Prediction_XAI.ipynb

    assets/

    .gitignore
    requirements.txt
    README.md

## Installation

Clone the repository and install the dependencies:

    pip install -r requirements.txt

## Run the Application

From the project root directory:

    streamlit run app/app.py

## Example Prediction

Sample patient result:

Prediction: No Heart Disease

Disease Probability: 16.55%

No Disease Probability: 83.45%

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- LIME
- Matplotlib
- Joblib
- Streamlit
- Google Colab

## Future Improvements

- Add additional disease prediction modules
- Use larger and more diverse datasets
- Improve probability calibration
- Add advanced risk visualization
- Add additional XAI techniques
- Improve user interface

## Project Purpose

This project demonstrates the practical application of Machine Learning and Explainable AI to a healthcare-related classification problem.

## Medical Disclaimer

This application is intended strictly for educational and research purposes.

It does not provide medical diagnosis, treatment recommendations, or professional medical advice.

Users should consult qualified healthcare professionals for medical decisions.