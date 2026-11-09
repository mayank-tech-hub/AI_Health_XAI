
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def get_preprocessor(obj):
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return step
    return obj if hasattr(obj, "feature_names_in_") else None


def verify_pipeline(name, model_path, dataset_path):
    print(f"\n--- {name} ---")

    model = joblib.load(model_path)
    df = pd.read_csv(dataset_path)

    preprocessor = get_preprocessor(model)

    if preprocessor is None:
        raise ValueError("Preprocessor not found")

    features = list(preprocessor.feature_names_in_)
    X = df[features].iloc[[0]]

    probabilities = model.predict_proba(X)[:, 1]

    print("Model loaded: SUCCESS")
    print("Features:", len(features))
    print("Probability:", round(float(probabilities[0]), 4))
    print("Prediction: SUCCESS")


def verify_separate_model(name, model_path, prep_path, dataset_path):
    print(f"\n--- {name} ---")

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    df = pd.read_csv(dataset_path)

    features = list(preprocessor.feature_names_in_)
    X = df[features].iloc[[0]]

    X_processed = preprocessor.transform(X)
    probabilities = model.predict_proba(X_processed)[:, 1]

    print("Model loaded: SUCCESS")
    print("Features:", len(features))
    print("Probability:", round(float(probabilities[0]), 4))
    print("Prediction: SUCCESS")


# Part 1: Original Heart
verify_pipeline(
    "Original Heart Disease",
    ROOT / "models/original_heart/heart_disease_logistic_pipeline.pkl",
    ROOT / "datasets/original_heart/dataset.csv"
)

# Part 2: Original Diabetes
verify_separate_model(
    "Original Diabetes",
    ROOT / "models/original_diabetes/diabetes_model.pkl",
    ROOT / "models/original_diabetes/diabetes_preprocessor.pkl",
    ROOT / "datasets/original_diabetes/dataset.csv"
)

# Part 3: Kaggle Heart
verify_pipeline(
    "Kaggle Heart Disease",
    ROOT / "models/kaggle_heart/kaggle_heart_disease_pipeline.pkl",
    ROOT / "datasets/kaggle_heart/dataset.csv"
)

# Part 4: Kaggle Diabetes

from xgboost import XGBClassifier

print("\n--- Kaggle Diabetes ---")

# Load native XGBoost JSON model
diabetes_model = XGBClassifier()
diabetes_model.load_model(
    ROOT / "models/kaggle_diabetes/kaggle_diabetes_model.json"
)

# Load existing preprocessor
diabetes_preprocessor = joblib.load(
    ROOT / "models/kaggle_diabetes/kaggle_diabetes_preprocessor.pkl"
)

# Load dataset
diabetes_df = pd.read_csv(
    ROOT / "datasets/kaggle_diabetes/dataset.csv"
)

# Prepare sample
diabetes_features = list(
    diabetes_preprocessor.feature_names_in_
)

X = diabetes_df[diabetes_features].iloc[[0]]

# Transform and predict
X_processed = diabetes_preprocessor.transform(X)

probability = float(
    diabetes_model.predict_proba(X_processed)[0, 1]
)

threshold = 0.55
prediction = int(probability >= threshold)

print("JSON Model Loaded: SUCCESS")
print("Preprocessor Loaded: SUCCESS")
print("Processed Shape:", X_processed.shape)
print("Diabetes Probability:", round(probability * 100, 2), "%")
print("Prediction:", "Diabetes Risk Detected" if prediction else "No Diabetes Risk Detected")
print("Kaggle Diabetes Verification: SUCCESS")


print("\nALL FOUR MODEL TESTS COMPLETED!")
