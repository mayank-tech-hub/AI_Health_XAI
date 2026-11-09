
import joblib
import pandas as pd
import shap
import numpy as np

from pathlib import Path
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent


def test_shap(name, model, preprocessor, dataset_path,
              background_path, background_processed=False,
              xgb_model=False):

    print(f"\n--- {name} SHAP TEST ---")

    df = pd.read_csv(dataset_path)

    features = list(preprocessor.feature_names_in_)
    X = df[features].iloc[[0]]

    background = joblib.load(background_path)

    # Transform patient input
    X_processed = preprocessor.transform(X)

    # Prepare background in the same feature space
    if background_processed:
        background_processed_data = background
    else:
        background_processed_data = preprocessor.transform(background)

    # Extract final estimator from pipeline
    if hasattr(model, "named_steps"):
        model_only = model.named_steps[
            list(model.named_steps.keys())[-1]
        ]
    else:
        model_only = model

    # Verify feature dimensions
    print("Patient shape:", X_processed.shape)
    print("Background shape:", background_processed_data.shape)

    if X_processed.shape[1] != background_processed_data.shape[1]:
        raise ValueError("Patient and background feature dimensions do not match!")

    # Calculate SHAP values
    if xgb_model:
        explainer = shap.TreeExplainer(model_only)
        shap_values = explainer.shap_values(X_processed)
    else:
        explainer = shap.Explainer(
            model_only.predict_proba,
            background_processed_data
        )
        shap_values = explainer(X_processed)

    # Prediction probability
    probability = model.predict_proba(X)[0, 1] \
        if hasattr(model, "named_steps") else \
        model.predict_proba(X_processed)[0, 1]

    print("Probability:", round(float(probability), 4))
    print("SHAP output shape:", np.asarray(shap_values.values if hasattr(shap_values, "values") else shap_values).shape)
    print("SHAP generated: SUCCESS")
    print("SHAP TEST: SUCCESS")


# Original Heart
original_heart = joblib.load(
    ROOT / "models/original_heart/heart_disease_logistic_pipeline.pkl"
)

test_shap(
    "Original Heart",
    original_heart,
    original_heart.named_steps["preprocessor"],
    ROOT / "datasets/original_heart/dataset.csv",
    ROOT / "models/original_heart/shap_background.pkl",
    background_processed=True
)


# Original Diabetes
original_diabetes = joblib.load(
    ROOT / "models/original_diabetes/diabetes_model.pkl"
)

original_diabetes_preprocessor = joblib.load(
    ROOT / "models/original_diabetes/diabetes_preprocessor.pkl"
)

test_shap(
    "Original Diabetes",
    original_diabetes,
    original_diabetes_preprocessor,
    ROOT / "datasets/original_diabetes/dataset.csv",
    ROOT / "models/original_diabetes/diabetes_shap_background.pkl",
    background_processed=False
)


# Kaggle Heart
kaggle_heart = joblib.load(
    ROOT / "models/kaggle_heart/kaggle_heart_disease_pipeline.pkl"
)

test_shap(
    "Kaggle Heart",
    kaggle_heart,
    kaggle_heart.named_steps["preprocessor"],
    ROOT / "datasets/kaggle_heart/dataset.csv",
    ROOT / "models/kaggle_heart/kaggle_heart_shap_background.pkl",
    background_processed=True
)


# Kaggle Diabetes
kaggle_diabetes = XGBClassifier()

kaggle_diabetes.load_model(
    ROOT / "models/kaggle_diabetes/kaggle_diabetes_model.json"
)

kaggle_diabetes_preprocessor = joblib.load(
    ROOT / "models/kaggle_diabetes/kaggle_diabetes_preprocessor.pkl"
)

test_shap(
    "Kaggle Diabetes",
    kaggle_diabetes,
    kaggle_diabetes_preprocessor,
    ROOT / "datasets/kaggle_diabetes/dataset.csv",
    ROOT / "models/kaggle_diabetes/kaggle_diabetes_shap_background.pkl",
    background_processed=True,
    xgb_model=True
)

print("\nALL FOUR SHAP TESTS COMPLETED!")
