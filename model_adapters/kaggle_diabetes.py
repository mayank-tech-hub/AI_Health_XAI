
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parent.parent

PREPROCESSOR_PATH = (
    ROOT
    / "models/kaggle_diabetes/kaggle_diabetes_preprocessor.pkl"
)

MODEL_PATH = (
    ROOT
    / "models/kaggle_diabetes/kaggle_diabetes_model.json"
)


def predict(input_data):

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    features = list(preprocessor.feature_names_in_)

    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    missing = [
        f for f in features
        if f not in input_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing Kaggle Diabetes features: {missing}"
        )

    X = input_data[features]

    transformed = preprocessor.transform(X)

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    if transformed.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Preprocessor produced {transformed.shape[1]} features; "
            f"model expects {model.n_features_in_}."
        )

    prediction = int(model.predict(transformed)[0])

    probabilities = model.predict_proba(transformed)[0]

    classes = list(model.classes_)

    if set(classes) != {0, 1}:
        raise ValueError(f"Unexpected classes: {classes}")

    class_0 = float(probabilities[classes.index(0)])
    class_1 = float(probabilities[classes.index(1)])

    return {
        "model": "Kaggle Diabetes",
        "prediction": prediction,
        "model_score": class_1,
        "score_label": "Class 1 model score",
        "class_probabilities": {
            "Class 0": class_0,
            "Class 1": class_1,
        },
        "class_labels": {
            0: "No Diabetes",
            1: "Diabetes",
        },
    }
