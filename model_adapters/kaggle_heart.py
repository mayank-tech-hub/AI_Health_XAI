
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models/kaggle_heart/kaggle_heart_disease_pipeline.pkl"
)


def predict(input_data):

    model = joblib.load(MODEL_PATH)

    features = list(model.feature_names_in_)

    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    missing = [
        f for f in features
        if f not in input_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing Kaggle Heart features: {missing}"
        )

    X = input_data[features]

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]

    classes = list(model.classes_)

    if set(classes) != {0, 1}:
        raise ValueError(f"Unexpected classes: {classes}")

    class_0 = float(probabilities[classes.index(0)])
    class_1 = float(probabilities[classes.index(1)])

    return {
        "model": "Kaggle Heart Disease",
        "prediction": prediction,
        "model_score": class_1,
        "score_label": "Class 1 model score",
        "class_probabilities": {
            "Class 0": class_0,
            "Class 1": class_1,
        },
        "class_labels": {
            0: "No Heart Disease",
            1: "Heart Disease",
        },
    }
