
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models/original_diabetes/diabetes_model.pkl"
)


def predict(input_data):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Diabetes model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    features = list(model.feature_names_in_)

    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    missing = [
        feature
        for feature in features
        if feature not in input_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing Original Diabetes features: {missing}"
        )

    X = input_data[features]

    prediction = int(model.predict(X)[0])

    probabilities = model.predict_proba(X)[0]

    classes = list(model.classes_)

    if set(classes) != {0, 1}:
        raise ValueError(
            f"Expected binary classes [0, 1], got: {classes}"
        )

    class_0_index = classes.index(0)
    class_1_index = classes.index(1)

    class_0_probability = float(
        probabilities[class_0_index]
    )

    class_1_probability = float(
        probabilities[class_1_index]
    )

    return {
        "model": "Original Pima Diabetes",

        "prediction": prediction,

        "model_score": class_1_probability,

        "score_label": "Class 1 model score",

        "class_probabilities": {
            "Class 0": class_0_probability,
            "Class 1": class_1_probability,
        },

        "class_labels": {
            0: "No Diabetes",
            1: "Diabetes",
        },
    }
