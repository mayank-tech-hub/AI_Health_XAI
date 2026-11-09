
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

CONFIG = {
    "original_heart": {
        "model": MODEL_DIR / "original_heart/heart_disease_logistic_pipeline.pkl",
        "background": MODEL_DIR / "original_heart/shap_background.pkl",
        "features": [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal"
        ],
        "kind": "pipeline_transformed",
    },

    "kaggle_heart": {
        "model": MODEL_DIR / "kaggle_heart/kaggle_heart_disease_pipeline.pkl",
        "background": MODEL_DIR / "kaggle_heart/kaggle_heart_shap_background.pkl",
        "features": [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal"
        ],
        "kind": "pipeline_transformed",
    },

    "original_diabetes": {
        "model": MODEL_DIR / "original_diabetes/diabetes_model.pkl",
        "background": MODEL_DIR / "original_diabetes/diabetes_shap_background.pkl",
        "features": [
            "Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"
        ],
        "kind": "pipeline_raw",
    },

    "kaggle_diabetes": {
        "model": MODEL_DIR / "kaggle_diabetes/kaggle_diabetes_model.json",
        "preprocessor": MODEL_DIR / "kaggle_diabetes/kaggle_diabetes_preprocessor.pkl",
        "background": MODEL_DIR / "kaggle_diabetes/kaggle_diabetes_shap_background.pkl",
        "features": [
            "gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level",
            "blood_glucose_level"
        ],
        "kind": "xgboost",
    },
}


# --------------------------------------------------
# LOAD MODEL AND BACKGROUND
# --------------------------------------------------

def _load_model_and_background(model_key):

    if model_key not in CONFIG:
        raise ValueError(f"Unknown model key: {model_key}")

    cfg = CONFIG[model_key]

    for key in ["model", "background"]:
        if not cfg[key].exists():
            raise FileNotFoundError(
                f"Missing {key} file: {cfg[key]}"
            )

    background = joblib.load(cfg["background"])

    if isinstance(background, pd.DataFrame):
        background = background.copy()
    else:
        background = np.asarray(background)

    if model_key == "kaggle_diabetes":

        if not cfg["preprocessor"].exists():
            raise FileNotFoundError(
                f"Missing preprocessor: {cfg['preprocessor']}"
            )

        preprocessor = joblib.load(cfg["preprocessor"])

        model = xgb.XGBClassifier()
        model.load_model(str(cfg["model"]))

        transformed_names = list(
            preprocessor.get_feature_names_out()
        )

        return (
            model,
            background,
            cfg["features"],
            transformed_names,
            "xgboost",
        )

    if cfg["kind"] == "pipeline_transformed":

        pipeline = joblib.load(cfg["model"])

        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        transformed_names = list(
            preprocessor.get_feature_names_out()
        )

        return (
            model,
            background,
            cfg["features"],
            transformed_names,
            "transformed",
        )

    pipeline = joblib.load(cfg["model"])

    return (
        pipeline,
        background,
        cfg["features"],
        cfg["features"],
        "raw",
    )


# --------------------------------------------------
# SHAP OUTPUT HELPERS
# --------------------------------------------------

def _class_one_values(values):

    values = np.asarray(values)

    if values.ndim == 3:

        if values.shape[-1] != 2:
            raise ValueError(
                f"Unexpected SHAP class dimension: {values.shape}"
            )

        values = values[:, :, 1]

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    return values


def _class_one_base_value(base_values):

    base = np.asarray(base_values)

    if base.ndim == 0:
        return float(base)

    if base.ndim == 1:
        return float(base[1] if len(base) == 2 else base[0])

    if base.ndim == 2:

        if base.shape[-1] == 2:
            return float(base[0, 1])

        return float(base[0, 0])

    raise ValueError(
        f"Unexpected SHAP base value shape: {base.shape}"
    )


# --------------------------------------------------
# GROUP TRANSFORMED FEATURES
# --------------------------------------------------

def _group_features(values, transformed_names, raw_features):

    grouped = {
        feature: 0.0 for feature in raw_features
    }

    for index, transformed_name in enumerate(transformed_names):

        name = str(transformed_name)

        matches = [
            feature
            for feature in raw_features
            if (
                name == feature
                or name.endswith("__" + feature)
                or name.startswith(feature + "_")
                or name.startswith(feature + "__")
                or ("__" + feature + "_") in name
            )
        ]

        if not matches:
            raise ValueError(
                f"Cannot map transformed feature '{name}' "
                "to an original feature."
            )

        original_feature = max(matches, key=len)

        grouped[original_feature] += float(values[index])

    return grouped


# --------------------------------------------------
# MAIN EXPLANATION FUNCTION
# --------------------------------------------------

def explain(model_key, input_data):
    """
    Calculate Class 1 SHAP contributions.

    Returns:
        prediction
        prediction_label
        model_output
        base_value
        top_features
        all_features
        grouped_contributions
        additivity_error
        validation_passed

    This function does not save patient input or explanation data.
    """

    (
        model,
        background,
        raw_features,
        transformed_names,
        mode,
    ) = _load_model_and_background(model_key)

    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    elif not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data)

    missing = [
        feature
        for feature in raw_features
        if feature not in input_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing input features: {missing}"
        )

    X = input_data[raw_features].copy()

    # ----------------------------------------------
    # RAW PIPELINE
    # ----------------------------------------------

    if mode == "raw":

        explainer = shap.Explainer(
            model.predict_proba,
            background,
        )

        explanation = explainer(X)

        values = _class_one_values(
            explanation.values
        )[0]

        base_value = _class_one_base_value(
            explanation.base_values
        )

        probabilities = model.predict_proba(X)

        prediction = model.predict(X)[0]

    # ----------------------------------------------
    # TRANSFORMED HEART PIPELINES
    # ----------------------------------------------

    elif mode == "transformed":

        pipeline = joblib.load(
            CONFIG[model_key]["model"]
        )

        preprocessor = pipeline.named_steps[
            "preprocessor"
        ]

        transformed = preprocessor.transform(X)

        explainer = shap.Explainer(
            model.predict_proba,
            background,
        )

        explanation = explainer(transformed)

        values = _class_one_values(
            explanation.values
        )[0]

        base_value = _class_one_base_value(
            explanation.base_values
        )

        probabilities = model.predict_proba(
            transformed
        )

        prediction = model.predict(
            transformed
        )[0]

    # ----------------------------------------------
    # KAGGLE DIABETES XGBOOST
    # ----------------------------------------------

    elif mode == "xgboost":

        preprocessor = joblib.load(
            CONFIG[model_key]["preprocessor"]
        )

        transformed = preprocessor.transform(X)

        explainer = shap.TreeExplainer(
            model,
            data=background,
            model_output="probability",
            feature_perturbation="interventional",
        )

        explanation = explainer(transformed)

        values = _class_one_values(
            explanation.values
        )[0]

        base_value = _class_one_base_value(
            explanation.base_values
        )

        probabilities = model.predict_proba(
            transformed
        )

        prediction = model.predict(
            transformed
        )[0]

    else:

        raise ValueError(
            f"Unsupported model mode: {mode}"
        )

    # ----------------------------------------------
    # PREDICTION VALIDATION
    # ----------------------------------------------

    prediction = int(prediction)

    if probabilities.shape[1] != 2:
        raise ValueError(
            "Expected a binary classification model."
        )

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            f"Class 1 not found in model classes: {classes}"
        )

    class_one_index = classes.index(1)

    model_output = float(
        probabilities[0][class_one_index]
    )

    prediction_label = (
        "Class 1" if prediction == 1 else "Class 0"
    )

    if not np.isfinite(model_output):
        raise ValueError(
            "Model returned an invalid probability."
        )

    # ----------------------------------------------
    # FEATURE VALIDATION
    # ----------------------------------------------

    if len(values) != len(transformed_names):
        raise ValueError(
            f"SHAP feature count {len(values)} does not "
            f"match transformed feature count "
            f"{len(transformed_names)}."
        )

    grouped = _group_features(
        values,
        transformed_names,
        raw_features,
    )

    raw_sum = float(np.sum(values))

    grouped_sum = float(
        sum(grouped.values())
    )

    if not np.isclose(
        raw_sum,
        grouped_sum,
        atol=1e-5,
    ):
        raise ValueError(
            "Feature grouping failed the SHAP sum check."
        )

    # ----------------------------------------------
    # ADDITIVITY VALIDATION
    # ----------------------------------------------

    additivity_error = abs(
        (base_value + grouped_sum) - model_output
    )

    if additivity_error > 0.02:
        raise ValueError(
            "SHAP additivity check failed. "
            f"Difference: {additivity_error:.6f}"
        )

    # ----------------------------------------------
    # RANK ALL FEATURE CONTRIBUTIONS
    # ----------------------------------------------

    ranked = sorted(
        grouped.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    all_features = [
        {
            "feature": feature,
            "contribution": float(contribution),
            "direction": (
                "pushed model output higher"
                if contribution > 0
                else "pushed model output lower"
                if contribution < 0
                else "no contribution"
            ),
        }
        for feature, contribution in ranked
    ]

    # Preserve existing top-five structure.
    top_features = all_features[:5]

    # ----------------------------------------------
    # FINAL RESULT
    # ----------------------------------------------

    return {
        "model_key": model_key,
        "prediction": prediction,
        "prediction_label": prediction_label,
        "model_output": model_output,
        "base_value": base_value,
        "top_features": top_features,
        "all_features": all_features,
        "grouped_contributions": grouped,
        "additivity_error": additivity_error,
        "validation_passed": True,
    }
