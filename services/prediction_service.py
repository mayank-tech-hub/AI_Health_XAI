
from model_adapters import (
    original_heart,
    kaggle_heart,
    original_diabetes,
    kaggle_diabetes,
)


ADAPTERS = {
    "original_heart": original_heart,
    "kaggle_heart": kaggle_heart,
    "original_diabetes": original_diabetes,
    "kaggle_diabetes": kaggle_diabetes,
}


def predict(model_key, input_data):

    if model_key not in ADAPTERS:
        raise ValueError(
            f"Unknown model key: {model_key}"
        )

    result = ADAPTERS[model_key].predict(input_data)

    prediction = int(result["prediction"])

    result["prediction"] = prediction

    result["prediction_label"] = (
        "Class 1" if prediction == 1 else "Class 0"
    )

    # Preserve two-class probabilities when supplied
    # by the model adapter.
    if "class_probabilities" in result:

        probabilities = result["class_probabilities"]

        if not isinstance(probabilities, dict):
            raise ValueError(
                "class_probabilities must be a dictionary."
            )

        class_0 = float(probabilities["Class 0"])
        class_1 = float(probabilities["Class 1"])

        if (
            not 0 <= class_0 <= 1
            or not 0 <= class_1 <= 1
        ):
            raise ValueError(
                "Invalid class probability values."
            )

        if abs(class_0 + class_1 - 1.0) > 1e-5:
            raise ValueError(
                "Class probabilities must sum to 1."
            )

    return result
