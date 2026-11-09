from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"

MODEL_REGISTRY = {
    "original_heart": {
        "name": "Original UCI Heart Disease",
        "type": "heart",
        "artifact": MODEL_DIR / "original_heart" / "heart_disease_logistic_pipeline.pkl",
    },
    "kaggle_heart": {
        "name": "Kaggle Heart Disease",
        "type": "heart",
        "artifact": MODEL_DIR / "kaggle_heart" / "kaggle_heart_disease_pipeline.pkl",
    },
    "original_diabetes": {
        "name": "Original Pima Diabetes",
        "type": "diabetes",
        "artifact": MODEL_DIR / "original_diabetes" / "diabetes_model.pkl",
    },
    "kaggle_diabetes": {
        "name": "Kaggle Diabetes",
        "type": "diabetes",
        "artifact": MODEL_DIR / "kaggle_diabetes" / "kaggle_diabetes_model.json",
    },
}
