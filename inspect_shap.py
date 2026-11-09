import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

files = [
    "models/original_heart/heart_disease_logistic_pipeline.pkl",
    "models/original_heart/shap_background.pkl",
    "models/original_diabetes/diabetes_model.pkl",
    "models/original_diabetes/diabetes_preprocessor.pkl",
    "models/original_diabetes/diabetes_shap_background.pkl",
    "models/kaggle_heart/kaggle_heart_disease_pipeline.pkl",
    "models/kaggle_heart/kaggle_heart_shap_background.pkl",
    "models/kaggle_diabetes/kaggle_diabetes_preprocessor.pkl",
    "models/kaggle_diabetes/kaggle_diabetes_shap_background.pkl",
]

for file in files:
    path = ROOT / file
    print(f"\n--- {file} ---")

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    try:
        obj = joblib.load(path)
        print("Type:", type(obj))

        if hasattr(obj, "named_steps"):
            print("Pipeline steps:", list(obj.named_steps.keys()))

        if hasattr(obj, "feature_names_in_"):
            print("Features:", list(obj.feature_names_in_))

        if hasattr(obj, "shape"):
            print("Shape:", obj.shape)

    except Exception as e:
        print("ERROR:", str(e))