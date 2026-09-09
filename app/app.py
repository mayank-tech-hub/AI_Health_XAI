
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# Page Configuration

st.set_page_config(
    page_title="AI Health XAI",
    page_icon="❤️",
    layout="wide"
)


# Load Saved Files

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "heart_disease_logistic_pipeline.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "heart_disease_preprocessor.pkl"
)

BACKGROUND_PATH = os.path.join(
    BASE_DIR,
    "models",
    "shap_background.pkl"
)

FEATURE_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "clean_feature_names.pkl"
)

MODEL_INFO_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_info.pkl"
)


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    background = joblib.load(BACKGROUND_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    model_info = joblib.load(MODEL_INFO_PATH)

    return (
        model,
        preprocessor,
        background,
        feature_names,
        model_info
    )


model, preprocessor, background, feature_names, model_info = load_model()


# SHAP Explainer

@st.cache_resource
def create_shap_explainer(_model, _background):

    logistic_model = _model.named_steps["model"]

    explainer = shap.LinearExplainer(
        logistic_model,
        _background
    )

    return explainer


shap_explainer = create_shap_explainer(
    model,
    background
)


# Header

st.title("❤️ AI-Based Early Disease Prediction")
st.subheader("Explainable AI (XAI) for Heart Disease")

st.write(
    """
This application uses a machine learning model to estimate
the probability of heart disease based on patient information.

The application also provides an explanation of the model's
prediction using SHAP (SHapley Additive exPlanations).
"""
)

st.warning(
    "This application is for educational and research purposes only. "
    "It is not a medical diagnosis or a substitute for professional "
    "medical advice."
)


# Sidebar - Model Information

with st.sidebar:

    st.header("Model Information")

    st.write(
        f"**Model:** {model_info['model_name']}"
    )

    st.write(
        f"**Dataset:** {model_info['dataset']}"
    )

    st.write(
        f"**Test Accuracy:** "
        f"{model_info['test_accuracy']:.2%}"
    )

    st.write(
        f"**Test Recall:** "
        f"{model_info['test_recall']:.2%}"
    )

    st.write(
        f"**Test ROC-AUC:** "
        f"{model_info['test_roc_auc']:.4f}"
    )


# Patient Information

st.header("Patient Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=55
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x:
            "Female (0)" if x == 0 else "Male (1)"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        options=[1, 2, 3, 4]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=130
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=250
    )


with col2:

    fbs = st.selectbox(
        "Fasting Blood Sugar",
        options=[0, 1],
        format_func=lambda x:
            "No (0)" if x == 0 else "Yes (1)"
    )

    restecg = st.selectbox(
        "Resting ECG",
        options=[0, 1, 2]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=40,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x:
            "No (0)" if x == 0 else "Yes (1)"
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )


with col3:

    slope = st.selectbox(
        "ST Slope",
        options=[1, 2, 3]
    )

    ca = st.selectbox(
        "Major Vessels (CA)",
        options=[0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thalassemia",
        options=[3, 6, 7]
    )


# Prediction Button

st.divider()

predict_button = st.button(
    "🔍 Predict Heart Disease",
    type="primary",
    use_container_width=True
)


if predict_button:

    # Create patient dataframe
    patient_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # Prediction

    prediction = model.predict(patient_data)[0]

    probabilities = model.predict_proba(
        patient_data
    )[0]

    no_disease_probability = probabilities[0]
    disease_probability = probabilities[1]


    # Display Result

    st.header("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️ Model Prediction: Disease Present"
            )

        else:

            st.success(
                "✅ Model Prediction: No Disease"
            )

    with result_col2:

        st.metric(
            "Disease Probability",
            f"{disease_probability:.2%}"
        )


    # Probability chart
    probability_df = pd.DataFrame(
        {
            "Outcome": [
                "No Disease",
                "Disease Present"
            ],
            "Probability": [
                no_disease_probability,
                disease_probability
            ]
        }
    )

    st.bar_chart(
        probability_df.set_index("Outcome")
    )


    # SHAP Explanation

    st.header("🔎 Explainable AI")

    st.write(
        """
The chart below shows which patient features influenced
the model's prediction. Positive SHAP values push the
prediction toward Disease Present, while negative values
push it toward No Disease.
"""
    )

    # Transform patient
    patient_transformed = preprocessor.transform(
        patient_data
    )

    # Calculate SHAP values
    shap_explanation = shap_explainer(
        patient_transformed
    )

    shap_values = shap_explanation.values[0]

    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values
    })

    explanation_df["Absolute SHAP"] = np.abs(
        explanation_df["SHAP Value"]
    )

    explanation_df = explanation_df.sort_values(
        by="Absolute SHAP",
        ascending=False
    ).head(10)


    # SHAP Bar Chart

    chart_df = explanation_df[
        ["Feature", "SHAP Value"]
    ].set_index("Feature")

    st.bar_chart(chart_df)


    # SHAP Explanation Table

    st.subheader("Top Contributing Features")

    display_df = explanation_df[
        ["Feature", "SHAP Value"]
    ].copy()

    display_df["Direction"] = np.where(
        display_df["SHAP Value"] > 0,
        "Increases Disease Probability",
        "Decreases Disease Probability"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# Footer

st.divider()

st.caption(
    "AI Health XAI | Educational Machine Learning Project"
)
