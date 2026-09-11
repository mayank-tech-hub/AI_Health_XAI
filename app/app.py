import streamlit as st
import os
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

#UI/UX Design
st.markdown('<script src="custom_css.css"></script>', unsafe_allow_html=True)

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


    # =========================================================
    # NEW SECTION: PERSONALIZED HEALTH GUIDANCE
    # =========================================================

    st.header("❤️ Health Guidance & Next Steps")

    # Base recommendations
    guidance = [
        "👨‍⚕️ Consider discussing the prediction and your risk factors with a qualified healthcare professional.",
        "🩺 Keep track of blood pressure, cholesterol and other relevant health measurements as advised by a healthcare professional.",
        "🥗 Follow a balanced, heart-healthy eating pattern with vegetables, fruits, whole grains and appropriate portions.",
        "🚶 Maintain regular physical activity according to your fitness level and healthcare professional's advice.",
        "🚭 Avoid smoking and tobacco exposure.",
        "😴 Maintain healthy sleep habits and work toward a healthy body weight where appropriate."
    ]

    # Prediction-based guidance
    if prediction == 1:

        guidance.insert(
            0,
            "📋 Arrange a healthcare professional evaluation to discuss the result and whether additional clinical assessment or testing is appropriate."
        )

    else:

        st.success(
            "✅ The model estimates a lower likelihood of heart disease for the entered information."
        )

        guidance.insert(
            0,
            "📋 Continue preventive heart-health habits and discuss routine cardiovascular screening with a healthcare professional."
        )


    # Blood Pressure Guidance
    if trestbps >= 140:

        guidance.append(
            "🩺 Your entered resting blood pressure is elevated. Discuss blood-pressure monitoring and evaluation with a healthcare professional."
        )

    elif trestbps >= 130:

        guidance.append(
            "🩺 Your entered resting blood pressure is above the ideal range. Regular monitoring and discussion with a healthcare professional may be useful."
        )


    # Cholesterol Guidance
    if chol >= 240:

        guidance.append(
            "🧪 Your entered cholesterol value is high. Discuss cholesterol assessment and heart-health management with a healthcare professional."
        )

    elif chol >= 200:

        guidance.append(
            "🧪 Your entered cholesterol value is above the desirable range. Consider discussing your complete lipid profile with a healthcare professional."
        )


    # Fasting Blood Sugar Guidance
    if fbs == 1:

        guidance.append(
            "🩸 Your entered fasting blood sugar is marked as elevated. Discuss blood-sugar assessment with a healthcare professional."
        )


    # Exercise Angina Guidance
    if exang == 1:

        guidance.append(
            "💓 Exercise-induced angina is marked as present. Discuss this symptom with a healthcare professional, particularly if it is new, worsening or concerning."
        )


    # Maximum Heart Rate Guidance
    if thalach < 100:

        guidance.append(
            "❤️ The entered maximum heart-rate value is relatively low. Discuss its clinical significance with a healthcare professional."
        )


    # Age-related preventive guidance
    if age >= 45:

        guidance.append(
            "📅 Regular cardiovascular risk assessment can be useful, especially as age-related risk factors increase."
        )


    # Display Guidance
    st.subheader("Recommended Next Steps")

    for item in guidance:

        st.write(item)


    # Urgent symptom information
    st.subheader("🚨 When to Seek Urgent Medical Help")

    st.write(
        """
If someone has severe or persistent chest pain/pressure,
difficulty breathing, fainting, sudden weakness, or other
serious or rapidly worsening symptoms, they should seek
urgent medical attention rather than relying on this
application's prediction.
"""
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
