import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st
from auth import require_login

st.set_page_config(page_title="AI Health XAI", page_icon="❤️", layout="wide")

require_login()


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Health XAI",
    page_icon="❤️",
    layout="wide"
)


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# =========================================================
# DISEASE SELECTION
# =========================================================

st.title("❤️ AI-Based Early Disease Prediction")
st.subheader("Explainable AI (XAI) Health Prediction System")

disease = st.selectbox(
    "Select Disease to Predict",
    [
        "Heart Disease",
        "Diabetes"
    ]
)

st.divider()


# =========================================================
# HEART DISEASE
# =========================================================

if disease == "Heart Disease":

    # -----------------------------------------------------
    # HEART DISEASE MODEL PATHS
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # LOAD HEART MODEL
    # -----------------------------------------------------

    @st.cache_resource
    def load_heart_model():

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


    (
        model,
        preprocessor,
        background,
        feature_names,
        model_info
    ) = load_heart_model()


    # -----------------------------------------------------
    # HEART SHAP EXPLAINER
    # -----------------------------------------------------

    @st.cache_resource
    def create_heart_shap_explainer(
        _model,
        _background
    ):

        logistic_model = _model.named_steps["model"]

        explainer = shap.LinearExplainer(
            logistic_model,
            _background
        )

        return explainer


    shap_explainer = create_heart_shap_explainer(
        model,
        background
    )


    # -----------------------------------------------------
    # HEART HEADER
    # -----------------------------------------------------

    st.header("🫀 Heart Disease Prediction")

    st.write(
        """
This application uses a machine learning model to estimate
the probability of heart disease based on patient information.

The prediction is also explained using SHAP
(Explainable AI).
"""
    )


    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    with st.sidebar:

        st.header("Heart Disease Model")

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


    # -----------------------------------------------------
    # HEART PATIENT INFORMATION
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # HEART PREDICTION BUTTON
    # -----------------------------------------------------

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

        prediction = model.predict(
            patient_data
        )[0]

        probabilities = model.predict_proba(
            patient_data
        )[0]

        no_disease_probability = probabilities[0]
        disease_probability = probabilities[1]


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

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


        # -------------------------------------------------
        # PROBABILITY CHART
        # -------------------------------------------------

        probability_df = pd.DataFrame({

            "Outcome": [
                "No Disease",
                "Disease Present"
            ],

            "Probability": [
                no_disease_probability,
                disease_probability
            ]

        })


        st.bar_chart(
            probability_df.set_index("Outcome")
        )


        # -------------------------------------------------
        # HEALTH GUIDANCE
        # -------------------------------------------------

        st.header("❤️ Health Guidance & Next Steps")

        st.info(
            """
These recommendations are general educational guidance.
They are not a medical diagnosis or treatment plan.
Clinical decisions should be made by a qualified
healthcare professional.
"""
        )


        guidance = [

            "👨‍⚕️ Consider discussing the prediction and your risk factors with a qualified healthcare professional.",

            "🩺 Keep track of blood pressure, cholesterol and other relevant health measurements as advised by a healthcare professional.",

            "🥗 Follow a balanced, heart-healthy eating pattern.",

            "🚶 Maintain regular physical activity according to your fitness level and healthcare professional's advice.",

            "🚭 Avoid smoking and tobacco exposure.",

            "😴 Maintain healthy sleep habits and a healthy lifestyle."

        ]


        if prediction == 1:

            st.warning(
                "⚠️ The model estimates a higher likelihood of heart disease for the entered information."
            )

            guidance.insert(
                0,
                "📋 Discuss the prediction with a healthcare professional and ask whether further clinical evaluation is appropriate."
            )

        else:

            st.success(
                "✅ The model estimates a lower likelihood of heart disease for the entered information."
            )

            guidance.insert(
                0,
                "📋 Continue preventive heart-health habits and discuss routine cardiovascular screening with a healthcare professional."
            )


        # Blood pressure

        if trestbps >= 140:

            guidance.append(
                "🩺 Your entered resting blood pressure is elevated. Discuss monitoring and evaluation with a healthcare professional."
            )

        elif trestbps >= 130:

            guidance.append(
                "🩺 Your entered resting blood pressure is above the ideal range. Regular monitoring may be useful."
            )


        # Cholesterol

        if chol >= 240:

            guidance.append(
                "🧪 Your entered cholesterol value is high. Discuss cholesterol assessment with a healthcare professional."
            )

        elif chol >= 200:

            guidance.append(
                "🧪 Your entered cholesterol value is above the desirable range. Consider discussing your lipid profile with a healthcare professional."
            )


        # Fasting blood sugar

        if fbs == 1:

            guidance.append(
                "🩸 Your entered fasting blood sugar is marked as elevated. Discuss blood-sugar assessment with a healthcare professional."
            )


        # Exercise angina

        if exang == 1:

            guidance.append(
                "💓 Exercise-induced angina is marked as present. Discuss this with a healthcare professional."
            )


        # Age

        if age >= 45:

            guidance.append(
                "📅 Regular cardiovascular risk assessment can be useful as age-related risk factors increase."
            )


        st.subheader("Recommended Next Steps")

        for item in guidance:

            st.write(item)


        # -------------------------------------------------
        # URGENT MEDICAL HELP
        # -------------------------------------------------

        st.subheader(
            "🚨 When to Seek Urgent Medical Help"
        )

        st.write(
            """
If someone has severe or persistent chest pain or pressure,
difficulty breathing, fainting, sudden weakness, or other
serious or rapidly worsening symptoms, they should seek
urgent medical attention rather than relying on this
application's prediction.
"""
        )


        # -------------------------------------------------
        # SHAP EXPLANATION
        # -------------------------------------------------

        st.header("🔎 Explainable AI")

        st.write(
            """
The chart below shows which patient features influenced
the model's prediction.

Positive SHAP values push the prediction toward
Disease Present, while negative values push it toward
No Disease.
"""
        )


        patient_transformed = preprocessor.transform(
            patient_data
        )


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


        # SHAP chart

        chart_df = explanation_df[
            ["Feature", "SHAP Value"]
        ].set_index("Feature")


        st.bar_chart(chart_df)


        # SHAP table

        st.subheader(
            "Top Contributing Features"
        )


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


# =========================================================
# DIABETES
# =========================================================

elif disease == "Diabetes":

    st.header("🩸 Diabetes Prediction")

    st.write(
        """
The Diabetes prediction module will use a separate
machine learning model trained specifically for diabetes
prediction.
"""
    )


    # -----------------------------------------------------
    # DIABETES PATIENT INFORMATION
    # -----------------------------------------------------

    st.header("Diabetes Patient Information")

    col1, col2 = st.columns(2)


    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0,
            max_value=300,
            value=120
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0,
            max_value=200,
            value=70
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0,
            max_value=100,
            value=20
        )


    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0,
            max_value=1000,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=80.0,
            value=25.0,
            step=0.1
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01
        )

        diabetes_age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )


    st.divider()


    # -----------------------------------------------------
    # DIABETES PREDICTION BUTTON
    # -----------------------------------------------------

    diabetes_button = st.button(
        "🔍 Predict Diabetes",
        type="primary",
        use_container_width=True
    )


    if diabetes_button:

        st.warning(
            """
The Diabetes prediction model has not been connected yet.
The input interface is ready, but a trained Diabetes model
must be added before a valid prediction can be produced.
"""
        )

        st.info(
            """
Next step: train and save a dedicated Diabetes Machine
Learning model and its SHAP explanation files. Then this
section can use that model for prediction.
"""
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Health XAI | Educational Machine Learning Project"
)
