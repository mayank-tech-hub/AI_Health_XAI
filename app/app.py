import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
import shap


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
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# =========================================================
# DISEASE SELECTION
# =========================================================

st.title(
    "❤️ AI-Based Early Disease Prediction"
)

st.subheader(
    "Explainable AI (XAI) Health Prediction System"
)

disease = st.selectbox(
    "Select Disease to Predict",
    [
        "Heart Disease",
        "Diabetes"
    ]
)

st.divider()


# =========================================================
# HEART DISEASE FILE PATHS
# =========================================================

HEART_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "heart_disease_logistic_pipeline.pkl"
)

HEART_PREPROCESSOR_PATH = os.path.join(
    MODELS_DIR,
    "heart_disease_preprocessor.pkl"
)

HEART_BACKGROUND_PATH = os.path.join(
    MODELS_DIR,
    "shap_background.pkl"
)

HEART_FEATURE_NAMES_PATH = os.path.join(
    MODELS_DIR,
    "clean_feature_names.pkl"
)

HEART_MODEL_INFO_PATH = os.path.join(
    MODELS_DIR,
    "model_info.pkl"
)


# =========================================================
# DIABETES FILE PATHS
# =========================================================

DIABETES_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "diabetes_model.pkl"
)

DIABETES_PREPROCESSOR_PATH = os.path.join(
    MODELS_DIR,
    "diabetes_preprocessor.pkl"
)

DIABETES_BACKGROUND_PATH = os.path.join(
    MODELS_DIR,
    "diabetes_shap_background.pkl"
)

DIABETES_FEATURE_NAMES_PATH = os.path.join(
    MODELS_DIR,
    "diabetes_feature_names.pkl"
)

DIABETES_MODEL_INFO_PATH = os.path.join(
    MODELS_DIR,
    "diabetes_model_info.pkl"
)


# =========================================================
# LOAD HEART MODEL FILES
# =========================================================

@st.cache_resource
def load_heart_files():

    model = joblib.load(
        HEART_MODEL_PATH
    )

    preprocessor = joblib.load(
        HEART_PREPROCESSOR_PATH
    )

    background = joblib.load(
        HEART_BACKGROUND_PATH
    )

    feature_names = joblib.load(
        HEART_FEATURE_NAMES_PATH
    )

    model_info = joblib.load(
        HEART_MODEL_INFO_PATH
    )

    return (
        model,
        preprocessor,
        background,
        feature_names,
        model_info
    )


# =========================================================
# LOAD DIABETES MODEL FILES
# =========================================================

@st.cache_resource
def load_diabetes_files():

    model = joblib.load(
        DIABETES_MODEL_PATH
    )

    preprocessor = joblib.load(
        DIABETES_PREPROCESSOR_PATH
    )

    background = joblib.load(
        DIABETES_BACKGROUND_PATH
    )

    feature_names = joblib.load(
        DIABETES_FEATURE_NAMES_PATH
    )

    model_info = joblib.load(
        DIABETES_MODEL_INFO_PATH
    )

    return (
        model,
        preprocessor,
        background,
        feature_names,
        model_info
    )


# =========================================================
# HEART SHAP EXPLAINER
# IMPORTANT:
# underscore parameters prevent Streamlit cache hashing error
# =========================================================

@st.cache_resource
def create_heart_shap_explainer(
    _model,
    _background
):

    logistic_model = _model.named_steps[
        "model"
    ]

    explainer = shap.LinearExplainer(
        logistic_model,
        _background
    )

    return explainer


# =========================================================
# DIABETES SHAP EXPLAINER
# IMPORTANT:
# underscore parameter prevents Streamlit cache hashing error
# =========================================================

@st.cache_resource
def create_diabetes_shap_explainer(
    _model
):

    explainer = shap.TreeExplainer(
        _model
    )

    return explainer


# =========================================================
# HEART DISEASE MODULE
# =========================================================

if disease == "Heart Disease":

    (
        heart_model,
        heart_preprocessor,
        heart_background,
        heart_feature_names,
        heart_model_info
    ) = load_heart_files()


    heart_shap_explainer = (
        create_heart_shap_explainer(
            heart_model,
            heart_background
        )
    )


    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.header(
        "🫀 Heart Disease Prediction"
    )

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

        st.header(
            "Heart Disease Model"
        )

        st.write(
            f"**Model:** "
            f"{heart_model_info['model_name']}"
        )

        st.write(
            f"**Dataset:** "
            f"{heart_model_info['dataset']}"
        )

        st.write(
            f"**Test Accuracy:** "
            f"{heart_model_info['test_accuracy']:.2%}"
        )

        st.write(
            f"**Test Recall:** "
            f"{heart_model_info['test_recall']:.2%}"
        )

        st.write(
            f"**Test ROC-AUC:** "
            f"{heart_model_info['test_roc_auc']:.4f}"
        )


    # -----------------------------------------------------
    # PATIENT INFORMATION
    # -----------------------------------------------------

    st.header(
        "Patient Information"
    )

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
    # HEART PREDICTION
    # -----------------------------------------------------

    st.divider()

    predict_heart = st.button(
        "🔍 Predict Heart Disease",
        type="primary",
        use_container_width=True
    )


    if predict_heart:

        patient_data = pd.DataFrame(
            [{
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
            }]
        )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction = int(
            heart_model.predict(
                patient_data
            )[0]
        )

        probabilities = (
            heart_model.predict_proba(
                patient_data
            )[0]
        )

        no_disease_probability = float(
            probabilities[0]
        )

        disease_probability = float(
            probabilities[1]
        )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.header(
            "Prediction Result"
        )

        result_col1, result_col2 = (
            st.columns(2)
        )


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
            probability_df.set_index(
                "Outcome"
            )
        )


        # -------------------------------------------------
        # HEALTH GUIDANCE
        # -------------------------------------------------

        st.header(
            "❤️ Health Guidance & Next Steps"
        )

        if prediction == 1:

            st.warning(
                "⚠️ The model estimates a higher likelihood "
                "of heart disease for the entered information."
            )

            guidance = [

                "👨‍⚕️ Discuss the prediction with a qualified healthcare professional.",

                "🩺 Ask whether further cardiovascular evaluation is appropriate.",

                "🥗 Follow a balanced, heart-healthy eating pattern.",

                "🚶 Maintain regular physical activity according to your fitness level and professional advice.",

                "🚭 Avoid smoking and tobacco exposure.",

                "😴 Maintain healthy sleep habits."

            ]

        else:

            st.success(
                "✅ The model estimates a lower likelihood "
                "of heart disease for the entered information."
            )

            guidance = [

                "👨‍⚕️ Continue routine health check-ups as appropriate.",

                "🥗 Follow a balanced, heart-healthy eating pattern.",

                "🚶 Maintain regular physical activity.",

                "🚭 Avoid smoking and tobacco exposure.",

                "😴 Maintain healthy sleep habits."

            ]


        if trestbps >= 140:

            guidance.append(
                "🩺 The entered resting blood pressure is elevated. "
                "Discuss monitoring with a healthcare professional."
            )

        elif trestbps >= 130:

            guidance.append(
                "🩺 The entered resting blood pressure is above "
                "the ideal range. Regular monitoring may be useful."
            )


        if chol >= 240:

            guidance.append(
                "🧪 The entered cholesterol value is high. "
                "Discuss cholesterol assessment with a healthcare professional."
            )

        elif chol >= 200:

            guidance.append(
                "🧪 The entered cholesterol value is above "
                "the desirable range. Consider discussing "
                "your lipid profile with a healthcare professional."
            )


        if fbs == 1:

            guidance.append(
                "🩸 The entered fasting blood sugar is marked as elevated. "
                "Discuss blood-sugar assessment with a healthcare professional."
            )


        if exang == 1:

            guidance.append(
                "💓 Exercise-induced angina is marked as present. "
                "Discuss this with a healthcare professional."
            )


        st.subheader(
            "Recommended Next Steps"
        )

        for item in guidance:

            st.write(item)


        # -------------------------------------------------
        # URGENT HELP
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
        # HEART SHAP
        # -------------------------------------------------

        st.header(
            "🔎 Explainable AI"
        )

        st.write(
            """
Positive SHAP values push the prediction toward
Disease Present, while negative SHAP values push it toward
No Disease.
"""
        )


        patient_transformed = (
            heart_preprocessor.transform(
                patient_data
            )
        )


        shap_explanation = (
            heart_shap_explainer(
                patient_transformed
            )
        )


        heart_shap_values = np.asarray(
            shap_explanation.values[0]
        ).reshape(-1)


        if len(heart_feature_names) == len(
            heart_shap_values
        ):

            explanation_df = pd.DataFrame(
                {
                    "Feature": heart_feature_names,
                    "SHAP Value": heart_shap_values
                }
            )

            explanation_df[
                "Absolute SHAP"
            ] = np.abs(
                explanation_df[
                    "SHAP Value"
                ]
            )

            explanation_df = (
                explanation_df
                .sort_values(
                    "Absolute SHAP",
                    ascending=False
                )
                .head(10)
            )


            st.subheader(
                "Top Contributing Features"
            )


            st.bar_chart(
                explanation_df[
                    [
                        "Feature",
                        "SHAP Value"
                    ]
                ].set_index(
                    "Feature"
                )
            )


            display_df = explanation_df[
                [
                    "Feature",
                    "SHAP Value"
                ]
            ].copy()


            display_df[
                "Direction"
            ] = np.where(
                display_df[
                    "SHAP Value"
                ] > 0,
                "Increases Disease Probability",
                "Decreases Disease Probability"
            )


            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.error(
                "Heart SHAP feature count does not match "
                "the saved feature names."
            )


# =========================================================
# DIABETES MODULE
# =========================================================

elif disease == "Diabetes":

    (
        diabetes_model,
        diabetes_preprocessor,
        diabetes_background,
        diabetes_feature_names,
        diabetes_model_info
    ) = load_diabetes_files()


    diabetes_shap_explainer = (
        create_diabetes_shap_explainer(
            diabetes_model
        )
    )


    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.header(
        "🩸 Diabetes Prediction"
    )

    st.write(
        """
Enter patient information to estimate diabetes
probability and view a SHAP-based explanation.
"""
    )


    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    with st.sidebar:

        st.header(
            "Diabetes Model"
        )

        st.write(
            f"**Model:** "
            f"{diabetes_model_info['model_name']}"
        )

        st.write(
            f"**Dataset:** "
            f"{diabetes_model_info['dataset']}"
        )

        st.write(
            f"**Test Accuracy:** "
            f"{diabetes_model_info['test_accuracy']:.2%}"
        )

        st.write(
            f"**Test Recall:** "
            f"{diabetes_model_info['test_recall']:.2%}"
        )

        st.write(
            f"**Test ROC-AUC:** "
            f"{diabetes_model_info['test_roc_auc']:.4f}"
        )


    # -----------------------------------------------------
    # DIABETES INPUTS
    # -----------------------------------------------------

    st.header(
        "Diabetes Patient Information"
    )

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


    # -----------------------------------------------------
    # DIABETES PREDICTION
    # -----------------------------------------------------

    st.divider()

    predict_diabetes = st.button(
        "🔍 Predict Diabetes",
        type="primary",
        use_container_width=True
    )


    if predict_diabetes:

        diabetes_patient = pd.DataFrame(
            [{
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction":
                    diabetes_pedigree,
                "Age": diabetes_age
            }]
        )


        # -------------------------------------------------
        # PREPROCESSING
        # -------------------------------------------------

        diabetes_patient_transformed = (
            diabetes_preprocessor.transform(
                diabetes_patient
            )
        )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        diabetes_probability = float(
            diabetes_model.predict_proba(
                diabetes_patient_transformed
            )[0, 1]
        )


        diabetes_threshold = float(
            diabetes_model_info.get(
                "prediction_threshold",
                0.42
            )
        )


        diabetes_prediction = int(
            diabetes_probability >=
            diabetes_threshold
        )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.header(
            "Prediction Result"
        )

        result_col1, result_col2 = (
            st.columns(2)
        )


        with result_col1:

            if diabetes_prediction == 1:

                st.error(
                    "⚠️ Model Prediction: Diabetes Likely"
                )

            else:

                st.success(
                    "✅ Model Prediction: Diabetes Less Likely"
                )


        with result_col2:

            st.metric(
                "Diabetes Probability",
                f"{diabetes_probability:.2%}"
            )


        st.caption(
            f"Prediction threshold: "
            f"{diabetes_threshold:.2f}"
        )


        # -------------------------------------------------
        # PROBABILITY CHART
        # -------------------------------------------------

        diabetes_probability_df = pd.DataFrame(
            {
                "Outcome": [
                    "No Diabetes",
                    "Diabetes"
                ],

                "Probability": [
                    1 - diabetes_probability,
                    diabetes_probability
                ]
            }
        )


        st.bar_chart(
            diabetes_probability_df.set_index(
                "Outcome"
            )
        )


        # -------------------------------------------------
        # HEALTH GUIDANCE
        # -------------------------------------------------

        st.header(
            "🩸 Health Guidance & Next Steps"
        )


        if diabetes_prediction == 1:

            st.warning(
                "⚠️ The model estimates a higher likelihood "
                "of diabetes for the entered information."
            )

            diabetes_guidance = [

                "👨‍⚕️ Discuss the prediction with a qualified healthcare professional.",

                "🧪 Consider appropriate blood-glucose testing and clinical evaluation as advised by a healthcare professional.",

                "🥗 Follow a balanced diet with appropriate portion control.",

                "🚶 Maintain regular physical activity according to your fitness level and professional advice.",

                "😴 Maintain healthy sleep habits."

            ]

        else:

            st.success(
                "✅ The model estimates a lower likelihood "
                "of diabetes for the entered information."
            )

            diabetes_guidance = [

                "👨‍⚕️ Continue routine health check-ups as appropriate.",

                "🥗 Maintain a balanced diet.",

                "🚶 Maintain regular physical activity.",

                "⚖️ Maintain a healthy lifestyle.",

                "😴 Maintain healthy sleep habits."

            ]


        if glucose >= 126:

            diabetes_guidance.append(
                "🩸 The entered glucose value is high. "
                "Discuss appropriate testing and evaluation "
                "with a healthcare professional."
            )

        elif glucose >= 100:

            diabetes_guidance.append(
                "🩸 The entered glucose value is above the "
                "normal fasting range if this measurement was "
                "fasting. Discuss it with a healthcare professional."
            )


        if bmi >= 30:

            diabetes_guidance.append(
                "⚖️ The entered BMI is in the obesity range. "
                "Discuss healthy weight-management strategies "
                "with a healthcare professional."
            )

        elif bmi >= 25:

            diabetes_guidance.append(
                "⚖️ The entered BMI is above the normal range. "
                "Healthy lifestyle and weight-management "
                "strategies may be useful."
            )


        st.subheader(
            "Recommended Next Steps"
        )


        for item in diabetes_guidance:

            st.write(item)


        # -------------------------------------------------
        # DIABETES SHAP
        # -------------------------------------------------

        st.header(
            "🔎 Explainable AI"
        )

        st.write(
            """
Positive SHAP values push the model toward Diabetes,
while negative SHAP values push it toward No Diabetes.
"""
        )


        raw_shap_values = (
            diabetes_shap_explainer.shap_values(
                diabetes_patient_transformed
            )
        )


        # -------------------------------------------------
        # HANDLE SHAP OUTPUT
        # -------------------------------------------------

        if isinstance(
            raw_shap_values,
            list
        ):

            # Older SHAP versions
            diabetes_patient_shap = np.asarray(
                raw_shap_values[1]
            )[0]

        else:

            shap_array = np.asarray(
                raw_shap_values
            )

            if shap_array.ndim == 3:

                # samples, features, classes
                diabetes_patient_shap = (
                    shap_array[0, :, 1]
                )

            elif shap_array.ndim == 2:

                diabetes_patient_shap = (
                    shap_array[0]
                )

            elif shap_array.ndim == 1:

                diabetes_patient_shap = (
                    shap_array
                )

            else:

                diabetes_patient_shap = (
                    shap_array.reshape(-1)
                )


        diabetes_patient_shap = np.asarray(
            diabetes_patient_shap
        ).reshape(-1)


        diabetes_feature_names = list(
            diabetes_feature_names
        )


        # -------------------------------------------------
        # SHAP RESULT
        # -------------------------------------------------

        if len(
            diabetes_feature_names
        ) != len(
            diabetes_patient_shap
        ):

            st.error(
                "Diabetes SHAP feature count does not "
                "match the saved feature names."
            )

        else:

            diabetes_explanation_df = (
                pd.DataFrame(
                    {
                        "Feature":
                            diabetes_feature_names,

                        "SHAP Value":
                            diabetes_patient_shap
                    }
                )
            )


            diabetes_explanation_df[
                "Absolute SHAP"
            ] = np.abs(
                diabetes_explanation_df[
                    "SHAP Value"
                ]
            )


            diabetes_explanation_df = (
                diabetes_explanation_df
                .sort_values(
                    "Absolute SHAP",
                    ascending=False
                )
                .head(8)
            )


            st.subheader(
                "Top Contributing Features"
            )


            st.bar_chart(
                diabetes_explanation_df[
                    [
                        "Feature",
                        "SHAP Value"
                    ]
                ].set_index(
                    "Feature"
                )
            )


            diabetes_display_df = (
                diabetes_explanation_df[
                    [
                        "Feature",
                        "SHAP Value"
                    ]
                ].copy()
            )


            diabetes_display_df[
                "Direction"
            ] = np.where(
                diabetes_display_df[
                    "SHAP Value"
                ] > 0,
                "Increases Diabetes Probability",
                "Decreases Diabetes Probability"
            )


            st.dataframe(
                diabetes_display_df,
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Health XAI | Educational Machine Learning Project"
)
