import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


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
# MAIN HEADER
# =========================================================

st.title("❤️ AI-Based Early Disease Prediction")

st.subheader(
    "Explainable AI (XAI) Health Prediction System"
)


# =========================================================
# DISEASE SELECTION
# =========================================================

disease = st.selectbox(
    "Select Disease to Predict",
    [
        "Heart Disease",
        "Diabetes"
    ]
)

st.divider()


# =========================================================
# HEART DISEASE MODULE
# =========================================================

if disease == "Heart Disease":

    # -----------------------------------------------------
    # FILE PATHS
    # -----------------------------------------------------

    heart_model_path = os.path.join(
        MODELS_DIR,
        "heart_disease_logistic_pipeline.pkl"
    )

    heart_preprocessor_path = os.path.join(
        MODELS_DIR,
        "heart_disease_preprocessor.pkl"
    )

    heart_background_path = os.path.join(
        MODELS_DIR,
        "shap_background.pkl"
    )

    heart_feature_names_path = os.path.join(
        MODELS_DIR,
        "clean_feature_names.pkl"
    )

    heart_model_info_path = os.path.join(
        MODELS_DIR,
        "model_info.pkl"
    )


    # -----------------------------------------------------
    # LOAD HEART MODEL
    # -----------------------------------------------------

    @st.cache_resource
    def load_heart_model():

        model = joblib.load(
            heart_model_path
        )

        preprocessor = joblib.load(
            heart_preprocessor_path
        )

        background = joblib.load(
            heart_background_path
        )

        feature_names = joblib.load(
            heart_feature_names_path
        )

        model_info = joblib.load(
            heart_model_info_path
        )

        return (
            model,
            preprocessor,
            background,
            feature_names,
            model_info
        )


    (
        heart_model,
        heart_preprocessor,
        heart_background,
        heart_feature_names,
        heart_model_info
    ) = load_heart_model()


    # -----------------------------------------------------
    # HEART SHAP EXPLAINER
    # -----------------------------------------------------

    @st.cache_resource
    def create_heart_explainer(
        model,
        background
    ):

        logistic_model = model.named_steps["model"]

        explainer = shap.LinearExplainer(
            logistic_model,
            background
        )

        return explainer


    heart_explainer = create_heart_explainer(
        heart_model,
        heart_background
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

        sex_label = st.selectbox(
            "Sex",
            [
                "Female (0)",
                "Male (1)"
            ]
        )

        if sex_label == "Female (0)":
            sex = 0
        else:
            sex = 1

        cp = st.selectbox(
            "Chest Pain Type",
            [1, 2, 3, 4]
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

        fbs_label = st.selectbox(
            "Fasting Blood Sugar",
            [
                "No (0)",
                "Yes (1)"
            ]
        )

        if fbs_label == "No (0)":
            fbs = 0
        else:
            fbs = 1

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2]
        )

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=40,
            max_value=250,
            value=150
        )

        exang_label = st.selectbox(
            "Exercise Induced Angina",
            [
                "No (0)",
                "Yes (1)"
            ]
        )

        if exang_label == "No (0)":
            exang = 0
        else:
            exang = 1

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
            [1, 2, 3]
        )

        ca = st.selectbox(
            "Major Vessels (CA)",
            [0, 1, 2, 3]
        )

        thal = st.selectbox(
            "Thalassemia",
            [3, 6, 7]
        )


    # -----------------------------------------------------
    # PREDICTION BUTTON
    # -----------------------------------------------------

    st.divider()

    predict_button = st.button(
        "🔍 Predict Heart Disease",
        type="primary",
        use_container_width=True
    )


    if predict_button:

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

        prediction = heart_model.predict(
            patient_data
        )[0]

        probabilities = heart_model.predict_proba(
            patient_data
        )[0]

        no_disease_probability = probabilities[0]

        disease_probability = probabilities[1]


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.header(
            "Prediction Result"
        )

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
                "The model estimates a higher likelihood of heart disease for the entered information."
            )

            guidance = [
                "👨‍⚕️ Discuss this prediction with a qualified healthcare professional.",
                "🩺 Consider appropriate cardiovascular evaluation as advised by a healthcare professional.",
                "🥗 Follow a balanced, heart-healthy eating pattern.",
                "🚶 Maintain regular physical activity according to your fitness level and professional advice.",
                "🚭 Avoid smoking and tobacco exposure.",
                "😴 Maintain healthy sleep habits."
            ]

        else:

            st.success(
                "The model estimates a lower likelihood of heart disease for the entered information."
            )

            guidance = [
                "👨‍⚕️ Continue routine health check-ups as appropriate.",
                "🥗 Follow a balanced and heart-healthy diet.",
                "🚶 Maintain regular physical activity.",
                "🚭 Avoid smoking and tobacco exposure.",
                "😴 Maintain healthy sleep habits."
            ]


        if trestbps >= 140:

            guidance.append(
                "🩺 The entered resting blood pressure is elevated. Discuss monitoring with a healthcare professional."
            )


        if chol >= 240:

            guidance.append(
                "🧪 The entered cholesterol value is high. Discuss your lipid profile with a healthcare professional."
            )


        if fbs == 1:

            guidance.append(
                "🩸 The entered fasting blood sugar is marked as elevated. Discuss blood-sugar assessment with a healthcare professional."
            )


        if exang == 1:

            guidance.append(
                "💓 Exercise-induced angina is marked as present. Discuss this with a healthcare professional."
            )


        st.subheader(
            "Recommended Next Steps"
        )


        for item in guidance:

            st.write(item)


        # -------------------------------------------------
        # SHAP
        # -------------------------------------------------

        st.header(
            "🔎 Explainable AI"
        )

        st.write(
            """
SHAP explains which patient features influenced the
model's prediction.

Positive SHAP values push the prediction toward
Disease Present, while negative values push it toward
No Disease.
"""
        )


        patient_transformed = (
            heart_preprocessor.transform(
                patient_data
            )
        )


        shap_explanation = heart_explainer(
            patient_transformed
        )


        shap_values = (
            shap_explanation.values[0]
        )


        heart_explanation_df = pd.DataFrame(
            {
                "Feature": heart_feature_names,
                "SHAP Value": shap_values
            }
        )


        heart_explanation_df[
            "Absolute SHAP"
        ] = np.abs(
            heart_explanation_df[
                "SHAP Value"
            ]
        )


        heart_explanation_df = (
            heart_explanation_df
            .sort_values(
                by="Absolute SHAP",
                ascending=False
            )
            .head(10)
        )


        st.subheader(
            "Top Contributing Features"
        )


        heart_chart_df = (
            heart_explanation_df[
                [
                    "Feature",
                    "SHAP Value"
                ]
            ]
            .set_index("Feature")
        )


        st.bar_chart(
            heart_chart_df
        )


        heart_display_df = (
            heart_explanation_df[
                [
                    "Feature",
                    "SHAP Value"
                ]
            ].copy()
        )


        heart_display_df[
            "Direction"
        ] = np.where(
            heart_display_df["SHAP Value"] > 0,
            "Increases Disease Probability",
            "Decreases Disease Probability"
        )


        st.dataframe(
            heart_display_df,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# DIABETES MODULE
# =========================================================

elif disease == "Diabetes":

    # -----------------------------------------------------
    # DIABETES FILE PATHS
    # -----------------------------------------------------

    diabetes_model_path = os.path.join(
        MODELS_DIR,
        "diabetes_model.pkl"
    )

    diabetes_preprocessor_path = os.path.join(
        MODELS_DIR,
        "diabetes_preprocessor.pkl"
    )

    diabetes_background_path = os.path.join(
        MODELS_DIR,
        "diabetes_shap_background.pkl"
    )

    diabetes_feature_names_path = os.path.join(
        MODELS_DIR,
        "diabetes_feature_names.pkl"
    )

    diabetes_info_path = os.path.join(
        MODELS_DIR,
        "diabetes_model_info.pkl"
    )


    # -----------------------------------------------------
    # LOAD DIABETES MODEL
    # -----------------------------------------------------

    @st.cache_resource
    def load_diabetes_model():

        model = joblib.load(
            diabetes_model_path
        )

        preprocessor = joblib.load(
            diabetes_preprocessor_path
        )

        background = joblib.load(
            diabetes_background_path
        )

        feature_names = joblib.load(
            diabetes_feature_names_path
        )

        model_info = joblib.load(
            diabetes_info_path
        )

        return (
            model,
            preprocessor,
            background,
            feature_names,
            model_info
        )


    (
        diabetes_model,
        diabetes_preprocessor,
        diabetes_background,
        diabetes_feature_names,
        diabetes_model_info
    ) = load_diabetes_model()


    # -----------------------------------------------------
    # DIABETES HEADER
    # -----------------------------------------------------

    st.header(
        "🩸 Diabetes Prediction"
    )

    st.write(
        """
This application uses a dedicated machine learning model
to estimate the probability of diabetes based on patient
information.

The prediction is also explained using SHAP
(Explainable AI).
"""
    )


    # -----------------------------------------------------
    # DIABETES SIDEBAR
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

    diabetes_col1, diabetes_col2 = st.columns(2)


    with diabetes_col1:

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


    with diabetes_col2:

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
    # DIABETES BUTTON
    # -----------------------------------------------------

    st.divider()

    diabetes_button = st.button(
        "🔍 Predict Diabetes",
        type="primary",
        use_container_width=True
    )


    if diabetes_button:

        diabetes_patient = pd.DataFrame(
            [{
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": diabetes_pedigree,
                "Age": diabetes_age
            }]
        )


        # -------------------------------------------------
        # DIABETES PROBABILITY
        # -------------------------------------------------

        diabetes_probability = (
            diabetes_model
            .predict_proba(
                diabetes_patient
            )[0, 1]
        )


        # -------------------------------------------------
        # THRESHOLD
        # -------------------------------------------------

        diabetes_threshold = float(
            diabetes_model_info.get(
                "prediction_threshold",
                0.42
            )
        )


        diabetes_prediction = int(
            diabetes_probability
            >= diabetes_threshold
        )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.header(
            "Prediction Result"
        )


        diabetes_result_col1, diabetes_result_col2 = (
            st.columns(2)
        )


        with diabetes_result_col1:

            if diabetes_prediction == 1:

                st.error(
                    "⚠️ Model Prediction: Diabetes Likely"
                )

            else:

                st.success(
                    "✅ Model Prediction: Diabetes Less Likely"
                )


        with diabetes_result_col2:

            st.metric(
                "Diabetes Probability",
                f"{diabetes_probability:.2%}"
            )


        st.caption(
            f"Prediction threshold used: "
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
                "The model estimates a higher likelihood of diabetes for the entered information."
            )

            diabetes_guidance = [
                "👨‍⚕️ Discuss this prediction with a qualified healthcare professional.",
                "🧪 Consider appropriate blood-glucose testing and clinical evaluation as advised by a healthcare professional.",
                "🥗 Follow a balanced diet with appropriate portion control.",
                "🚶 Maintain regular physical activity according to your fitness level and professional advice.",
                "💧 Maintain adequate hydration.",
                "😴 Maintain healthy sleep habits."
            ]

        else:

            st.success(
                "The model estimates a lower likelihood of diabetes for the entered information."
            )

            diabetes_guidance = [
                "👨‍⚕️ Continue routine health check-ups as appropriate.",
                "🥗 Maintain a balanced diet.",
                "🚶 Maintain regular physical activity.",
                "⚖️ Maintain a healthy weight where appropriate.",
                "😴 Maintain healthy sleep habits."
            ]


        if glucose >= 126:

            diabetes_guidance.append(
                "🩸 The entered glucose value is high. Discuss appropriate glucose testing and evaluation with a healthcare professional."
            )

        elif glucose >= 100:

            diabetes_guidance.append(
                "🩸 The entered glucose value is above the normal fasting range if this measurement was fasting. Discuss it with a healthcare professional."
            )


        if bmi >= 30:

            diabetes_guidance.append(
                "⚖️ The entered BMI is in the obesity range. Discuss healthy weight-management strategies with a healthcare professional."
            )

        elif bmi >= 25:

            diabetes_guidance.append(
                "⚖️ The entered BMI is above the normal range. Healthy lifestyle and weight-management strategies may be useful."
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
SHAP explains which patient features influenced the
Diabetes prediction.

Positive SHAP values push the prediction toward
Diabetes, while negative values push it toward
No Diabetes.
"""
        )


        # -------------------------------------------------
        # TRANSFORM PATIENT
        # -------------------------------------------------

        diabetes_patient_transformed = (
            diabetes_preprocessor.transform(
                diabetes_patient
            )
        )


        diabetes_patient_transformed = np.asarray(
            diabetes_patient_transformed
        )


        # -------------------------------------------------
        # GET XGBOOST MODEL
        # -------------------------------------------------

        diabetes_model_step = (
            diabetes_model.named_steps["model"]
        )


        # -------------------------------------------------
        # SHAP EXPLAINER
        # -------------------------------------------------

        diabetes_explainer = shap.TreeExplainer(
            diabetes_model_step
        )


        raw_shap_values = (
            diabetes_explainer.shap_values(
                diabetes_patient_transformed
            )
        )


        raw_shap_values = np.asarray(
            raw_shap_values
        )


        # -------------------------------------------------
        # SHAP SHAPE HANDLING
        # -------------------------------------------------

        if raw_shap_values.ndim == 3:

            # Shape: samples, features, classes

            diabetes_shap_values = (
                raw_shap_values[0, :, 1]
            )

        elif raw_shap_values.ndim == 2:

            # Shape: samples, features

            diabetes_shap_values = (
                raw_shap_values[0]
            )

        else:

            diabetes_shap_values = (
                raw_shap_values
            )


        diabetes_shap_values = np.asarray(
            diabetes_shap_values
        ).reshape(-1)


        # -------------------------------------------------
        # SHAP FEATURE NAMES
        # -------------------------------------------------

        diabetes_feature_names = list(
            diabetes_feature_names
        )


        # Safety check

        if len(diabetes_feature_names) != len(
            diabetes_shap_values
        ):

            st.error(
                "SHAP feature count does not match the saved feature names."
            )

        else:

            # -------------------------------------------------
            # SHAP DATAFRAME
            # -------------------------------------------------

            diabetes_explanation_df = pd.DataFrame(
                {
                    "Feature": diabetes_feature_names,
                    "SHAP Value": diabetes_shap_values
                }
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
                    by="Absolute SHAP",
                    ascending=False
                )
                .head(8)
            )


            # -------------------------------------------------
            # SHAP CHART
            # -------------------------------------------------

            st.subheader(
                "Top Contributing Features"
            )


            diabetes_chart_df = (
                diabetes_explanation_df[
                    [
                        "Feature",
                        "SHAP Value"
                    ]
                ]
                .set_index("Feature")
            )


            st.bar_chart(
                diabetes_chart_df
            )


            # -------------------------------------------------
            # SHAP TABLE
            # -------------------------------------------------

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
                diabetes_display_df["SHAP Value"] > 0,
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
