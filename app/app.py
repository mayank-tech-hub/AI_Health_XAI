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
    os.path.dirname(os.path.abspath(__file__))
)


# =========================================================
# DISEASE SELECTION
# =========================================================

st.title("❤️ AI-Based Early Disease Prediction")

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

        model = joblib.load(
            MODEL_PATH
        )

        preprocessor = joblib.load(
            PREPROCESSOR_PATH
        )

        background = joblib.load(
            BACKGROUND_PATH
        )

        feature_names = joblib.load(
            FEATURE_NAMES_PATH
        )

        model_info = joblib.load(
            MODEL_INFO_PATH
        )

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

        logistic_model = _model.named_steps[
            "model"
        ]

        explainer = shap.LinearExplainer(
            logistic_model,
            _background
        )

        return explainer


    shap_explainer = (
        create_heart_shap_explainer(
            model,
            background
        )
    )


    # -----------------------------------------------------
    # HEART HEADER
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
    # HEART SIDEBAR
    # -----------------------------------------------------

    with st.sidebar:

        st.header(
            "Heart Disease Model"
        )

        st.write(
            f"**Model:** "
            f"{model_info['model_name']}"
        )

        st.write(
            f"**Dataset:** "
            f"{model_info['dataset']}"
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
            "Female (0)"
            if x == 0
            else "Male (1)"
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
            "No (0)"
            if x == 0
            else "Yes (1)"
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
            "No (0)"
            if x == 0
            else "Yes (1)"
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
        # HEART PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            patient_data
        )[0]

        probabilities = model.predict_proba(
            patient_data
        )[0]

        no_disease_probability = (
            probabilities[0]
        )

        disease_probability = (
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
                    "⚠️ Model Prediction: "
                    "Disease Present"
                )

            else:

                st.success(
                    "✅ Model Prediction: "
                    "No Disease"
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


        st.subheader(
            "Recommended Next Steps"
        )

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
        # HEART SHAP
        # -------------------------------------------------

        st.header(
            "🔎 Explainable AI"
        )

        st.write(
            """
The chart below shows which patient features influenced
the model's prediction.

Positive SHAP values push the prediction toward
Disease Present, while negative values push it toward
No Disease.
"""
        )


        patient_transformed = (
            preprocessor.transform(
                patient_data
            )
        )


        shap_explanation = (
            shap_explainer(
                patient_transformed
            )
        )


        heart_shap_values = np.asarray(
            shap_explanation.values[0]
        ).reshape(-1)


        if len(feature_names) == len(
            heart_shap_values
        ):

            explanation_df = pd.DataFrame(
                {
                    "Feature": feature_names,
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
# DIABETES
# =========================================================

elif disease == "Diabetes":

    # -----------------------------------------------------
    # DIABETES MODEL PATHS
    # -----------------------------------------------------

    DIABETES_MODEL_PATH = os.path.join(
        BASE_DIR,
        "models",
        "diabetes_model.pkl"
    )

    DIABETES_PREPROCESSOR_PATH = os.path.join(
        BASE_DIR,
        "models",
        "diabetes_preprocessor.pkl"
    )

    DIABETES_BACKGROUND_PATH = os.path.join(
        BASE_DIR,
        "models",
        "diabetes_shap_background.pkl"
    )

    DIABETES_FEATURE_NAMES_PATH = os.path.join(
        BASE_DIR,
        "models",
        "diabetes_feature_names.pkl"
    )

    DIABETES_MODEL_INFO_PATH = os.path.join(
        BASE_DIR,
        "models",
        "diabetes_model_info.pkl"
    )


    # -----------------------------------------------------
    # LOAD DIABETES MODEL
    # -----------------------------------------------------

    @st.cache_resource
    def load_diabetes_model():

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


    (
        diabetes_model,
        diabetes_preprocessor,
        diabetes_background,
        diabetes_feature_names,
        diabetes_model_info
    ) = load_diabetes_model()


    # -----------------------------------------------------
    # GET DIABETES ESTIMATOR
    # -----------------------------------------------------

    if hasattr(
        diabetes_model,
        "named_steps"
    ):

        diabetes_estimator = (
            diabetes_model.named_steps.get(
                "model",
                diabetes_model
            )
        )

        if (
            "preprocessor"
            in diabetes_model.named_steps
        ):

            diabetes_model_preprocessor = (
                diabetes_model.named_steps[
                    "preprocessor"
                ]
            )

        else:

            diabetes_model_preprocessor = (
                diabetes_preprocessor
            )

    else:

        diabetes_estimator = (
            diabetes_model
        )

        diabetes_model_preprocessor = (
            diabetes_preprocessor
        )


    # -----------------------------------------------------
    # DIABETES SHAP EXPLAINER FUNCTION
    # -----------------------------------------------------

    def create_diabetes_shap_explainer(
        estimator,
        background
    ):

        # IMPORTANT:
        # This function is intentionally NOT cached.
        #
        # Streamlit cache tries to hash sklearn estimator
        # objects such as ExtraTreesClassifier.
        # That causes:
        #
        # UnhashableParamError
        #
        # PermutationExplainer is model-agnostic and works
        # with the saved Extra Trees model.

        return shap.PermutationExplainer(
            estimator.predict_proba,
            background
        )


    # -----------------------------------------------------
    # DIABETES HEADER
    # -----------------------------------------------------

    st.header(
        "🩸 Diabetes Prediction"
    )

    st.write(
        """
This module uses a dedicated machine learning model to
estimate the probability of diabetes from patient information.

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
    # DIABETES PATIENT INFORMATION
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
    # DIABETES PREDICTION BUTTON
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
                "DiabetesPedigreeFunction":
                    diabetes_pedigree,
                "Age": diabetes_age
            }]
        )


        # -------------------------------------------------
        # DIABETES PREDICTION
        # -------------------------------------------------

        if hasattr(
            diabetes_model,
            "named_steps"
        ):

            diabetes_probability = float(
                diabetes_model.predict_proba(
                    diabetes_patient
                )[0, 1]
            )

        else:

            diabetes_patient_transformed = (
                diabetes_model_preprocessor.transform(
                    diabetes_patient
                )
            )

            diabetes_probability = float(
                diabetes_model.predict_proba(
                    diabetes_patient_transformed
                )[0, 1]
            )


        # -------------------------------------------------
        # THRESHOLD
        # -------------------------------------------------

        diabetes_threshold = float(
            diabetes_model_info.get(
                "prediction_threshold",
                0.50
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

        result_col1, result_col2 = (
            st.columns(2)
        )


        with result_col1:

            if diabetes_prediction == 1:

                st.error(
                    "⚠️ Model Prediction: "
                    "Diabetes Likely"
                )

            else:

                st.success(
                    "✅ Model Prediction: "
                    "Diabetes Less Likely"
                )


        with result_col2:

            st.metric(
                "Diabetes Probability",
                f"{diabetes_probability:.2%}"
            )


        st.caption(
            f"Model threshold used: "
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
                "⚠️ The model estimates a higher "
                "likelihood of diabetes for the "
                "entered information."
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
                "✅ The model estimates a lower "
                "likelihood of diabetes for the "
                "entered information."
            )

            diabetes_guidance = [

                "👨‍⚕️ Continue routine health check-ups as appropriate.",

                "🥗 Maintain a balanced diet.",

                "🚶 Maintain regular physical activity.",

                "⚖️ Maintain a healthy lifestyle and healthy weight where appropriate.",

                "😴 Maintain healthy sleep habits."

            ]


        if glucose >= 126:

            diabetes_guidance.append(
                "🩸 The entered glucose value is high. "
                "Discuss appropriate testing and "
                "evaluation with a healthcare professional."
            )

        elif glucose >= 100:

            diabetes_guidance.append(
                "🩸 The entered glucose value is above "
                "the normal fasting range if this "
                "measurement was fasting. Discuss it "
                "with a healthcare professional."
            )


        if bmi >= 30:

            diabetes_guidance.append(
                "⚖️ The entered BMI is in the obesity "
                "range. Discuss healthy weight-management "
                "strategies with a healthcare professional."
            )

        elif bmi >= 25:

            diabetes_guidance.append(
                "⚖️ The entered BMI is above the normal "
                "range. Healthy lifestyle and "
                "weight-management strategies may be useful."
            )


        st.subheader(
            "Recommended Next Steps"
        )


        for item in diabetes_guidance:

            st.write(item)


        # -------------------------------------------------
        # DIABETES SHAP EXPLANATION
        # -------------------------------------------------

        st.header(
            "🔎 Explainable AI"
        )

        st.write(
            """
SHAP explains which patient features influenced the
Diabetes prediction.

Positive SHAP values push the model toward Diabetes,
while negative SHAP values push it toward No Diabetes.
"""
        )


        # -------------------------------------------------
        # PREPARE PATIENT FOR SHAP
        # -------------------------------------------------

        diabetes_patient_for_shap = (
            diabetes_model_preprocessor.transform(
                diabetes_patient
            )
        )


        # -------------------------------------------------
        # PREPARE SHAP BACKGROUND
        # -------------------------------------------------

        diabetes_background_for_shap = (
            diabetes_background.copy()
        )


        if isinstance(
            diabetes_background_for_shap,
            pd.DataFrame
        ):

            expected_features = getattr(
                diabetes_estimator,
                "n_features_in_",
                None
            )


            if (
                expected_features is not None
                and
                diabetes_background_for_shap.shape[1]
                == expected_features
            ):

                # Background already has the same
                # number of features expected by
                # the estimator.

                diabetes_background_for_shap = (
                    diabetes_background_for_shap.to_numpy()
                )

            else:

                # Transform only when required.

                diabetes_background_for_shap = (
                    diabetes_model_preprocessor.transform(
                        diabetes_background_for_shap
                    )
                )

        else:

            diabetes_background_for_shap = (
                np.asarray(
                    diabetes_background_for_shap
                )
            )


        # -------------------------------------------------
        # LIMIT BACKGROUND SIZE
        # -------------------------------------------------

        if len(
            diabetes_background_for_shap
        ) > 50:

            diabetes_background_for_shap = (
                diabetes_background_for_shap[:50]
            )


        # -------------------------------------------------
        # CREATE DIABETES SHAP EXPLAINER
        # -------------------------------------------------

        try:

            diabetes_shap_explainer = (
                create_diabetes_shap_explainer(
                    diabetes_estimator,
                    diabetes_background_for_shap
                )
            )


            # -------------------------------------------------
            # CALCULATE SHAP VALUES
            # -------------------------------------------------

            diabetes_shap_explanation = (
                diabetes_shap_explainer(
                    diabetes_patient_for_shap
                )
            )


            raw_shap_values = np.asarray(
                diabetes_shap_explanation.values
            )


        except Exception as shap_error:

            st.error(
                "Diabetes SHAP explanation "
                "could not be generated."
            )

            st.caption(
                f"SHAP error: "
                f"{type(shap_error).__name__}: "
                f"{shap_error}"
            )

            raw_shap_values = None


        # -------------------------------------------------
        # HANDLE SHAP OUTPUT
        # -------------------------------------------------

        if raw_shap_values is None:

            diabetes_patient_shap = (
                np.array([])
            )

        elif raw_shap_values.ndim == 3:

            # Shape:
            # samples, features, classes

            if raw_shap_values.shape[2] > 1:

                # Class 1 = Diabetes

                diabetes_patient_shap = (
                    raw_shap_values[
                        0,
                        :,
                        1
                    ]
                )

            else:

                diabetes_patient_shap = (
                    raw_shap_values[
                        0,
                        :,
                        0
                    ]
                )

        elif raw_shap_values.ndim == 2:

            diabetes_patient_shap = (
                raw_shap_values[0]
            )

        else:

            diabetes_patient_shap = (
                raw_shap_values.reshape(-1)
            )


        diabetes_patient_shap = np.asarray(
            diabetes_patient_shap
        ).reshape(-1)


        # -------------------------------------------------
        # SHAP RESULT
        # -------------------------------------------------

        if len(
            diabetes_feature_names
        ) != len(
            diabetes_patient_shap
        ):

            if len(
                diabetes_patient_shap
            ) == 0:

                # SHAP error has already been
                # displayed above.

                pass

            else:

                st.error(
                    "Diabetes SHAP feature count "
                    "does not match the saved "
                    "feature names."
                )

        else:

            diabetes_explanation_df = (
                pd.DataFrame(
                    {
                        "Feature":
                            list(
                                diabetes_feature_names
                            ),

                        "SHAP Value":
                            diabetes_patient_shap
                    }
                )
            )


            # -------------------------------------------------
            # ABSOLUTE SHAP
            # -------------------------------------------------

            diabetes_explanation_df[
                "Absolute SHAP"
            ] = np.abs(
                diabetes_explanation_df[
                    "SHAP Value"
                ]
            )


            # -------------------------------------------------
            # TOP FEATURES
            # -------------------------------------------------

            diabetes_explanation_df = (
                diabetes_explanation_df
                .sort_values(
                    by="Absolute SHAP",
                    ascending=False
                )
                .head(8)
            )


            # -------------------------------------------------
            # SHAP BAR CHART
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
                .set_index(
                    "Feature"
                )
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
