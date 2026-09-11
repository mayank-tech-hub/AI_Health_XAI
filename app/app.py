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
# =========================================================
# HEART DISEASE
# =========================================================
# =========================================================

if disease == "Heart Disease":

    # -----------------------------------------------------
    # HEART FILE PATHS
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # LOAD HEART MODEL
    # -----------------------------------------------------

    @st.cache_resource
    def load_heart_model():

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
    def create_heart_shap_explainer(
        _model,
        _background
    ):

        logistic_model = (
            _model.named_steps["model"]
        )

        explainer = shap.LinearExplainer(
            logistic_model,
            _background
        )

        return explainer


    heart_shap_explainer = (
        create_heart_shap_explainer(
            heart_model,
            heart_background
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

       
