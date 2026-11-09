
import math
import uuid

import pandas as pd
import streamlit as st

from services.prediction_service import predict
from services.xai_service import explain
from services.assessment_service import save_assessment
from services.health_guidance import build_health_guidance


HEART_MODELS = {
    "Original UCI Heart Disease": "original_heart",
    "Kaggle Heart Disease": "kaggle_heart",
}

DIABETES_MODELS = {
    "Original Pima Diabetes": "original_diabetes",
    "Kaggle Diabetes": "kaggle_diabetes",
}


# --------------------------------------------------
# JSON CONVERSION
# --------------------------------------------------

def _json_safe(value):

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())

    if hasattr(value, "item"):
        return _json_safe(value.item())

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if value is None or isinstance(value, (str, int, bool)):
        return value

    return str(value)


# --------------------------------------------------
# RESET FUNCTIONS
# --------------------------------------------------

def _clear_heart():

    st.session_state.pop("heart_review", None)
    st.session_state.pop("heart_result", None)

    st.session_state["heart_form_version"] = (
        st.session_state.get("heart_form_version", 0) + 1
    )


def _clear_diabetes():

    st.session_state.pop("diabetes_review", None)
    st.session_state.pop("diabetes_result", None)

    st.session_state["diabetes_form_version"] = (
        st.session_state.get("diabetes_form_version", 0) + 1
    )


# --------------------------------------------------
# PREDICTION + XAI
# --------------------------------------------------

def _run_assessment(model_key, data):

    result = predict(model_key, data)
    xai_result = explain(model_key, data)

    if result["prediction"] != xai_result["prediction"]:
        raise ValueError(
            "Prediction mismatch between prediction and XAI services."
        )

    if abs(
        result["model_score"] - xai_result["model_output"]
    ) > 1e-6:
        raise ValueError(
            "Model score mismatch between prediction and XAI services."
        )

    if not xai_result.get("validation_passed", False):
        raise ValueError("SHAP validation did not pass.")

    probabilities = result.get("class_probabilities")

    if not isinstance(probabilities, dict):
        raise ValueError(
            f"Class probabilities are missing for model: {model_key}"
        )

    class_0 = float(probabilities["Class 0"])
    class_1 = float(probabilities["Class 1"])

    if not (
        math.isfinite(class_0)
        and math.isfinite(class_1)
        and 0 <= class_0 <= 1
        and 0 <= class_1 <= 1
    ):
        raise ValueError("Invalid class probability values.")

    if abs((class_0 + class_1) - 1.0) > 1e-5:
        raise ValueError("Class probabilities must sum to 1.")

    xai_result["class_probabilities"] = {
        "Class 0": class_0,
        "Class 1": class_1,
    }

    xai_result["class_labels"] = result.get(
        "class_labels",
        {
            0: "Class 0",
            1: "Class 1",
        },
    )

    result["xai"] = xai_result

    return result


# --------------------------------------------------
# RESULT DISPLAY + DATABASE SAVE
# --------------------------------------------------

def _show_result(result, assessment_type, model_key, input_data):

    st.divider()
    st.subheader("Assessment Result")

    st.caption(f'Model: {result["model"]}')

    prediction = int(result["prediction"])
    score = float(result["model_score"])
    xai_result = result.get("xai") or {}

    probabilities = xai_result.get("class_probabilities") or {}

    class_0 = float(probabilities.get("Class 0", 1.0 - score))
    class_1 = float(probabilities.get("Class 1", score))

    class_labels = xai_result.get("class_labels") or {}

    class_0_label = class_labels.get(
        0,
        class_labels.get("0", "Class 0"),
    )

    class_1_label = class_labels.get(
        1,
        class_labels.get("1", "Class 1"),
    )

    # ----------------------------------------------
    # 1. PREDICTION SUMMARY
    # ----------------------------------------------

    result_col, score_col = st.columns([1.2, 1])

    with result_col:

        if prediction == 1:
            st.error("Model Output: Class 1")
        else:
            st.success("Model Output: Class 0")

    with score_col:

        st.metric(
            "Class 1 Model Score",
            f"{class_1 * 100:.2f}%",
        )

    # ----------------------------------------------
    # 2. CLASS PROBABILITY COMPARISON GRAPH
    # ----------------------------------------------

    st.markdown("### Disease vs No Disease Comparison")

    probability_df = pd.DataFrame(
        {
            "Class": [
                class_1_label,
                class_0_label,
            ],
            "Probability (%)": [
                class_1 * 100,
                class_0 * 100,
            ],
        }
    ).set_index("Class")

    st.bar_chart(
        probability_df,
        horizontal=True,
        use_container_width=True,
    )

    # ----------------------------------------------
    # 3. EXPLAINABLE AI
    # ----------------------------------------------

    if xai_result:

        st.divider()
        st.subheader("Explainable AI (XAI)")

        st.markdown("### SHAP Feature Contribution Graph")

        all_features = (
            xai_result.get("all_features")
            or xai_result.get("top_features", [])
        )

        chart_items = sorted(
            all_features,
            key=lambda item: abs(
                float(item.get("contribution", 0.0))
            ),
            reverse=True,
        )

        if chart_items:

            chart_df = pd.DataFrame({
                "Feature": [
                    str(item["feature"])
                    for item in chart_items
                ],
                "SHAP Contribution": [
                    float(item["contribution"])
                    for item in chart_items
                ],
            })

            chart_df = chart_df.set_index("Feature")

            st.bar_chart(
                chart_df,
                horizontal=True,
                use_container_width=True,
            )

            with st.expander(
                "View detailed SHAP contribution values"
            ):

                detail_df = pd.DataFrame([
                    {
                        "Feature": item["feature"],
                        "Contribution": round(
                            float(item["contribution"]), 6
                        ),
                        "Direction": item.get("direction", ""),
                    }
                    for item in chart_items
                ])

                st.dataframe(
                    detail_df,
                    use_container_width=True,
                    hide_index=True,
                )

        else:

            st.info(
                "SHAP feature contributions are not available."
            )


    # ----------------------------------------------
    # 4. HEALTH GUIDANCE
    # ----------------------------------------------

    st.divider()
    st.subheader("Health Guidance & Next Steps")

    guidance = build_health_guidance(
        assessment_type=assessment_type,
        model_key=model_key,
        prediction=prediction,
        model_score=score,
        input_data=input_data,
        xai_result=xai_result,
    )

    st.markdown("### What This Result Means")
    st.write(guidance["summary"])

    # ----------------------------------------------
    # HEALTH GUIDANCE SECTIONS
    # ----------------------------------------------

    sections = guidance.get("sections", [])

    if isinstance(sections, dict):
        sections = [sections]
    elif not isinstance(sections, (list, tuple)):
        sections = []

    for section in sections:

        if not isinstance(section, dict):
            continue

        section_title = section.get(
            "title",
            "Additional Information"
        )

        st.markdown(f"### {section_title}")

        items = section.get("items", [])

        if isinstance(items, str):
            items = [items]

        elif isinstance(items, (list, tuple)):
            items = list(items)

            if (
                len(items) > 1
                and all(
                    isinstance(item, str) and len(item) == 1
                    for item in items
                )
            ):
                items = ["".join(items)]

        else:
            items = []

        for item in items:
            if item is not None:
                st.markdown(f"- {item}")

    # ----------------------------------------------
    # IMPORTANT CONSIDERATIONS
    # ----------------------------------------------

    alerts = guidance.get("alerts", [])

    if isinstance(alerts, str):
        alerts = [alerts]

    elif not isinstance(alerts, (list, tuple)):
        alerts = []

    if alerts:

        st.markdown("### Important Considerations")

        for alert in alerts:

            if alert is not None:
                st.markdown(f"- {alert}")

    # ----------------------------------------------
    # 5. DATABASE SAVE
    # ----------------------------------------------

    st.divider()
    st.subheader("Save Assessment")

    user = st.session_state.get("user")

    if not user or not user.get("id"):
        st.error("User session not found. Please log in again.")
        return

    save_token = result.get("_save_token")

    if not save_token:
        st.error("Assessment save reference is missing.")
        return

    saved_key = f"assessment_saved_{save_token}"

    if saved_key in st.session_state:

        st.success(
            f'Assessment saved successfully. '
            f'Assessment ID: {st.session_state[saved_key]}'
        )

        st.caption(
            "You can view this record in Assessment History."
        )
        return

    consent_key = f"assessment_consent_{save_token}"

    consent = st.checkbox(
        "I consent to saving my assessment inputs, model result "
        "and XAI explanation in my account history.",
        key=consent_key,
    )

    if st.button(
        "Save Assessment",
        key=f"save_assessment_{save_token}",
        type="primary",
        use_container_width=True,
        disabled=not consent,
    ):

        try:

            with st.spinner(
                "Saving assessment to your account..."
            ):

                assessment_id = save_assessment(
                    user_id=int(user["id"]),
                    assessment_type=assessment_type,
                    model_key=model_key,
                    prediction=int(result["prediction"]),
                    model_score=float(result["model_score"]),
                    input_data=_json_safe(input_data),
                    explanation=_json_safe(xai_result),
                    consent_given=True,
                )

            if assessment_id:

                st.session_state[saved_key] = assessment_id

                st.success(
                    f"Assessment saved successfully. "
                    f"Assessment ID: {assessment_id}"
                )

                st.caption(
                    "You can view this record in Assessment History."
                )

            else:

                st.error(
                    "Assessment could not be saved. Please try again."
                )

        except Exception:

            st.error(
                "Something went wrong while saving the assessment. "
                "Please try again."
            )


# --------------------------------------------------
# HEART ASSESSMENT FORM
# --------------------------------------------------

def heart_form():

    st.subheader("Heart Disease Assessment")

    st.write(
        "Enter the required information to generate a "
        "machine-learning-based heart risk assessment."
    )

    if "heart_form_version" not in st.session_state:
        st.session_state["heart_form_version"] = 0

    # Model selection
    with st.container(border=True):

        st.markdown("### Select Prediction Model")

        model_name = st.selectbox(
            "Heart Disease Model",
            list(HEART_MODELS.keys()),
            key="heart_model_choice",
            on_change=_clear_heart,
        )

        model_key = HEART_MODELS[model_name]

        if model_key == "original_heart":
            st.caption(
                "Original UCI Heart Disease dataset model."
            )
        else:
            st.caption(
                "Kaggle Heart Disease dataset model."
            )

    version = st.session_state["heart_form_version"]
    suffix = f"_v{version}"

    st.divider()

    st.markdown("### Assessment Information")

    st.caption(
        "Enter values in the specified units. "
        "Categorical fields use dataset codes."
    )

    # Input form
    with st.form(f"heart_form{suffix}"):

        with st.container(border=True):

            st.markdown("#### 1. Personal Information")

            c1, c2 = st.columns(2)

            with c1:
                age = st.number_input(
                    "Age (years)",
                    18, 100, 45,
                    key="h_age" + suffix
                )

            with c2:
                sex = st.selectbox(
                    "Sex (dataset code)",
                    [0, 1],
                    key="h_sex" + suffix
                )

        with st.container(border=True):

            st.markdown("#### 2. Clinical Measurements")

            c1, c2, c3 = st.columns(3)

            with c1:
                trestbps = st.number_input(
                    "Resting Blood Pressure (mm Hg)",
                    50, 260, 120,
                    key="h_bp" + suffix
                )

                chol = st.number_input(
                    "Serum Cholesterol (mg/dL)",
                    50, 700, 200,
                    key="h_chol" + suffix
                )

            with c2:
                thalach = st.number_input(
                    "Maximum Heart Rate (bpm)",
                    40, 250, 150,
                    key="h_hr" + suffix
                )

                oldpeak = st.number_input(
                    "Oldpeak (ST Depression)",
                    0.0, 15.0, 1.0,
                    step=0.1,
                    key="h_oldpeak" + suffix
                )

            with c3:
                fbs = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dL",
                    [0, 1],
                    key="h_fbs" + suffix
                )

                exang = st.selectbox(
                    "Exercise-Induced Angina",
                    [0, 1],
                    key="h_exang" + suffix
                )

        with st.container(border=True):

            st.markdown("#### 3. Categorical Measurements")

            c1, c2 = st.columns(2)

            with c1:

                cp = st.selectbox(
                    "Chest Pain Type (dataset code)",
                    [1, 2, 3, 4],
                    key="h_cp" + suffix
                )

                restecg = st.selectbox(
                    "Resting ECG (dataset code)",
                    [0, 1, 2],
                    key="h_ecg" + suffix
                )

            with c2:

                slope = st.selectbox(
                    "ST Slope (dataset code)",
                    [1, 2, 3],
                    key="h_slope" + suffix
                )

                ca = st.selectbox(
                    "CA (dataset code)",
                    [0, 1, 2, 3],
                    key="h_ca" + suffix
                )

            thal = st.selectbox(
                "Thal (dataset code)",
                [3, 6, 7],
                key="h_thal" + suffix
            )

        st.caption(
            "Please verify all entered values before submitting."
        )

        submitted = st.form_submit_button(
            "Review Assessment",
            use_container_width=True,
            type="primary"
        )

    # Store submitted information
    if submitted:

        data = {
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
            "thal": thal,
        }

        st.session_state.pop("heart_result", None)

        st.session_state["heart_review"] = {
            "model_name": model_name,
            "model_key": model_key,
            "data": data,
        }

    # Review section
    review = st.session_state.get("heart_review")

    if review and review["model_key"] != model_key:

        _clear_heart()
        review = None

    if review:

        st.divider()

        with st.container(border=True):

            st.markdown("### Review Your Assessment")

            st.caption(
                f'Selected model: {review["model_name"]}'
            )

            st.dataframe(
                [
                    {"Feature": k, "Entered Value": v}
                    for k, v in review["data"].items()
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Confirm the information before generating the assessment."
            )

            c1, c2 = st.columns(2)

            with c1:

                confirm = st.button(
                    "Confirm and Predict",
                    key="heart_confirm",
                    type="primary",
                    use_container_width=True,
                )

            with c2:

                reset = st.button(
                    "Reset Assessment",
                    key="heart_reset",
                    use_container_width=True,
                )

            if confirm:

                try:

                    with st.spinner(
                        "Generating prediction and SHAP explanation..."
                    ):

                        result = _run_assessment(
                            review["model_key"],
                            review["data"],
                        )

                    result["_save_token"] = uuid.uuid4().hex

                    st.session_state["heart_result"] = result

                    st.rerun()

                except Exception as exc:

                    st.session_state.pop("heart_result", None)

                    st.error(f"Assessment failed: {exc}")

            if reset:

                _clear_heart()
                st.rerun()

    # Display result and database save section
    result = st.session_state.get("heart_result")

    if result:

        review = st.session_state.get("heart_review")

        if review:

            _show_result(
                result,
                "heart",
                review["model_key"],
                review["data"],
            )


# --------------------------------------------------
# DIABETES ASSESSMENT FORM
# --------------------------------------------------

def diabetes_form():

    st.subheader("Diabetes Assessment")

    st.write(
        "Enter the required information to generate a "
        "machine-learning-based diabetes risk assessment."
    )

    if "diabetes_form_version" not in st.session_state:
        st.session_state["diabetes_form_version"] = 0

    # Model selection
    with st.container(border=True):

        st.markdown("### Select Prediction Model")

        model_name = st.selectbox(
            "Diabetes Model",
            list(DIABETES_MODELS.keys()),
            key="diabetes_model_choice",
            on_change=_clear_diabetes,
        )

        model_key = DIABETES_MODELS[model_name]

        if model_key == "original_diabetes":
            st.caption(
                "Original Pima Indians Diabetes dataset model."
            )
        else:
            st.caption(
                "Kaggle Diabetes dataset model."
            )

    version = st.session_state["diabetes_form_version"]
    suffix = f"_v{version}"

    st.divider()

    st.markdown("### Assessment Information")

    with st.form(f"diabetes_form{suffix}"):

        if model_key == "original_diabetes":

            with st.container(border=True):

                st.markdown("#### 1. Personal Information")

                c1, c2 = st.columns(2)

                with c1:

                    pregnancies = st.number_input(
                        "Pregnancies (count)",
                        0, 30, 1,
                        key="d_preg" + suffix
                    )

                    age = st.number_input(
                        "Age (years)",
                        18, 120, 30,
                        key="d_age" + suffix
                    )

                with c2:

                    pedigree = st.number_input(
                        "Diabetes Pedigree Function",
                        0.0, 5.0, 0.5,
                        step=0.01,
                        key="d_pedigree" + suffix
                    )

            with st.container(border=True):

                st.markdown("#### 2. Clinical Measurements")

                c1, c2 = st.columns(2)

                with c1:

                    glucose = st.number_input(
                        "Glucose (mg/dL)",
                        1.0, 400.0, 120.0,
                        key="d_glucose" + suffix
                    )

                    bp = st.number_input(
                        "Blood Pressure (mm Hg)",
                        1.0, 200.0, 70.0,
                        key="d_bp" + suffix
                    )

                with c2:

                    skin = st.number_input(
                        "Skin Thickness (mm)",
                        1.0, 150.0, 20.0,
                        key="d_skin" + suffix
                    )

                    insulin = st.number_input(
                        "Insulin (mu U/ml)",
                        1.0, 1500.0, 80.0,
                        key="d_insulin" + suffix
                    )

                bmi = st.number_input(
                    "BMI (kg/m²)",
                    1.0, 100.0, 25.0,
                    key="d_bmi" + suffix
                )

            data = {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": bp,
                "SkinThickness": skin,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": pedigree,
                "Age": age,
            }

        else:

            with st.container(border=True):

                st.markdown("#### 1. Personal Information")

                c1, c2 = st.columns(2)

                with c1:

                    gender = st.selectbox(
                        "Gender",
                        ["Female", "Male", "Other"],
                        key="d_gender" + suffix
                    )

                    age = st.number_input(
                        "Age (years)",
                        0.0, 100.0, 30.0,
                        step=0.1,
                        key="d_age" + suffix
                    )

                with c2:

                    smoking = st.selectbox(
                        "Smoking History",
                        [
                            "No Info",
                            "current",
                            "ever",
                            "former",
                            "never",
                            "not current"
                        ],
                        key="d_smoking" + suffix
                    )

                    bmi = st.number_input(
                        "BMI (kg/m²)",
                        5.0, 120.0, 25.0,
                        key="d_bmi" + suffix
                    )

            with st.container(border=True):

                st.markdown("#### 2. Clinical Measurements")

                c1, c2 = st.columns(2)

                with c1:

                    hypertension = st.selectbox(
                        "Hypertension",
                        [0, 1],
                        format_func=lambda x: "Yes" if x else "No",
                        key="d_hyper" + suffix
                    )

                    heart_disease = st.selectbox(
                        "Existing Heart Disease",
                        [0, 1],
                        format_func=lambda x: "Yes" if x else "No",
                        key="d_heart" + suffix
                    )

                with c2:

                    hba1c = st.number_input(
                        "HbA1c Level (%)",
                        2.0, 20.0, 5.5,
                        step=0.1,
                        key="d_hba1c" + suffix
                    )

                    glucose = st.number_input(
                        "Blood Glucose Level (mg/dL)",
                        40, 500, 100,
                        key="d_glucose" + suffix
                    )

            data = {
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "smoking_history": smoking,
                "bmi": bmi,
                "HbA1c_level": hba1c,
                "blood_glucose_level": glucose,
            }

        st.caption(
            "Please verify all entered values before submitting."
        )

        submitted = st.form_submit_button(
            "Review Assessment",
            use_container_width=True,
            type="primary"
        )

    # Store review data
    if submitted:

        st.session_state.pop("diabetes_result", None)

        st.session_state["diabetes_review"] = {
            "model_name": model_name,
            "model_key": model_key,
            "data": data,
        }

    review = st.session_state.get("diabetes_review")

    if review and review["model_key"] != model_key:

        _clear_diabetes()
        review = None

    # Review section
    if review:

        st.divider()

        with st.container(border=True):

            st.markdown("### Review Your Assessment")

            st.caption(
                f'Selected model: {review["model_name"]}'
            )

            st.dataframe(
                [
                    {"Feature": k, "Entered Value": v}
                    for k, v in review["data"].items()
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Confirm the information before generating the assessment."
            )

            c1, c2 = st.columns(2)

            with c1:

                confirm = st.button(
                    "Confirm and Predict",
                    key="diabetes_confirm",
                    type="primary",
                    use_container_width=True,
                )

            with c2:

                reset = st.button(
                    "Reset Assessment",
                    key="diabetes_reset",
                    use_container_width=True,
                )

            if confirm:

                try:

                    with st.spinner(
                        "Generating prediction and SHAP explanation..."
                    ):

                        result = _run_assessment(
                            review["model_key"],
                            review["data"],
                        )

                    result["_save_token"] = uuid.uuid4().hex

                    st.session_state["diabetes_result"] = result

                    st.rerun()

                except Exception as exc:

                    st.session_state.pop("diabetes_result", None)

                    st.error(f"Assessment failed: {exc}")

            if reset:

                _clear_diabetes()
                st.rerun()

    # Result and database saving
    result = st.session_state.get("diabetes_result")

    if result:

        review = st.session_state.get("diabetes_review")

        if review:

            _show_result(
                result,
                "diabetes",
                review["model_key"],
                review["data"],
            )
