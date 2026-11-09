
def _top_feature_names(xai_result, limit=3):
    items = xai_result.get("top_features") or []

    return [
        str(item.get("feature", ""))
        for item in items[:limit]
        if isinstance(item, dict) and item.get("feature")
    ]


def build_health_guidance(
    assessment_type,
    model_key,
    prediction,
    model_score,
    input_data,
    xai_result=None,
):
    xai_result = xai_result or {}
    input_data = input_data or {}

    top_names = _top_feature_names(xai_result)
    model_class_one = int(prediction) == 1

    # -----------------------------------------
    # HEART HEALTH GUIDANCE
    # -----------------------------------------

    if assessment_type == "heart":

        if model_class_one:
            summary = (
                "The model assigned this input to Class 1. "
                "Discuss this result with a healthcare professional, "
                "especially if you have symptoms."
            )
        else:
            summary = (
                "The model assigned this input to Class 0. "
                "This does not rule out heart disease or guarantee "
                "that there is no health concern."
            )

        sections = [
            {
                "title": "Practical Heart-Health Habits",
                "items": [
                    "Discuss blood pressure and cholesterol measurements with a clinician.",
                    "Choose balanced meals with vegetables, fruits, whole grains and pulses.",
                    "Build regular physical activity gradually according to your health.",
                    "Avoid tobacco and seek professional support if you smoke.",
                ],
            },
            {
                "title": "What to Review with Your Clinician",
                "items": [
                    "Review blood pressure, cholesterol, symptoms and family history.",
                    "Ask whether further examination or tests are appropriate.",
                    "Do not stop or change prescribed medicines based on those results.",
                ],
            },
        ]

        alerts = []

        bp = input_data.get("trestbps")

        if isinstance(bp, (int, float)) and bp >= 180:
            alerts.append(
                "The entered resting blood pressure is very high "
                "(180 mm Hg or above). Repeat it after resting and "
                "seek prompt medical advice. If it is 180 systolic "
                "and/or 120 diastolic with chest pain, breathlessness, "
                "weakness, vision change or difficulty speaking, "
                "seek emergency care."
            )

        if input_data.get("exang") == 1:
            alerts.append(
                "Exercise-related angina was recorded in the input. "
                "Discuss this with a healthcare professional."
            )

        if input_data.get("cp") in (1, 4):
            alerts.append(
                "A chest-pain category was selected. Dataset codes "
                "are not a clinical interpretation. Explain any "
                "chest symptoms to a clinician."
            )

        if top_names:
            sections.append(
                {
                    "title": "Model Features to Discuss",
                    "items": [
                        "The model was most influenced by: "
                        + ", ".join(top_names) + "."
                    ],
                }
            )

    # -----------------------------------------
    # DIABETES GUIDANCE
    # -----------------------------------------

    elif assessment_type == "diabetes":

        if model_class_one:
            summary = (
                "The model assigned this input to Class 1. "
                "Discuss appropriate laboratory testing and your "
                "individual health context with a healthcare professional."
            )
        else:
            summary = (
                "The model assigned this input to Class 0. "
                "This does not exclude diabetes or replace "
                "recommended screening and laboratory tests."
            )

        sections = [
            {
                "title": "Practical Metabolic-Health Habits",
                "items": [
                    "Prefer balanced meals with vegetables, pulses, whole grains and suitable protein.",
                    "Limit frequent sugary drinks and highly refined snacks.",
                    "Aim for regular physical activity appropriate for your health.",
                    "Maintain a consistent sleep routine.",
                    "Avoid crash diets and extreme food restrictions.",
                    "Do not change diabetes medicines based on this app.",
                ],
            },
            {
                "title": "What to Review with Your Clinician",
                "items": [
                    "Ask whether confirmatory HbA1c or fasting plasma glucose testing is appropriate.",
                    "Review family history, symptoms, medicines and previous glucose results.",
                ],
            },
        ]

        alerts = []

        glucose = input_data.get(
            "blood_glucose_level",
            input_data.get("Glucose"),
        )

        hba1c = input_data.get("HbA1c_level")

        if isinstance(hba1c, (int, float)) and hba1c >= 6.5:
            alerts.append(
                "The entered HbA1c value is at or above a commonly "
                "used laboratory diabetes threshold (6.5%). Diagnosis "
                "requires appropriate clinical confirmation. Discuss "
                "this result with a healthcare professional."
            )

        if isinstance(glucose, (int, float)) and glucose >= 200:
            alerts.append(
                "The entered glucose value is 200 mg/dL or higher. "
                "Its meaning depends on whether it was random, fasting "
                "or measured after a meal and on symptoms. Discuss it "
                "with a healthcare professional."
            )

        if top_names:
            sections.append(
                {
                    "title": "Model Features to Discuss",
                    "items": [
                        "The model was most influenced by: "
                        + ", ".join(top_names) + ".",
                        "These are statistical contributors, not "
                        "a diagnosis or proof of medical causation.",
                    ],
                }
            )

    else:
        summary = (
            "The assessment type was not recognised. "
            "Seek guidance from a qualified healthcare professional."
        )

        sections = []
        alerts = []

    # -----------------------------------------
    # FINAL RESPONSE
    # -----------------------------------------

    return {
        "summary": summary,
        "sections": sections,
        "alerts": alerts,
        "disclaimer": (
            "Guidance is general education, not an individual "
            "treatment plan. If you have severe or sudden chest pain, "
            "severe breathlessness, fainting, signs of stroke, or "
            "another emergency symptom, seek emergency medical care "
            "immediately. In India, call 112."
        ),
    }
