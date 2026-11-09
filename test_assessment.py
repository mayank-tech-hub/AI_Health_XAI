from services.assessment_service import save_assessment

print("1. Consent rejection test")

result = save_assessment(
    user_id=1,
    assessment_type="heart",
    model_key="original_heart",
    prediction=0,
    model_score=0.15,
    input_data={"age": 45, "sex": 1},
    explanation={"top_features": []},
    consent_given=False,
)

print(result)

print("\n2. Valid assessment save test")

result = save_assessment(
    user_id=1,
    assessment_type="heart",
    model_key="original_heart",
    prediction=0,
    model_score=0.15,
    input_data={"age": 45, "sex": 1},
    explanation={"top_features": []},
    consent_given=True,
)

print(result)
