from services.assessment_service import get_user_assessments

print("1. Complete history")
records = get_user_assessments(user_id=1)

for record in records:
    print({
        "id": record["id"],
        "type": record["assessment_type"],
        "model": record["model_key"],
        "prediction": record["prediction"],
        "date": record["created_at"],
    })

print("\n2. Heart history")
heart_records = get_user_assessments(
    user_id=1,
    assessment_type="heart"
)
print("Heart records:", len(heart_records))

print("\n3. Diabetes history")
diabetes_records = get_user_assessments(
    user_id=1,
    assessment_type="diabetes"
)
print("Diabetes records:", len(diabetes_records))

print("\n4. Invalid user")
print("Invalid user records:", len(get_user_assessments(user_id=99999)))
