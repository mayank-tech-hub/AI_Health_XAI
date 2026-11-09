from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from database.connection import get_engine
from database.models import Assessment, User


ALLOWED_MODELS = {
    "original_heart",
    "kaggle_heart",
    "original_diabetes",
    "kaggle_diabetes",
}


def save_assessment(
    user_id,
    assessment_type,
    model_key,
    prediction,
    model_score,
    input_data,
    explanation,
    consent_given,
):
    if not consent_given:
        return {
            "success": False,
            "message": "Consent is required to save assessment data.",
        }

    if assessment_type not in ("heart", "diabetes"):
        return {"success": False, "message": "Invalid assessment type."}

    if model_key not in ALLOWED_MODELS:
        return {"success": False, "message": "Invalid model selection."}

    if not isinstance(prediction, int) or prediction not in (0, 1):
        return {"success": False, "message": "Invalid prediction value."}

    if not isinstance(input_data, dict):
        return {"success": False, "message": "Invalid assessment input."}

    if explanation is not None and not isinstance(explanation, dict):
        return {"success": False, "message": "Invalid explanation data."}

    if assessment_type == "heart" and model_key not in (
        "original_heart", "kaggle_heart"
    ):
        return {"success": False, "message": "Heart model mismatch."}

    if assessment_type == "diabetes" and model_key not in (
        "original_diabetes", "kaggle_diabetes"
    ):
        return {"success": False, "message": "Diabetes model mismatch."}

    if model_score is not None:
        if (
            isinstance(model_score, bool)
            or not isinstance(model_score, (int, float))
            or not 0 <= model_score <= 1
        ):
            return {"success": False, "message": "Invalid model score."}

    try:
        engine = get_engine()

        with Session(engine) as session:
            user = session.scalar(
                select(User).where(
                    User.id == user_id,
                    User.is_active.is_(True),
                )
            )

            if user is None:
                return {"success": False, "message": "Active user not found."}

            assessment = Assessment(
                user_id=user_id,
                assessment_type=assessment_type,
                model_key=model_key,
                prediction=prediction,
                model_score=model_score,
                input_data=input_data,
                explanation=explanation,
                consent_given=True,
                consent_at=datetime.now(timezone.utc),
            )

            session.add(assessment)
            session.commit()
            assessment_id = assessment.id

        return {
            "success": True,
            "message": "Assessment saved successfully.",
            "assessment_id": assessment_id,
        }

    except SQLAlchemyError:
        return {
            "success": False,
            "message": "Database error while saving assessment.",
        }

def get_user_assessments(user_id, assessment_type=None, limit=50):
    if not isinstance(user_id, int) or user_id <= 0:
        return []

    if assessment_type not in (None, "heart", "diabetes"):
        return []

    if not isinstance(limit, int) or not 1 <= limit <= 100:
        return []

    try:
        engine = get_engine()

        with Session(engine) as session:
            query = select(Assessment).where(
                Assessment.user_id == user_id
            )

            if assessment_type:
                query = query.where(
                    Assessment.assessment_type == assessment_type
                )

            query = query.order_by(
                Assessment.created_at.desc()
            ).limit(limit)

            records = session.scalars(query).all()

            return [
                {
                    "id": item.id,
                    "assessment_type": item.assessment_type,
                    "model_key": item.model_key,
                    "prediction": item.prediction,
                    "model_score": item.model_score,
                    "input_data": item.input_data,
                    "explanation": item.explanation,
                    "created_at": item.created_at,
                }
                for item in records
            ]

    except SQLAlchemyError:
        return []