from sqlalchemy import select
from sqlalchemy.orm import Session
from database.connection import get_engine
from database.models import User, Assessment

engine = get_engine()

with Session(engine) as session:
    print("REGISTERED USERS")
    users = session.scalars(select(User)).all()

    for user in users:
        print({
            "id": user.id,
            "name": user.full_name,
            "email": user.email,
            "active": user.is_active
        })

    print("\nSAVED ASSESSMENTS")
    assessments = session.scalars(select(Assessment)).all()

    for item in assessments:
        print({
            "id": item.id,
            "user_id": item.user_id,
            "type": item.assessment_type,
            "model": item.model_key,
            "prediction": item.prediction
        })
