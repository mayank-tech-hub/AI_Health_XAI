
from sqlalchemy import inspect

from database.connection import get_engine
from database.models import Base


def migrate_pending_registrations():

    engine = get_engine()

    try:
        with engine.begin() as connection:

            inspector = inspect(connection)
            existing_tables = inspector.get_table_names()

            if "users" not in existing_tables:
                raise RuntimeError(
                    "Users table not found. Migration cancelled."
                )

            if "assessments" not in existing_tables:
                raise RuntimeError(
                    "Assessments table not found. Migration cancelled."
                )

            # Creates only missing tables.
            # Existing users and assessments remain untouched.
            Base.metadata.create_all(bind=connection)

            inspector = inspect(connection)
            tables = inspector.get_table_names()

            if "pending_registrations" not in tables:
                raise RuntimeError(
                    "Pending registrations table was not created."
                )

        print("Pending registration migration successful.")
        print("Existing users and assessments preserved.")

    except Exception as exc:
        print("Migration failed:", str(exc))
        raise


if __name__ == "__main__":
    migrate_pending_registrations()
