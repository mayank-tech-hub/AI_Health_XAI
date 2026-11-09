
from sqlalchemy import inspect, text

from database.connection import get_engine
from database.models import Base


def migrate_email_verification():
    engine = get_engine()

    try:
        with engine.begin() as connection:

            inspector = inspect(connection)
            tables = inspector.get_table_names()

            if "users" not in tables:
                raise RuntimeError(
                    "Users table not found. Migration cancelled."
                )

            # Preserve all existing users as verified.
            connection.execute(
                text("""
                    ALTER TABLE users
                    ADD COLUMN IF NOT EXISTS email_verified
                    BOOLEAN NOT NULL DEFAULT TRUE
                """)
            )

            # Create only missing tables, including email_verifications.
            Base.metadata.create_all(bind=connection)

            # Confirm migration.
            inspector = inspect(connection)

            user_columns = {
                column["name"]
                for column in inspector.get_columns("users")
            }

            if "email_verified" not in user_columns:
                raise RuntimeError(
                    "email_verified column was not created."
                )

            if "email_verifications" not in inspector.get_table_names():
                raise RuntimeError(
                    "email_verifications table was not created."
                )

        print("Email verification migration successful.")
        print("Existing users and assessments preserved.")

    except Exception as exc:
        print("Migration failed:", str(exc))
        raise


if __name__ == "__main__":
    migrate_email_verification()
