
from sqlalchemy import inspect, text

from database.connection import get_engine


def migrate_otp_resend_limits():
    engine = get_engine()

    try:
        with engine.begin() as connection:

            inspector = inspect(connection)
            tables = inspector.get_table_names()

            if "email_verifications" not in tables:
                raise RuntimeError(
                    "Email verification table not found. Migration cancelled."
                )

            connection.execute(
                text("""
                    ALTER TABLE email_verifications
                    ADD COLUMN IF NOT EXISTS resend_count
                    INTEGER NOT NULL DEFAULT 0
                """)
            )

            connection.execute(
                text("""
                    ALTER TABLE email_verifications
                    ADD COLUMN IF NOT EXISTS resend_window_started_at
                    TIMESTAMP WITH TIME ZONE
                    NOT NULL DEFAULT NOW()
                """)
            )

            inspector = inspect(connection)

            columns = {
                column["name"]
                for column in inspector.get_columns(
                    "email_verifications"
                )
            }

            required_columns = {
                "resend_count",
                "resend_window_started_at",
            }

            if not required_columns.issubset(columns):
                raise RuntimeError(
                    "Resend limit columns verification failed."
                )

        print("OTP resend-limit migration successful.")
        print("Existing OTP records preserved.")

    except Exception as exc:
        print("Migration failed:", str(exc))
        raise


if __name__ == "__main__":
    migrate_otp_resend_limits()
