from database.connection import get_engine
from database.models import Base


def initialize_database():
    engine = get_engine()

    Base.metadata.create_all(bind=engine)

    print("Database tables created/verified successfully.")

    with engine.connect() as connection:
        from sqlalchemy import inspect

        inspector = inspect(connection)
        tables = inspector.get_table_names()

    print("Available tables:", tables)


if __name__ == "__main__":
    initialize_database()
