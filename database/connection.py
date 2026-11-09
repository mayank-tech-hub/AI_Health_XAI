import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


@st.cache_resource
def get_engine():
    database_url = st.secrets.get("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is missing.")

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=0,
        pool_recycle=300,
        connect_args={"connect_timeout": 10},
    )


def test_connection():
    try:
        engine = get_engine()

        with engine.connect() as connection:
            version = connection.execute(
                text("SELECT version()")
            ).scalar()

        return {
            "success": True,
            "message": "Neon PostgreSQL connected successfully.",
            "version": version,
        }

    except (SQLAlchemyError, ValueError) as exc:
        return {
            "success": False,
            "message": str(exc),
        }
