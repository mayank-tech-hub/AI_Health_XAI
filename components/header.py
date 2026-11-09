
import streamlit as st

PROJECT_NAME = (
    "AI-Based Early Health Risk Detection "
    "with Explainable AI (XAI)"
)


def render_header(section):

    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-eyebrow">
                AI HEALTH XAI &nbsp; / &nbsp; HEALTH ANALYTICS
            </div>
            <h1>{section}</h1>
            <div class="page-description">
                {PROJECT_NAME}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
