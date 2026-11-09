
import streamlit as st
from pathlib import Path

from config.settings import PROJECT_NAME, DISCLAIMER
from app_pages.auth import show_auth_page


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title=PROJECT_NAME,
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# RESPONSIVE CSS
# --------------------------------------------------

css_path = Path("assets/styles/responsive.css")

if css_path.exists():
   css_content = css_path.read_text(encoding="utf-8")

st.markdown(
        f"<style>{css_content}</style>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

if not st.session_state.get("authenticated", False):

    show_auth_page()
    st.stop()


user = st.session_state.get("user")

if not user:

    st.session_state.clear()
    st.rerun()


# --------------------------------------------------
# BRANDING
# --------------------------------------------------

st.logo(
    "assets/images/ai_health_xai_logo.png",
    size="large",
)


# --------------------------------------------------
# NAVIGATION PAGES
# --------------------------------------------------

dashboard = st.Page(
    "app_pages/dashboard.py",
    title="Dashboard",
    icon="🏠",
    default=True,
)

heart = st.Page(
    "app_pages/heart_assessment.py",
    title="Heart Disease Assessment",
    icon="❤️",
)

diabetes = st.Page(
    "app_pages/diabetes_assessment.py",
    title="Diabetes Assessment",
    icon="🩺",
)

history = st.Page(
    "app_pages/history.py",
    title="Assessment History",
    icon="📋",
)

reports = st.Page(
    "app_pages/reports.py",
    title="Reports",
    icon="📄",
)

profile = st.Page(
    "app_pages/profile.py",
    title="My Profile",
    icon="👤",
)

settings = st.Page(
    "app_pages/settings.py",
    title="Settings",
    icon="⚙️",
)

logout = st.Page(
    "app_pages/logout.py",
    title="Logout",
    icon="🚪",
)


# --------------------------------------------------
# NAVIGATION GROUPS
# --------------------------------------------------

navigation = st.navigation(
    {
        "WORKSPACE": [
            dashboard,
            heart,
            diabetes,
            history,
            reports,
        ],
        "ACCOUNT": [
            profile,
            settings,
            logout,
        ],
    },
    position="sidebar",
)


# --------------------------------------------------
# SIDEBAR ACCOUNT INFORMATION
# --------------------------------------------------

with st.sidebar:

    st.divider()

    with st.container(border=True):

        st.caption("LOGGED IN AS")

        st.markdown(
            f"**{user.get('full_name', 'User')}**"
        )

        st.caption(
            user.get("email", "")
        )

        st.caption("Account Status: Active")

    st.divider()

    st.caption("AI HEALTH XAI")

    st.caption(
        "AI-Based Early Health Risk Detection "
        "with Explainable AI (XAI)"
    )

    st.divider()

    st.caption(DISCLAIMER)


# --------------------------------------------------
# RUN NAVIGATION
# --------------------------------------------------

navigation.run()
