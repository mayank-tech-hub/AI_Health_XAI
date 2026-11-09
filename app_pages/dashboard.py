
import streamlit as st

from components.header import render_header
from services.assessment_service import get_user_assessments


# ==================================================
# DASHBOARD
# ==================================================

render_header("Dashboard")

user = st.session_state.get("user", {})

if not user or not user.get("id"):
    st.error("User session not found. Please log in again.")
    st.stop()


# ==================================================
# CUSTOM DASHBOARD STYLES
# ==================================================

st.markdown("""
<style>
.dashboard-welcome {
    padding: 22px 24px;
    border-radius: 16px;
    border: 1px solid var(--brand-border);
    background: var(--secondary-background-color);
    margin-bottom: 20px;
}

.dashboard-welcome h2 {
    margin: 0 0 7px 0;
    font-size: 1.55rem;
}

.dashboard-welcome p {
    margin: 0;
    opacity: 0.72;
    font-size: 0.92rem;
}

.section-label {
    font-size: 1.15rem;
    font-weight: 700;
    margin: 22px 0 12px 0;
}

.quick-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid var(--brand-border);
    background: var(--secondary-background-color);
    min-height: 165px;
    margin-bottom: 10px;
}

.quick-icon {
    font-size: 27px;
    margin-bottom: 10px;
}

.quick-card h3 {
    margin: 0 0 8px 0;
    font-size: 1.1rem;
}

.quick-card p {
    opacity: 0.72;
    font-size: 0.88rem;
    line-height: 1.55;
    margin: 0;
}

.activity-title {
    font-weight: 700;
    font-size: 1rem;
}

.activity-meta {
    font-size: 0.82rem;
    opacity: 0.68;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# FETCH ASSESSMENTS
# ==================================================

try:
    with st.spinner("Loading your dashboard..."):
        assessments = get_user_assessments(
            user_id=int(user["id"]),
            limit=50,
        )

    if assessments is None:
        assessments = []

except Exception:
    assessments = []
    st.error("Unable to load dashboard statistics. Please refresh.")


# ==================================================
# WELCOME SECTION
# ==================================================

full_name = user.get("full_name", user.get("name", "User"))

first_name = full_name.strip().split()[0] if full_name.strip() else "User"

st.markdown(
    f"""
    <div class="dashboard-welcome">
        <h2>Welcome back, {first_name} 👋</h2>
        <p>
            Here's an overview of your health assessment activity.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# STATISTICS
# ==================================================

total_assessments = len(assessments)

heart_assessments = sum(
    1 for item in assessments
    if item.get("assessment_type") == "heart"
)

diabetes_assessments = sum(
    1 for item in assessments
    if item.get("assessment_type") == "diabetes"
)

st.markdown(
    '<div class="section-label">Assessment Overview</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.metric(
            label="Total Assessments",
            value=total_assessments,
            delta=None,
        )
        st.caption("All saved assessments")

with col2:
    with st.container(border=True):
        st.metric(
            label="Heart Assessments",
            value=heart_assessments,
            delta=None,
        )
        st.caption("Heart assessment records")

with col3:
    with st.container(border=True):
        st.metric(
            label="Diabetes Assessments",
            value=diabetes_assessments,
            delta=None,
        )
        st.caption("Diabetes assessment records")


# ==================================================
# QUICK ACCESS
# ==================================================

st.markdown(
    '<div class="section-label">Quick Access</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("""
        <div class="quick-card">
            <div class="quick-icon">❤️</div>
            <h3>Heart Risk Assessment</h3>
            <p>
                Explore heart risk assessment and
                explainable AI results.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Start Heart Assessment",
            use_container_width=True,
            type="primary",
            key="dashboard_heart",
        ):
            st.switch_page("app_pages/heart_assessment.py")

with col2:
    with st.container(border=True):
        st.markdown("""
        <div class="quick-card">
            <div class="quick-icon">🩸</div>
            <h3>Diabetes Risk Assessment</h3>
            <p>
                Explore diabetes risk assessment and
                model explanation results.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Start Diabetes Assessment",
            use_container_width=True,
            type="primary",
            key="dashboard_diabetes",
        ):
            st.switch_page("app_pages/diabetes_assessment.py")


# ==================================================
# RECENT ACTIVITY
# ==================================================

st.markdown(
    '<div class="section-label">Recent Assessment Activity</div>',
    unsafe_allow_html=True,
)

if assessments:

    recent_assessments = sorted(
        assessments,
        key=lambda item: str(item.get("created_at") or ""),
        reverse=True,
    )[:5]

    for item in recent_assessments:

        assessment_type = item.get("assessment_type", "unknown")

        if assessment_type == "heart":
            title = "Heart Assessment"
            icon = "❤️"
        elif assessment_type == "diabetes":
            title = "Diabetes Assessment"
            icon = "🩸"
        else:
            title = "Assessment"
            icon = "📋"

        prediction = item.get("prediction")

        if prediction in (0, 1):
            prediction_text = f"Class {prediction}"
        else:
            prediction_text = "Not available"

        created_at = item.get("created_at")

        if created_at:
            try:
                date_text = created_at.strftime(
                    "%d %b %Y, %I:%M %p"
                )
            except AttributeError:
                date_text = str(created_at)
        else:
            date_text = "Date unavailable"

        with st.container(border=True):

            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(
                    f"**{icon} {title}**"
                )
                st.caption(
                    f"Model: {item.get('model_key', 'Unknown')}"
                )
                st.caption(date_text)

            with col2:
                st.caption("Prediction")
                st.markdown(f"**{prediction_text}**")

                score = item.get("model_score")

                if score is not None:
                    st.caption(
                        f"Model score: {float(score) * 100:.2f}%"
                    )

    if st.button(
        "View All Assessment History",
        use_container_width=True,
        key="dashboard_history",
    ):
        st.switch_page("app_pages/history.py")

else:

    with st.container(border=True):
        st.markdown("### 📋 No assessments yet")
        st.write(
            "Your recent assessment activity will appear here."
        )

        st.caption(
            "Start with a Heart or Diabetes Assessment."
        )


# ==================================================
# REPORTS
# ==================================================

st.markdown(
    '<div class="section-label">Reports & Records</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("**📄 Your Reports**")
        st.caption(
            "Access your assessment reports and saved records."
        )

    with col2:
        if st.button(
            "Open Reports",
            use_container_width=True,
            key="dashboard_reports",
        ):
            st.switch_page("app_pages/reports.py")


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "AI-Based Early Health Risk Detection with Explainable AI (XAI)"
)
