
import json
from datetime import datetime, timezone

import streamlit as st

from components.header import render_header
from services.assessment_service import get_user_assessments


# --------------------------------------------------
# DATE FORMAT
# --------------------------------------------------

def format_date(value):

    if value is None:
        return "Date unavailable"

    if isinstance(value, str):

        try:
            value = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except ValueError:
            return value

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone().strftime(
        "%d %b %Y, %I:%M %p"
    )


# --------------------------------------------------
# HISTORY PAGE
# --------------------------------------------------

def show_history():

    render_header("Assessment History")

    user = st.session_state.get("user")

    if not user or not user.get("id"):

        st.error("Please login to view your assessment history.")
        return

    st.subheader("Your Previous Assessments")

    st.caption(
        "View, filter and review your saved assessment records."
    )

    # ----------------------------------------------
    # FETCH COMPLETE HISTORY
    # ----------------------------------------------

    try:

        with st.spinner("Loading assessment history..."):

            all_records = get_user_assessments(
                user_id=int(user["id"]),
                limit=100,
            )

    except Exception:

        st.error(
            "Unable to load assessment history. Please try again."
        )
        return

    if all_records is None:
        st.error("Assessment history could not be retrieved.")
        return

    # ----------------------------------------------
    # GLOBAL STATISTICS
    # ----------------------------------------------

    total = len(all_records)

    heart_count = sum(
        1 for item in all_records
        if item.get("assessment_type") == "heart"
    )

    diabetes_count = sum(
        1 for item in all_records
        if item.get("assessment_type") == "diabetes"
    )

    st.markdown("### Assessment Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Assessments", total)

    with col2:
        st.metric("Heart Assessments", heart_count)

    with col3:
        st.metric("Diabetes Assessments", diabetes_count)

    st.divider()

    # ----------------------------------------------
    # FILTER AND SORT
    # ----------------------------------------------

    st.markdown("### Assessment Records")

    filter_col, sort_col = st.columns(2)

    with filter_col:

        filter_option = st.selectbox(
            "Filter by assessment type",
            ["All Assessments", "Heart", "Diabetes"],
            key="history_filter",
        )

    with sort_col:

        sort_option = st.selectbox(
            "Sort by date",
            ["Newest First", "Oldest First"],
            key="history_sort",
        )

    assessment_type = {
        "All Assessments": None,
        "Heart": "heart",
        "Diabetes": "diabetes",
    }[filter_option]

    records = [
        item for item in all_records
        if (
            assessment_type is None
            or item.get("assessment_type") == assessment_type
        )
    ]

    records.sort(
        key=lambda item: (
            item.get("created_at") is not None,
            str(item.get("created_at") or "")
        ),
        reverse=(sort_option == "Newest First"),
    )

    st.caption(
        f"Showing {len(records)} of {total} saved assessments."
    )

    # ----------------------------------------------
    # EMPTY STATE
    # ----------------------------------------------

    if not records:

        st.info(
            "No assessments found for the selected filter."
        )

        st.caption(
            "Complete a Heart or Diabetes Assessment "
            "to create a history record."
        )

        st.divider()

        st.caption(
            "Model scores are model outputs and should not "
            "be interpreted as calibrated medical probabilities."
        )

        return

    # ----------------------------------------------
    # DISPLAY ASSESSMENT CARDS
    # ----------------------------------------------

    for record in records:

        assessment_type = record.get(
            "assessment_type", "unknown"
        )

        if assessment_type == "heart":
            assessment_name = "Heart Assessment"
        elif assessment_type == "diabetes":
            assessment_name = "Diabetes Assessment"
        else:
            assessment_name = "Assessment"

        prediction = record.get("prediction")
        score = record.get("model_score")

        with st.container(border=True):

            # Header
            header_col, date_col = st.columns([3, 1])

            with header_col:

                st.markdown(f"### {assessment_name}")

                st.caption(
                    f"Model: {record.get('model_key', 'Unknown')}"
                )

                st.caption(
                    f"Assessment ID: {record.get('id', 'N/A')}"
                )

            with date_col:

                st.caption(
                    format_date(record.get("created_at"))
                )

            st.divider()

            # Result summary
            result_col, score_col = st.columns(2)

            with result_col:

                st.markdown("**Prediction**")

                if prediction == 1:
                    st.warning("Class 1")
                elif prediction == 0:
                    st.success("Class 0")
                else:
                    st.write("Not available")

            with score_col:

                st.markdown("**Class 1 Model Score**")

                if score is not None:
                    st.metric(
                        "Model output",
                        f"{float(score) * 100:.2f}%"
                    )
                else:
                    st.write("Not available")

            # Details
            with st.expander("View Assessment Details"):

                st.markdown("#### Input Data")

                input_data = record.get("input_data")

                if input_data:

                    if isinstance(input_data, str):
                        try:
                            input_data = json.loads(input_data)
                        except json.JSONDecodeError:
                            st.text(input_data)

                    if isinstance(input_data, dict):
                        st.dataframe(
                            [
                                {
                                    "Feature": key,
                                    "Entered Value": value
                                }
                                for key, value in input_data.items()
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )

                else:

                    st.info("No input details available.")

                st.divider()

                st.markdown("#### Explainable AI (XAI)")

                explanation = record.get("explanation")

                if explanation:

                    if isinstance(explanation, str):
                        try:
                            explanation = json.loads(explanation)
                        except json.JSONDecodeError:
                            st.text(explanation)
                            explanation = None

                    if isinstance(explanation, dict):

                        st.markdown("**Top Feature Contributions**")

                        top_features = explanation.get(
                            "top_features", []
                        )

                        if top_features:

                            for item in top_features:

                                st.markdown(
                                    f"**{item.get('feature', 'Feature')}**: "
                                    f"{item.get('contribution', 0):+.5f}"
                                )

                                st.caption(
                                    item.get("direction", "")
                                )

                        else:

                            st.info(
                                "Feature contribution details unavailable."
                            )

                        if explanation.get("additivity_error") is not None:

                            st.caption(
                                "SHAP additivity error: "
                                f"{float(explanation['additivity_error']):.2e}"
                            )

                        with st.expander("View Complete XAI Data"):
                            st.json(explanation)

                else:

                    st.info("No XAI explanation available.")

    # ----------------------------------------------
    # FOOTER
    # ----------------------------------------------

    st.divider()

    st.caption(
        "Model scores are model outputs and should not be "
        "interpreted as calibrated medical probabilities."
    )


show_history()
