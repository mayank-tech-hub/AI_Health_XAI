
import streamlit as st
import json
import csv
import io

from components.header import render_header
from services.assessment_service import get_user_assessments


def show_settings():
    render_header("Settings")

    user = st.session_state.get("user", {})

    if not user:
        st.error("User information not found. Please log in again.")
        return

    user_id = user.get("id")
    full_name = user.get("full_name", "Not available")
    email = user.get("email", "Not available")

    st.subheader("1. Account Information")

    col1, col2 = st.columns(2)

    with col1:
        st.text_input("Full Name", value=str(full_name), disabled=True)

    with col2:
        st.text_input("Email Address", value=str(email), disabled=True)

    st.text_input("User ID", value=str(user_id), disabled=True)

    st.caption("Account details are displayed as read-only information.")

    st.divider()

    st.subheader("2. Privacy & Data")

    st.markdown(
        """
        - Assessment records are associated with your account.
        - Health assessment data is saved only when you explicitly provide consent.
        - You can review your saved records from Assessment History.
        - Reports generated as PDFs are downloaded to your device.
        """
    )

    st.info(
        "Your assessment results are intended for educational "
        "and health-risk screening purposes, not medical diagnosis."
    )

    st.divider()

    st.subheader("3. Export Your Data")

    st.write(
        "Download a copy of the assessment records saved under your account."
    )

    if st.button("Load My Assessment Data", type="primary"):
        records = get_user_assessments(user_id, limit=100)
        st.session_state["settings_export_records"] = records
        st.success(f"{len(records)} records loaded.")

    records = st.session_state.get("settings_export_records", [])

    if records:
        json_data = json.dumps(
            records,
            indent=4,
            default=str
        )

        st.download_button(
            label="Download Data as JSON",
            data=json_data,
            file_name="my_assessment_data.json",
            mime="application/json"
        )

        output = io.StringIO()
        if records:
            fieldnames = list(records[0].keys())
            writer = csv.DictWriter(
                output,
                fieldnames=fieldnames
            )
            writer.writeheader()

            for record in records:
                writer.writerow({
                    key: json.dumps(value, default=str)
                    if isinstance(value, (dict, list))
                    else str(value) if value is not None else ""
                    for key, value in record.items()
                })

        st.download_button(
            label="Download Data as CSV",
            data=output.getvalue(),
            file_name="my_assessment_data.csv",
            mime="text/csv"
        )

    st.divider()

    st.subheader("4. Security")

    st.write("Password changes are available from My Profile.")
    st.write("Account deletion is currently unavailable.")


show_settings()
