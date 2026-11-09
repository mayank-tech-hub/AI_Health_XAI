
import streamlit as st

from components.header import render_header
from services.assessment_service import get_user_assessments
from services.auth_service import (
    update_user_profile,
    change_user_password,
)


def show_profile():

    render_header("My Profile")

    user = st.session_state.get("user", {})

    if not user:
        st.error("User information not found. Please log in again.")
        return

    user_id = user.get("id")
    full_name = user.get("full_name", "")
    email = user.get("email", "")

    # ------------------------------------------
    # ACCOUNT INFORMATION
    # ------------------------------------------

    st.subheader("Account Information")

    with st.container(border=True):

        col1, col2 = st.columns(2)

        with col1:
            st.text_input(
                "Full Name",
                value=full_name,
                disabled=True,
                key="profile_display_name",
            )

        with col2:
            st.text_input(
                "Email Address",
                value=email,
                disabled=True,
                key="profile_display_email",
            )

        st.text_input(
            "User ID",
            value=str(user_id),
            disabled=True,
            key="profile_display_id",
        )

        st.success("Your account is active.")

    # ------------------------------------------
    # EDIT PROFILE
    # ------------------------------------------

    st.divider()
    st.subheader("Edit Profile")

    with st.form("edit_profile_form"):

        new_name = st.text_input(
            "Full Name",
            value=full_name,
            max_chars=100,
        )

        update_profile = st.form_submit_button(
            "Save Profile Changes",
            type="primary",
            use_container_width=True,
        )

    if update_profile:

        if new_name.strip() == full_name:
            st.info("No changes detected.")

        else:
            result = update_user_profile(
                user_id,
                new_name,
            )

            if result["success"]:

                updated_user = dict(user)
                updated_user["full_name"] = result["full_name"]

                st.session_state["user"] = updated_user

                st.success(result["message"])
                st.rerun()

            else:
                st.error(result["message"])

    # ------------------------------------------
    # CHANGE PASSWORD
    # ------------------------------------------

    st.divider()
    st.subheader("Change Password")

    st.caption(
        "Your new password must contain at least 12 characters."
    )

    with st.form("change_password_form"):

        current_password = st.text_input(
            "Current Password",
            type="password",
        )

        new_password = st.text_input(
            "New Password",
            type="password",
        )

        confirm_password = st.text_input(
            "Confirm New Password",
            type="password",
        )

        change_password = st.form_submit_button(
            "Update Password",
            type="primary",
            use_container_width=True,
        )

    if change_password:

        if not current_password or not new_password or not confirm_password:
            st.error("Please complete all password fields.")

        elif new_password != confirm_password:
            st.error("New password and confirmation do not match.")

        else:
            result = change_user_password(
                user_id,
                current_password,
                new_password,
            )

            if result["success"]:
                st.success(result["message"])

            else:
                st.error(result["message"])

    # ------------------------------------------
    # ASSESSMENT STATISTICS
    # ------------------------------------------

    st.divider()
    st.subheader("My Assessment Statistics")

    records = get_user_assessments(user_id, limit=100)

    heart_count = sum(
        1 for record in records
        if record.get("assessment_type") == "heart"
    )

    diabetes_count = sum(
        1 for record in records
        if record.get("assessment_type") == "diabetes"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Assessments", len(records))
    col2.metric("Heart Assessments", heart_count)
    col3.metric("Diabetes Assessments", diabetes_count)

    # ------------------------------------------
    # ACCOUNT DETAILS
    # ------------------------------------------

    st.divider()
    st.subheader("Account Details")

    st.write("**Account Status:** Active")
    st.write("**Authentication:** Argon2 Password Hashing")
    st.write("**Data Storage:** Neon PostgreSQL")

    # ------------------------------------------
    # PRIVACY
    # ------------------------------------------

    st.divider()
    st.subheader("Privacy & Security")

    st.info(
        "Your assessment records are associated with your account. "
        "Health assessment data is stored only when you explicitly "
        "provide consent."
    )


show_profile()
