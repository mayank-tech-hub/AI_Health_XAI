
import streamlit as st
from pathlib import Path

from services.auth_service import (
    register_user,
    login_user,
    verify_email_otp,
    resend_verification_otp,
)

LOGO_PATH = Path("assets/images/ai_health_xai_logo.png")


def show_auth_page():

    st.markdown("""
    <style>
    .block-container {
        width: 100%;
        max-width: 1250px;
        padding: 1.5rem 2rem 1rem;
    }

    div[data-testid="stHorizontalBlock"] {
        align-items: flex-start !important;
        gap: 2rem !important;
    }

    div[data-testid="column"] {
        min-width: 0 !important;
    }

    div[data-testid="stImage"] {
        width: 100%;
    }

    div[data-testid="stImage"] img {
        width: 100%;
        height: auto;
        max-width: 100%;
        object-fit: contain;
        border-radius: 12px;
    }

    .brand-features {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
    }

    .feature-badge {
        flex: 1 1 100px;
        min-width: 0;
        text-align: center;
        padding: 10px 5px;
        border: 1px solid var(--primary-color);
        border-radius: 9px;
        background: var(--secondary-background-color);
        color: var(--text-color);
        font-size: 12px;
        font-weight: 600;
    }

    .welcome-title {
        font-size: clamp(26px, 3vw, 36px);
        font-weight: 750;
        color: var(--text-color);
        margin: 0 0 5px 0;
    }

    .welcome-subtitle {
        color: var(--text-color);
        opacity: 0.75;
        font-size: 15px;
        margin-bottom: 20px;
    }

    div[data-testid="stForm"] {
        width: 100%;
        box-sizing: border-box;
        border: 1px solid var(--primary-color);
        border-radius: 14px;
        padding: 20px;
        background: var(--secondary-background-color);
    }

    div[data-testid="stTextInput"] input {
        border-radius: 9px;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: var(--primary-color);
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 700;
        min-height: 45px;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        filter: brightness(1.08);
        border: 1px solid var(--text-color);
    }

    @media screen and (max-width: 900px) {
        .block-container {
            padding: 1.2rem 1rem;
        }

        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 1.2rem !important;
        }

        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }

        .welcome-title {
            font-size: 28px;
        }
    }

    @media screen and (max-width: 480px) {
        .block-container {
            padding: 0.8rem 0.65rem;
        }

        div[data-testid="stForm"] {
            padding: 12px;
        }

        .brand-features {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 7px;
            margin-top: 8px;
        }

        .feature-badge {
            padding: 9px 3px;
            font-size: 11px;
        }

        .welcome-title {
            font-size: 24px;
        }

        .welcome-subtitle {
            font-size: 13px;
            margin-bottom: 12px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    left, right = st.columns(
        [0.95, 1.05],
        gap="large",
        vertical_alignment="top",
    )

    # ------------------------------------------
    # LEFT: BRANDING
    # ------------------------------------------

    with left:

        if LOGO_PATH.exists():
            st.image(
                str(LOGO_PATH),
                use_container_width=True,
            )
        else:
            st.error("Logo file not found.")

        st.markdown("""
        <div class="brand-features">
            <div class="feature-badge">❤️ Heart Risk</div>
            <div class="feature-badge">💧 Diabetes Risk</div>
            <div class="feature-badge">🧠 Explainable AI</div>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------
    # RIGHT: AUTHENTICATION
    # ------------------------------------------

    with right:

        st.markdown(
            '<div class="welcome-title">Welcome Back</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="welcome-subtitle">'
            'Login to your account or create a new account.'
            '</div>',
            unsafe_allow_html=True,
        )

        login_tab, register_tab = st.tabs(["Login", "Register"])

        # --------------------------------------
        # LOGIN
        # --------------------------------------

        with login_tab:

            with st.form("login_form"):

                email = st.text_input(
                    "Email address",
                    placeholder="Enter your email",
                    key="login_email",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    key="login_password",
                )

                submitted = st.form_submit_button(
                    "Login",
                    use_container_width=True,
                )

                if submitted:

                    result = login_user(email, password)

                    if result["success"]:

                        st.session_state["authenticated"] = True
                        st.session_state["user"] = result["user"]

                        st.rerun()

                    elif result.get("requires_verification"):

                        st.session_state["pending_verification_email"] = (
                            email.strip().lower()
                        )

                        st.session_state["register_email"] = (
                            email.strip().lower()
                        )

                        st.warning(
                            "Your email is not verified. "
                            "Open the Register tab to verify it."
                        )

                    else:
                        st.error(result["message"])

        # --------------------------------------
        # REGISTER AND OTP VERIFICATION
        # --------------------------------------

        with register_tab:

            pending_email = st.session_state.get(
                "pending_verification_email"
            )

            if pending_email:

                st.subheader("Verify Your Email")

                st.info(
                    f"Verification email: {pending_email}"
                )

                st.caption(
                    "Enter the 6-digit code sent to your email. "
                    "The OTP expires in 10 minutes. "
                    "The temporary registration expires in 30 minutes."
                )

                # Display the previous registration/email status.
                notice = st.session_state.pop(
                    "auth_notice",
                    None,
                )

                notice_type = st.session_state.pop(
                    "auth_notice_type",
                    "info",
                )

                if notice:
                    if notice_type == "success":
                        st.success(notice)
                    elif notice_type == "warning":
                        st.warning(notice)
                    else:
                        st.info(notice)

                with st.form("verify_otp_form"):

                    otp = st.text_input(
                        "6-digit verification code",
                        max_chars=6,
                        placeholder="Enter OTP",
                    )

                    verify_submitted = st.form_submit_button(
                        "Verify Email",
                        use_container_width=True,
                    )

                    if verify_submitted:

                        result = verify_email_otp(
                            pending_email,
                            otp,
                        )

                        if result["success"]:

                            st.session_state.pop(
                                "pending_verification_email",
                                None,
                            )

                            st.session_state["auth_notice"] = (
                                result["message"]
                            )
                            st.session_state["auth_notice_type"] = "success"

                            st.rerun()

                        else:
                            st.error(result["message"])

                if st.button(
                    "Resend Verification Code",
                    use_container_width=True,
                    key="resend_otp_button",
                ):

                    result = resend_verification_otp(
                        pending_email
                    )

                    if result["success"]:
                        st.success(result["message"])
                    else:
                        st.error(result["message"])

                if st.button(
                    "Cancel Verification",
                    use_container_width=True,
                    key="cancel_verification_button",
                ):

                    st.session_state.pop(
                        "pending_verification_email",
                        None,
                    )

                    st.rerun()

            else:

                with st.form("register_form"):

                    full_name = st.text_input(
                        "Full name",
                        placeholder="Enter your full name",
                        key="register_name",
                    )

                    new_email = st.text_input(
                        "Email address",
                        placeholder="Enter your email",
                        key="register_email",
                    )

                    new_password = st.text_input(
                        "Create password",
                        type="password",
                        placeholder="Minimum 12 characters",
                        key="register_password",
                    )

                    confirm_password = st.text_input(
                        "Confirm password",
                        type="password",
                        placeholder="Re-enter your password",
                        key="confirm_password",
                    )

                    submitted = st.form_submit_button(
                        "Create Account",
                        use_container_width=True,
                    )

                    if submitted:

                        if new_password != confirm_password:

                            st.error("Passwords do not match.")

                        else:

                            result = register_user(
                                full_name,
                                new_email,
                                new_password,
                            )

                            if (
                                result["success"]
                                or result.get("requires_verification")
                            ):

                                normalized_email = new_email.strip().lower()

                                st.session_state[
                                    "pending_verification_email"
                                ] = normalized_email

                                st.session_state["auth_notice"] = (
                                    result["message"]
                                )

                                st.session_state["auth_notice_type"] = (
                                    "success"
                                    if result.get("email_sent")
                                    else "warning"
                                )

                                st.rerun()

                            else:
                                st.error(result["message"])
