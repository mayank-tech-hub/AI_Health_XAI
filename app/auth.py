"""
auth.py
Self-contained login/signup for a Streamlit app, supporting two methods:
  - Email + Password
  - Mobile number + OTP

Storage: SQLite (users.db, created automatically next to this file).
Passwords: salted + hashed with PBKDF2-HMAC-SHA256 (stdlib only, no extra installs).

IMPORTANT — About OTP delivery:
Sending a real SMS requires a paid gateway (Twilio, MSG91, Fast2SMS, AWS SNS,
Firebase Phone Auth, etc.) and API credentials. That integration is NOT
included here. Instead, this module generates and verifies a real one-time
code, but displays it on-screen in a clearly-labeled "demo mode" banner so
you can test the full flow without a gateway.

To go live with real SMS delivery: implement the body of `_send_otp_sms()`
below with your provider's API call, and remove the on-screen OTP display.

Usage in your main app file (e.g. app.py):

    from auth import require_login

    require_login()   # <-- put this near the top, before your app content

    # ... rest of your existing Streamlit app code goes here ...
"""

import streamlit as st
import sqlite3
import hashlib
import hmac
import os
import re
import secrets
import time

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")

OTP_LENGTH = 6
OTP_VALIDITY_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Design: styling for the login/signup screen
# ---------------------------------------------------------------------------

def _inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --auth-bg: #F6F8F7;
            --auth-surface: #FFFFFF;
            --auth-primary: #0F4C4C;
            --auth-primary-dark: #0B2E33;
            --auth-accent: #E4573D;
            --auth-text: #1B2B2B;
            --auth-muted: #5B6D6D;
            --auth-border: #E1E8E7;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--auth-bg);
        }

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
            color: var(--auth-text);
        }

        .auth-hero {
            background: var(--auth-primary-dark);
            border-radius: 12px;
            padding: 48px 36px;
            height: 100%;
            min-height: 480px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            color: #EAF2F1;
        }

        .auth-hero h1 {
            font-family: 'Source Serif 4', serif;
            font-weight: 600;
            font-size: 2.1rem;
            line-height: 1.25;
            margin: 24px 0 12px 0;
            color: #FFFFFF;
        }

        .auth-hero p {
            font-size: 0.98rem;
            line-height: 1.6;
            color: #B9CCCA;
            max-width: 34ch;
        }

        .auth-hero .auth-tag {
            font-size: 0.85rem;
            color: #7FA8A4;
            margin-top: 32px;
        }

        .pulse-line-scroll {
            height: 56px;
            width: 100%;
            overflow: hidden;
            background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 160 60'><path d='M0 30 H60 L72 10 L84 50 L96 30 L108 30 L120 6 L132 54 L144 30 H160' fill='none' stroke='%23E4573D' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
            background-repeat: repeat-x;
            background-position: 0 0;
            background-size: 160px 56px;
            animation: scroll-pulse 2.6s linear infinite;
        }

        @keyframes scroll-pulse {
            to { background-position: -160px 0; }
        }

        @media (prefers-reduced-motion: reduce) {
            .pulse-line-scroll { animation: none; }
        }

        .auth-card-heading {
            font-family: 'Source Serif 4', serif;
            font-weight: 600;
            font-size: 1.5rem;
            color: var(--auth-text);
            margin-bottom: 2px;
        }

        .auth-card-subheading {
            color: var(--auth-muted);
            font-size: 0.92rem;
            margin-bottom: 20px;
        }

        [data-testid="stForm"] {
            background: var(--auth-surface);
            border: 1px solid var(--auth-border);
            border-radius: 10px;
            padding: 28px 28px 12px 28px;
        }

        /* Fix: widget labels must stay visible against the white card */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] * {
            color: var(--auth-text) !important;
            opacity: 1 !important;
        }

        .stTextInput input {
            border-radius: 6px;
            border: 1px solid var(--auth-border);
            font-family: 'IBM Plex Sans', sans-serif;
            color: var(--auth-text) !important;
        }

        .stTextInput input:focus {
            border-color: var(--auth-primary);
            box-shadow: 0 0 0 1px var(--auth-primary);
        }

        [data-testid="stTextInputRootElement"] button {
            background: transparent !important;
        }

        [data-testid="stFormSubmitButton"] button {
            background: var(--auth-accent);
            color: #FFFFFF;
            border: none;
            border-radius: 6px;
            font-weight: 500;
            padding: 0.55rem 1rem;
        }

        [data-testid="stFormSubmitButton"] button:hover {
            background: #C8462F;
            color: #FFFFFF;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1px solid var(--auth-border);
        }

        .stTabs [data-baseweb="tab"] {
            font-family: 'IBM Plex Sans', sans-serif;
            font-weight: 500;
            color: var(--auth-muted);
        }

        .stTabs [aria-selected="true"] {
            color: var(--auth-primary) !important;
            border-bottom-color: var(--auth-primary) !important;
        }

        /* Method switch (Email/Password vs Mobile/OTP) */
        .stRadio [role="radiogroup"] {
            gap: 4px;
        }

        .stRadio label p {
            color: var(--auth-text) !important;
            font-size: 0.92rem;
        }

        [data-testid="stSidebar"] {
            background: var(--auth-primary-dark);
        }

        [data-testid="stSidebar"] * {
            color: #EAF2F1 !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            font-size: 1.05rem;
        }

        [data-testid="stSidebar"] button {
            background: rgba(255, 255, 255, 0.10) !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 500;
            padding: 0.6rem 1.4rem !important;
        }

        [data-testid="stSidebar"] button:hover {
            background: rgba(255, 255, 255, 0.16) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero_panel():
    st.markdown(
        """
        <div class="auth-hero">
            <div>
                <div class="pulse-line-scroll"></div>
                <h1>Know Your Health<br/>Early.</h1>
                <p>
                    A machine-learning model estimates heart disease risk from
                    your clinical measurements, and explains exactly which
                    factors shaped that estimate.
                </p>
            </div>
            <div class="auth-tag">AI Health XAI — an educational tool, not a diagnosis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            mobile TEXT UNIQUE,
            salt TEXT,
            password_hash TEXT,
            auth_method TEXT NOT NULL DEFAULT 'password'
        )
        """
    )
    # Migration safety net: if an older users.db exists without these
    # columns, add them so upgrading doesn't break existing installs.
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
    if "mobile" not in existing_columns:
        conn.execute("ALTER TABLE users ADD COLUMN mobile TEXT")
    if "auth_method" not in existing_columns:
        conn.execute("ALTER TABLE users ADD COLUMN auth_method TEXT DEFAULT 'password'")
    conn.commit()
    return conn


def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return dk.hex()


def _create_user_email(username: str, email: str, password: str) -> tuple[bool, str]:
    conn = _get_connection()
    try:
        salt = secrets.token_bytes(16)
        password_hash = _hash_password(password, salt)
        conn.execute(
            """
            INSERT INTO users (username, email, mobile, salt, password_hash, auth_method)
            VALUES (?, ?, NULL, ?, ?, 'password')
            """,
            (username, email, salt.hex(), password_hash),
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."
    finally:
        conn.close()


def _create_user_mobile(username: str, mobile: str) -> tuple[bool, str]:
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO users (username, email, mobile, salt, password_hash, auth_method)
            VALUES (?, NULL, ?, NULL, NULL, 'otp')
            """,
            (username, mobile),
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "That username or mobile number is already registered."
    finally:
        conn.close()


def _verify_user_password(username: str, password: str) -> bool:
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT salt, password_hash FROM users WHERE username = ? AND auth_method = 'password'",
            (username,),
        ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            return False
        salt_hex, stored_hash = row
        salt = bytes.fromhex(salt_hex)
        candidate_hash = _hash_password(password, salt)
        return hmac.compare_digest(candidate_hash, stored_hash)
    finally:
        conn.close()


def _find_username_by_mobile(mobile: str) -> str | None:
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT username FROM users WHERE mobile = ? AND auth_method = 'otp'",
            (mobile,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _mobile_registered(mobile: str) -> bool:
    return _find_username_by_mobile(mobile) is not None


# ---------------------------------------------------------------------------
# OTP helpers
# ---------------------------------------------------------------------------

def _generate_otp() -> str:
    return "".join(str(secrets.randbelow(10)) for _ in range(OTP_LENGTH))


def _send_otp_sms(mobile: str, otp: str):
    """
    Stub for real SMS delivery. Replace the body of this function with a
    call to your SMS provider's API (Twilio, MSG91, Fast2SMS, AWS SNS, etc.)
    to actually text the OTP to `mobile`. Currently a no-op — the calling
    code displays the OTP on-screen instead, for local testing.
    """
    return True


def _start_otp_challenge(prefix: str, mobile: str):
    otp = _generate_otp()
    st.session_state[f"{prefix}_otp_code"] = otp
    st.session_state[f"{prefix}_otp_expiry"] = time.time() + OTP_VALIDITY_SECONDS
    st.session_state[f"{prefix}_otp_mobile"] = mobile
    st.session_state[f"{prefix}_otp_stage"] = "verify"
    _send_otp_sms(mobile, otp)


def _otp_is_valid(prefix: str, entered_otp: str) -> bool:
    stored_otp = st.session_state.get(f"{prefix}_otp_code")
    expiry = st.session_state.get(f"{prefix}_otp_expiry", 0)
    if stored_otp is None:
        return False
    if time.time() > expiry:
        return False
    return hmac.compare_digest(entered_otp, stored_otp)


def _reset_otp_challenge(prefix: str):
    for key in ("otp_code", "otp_expiry", "otp_mobile", "otp_stage"):
        st.session_state.pop(f"{prefix}_{key}", None)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _valid_email(email: str) -> bool:
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email) is not None


def _valid_username(username: str) -> bool:
    return re.match(r"^[A-Za-z0-9_]{3,20}$", username) is not None


def _valid_mobile(mobile: str) -> bool:
    return re.match(r"^\+?[0-9]{10,15}$", mobile) is not None


def _password_strength_ok(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
        return False, "Password must include at least one letter and one number."
    return True, ""


# ---------------------------------------------------------------------------
# UI: Login — Email & Password
# ---------------------------------------------------------------------------

def _render_login_email_form():
    with st.form("login_email_form", clear_on_submit=False):
        username = st.text_input("Username", key="login_email_username")
        password = st.text_input("Password", type="password", key="login_email_password")
        submitted = st.form_submit_button("Log in", use_container_width=True)

    if submitted:
        if not username or not password:
            st.error("Please enter both username and password.")
        elif _verify_user_password(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password.")


# ---------------------------------------------------------------------------
# UI: Login — Mobile & OTP
# ---------------------------------------------------------------------------

def _render_login_mobile_form():
    prefix = "login"
    stage = st.session_state.get(f"{prefix}_otp_stage", "request")

    if stage == "request":
        with st.form("login_mobile_request_form", clear_on_submit=False):
            mobile = st.text_input("Mobile number", placeholder="+91XXXXXXXXXX", key="login_mobile_number_input")
            submitted = st.form_submit_button("Send OTP", use_container_width=True)

        if submitted:
            if not _valid_mobile(mobile):
                st.error("Please enter a valid mobile number (10-15 digits, optional +country code).")
            elif not _mobile_registered(mobile):
                st.error("No account found with this mobile number. Please sign up first.")
            else:
                _start_otp_challenge(prefix, mobile)
                st.rerun()

    else:
        mobile = st.session_state.get(f"{prefix}_otp_mobile", "")
        st.info(
            f"Demo mode: OTP for **{mobile}** is **{st.session_state.get(f'{prefix}_otp_code')}** "
            f"(valid for 5 minutes). A production build would text this instead of showing it here."
        )

        with st.form("login_mobile_verify_form", clear_on_submit=False):
            otp_entered = st.text_input("Enter OTP", max_chars=OTP_LENGTH, key="login_otp_input")
            verify_submitted = st.form_submit_button("Verify & log in", use_container_width=True)

        if verify_submitted:
            if _otp_is_valid(prefix, otp_entered):
                username = _find_username_by_mobile(mobile)
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                _reset_otp_challenge(prefix)
                st.rerun()
            else:
                st.error("Incorrect or expired OTP. Please try again or resend.")

        col_resend, col_change = st.columns(2)
        with col_resend:
            if st.button("Resend OTP", key="login_resend_otp", use_container_width=True):
                _start_otp_challenge(prefix, mobile)
                st.rerun()
        with col_change:
            if st.button("Change number", key="login_change_number", use_container_width=True):
                _reset_otp_challenge(prefix)
                st.rerun()


# ---------------------------------------------------------------------------
# UI: Login — wrapper
# ---------------------------------------------------------------------------

def _render_login_form():
    st.markdown('<div class="auth-card-heading">Welcome back</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-card-subheading">Log in to run a new prediction.</div>', unsafe_allow_html=True)

    method = st.radio(
        "Login method",
        options=["Email & Password", "Mobile & OTP"],
        key="login_method",
        horizontal=True,
        label_visibility="collapsed",
    )

    if method == "Email & Password":
        _render_login_email_form()
    else:
        _render_login_mobile_form()


# ---------------------------------------------------------------------------
# UI: Signup — Email & Password
# ---------------------------------------------------------------------------

def _render_signup_email_form():
    with st.form("signup_email_form", clear_on_submit=False):
        username = st.text_input("Choose a username", key="signup_email_username")
        email = st.text_input("Email address", key="signup_email_email")
        password = st.text_input("Choose a password", type="password", key="signup_email_password")
        confirm_password = st.text_input("Confirm password", type="password", key="signup_email_confirm")
        submitted = st.form_submit_button("Create account", use_container_width=True)

    if submitted:
        if not username or not email or not password or not confirm_password:
            st.error("Please fill in all fields.")
            return

        if not _valid_username(username):
            st.error("Username must be 3-20 characters: letters, numbers, underscores only.")
            return

        if not _valid_email(email):
            st.error("Please enter a valid email address.")
            return

        if password != confirm_password:
            st.error("Passwords do not match.")
            return

        strong_enough, message = _password_strength_ok(password)
        if not strong_enough:
            st.error(message)
            return

        success, message = _create_user_email(username, email, password)
        if success:
            st.success(f"{message} You can now log in.")
        else:
            st.error(message)


# ---------------------------------------------------------------------------
# UI: Signup — Mobile & OTP
# ---------------------------------------------------------------------------

def _render_signup_mobile_form():
    prefix = "signup"
    stage = st.session_state.get(f"{prefix}_otp_stage", "request")

    if stage == "request":
        with st.form("signup_mobile_request_form", clear_on_submit=False):
            username = st.text_input("Choose a username", key="signup_mobile_username_input")
            mobile = st.text_input("Mobile number", placeholder="+91XXXXXXXXXX", key="signup_mobile_number_input")
            submitted = st.form_submit_button("Send OTP", use_container_width=True)

        if submitted:
            if not username or not mobile:
                st.error("Please enter both a username and a mobile number.")
            elif not _valid_username(username):
                st.error("Username must be 3-20 characters: letters, numbers, underscores only.")
            elif not _valid_mobile(mobile):
                st.error("Please enter a valid mobile number (10-15 digits, optional +country code).")
            elif _mobile_registered(mobile):
                st.error("This mobile number is already registered. Try logging in instead.")
            else:
                st.session_state[f"{prefix}_pending_username"] = username
                _start_otp_challenge(prefix, mobile)
                st.rerun()

    else:
        mobile = st.session_state.get(f"{prefix}_otp_mobile", "")
        username = st.session_state.get(f"{prefix}_pending_username", "")
        st.info(
            f"Demo mode: OTP for **{mobile}** is **{st.session_state.get(f'{prefix}_otp_code')}** "
            f"(valid for 5 minutes). A production build would text this instead of showing it here."
        )

        with st.form("signup_mobile_verify_form", clear_on_submit=False):
            otp_entered = st.text_input("Enter OTP", max_chars=OTP_LENGTH, key="signup_otp_input")
            verify_submitted = st.form_submit_button("Verify & create account", use_container_width=True)

        if verify_submitted:
            if _otp_is_valid(prefix, otp_entered):
                success, message = _create_user_mobile(username, mobile)
                _reset_otp_challenge(prefix)
                st.session_state.pop(f"{prefix}_pending_username", None)
                if success:
                    st.success(f"{message} You can now log in with your mobile number.")
                else:
                    st.error(message)
            else:
                st.error("Incorrect or expired OTP. Please try again or resend.")

        col_resend, col_change = st.columns(2)
        with col_resend:
            if st.button("Resend OTP", key="signup_resend_otp", use_container_width=True):
                _start_otp_challenge(prefix, mobile)
                st.rerun()
        with col_change:
            if st.button("Change number", key="signup_change_number", use_container_width=True):
                _reset_otp_challenge(prefix)
                st.session_state.pop(f"{prefix}_pending_username", None)
                st.rerun()


# ---------------------------------------------------------------------------
# UI: Signup — wrapper
# ---------------------------------------------------------------------------

def _render_signup_form():
    st.markdown('<div class="auth-card-heading">Create your account</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-card-subheading">Takes under a minute.</div>', unsafe_allow_html=True)

    method = st.radio(
        "Signup method",
        options=["Email & Password", "Mobile & OTP"],
        key="signup_method",
        horizontal=True,
        label_visibility="collapsed",
    )

    if method == "Email & Password":
        _render_signup_email_form()
    else:
        _render_signup_mobile_form()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def require_login():
    """
    Call this at the top of your Streamlit app.
    Blocks (via st.stop()) until the user is authenticated.
    Shows a sidebar logout control once authenticated.
    """

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    _inject_styles()

    if st.session_state["authenticated"]:
        with st.sidebar:
            st.write(f"Logged in as **{st.session_state.get('username', '')}**")
            if st.button("Log out"):
                st.session_state["authenticated"] = False
                st.session_state.pop("username", None)
                st.rerun()
        return  # authenticated -> let the rest of the app run

    # Not authenticated -> show the split login/signup screen and stop here
    left, right = st.columns([0.45, 0.55], gap="large")

    with left:
        _render_hero_panel()

    with right:
        st.write("")  # small top spacer to align with hero panel
        tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

        with tab_login:
            _render_login_form()

        with tab_signup:
            _render_signup_form()

    st.stop()
