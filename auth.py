"""
auth.py
Simple, self-contained login/signup for a Streamlit app.

Storage: SQLite (users.db, created automatically next to this file).
Passwords: salted + hashed with PBKDF2-HMAC-SHA256 (stdlib only, no extra installs).

Usage in your main app file (e.g. app.py):

    from auth import require_login

    require_login()   # <-- put this near the top, before your app content

    # ... rest of your existing Streamlit app code goes here ...

That's it. require_login() will block execution (via st.stop()) and show a
login/signup screen until the user is authenticated. Once logged in, it also
adds a "Logged in as ..." + "Log out" control in the sidebar.
"""

import streamlit as st
import sqlite3
import hashlib
import hmac
import os
import re
import secrets

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")


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

        .pulse-line {
            width: 100%;
            height: auto;
            overflow: visible;
        }

        .pulse-line path {
            stroke-dasharray: 900;
            stroke-dashoffset: 900;
            animation: draw-pulse 1.8s ease-out forwards;
        }

        @keyframes draw-pulse {
            to { stroke-dashoffset: 0; }
        }

        @media (prefers-reduced-motion: reduce) {
            .pulse-line path { animation: none; stroke-dashoffset: 0; }
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

        .stTextInput input {
            border-radius: 6px;
            border: 1px solid var(--auth-border);
            font-family: 'IBM Plex Sans', sans-serif;
        }

        .stTextInput input:focus {
            border-color: var(--auth-primary);
            box-shadow: 0 0 0 1px var(--auth-primary);
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

        [data-testid="stSidebar"] {
            background: var(--auth-primary-dark);
        }

        [data-testid="stSidebar"] * {
            color: #EAF2F1 !important;
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
                <svg class="pulse-line" viewBox="0 0 320 60" xmlns="http://www.w3.org/2000/svg">
                    <path d="M0 30 H90 L102 10 L114 50 L126 30 L138 30 L150 6 L162 54 L174 30 L320 30"
                          fill="none" stroke="#E4573D" stroke-width="2.5"
                          stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <h1>Know your heart,<br/>early.</h1>
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
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )
    return conn


def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return dk.hex()


def _create_user(username: str, email: str, password: str) -> tuple[bool, str]:
    conn = _get_connection()
    try:
        salt = secrets.token_bytes(16)
        password_hash = _hash_password(password, salt)
        conn.execute(
            "INSERT INTO users (username, email, salt, password_hash) VALUES (?, ?, ?, ?)",
            (username, email, salt.hex(), password_hash),
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."
    finally:
        conn.close()


def _verify_user(username: str, password: str) -> bool:
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT salt, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            return False
        salt_hex, stored_hash = row
        salt = bytes.fromhex(salt_hex)
        candidate_hash = _hash_password(password, salt)
        return hmac.compare_digest(candidate_hash, stored_hash)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _valid_email(email: str) -> bool:
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email) is not None


def _valid_username(username: str) -> bool:
    return re.match(r"^[A-Za-z0-9_]{3,20}$", username) is not None


def _password_strength_ok(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
        return False, "Password must include at least one letter and one number."
    return True, ""


# ---------------------------------------------------------------------------
# UI: Login form
# ---------------------------------------------------------------------------

def _render_login_form():
    st.markdown('<div class="auth-card-heading">Welcome back</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-card-subheading">Log in to run a new prediction.</div>', unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In", use_container_width=True)

    if submitted:
        if not username or not password:
            st.error("Please enter both username and password.")
        elif _verify_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password.")


# ---------------------------------------------------------------------------
# UI: Signup form
# ---------------------------------------------------------------------------

def _render_signup_form():
    st.markdown('<div class="auth-card-heading">Create your account</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-card-subheading">Takes under a minute.</div>', unsafe_allow_html=True)

    with st.form("signup_form", clear_on_submit=False):
        username = st.text_input("Choose a username")
        email = st.text_input("Email address")
        password = st.text_input("Choose a password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Sign Up", use_container_width=True)

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

        success, message = _create_user(username, email, password)
        if success:
            st.success(f"{message} You can now log in.")
        else:
            st.error(message)


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
