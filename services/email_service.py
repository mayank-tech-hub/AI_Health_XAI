
import smtplib
import streamlit as st

from email.message import EmailMessage


def send_otp_email(to_email: str, otp: str) -> tuple[bool, str]:
    """
    Send a verification OTP using Gmail SMTP.
    """

    try:
        smtp_email = st.secrets["SMTP_EMAIL"]
        smtp_password = st.secrets["SMTP_PASSWORD"]
        smtp_server = st.secrets.get(
            "SMTP_SERVER", "smtp.gmail.com"
        )
        smtp_port = int(
            st.secrets.get("SMTP_PORT", 587)
        )

        if not smtp_email or not smtp_password:
            return False, "Email configuration is incomplete."

        message = EmailMessage()

        message["Subject"] = "Verify Your Email - AI Health XAI"
        message["From"] = smtp_email
        message["To"] = to_email

        message.set_content(
            f"""
Hello,

Your email verification code for AI-Based Early Health Risk
Detection with Explainable AI (XAI) is:

{otp}

This code is valid for 10 minutes.

If you did not request this verification, please ignore this email.

Regards,
AI Health XAI Team
"""
        )

        with smtplib.SMTP(
            smtp_server,
            smtp_port,
            timeout=20,
        ) as server:

            server.ehlo()
            server.starttls()
            server.ehlo()

            server.login(
                smtp_email,
                smtp_password,
            )

            server.send_message(message)

        return True, "Verification email sent successfully."

    except Exception:
        # Do not expose SMTP credentials or internal errors.
        return False, "Unable to send verification email. Please try again."
