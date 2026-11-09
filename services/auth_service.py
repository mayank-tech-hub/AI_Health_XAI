
import re
import secrets
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHash,
    VerifyMismatchError,
    VerificationError,
)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from database.connection import get_engine
from database.models import User, EmailVerification, PendingRegistration
from services.email_service import send_otp_email


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

password_hasher = PasswordHasher()

MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_BYTES = 1024
MAX_NAME_LENGTH = 100

OTP_LENGTH = 6
OTP_EXPIRY_MINUTES = 10
PENDING_REGISTRATION_EXPIRY_MINUTES = 30
OTP_MAX_ATTEMPTS = 5
OTP_RESEND_COOLDOWN_SECONDS = 60
OTP_MAX_RESENDS = 5
OTP_RESEND_WINDOW_HOURS = 24

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# --------------------------------------------------
# PASSWORD AND OTP UTILITIES
# --------------------------------------------------

def hash_password(password):
    return password_hasher.hash(password)


def verify_password(password, password_hash):
    try:
        return password_hasher.verify(password_hash, password)
    except (
        VerifyMismatchError,
        VerificationError,
        InvalidHash,
        ValueError,
    ):
        return False


def hash_otp(otp):
    return password_hasher.hash(otp)


def verify_otp(otp, otp_hash):
    try:
        return password_hasher.verify(otp_hash, otp)
    except (
        VerifyMismatchError,
        VerificationError,
        InvalidHash,
        ValueError,
    ):
        return False


def generate_otp():
    return f"{secrets.randbelow(10 ** OTP_LENGTH):0{OTP_LENGTH}d}"


def validate_password(password):
    if not isinstance(password, str):
        return False, "Invalid password."

    if len(password) < MIN_PASSWORD_LENGTH:
        return False, (
            f"Password must contain at least "
            f"{MIN_PASSWORD_LENGTH} characters."
        )

    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return False, "Password is too long."

    return True, "Password is valid."


def normalize_email(email):
    if not isinstance(email, str):
        return None

    email = email.strip().lower()

    if not EMAIL_PATTERN.fullmatch(email):
        return None

    return email


# --------------------------------------------------
# USER REGISTRATION - TEMPORARY RECORD ONLY
# --------------------------------------------------

def register_user(full_name, email, password):
    if not isinstance(full_name, str):
        return {"success": False, "message": "Enter a valid full name."}

    full_name = full_name.strip()
    email = normalize_email(email)

    if not full_name or len(full_name) > MAX_NAME_LENGTH:
        return {"success": False, "message": "Enter a valid full name."}

    if not email:
        return {"success": False, "message": "Enter a valid email address."}

    valid, message = validate_password(password)
    if not valid:
        return {"success": False, "message": message}

    try:
        engine = get_engine()
        now = datetime.now(timezone.utc)

        with Session(engine) as session:
            # A permanent account must not be registered again.
            existing_user = session.scalar(
                select(User).where(User.email == email)
            )

            if existing_user:
                if existing_user.email_verified:
                    return {
                        "success": False,
                        "message": "This email is already registered. Please login.",
                    }

                # Preserve compatibility with old unverified accounts.
                return {
                    "success": False,
                    "requires_verification": True,
                    "message": (
                        "An earlier unverified account exists for this email. "
                        "Use its existing verification process."
                    ),
                }

            # Remove an expired temporary registration for this email.
            pending = session.scalar(
                select(PendingRegistration)
                .where(PendingRegistration.email == email)
                .with_for_update()
            )

            if pending:
                if pending.expires_at > now:
                    return {
                        "success": False,
                        "requires_verification": True,
                        "message": (
                            "A registration is already pending for this email. "
                            "Please use the resend code option."
                        ),
                    }

                session.delete(pending)
                session.flush()

            otp = generate_otp()

            pending = PendingRegistration(
                full_name=full_name,
                email=email,
                password_hash=hash_password(password),
                otp_hash=hash_otp(otp),
                expires_at=now + timedelta(
                    minutes=PENDING_REGISTRATION_EXPIRY_MINUTES
                ),
                attempts=0,
                last_sent_at=now,
                resend_count=0,
                resend_window_started_at=now,
            )

            session.add(pending)
            session.commit()

        # Send after saving the temporary registration.
        email_sent, _ = send_otp_email(email, otp)

        return {
            "success": True,
            "requires_verification": True,
            "email_sent": email_sent,
            "message": (
                "Verification code sent. It is valid for 10 minutes."
                if email_sent
                else (
                    "Registration is pending, but the email could not be sent. "
                    "Please use the resend option."
                ),
            ),
        }

    except IntegrityError:
        return {
            "success": False,
            "message": (
                "A registration already exists for this email. "
                "Please try again."
            ),
        }

    except SQLAlchemyError:
        return {
            "success": False,
            "message": "Registration failed. Please try again.",
        }


# --------------------------------------------------
# VERIFY EMAIL OTP
# --------------------------------------------------

def verify_email_otp(email, otp):
    email = normalize_email(email)

    if not email or not isinstance(otp, str):
        return {
            "success": False,
            "message": "Enter your email and verification code.",
        }

    otp = otp.strip()

    if not re.fullmatch(r"\d{6}", otp):
        return {
            "success": False,
            "message": "Enter a valid 6-digit code.",
        }

    try:
        engine = get_engine()
        now = datetime.now(timezone.utc)

        with Session(engine) as session:
            # First check the new temporary registration flow.
            pending = session.scalar(
                select(PendingRegistration)
                .where(PendingRegistration.email == email)
                .with_for_update()
            )

            if pending:
                # The entire pending registration expires after 30 minutes.
                if pending.expires_at <= now:
                    session.delete(pending)
                    session.commit()
                    return {
                        "success": False,
                        "message": (
                            "Registration expired. Please register again."
                        ),
                    }

                # OTP validity is shorter than registration validity.
                otp_expiry = pending.last_sent_at + timedelta(
                    minutes=OTP_EXPIRY_MINUTES
                )

                if otp_expiry <= now:
                    return {
                        "success": False,
                        "message": (
                            "Verification code expired. Please request a new code."
                        ),
                    }

                if pending.attempts >= OTP_MAX_ATTEMPTS:
                    return {
                        "success": False,
                        "message": (
                            "Maximum attempts reached. Please request a new code."
                        ),
                    }

                if not verify_otp(otp, pending.otp_hash):
                    pending.attempts += 1
                    remaining = OTP_MAX_ATTEMPTS - pending.attempts
                    session.commit()

                    return {
                        "success": False,
                        "message": (
                            f"Incorrect code. {remaining} attempts remaining."
                            if remaining > 0
                            else (
                                "Maximum attempts reached. "
                                "Please request a new code."
                            )
                        ),
                    }

                # Recheck permanent account before creating it.
                existing_user = session.scalar(
                    select(User)
                    .where(User.email == email)
                    .with_for_update()
                )

                if existing_user:
                    return {
                        "success": False,
                        "message": (
                            "This email already has an account. Please login."
                        ),
                    }

                # Create the permanent account only after OTP success.
                user = User(
                    full_name=pending.full_name,
                    email=pending.email,
                    password_hash=pending.password_hash,
                    is_active=True,
                    email_verified=True,
                )

                session.add(user)
                session.delete(pending)
                session.commit()

                return {
                    "success": True,
                    "message": (
                        "Email verified and account created successfully. "
                        "You can now login."
                    ),
                }

            # Legacy verification support for older unverified User records.
            user = session.scalar(
                select(User)
                .where(User.email == email)
                .with_for_update()
            )

            if not user:
                return {
                    "success": False,
                    "message": (
                        "Verification request not found. "
                        "Please register again."
                    ),
                }

            if not user.is_active:
                return {
                    "success": False,
                    "message": "Account is inactive.",
                }

            if user.email_verified:
                return {
                    "success": True,
                    "message": "Email is already verified. Please login.",
                }

            verification = session.scalar(
                select(EmailVerification)
                .where(EmailVerification.user_id == user.id)
                .with_for_update()
            )

            if not verification:
                return {
                    "success": False,
                    "message": "Verification request not found.",
                }

            if verification.expires_at <= now:
                return {
                    "success": False,
                    "message": "Verification code expired. Please request a new code.",
                }

            if verification.attempts >= OTP_MAX_ATTEMPTS:
                return {
                    "success": False,
                    "message": "Maximum attempts reached. Please request a new code.",
                }

            if not verify_otp(otp, verification.otp_hash):
                verification.attempts += 1
                remaining = OTP_MAX_ATTEMPTS - verification.attempts
                session.commit()

                return {
                    "success": False,
                    "message": (
                        f"Incorrect code. {remaining} attempts remaining."
                        if remaining > 0
                        else "Maximum attempts reached. Please request a new code."
                    ),
                }

            user.email_verified = True
            session.delete(verification)
            session.commit()

        return {
            "success": True,
            "message": "Email verified successfully. You can now login.",
        }

    except SQLAlchemyError:
        return {
            "success": False,
            "message": "Verification failed. Please try again.",
        }


# --------------------------------------------------
# RESEND VERIFICATION OTP
# --------------------------------------------------

def resend_verification_otp(email):
    email = normalize_email(email)

    if not email:
        return {
            "success": False,
            "message": "Enter a valid email address.",
        }

    try:
        engine = get_engine()
        now = datetime.now(timezone.utc)

        with Session(engine) as session:
            # New temporary registration flow.
            pending = session.scalar(
                select(PendingRegistration)
                .where(PendingRegistration.email == email)
                .with_for_update()
            )

            if pending:
                if pending.expires_at <= now:
                    session.delete(pending)
                    session.commit()
                    return {
                        "success": False,
                        "message": (
                            "Registration expired. Please register again."
                        ),
                    }

                window_elapsed = (
                    now - pending.resend_window_started_at
                ).total_seconds()

                if window_elapsed >= OTP_RESEND_WINDOW_HOURS * 3600:
                    pending.resend_count = 0
                    pending.resend_window_started_at = now

                elapsed = (
                    now - pending.last_sent_at
                ).total_seconds()

                if elapsed < OTP_RESEND_COOLDOWN_SECONDS:
                    wait_seconds = int(
                        OTP_RESEND_COOLDOWN_SECONDS - elapsed
                    ) + 1

                    return {
                        "success": False,
                        "message": (
                            f"Please wait {wait_seconds} seconds "
                            "before requesting another code."
                        ),
                    }

                if pending.resend_count >= OTP_MAX_RESENDS:
                    return {
                        "success": False,
                        "message": (
                            "Daily resend limit reached. "
                            "Please try again after 24 hours."
                        ),
                    }

                otp = generate_otp()

                # Send before replacing the existing OTP.
                email_sent, _ = send_otp_email(email, otp)

                if not email_sent:
                    return {
                        "success": False,
                        "message": (
                            "Unable to send email. Your previous code "
                            "remains unchanged."
                        ),
                    }

                pending.otp_hash = hash_otp(otp)
                pending.last_sent_at = now
                pending.attempts = 0
                pending.resend_count += 1

                session.commit()

                return {
                    "success": True,
                    "message": "A new verification code has been sent.",
                }

            # Legacy resend support for older unverified User records.
            user = session.scalar(
                select(User)
                .where(User.email == email)
                .with_for_update()
            )

            if not user or not user.is_active:
                return {
                    "success": False,
                    "message": "Unable to process verification request.",
                }

            if user.email_verified:
                return {
                    "success": False,
                    "message": "Email is already verified. Please login.",
                }

            verification = session.scalar(
                select(EmailVerification)
                .where(EmailVerification.user_id == user.id)
                .with_for_update()
            )

            if not verification:
                return {
                    "success": False,
                    "message": "Verification request not found.",
                }

            window_elapsed = (
                now - verification.resend_window_started_at
            ).total_seconds()

            if window_elapsed >= OTP_RESEND_WINDOW_HOURS * 3600:
                verification.resend_count = 0
                verification.resend_window_started_at = now

            elapsed = (
                now - verification.last_sent_at
            ).total_seconds()

            if elapsed < OTP_RESEND_COOLDOWN_SECONDS:
                wait_seconds = int(
                    OTP_RESEND_COOLDOWN_SECONDS - elapsed
                ) + 1

                return {
                    "success": False,
                    "message": (
                        f"Please wait {wait_seconds} seconds "
                        "before requesting another code."
                    ),
                }

            if verification.resend_count >= OTP_MAX_RESENDS:
                return {
                    "success": False,
                    "message": (
                        "Daily resend limit reached. "
                        "Please try again after 24 hours."
                    ),
                }

            otp = generate_otp()
            email_sent, _ = send_otp_email(email, otp)

            if not email_sent:
                return {
                    "success": False,
                    "message": "Unable to send email. Please try again later.",
                }

            verification.otp_hash = hash_otp(otp)
            verification.expires_at = now + timedelta(
                minutes=OTP_EXPIRY_MINUTES
            )
            verification.attempts = 0
            verification.last_sent_at = now
            verification.resend_count += 1

            session.commit()

        return {
            "success": True,
            "message": "A new verification code has been sent.",
        }

    except SQLAlchemyError:
        return {
            "success": False,
            "message": "Unable to resend verification code. Please try again.",
        }


# --------------------------------------------------
# USER LOGIN
# --------------------------------------------------

def login_user(email, password):
    email = normalize_email(email)

    if not email or not isinstance(password, str) or not password:
        return {
            "success": False,
            "message": "Email and password are required.",
        }

    try:
        engine = get_engine()

        with Session(engine) as session:
            user = session.scalar(
                select(User).where(User.email == email)
            )

            if not user:
                return {
                    "success": False,
                    "message": "Invalid email or password.",
                }

            if not user.is_active:
                return {
                    "success": False,
                    "message": "Account is inactive.",
                }

            if not verify_password(password, user.password_hash):
                return {
                    "success": False,
                    "message": "Invalid email or password.",
                }

            if not user.email_verified:
                return {
                    "success": False,
                    "requires_verification": True,
                    "message": "Please verify your email before logging in.",
                }

            user_data = {
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email,
            }

        return {
            "success": True,
            "message": "Login successful.",
            "user": user_data,
        }

    except SQLAlchemyError:
        return {
            "success": False,
            "message": "Database connection error.",
        }


# --------------------------------------------------
# UPDATE USER PROFILE
# --------------------------------------------------

def update_user_profile(user_id, full_name):
    if not isinstance(full_name, str):
        return {
            "success": False,
            "message": "Enter a valid full name.",
        }

    full_name = full_name.strip()

    if not full_name or len(full_name) > MAX_NAME_LENGTH:
        return {
            "success": False,
            "message": "Enter a valid full name (1-100 characters).",
        }

    try:
        engine = get_engine()

        with Session(engine) as session:
            user = session.get(User, user_id)

            if not user or not user.is_active:
                return {
                    "success": False,
                    "message": "Active user account not found.",
                }

            user.full_name = full_name
            session.commit()

        return {
            "success": True,
            "message": "Profile updated successfully.",
            "full_name": full_name,
        }

    except SQLAlchemyError:
        return {
            "success": False,
            "message": "Unable to update profile. Please try again.",
        }


# --------------------------------------------------
# CHANGE USER PASSWORD
# --------------------------------------------------

def change_user_password(user_id, current_password, new_password):
    if (
        not isinstance(current_password, str)
        or not isinstance(new_password, str)
        or not current_password
        or not new_password
    ):
        return {
            "success": False,
            "message": "All password fields are required.",
        }

    valid, message = validate_password(new_password)

    if not valid:
        return {
            "success": False,
            "message": message,
        }

    if current_password == new_password:
        return {
            "success": False,
            "message": (
                "New password must be different from current password."
            ),
        }

    try:
        engine = get_engine()

        with Session(engine) as session:
            user = session.get(User, user_id)

            if not user or not user.is_active:
                return {
                    "success": False,
                    "message": "Active user account not found.",
                }

            if not verify_password(
                current_password,
                user.password_hash,
            ):
                return {
                    "success": False,
                    "message": "Current password is incorrect.",
                }

            user.password_hash = hash_password(new_password)
            session.commit()

        return {
            "success": True,
            "message": "Password changed successfully.",
        }

    except SQLAlchemyError:
        return {
            "success": False,
            "message": "Unable to change password. Please try again.",
        }
