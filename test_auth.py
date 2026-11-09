from services.auth_service import register_user, login_user

email = "test_user_2026@example.com"
password = "TestPassword@12345"

print("\n1. Registration Test")
registration = register_user("Test User", email, password)
print(registration)

print("\n2. Correct Login Test")
login = login_user(email, password)
print(login)

print("\n3. Wrong Password Test")
wrong_login = login_user(email, "WrongPassword@12345")
print(wrong_login)

print("\n4. Duplicate Registration Test")
duplicate = register_user("Test User", email, password)
print(duplicate)
