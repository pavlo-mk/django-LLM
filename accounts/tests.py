from django.contrib.auth import get_user_model
from django.contrib.auth.tokens import default_token_generator
from django.core import mail
from django.test import TestCase, override_settings
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode

User = get_user_model()


class AuthViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="alice",
            email="alice@example.com",
            password="secure-pass-123",
        )

    def test_login_with_username(self):
        res = self.client.post(
            "/api/auth/login/",
            data={"username": "alice", "password": "secure-pass-123"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["user"]["username"], "alice")
        self.assertEqual(body["user"]["email"], "alice@example.com")
        self.assertEqual(body["user"]["id"], self.user.pk)

    def test_login_with_email(self):
        res = self.client.post(
            "/api/auth/login/",
            data={"email": "alice@example.com", "password": "secure-pass-123"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["user"]["email"], "alice@example.com")

    def test_login_with_email_is_case_insensitive(self):
        res = self.client.post(
            "/api/auth/login/",
            data={"email": "ALICE@EXAMPLE.COM", "password": "secure-pass-123"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["user"]["username"], "alice")

    def test_login_establishes_session(self):
        self.client.post(
            "/api/auth/login/",
            data={"username": "alice", "password": "secure-pass-123"},
            content_type="application/json",
        )
        me = self.client.get("/api/auth/me/")
        self.assertEqual(me.json()["user"]["username"], "alice")

    def test_login_rejects_unknown_user(self):
        res = self.client.post(
            "/api/auth/login/",
            data={"username": "nobody", "password": "secure-pass-123"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["error"], "invalid credentials")

    def test_login_rejects_bad_password(self):
        res = self.client.post(
            "/api/auth/login/",
            data={"username": "alice", "password": "wrong"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["error"], "invalid credentials")

    def test_login_rejects_inactive_user(self):
        self.user.is_active = False
        self.user.save(update_fields=["is_active"])
        res = self.client.post(
            "/api/auth/login/",
            data={"username": "alice", "password": "secure-pass-123"},
            content_type="application/json",
        )
        # Django's authenticate() returns None for inactive users.
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["error"], "invalid credentials")

    def test_login_requires_fields(self):
        res = self.client.post(
            "/api/auth/login/",
            data={"username": "alice"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)

    def test_login_rejects_empty_password(self):
        res = self.client.post(
            "/api/auth/login/",
            data={"username": "alice", "password": ""},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)

    def test_login_rejects_invalid_json(self):
        res = self.client.post(
            "/api/auth/login/",
            data="not-json",
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)

    def test_login_rejects_get(self):
        res = self.client.get("/api/auth/login/")
        self.assertEqual(res.status_code, 405)

    def test_me_unauthenticated(self):
        res = self.client.get("/api/auth/me/")
        self.assertEqual(res.status_code, 200)
        self.assertIsNone(res.json()["user"])

    def test_me_authenticated(self):
        self.client.login(username="alice", password="secure-pass-123")
        res = self.client.get("/api/auth/me/")
        self.assertEqual(res.json()["user"]["username"], "alice")

    def test_logout(self):
        self.client.login(username="alice", password="secure-pass-123")
        res = self.client.post("/api/auth/logout/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["detail"], "logged out")
        me = self.client.get("/api/auth/me/")
        self.assertIsNone(me.json()["user"])

    def test_logout_when_unauthenticated(self):
        res = self.client.post("/api/auth/logout/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["detail"], "logged out")

    def test_logout_rejects_get(self):
        res = self.client.get("/api/auth/logout/")
        self.assertEqual(res.status_code, 405)

    @override_settings(
        EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
        PASSWORD_RESET_FRONTEND_URL="http://localhost:8000/reset-password",
    )
    def test_password_reset_sends_email(self):
        res = self.client.post(
            "/api/auth/password-reset/",
            data={"email": "alice@example.com"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(
            res.json()["detail"],
            "if an account exists, a reset email was sent",
        )
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("alice@example.com", mail.outbox[0].to)
        self.assertIn("uid=", mail.outbox[0].body)
        self.assertIn("token=", mail.outbox[0].body)
        self.assertIn("http://localhost:8000/reset-password", mail.outbox[0].body)

    def test_password_reset_unknown_email_still_succeeds(self):
        res = self.client.post(
            "/api/auth/password-reset/",
            data={"email": "missing@example.com"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(
            res.json()["detail"],
            "if an account exists, a reset email was sent",
        )

    def test_password_reset_requires_email(self):
        res = self.client.post(
            "/api/auth/password-reset/",
            data={"email": ""},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)

    def test_password_reset_rejects_get(self):
        res = self.client.get("/api/auth/password-reset/")
        self.assertEqual(res.status_code, 405)

    def test_password_reset_confirm(self):
        uid = urlsafe_base64_encode(force_bytes(self.user.pk))
        token = default_token_generator.make_token(self.user)
        res = self.client.post(
            "/api/auth/password-reset/confirm/",
            data={"uid": uid, "token": token, "password": "new-secure-pass-456"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["detail"], "password updated")
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password("new-secure-pass-456"))

    def test_password_reset_confirm_rejects_bad_token(self):
        uid = urlsafe_base64_encode(force_bytes(self.user.pk))
        res = self.client.post(
            "/api/auth/password-reset/confirm/",
            data={"uid": uid, "token": "bad-token", "password": "new-secure-pass-456"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["error"], "invalid or expired link")

    def test_password_reset_confirm_rejects_invalid_uid(self):
        res = self.client.post(
            "/api/auth/password-reset/confirm/",
            data={"uid": "not-valid", "token": "t", "password": "new-secure-pass-456"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["error"], "invalid or expired link")

    def test_password_reset_confirm_rejects_unknown_user(self):
        uid = urlsafe_base64_encode(force_bytes(99999))
        token = default_token_generator.make_token(self.user)
        res = self.client.post(
            "/api/auth/password-reset/confirm/",
            data={"uid": uid, "token": token, "password": "new-secure-pass-456"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["error"], "invalid or expired link")

    def test_password_reset_confirm_requires_fields(self):
        res = self.client.post(
            "/api/auth/password-reset/confirm/",
            data={"uid": "abc"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)

    def test_password_reset_confirm_rejects_weak_password(self):
        uid = urlsafe_base64_encode(force_bytes(self.user.pk))
        token = default_token_generator.make_token(self.user)
        res = self.client.post(
            "/api/auth/password-reset/confirm/",
            data={"uid": uid, "token": token, "password": "123"},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("errors", res.json())

    def test_password_reset_confirm_rejects_get(self):
        res = self.client.get("/api/auth/password-reset/confirm/")
        self.assertEqual(res.status_code, 405)
