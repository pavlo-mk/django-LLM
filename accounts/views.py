import json

from django.contrib.auth import authenticate, get_user_model, login, logout
from django.contrib.auth.forms import PasswordResetForm, SetPasswordForm
from django.contrib.auth.tokens import default_token_generator
from django.http import HttpResponseBadRequest, JsonResponse
from django.utils.http import urlsafe_base64_decode
from django.views.decorators.http import require_GET, require_POST

User = get_user_model()


def _parse_json(request) -> dict:
    try:
        return json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return {}


def _user_payload(user) -> dict:
    return {
        "id": user.pk,
        "username": user.username,
        "email": user.email,
    }


def _lookup_user(identifier: str):
    """Resolve a login identifier to a User (username or email)."""
    if "@" in identifier:
        return User.objects.filter(email__iexact=identifier).first()
    return User.objects.filter(username__iexact=identifier).first()


@require_POST
def login_view(request):
    """Authenticate with username/email + password and start a session."""
    payload = _parse_json(request)
    identifier = (payload.get("username") or payload.get("email") or "").strip()
    password = payload.get("password") or ""
    if not identifier or not password:
        return HttpResponseBadRequest("username (or email) and password are required")

    user = _lookup_user(identifier)
    if user is None:
        return JsonResponse({"error": "invalid credentials"}, status=401)

    authed = authenticate(request, username=user.username, password=password)
    if authed is None:
        return JsonResponse({"error": "invalid credentials"}, status=401)
    if not authed.is_active:
        return JsonResponse({"error": "account is inactive"}, status=403)

    login(request, authed)
    return JsonResponse({"user": _user_payload(authed)})


@require_POST
def logout_view(request):
    """End the current session."""
    logout(request)
    return JsonResponse({"detail": "logged out"})


@require_GET
def me_view(request):
    """Return the currently authenticated user, if any."""
    if not request.user.is_authenticated:
        return JsonResponse({"user": None})
    return JsonResponse({"user": _user_payload(request.user)})


@require_POST
def password_reset_request(request):
    """Send a password-reset email when the address matches an account."""
    payload = _parse_json(request)
    email = (payload.get("email") or "").strip()
    if not email:
        return HttpResponseBadRequest("email is required")

    form = PasswordResetForm({"email": email})
    if form.is_valid():
        from django.conf import settings

        form.save(
            request=request,
            use_https=request.is_secure(),
            from_email=settings.DEFAULT_FROM_EMAIL,
            email_template_name="accounts/password_reset_email.txt",
            subject_template_name="accounts/password_reset_subject.txt",
            extra_email_context={"frontend_url": settings.PASSWORD_RESET_FRONTEND_URL},
        )
    # Always succeed so callers cannot probe for registered emails.
    return JsonResponse({"detail": "if an account exists, a reset email was sent"})


@require_POST
def password_reset_confirm(request):
    """Set a new password using the uid + token from the reset email."""
    payload = _parse_json(request)
    uid = (payload.get("uid") or "").strip()
    token = (payload.get("token") or "").strip()
    password = payload.get("password") or ""
    if not uid or not token or not password:
        return HttpResponseBadRequest("uid, token, and password are required")

    try:
        user_id = urlsafe_base64_decode(uid).decode()
        user = User.objects.get(pk=user_id)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        return JsonResponse({"error": "invalid or expired link"}, status=400)

    if not default_token_generator.check_token(user, token):
        return JsonResponse({"error": "invalid or expired link"}, status=400)

    form = SetPasswordForm(user, {"new_password1": password, "new_password2": password})
    if not form.is_valid():
        return JsonResponse({"errors": form.errors.get_json_data()}, status=400)

    form.save()
    return JsonResponse({"detail": "password updated"})
