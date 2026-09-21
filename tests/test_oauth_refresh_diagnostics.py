"""Tests for secret-safe OAuth refresh diagnostics."""

from hashlib import sha256
from logging import DEBUG, WARNING, getLogger
from typing import Any, ClassVar

from aiohttp import ClientError
import pytest

from carrier_api.oauth_refresh_diagnostics import (
    OauthRefreshDiagnostics,
    build_oauth_refresh_diagnostics,
    collect_oauth_refresh_diagnostics,
    emit_oauth_refresh_diagnostics,
    log_oauth_refresh_response,
)

_SECRET_ACCESS_TOKEN = "secret-access-token-aaa"
_SECRET_PASSWORD = "super-secret-password-xyz"
_SECRET_AUTHORIZATION = "Bearer secret-auth-header"
_SECRET_COOKIE = "session=secret-cookie-value"
_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.secret-signature-value"


def _assert_no_secrets(payload: object) -> None:
    """Fail if a diagnostic payload contains known secrets.

    Args:
        payload: Diagnostic record, log text, or interpolated arguments.
    """
    rendered = str(payload)
    for secret in (
        _SECRET_ACCESS_TOKEN,
        _SECRET_PASSWORD,
        _SECRET_AUTHORIZATION,
        _SECRET_COOKIE,
        _JWT,
        "secret-in-html-body",
    ):
        assert secret not in rendered


def test_build_success_json_object_hashes_body_without_tokens() -> None:
    """Hash a successful JSON body and ignore token fields."""
    raw_body = b'{"access_token":"secret-access-token-aaa","refresh_token":"secret"}'
    diagnostics = build_oauth_refresh_diagnostics(
        status=200,
        content_type="application/json",
        headers={"X-Request-Id": "req-1", "Authorization": _SECRET_AUTHORIZATION},
        raw_body=raw_body,
        parsed={
            "access_token": _SECRET_ACCESS_TOKEN,
            "refresh_token": "secret-rotated-refresh-ccc",
            "expires_in": 3600,
        },
    )

    assert diagnostics["status"] == 200
    assert diagnostics["content_type"] == "application/json"
    assert diagnostics["body_class"] == "json_object"
    assert diagnostics["body_length"] == len(raw_body)
    assert diagnostics["body_sha256"] == sha256(raw_body).hexdigest()
    assert diagnostics["oauth_error"] is None
    assert diagnostics["request_ids"] == {"x-request-id": "req-1"}
    _assert_no_secrets(diagnostics)


def test_build_invalid_grant_keeps_safe_description() -> None:
    """Keep RFC invalid_grant and a human-readable description."""
    diagnostics = build_oauth_refresh_diagnostics(
        status=400,
        content_type="application/json; charset=utf-8",
        headers={"x-okta-request-id": "okta-123"},
        raw_body=b'{"error":"invalid_grant"}',
        parsed={
            "error": "invalid_grant",
            "error_description": "The refresh token is invalid or expired.",
        },
    )

    assert diagnostics["oauth_error"] == "invalid_grant"
    assert diagnostics["oauth_error_description"] == "The refresh token is invalid or expired."
    assert diagnostics["request_ids"] == {"x-okta-request-id": "okta-123"}
    assert diagnostics["body_class"] == "json_object"


@pytest.mark.parametrize(
    ("error_code", "description"),
    [
        ("temporarily_unavailable", "The authorization server is busy."),
        ("invalid_client", "Client authentication failed."),
        ("server_error", "Unexpected authorization server error."),
    ],
)
def test_build_other_json_errors_are_allowlisted(error_code: str, description: str) -> None:
    """Keep other allowlisted OAuth error codes and safe descriptions.

    Args:
        error_code: Allowlisted OAuth error.
        description: Safe error_description to log.
    """
    diagnostics = build_oauth_refresh_diagnostics(
        status=400,
        content_type="application/json",
        headers=None,
        raw_body=None,
        parsed={"error": error_code, "error_description": description},
    )

    assert diagnostics["oauth_error"] == error_code
    assert diagnostics["oauth_error_description"] == description


def test_build_html_challenge_uses_content_type_and_hash() -> None:
    """Classify HTML challenges and hash the body without logging it."""
    raw_body = b"<html><body>secret-in-html-body</body></html>"
    diagnostics = build_oauth_refresh_diagnostics(
        status=403,
        content_type="text/html; charset=utf-8",
        headers={"X-Correlation-Id": "corr-42"},
        raw_body=raw_body,
        parsed=None,
    )

    assert diagnostics["body_class"] == "html"
    assert diagnostics["body_length"] == len(raw_body)
    assert diagnostics["body_sha256"] == sha256(raw_body).hexdigest()
    assert diagnostics["request_ids"] == {"x-correlation-id": "corr-42"}
    _assert_no_secrets(diagnostics)


def test_build_empty_and_malformed_bodies() -> None:
    """Classify empty and malformed JSON bodies."""
    empty = build_oauth_refresh_diagnostics(
        status=400,
        content_type="application/json",
        headers=None,
        raw_body=b"",
        parsed=None,
    )
    malformed = build_oauth_refresh_diagnostics(
        status=400,
        content_type="application/json",
        headers=None,
        raw_body=b"{not json",
        parsed=None,
    )

    assert empty["body_class"] == "empty"
    assert empty["body_length"] == 0
    assert empty["body_sha256"] == sha256(b"").hexdigest()
    assert malformed["body_class"] == "malformed"
    assert "{not json" not in str(malformed)


def test_build_rejects_adversarial_secret_payloads() -> None:
    """Drop unsafe error fields, headers, and token-shaped request IDs."""
    diagnostics = build_oauth_refresh_diagnostics(
        status=400,
        content_type="application/json",
        headers={
            "Authorization": _SECRET_AUTHORIZATION,
            "Cookie": _SECRET_COOKIE,
            "X-Request-Id": _JWT,
            "X-Evil-Token": _SECRET_ACCESS_TOKEN,
            "request-id": "req-safe-1",
        },
        raw_body=b'{"access_token":"secret-access-token-aaa","password":"super-secret-password-xyz"}',
        parsed={
            "error": _JWT,
            "error_description": f"password={_SECRET_PASSWORD}",
            "access_token": _SECRET_ACCESS_TOKEN,
            "refresh_token": "secret-rotated-refresh-ccc",
        },
    )

    assert diagnostics["oauth_error"] is None
    assert diagnostics["oauth_error_description"] is None
    assert diagnostics["request_ids"] == {"request-id": "req-safe-1"}
    _assert_no_secrets(diagnostics)


@pytest.mark.parametrize(
    ("parsed", "expected_class"),
    [
        (["invalid_grant"], "json_array"),
        ("invalid_grant", "json_other"),
        (None, "unavailable"),
    ],
)
def test_build_classifies_non_object_json(parsed: object, expected_class: str) -> None:
    """Classify non-object JSON values without treating them as objects.

    Args:
        parsed: Decoded JSON value.
        expected_class: Expected body class.
    """
    diagnostics = build_oauth_refresh_diagnostics(
        status=400,
        content_type="application/json",
        headers=None,
        raw_body=None,
        parsed=parsed,
    )

    assert diagnostics["body_class"] == expected_class
    assert diagnostics["oauth_error"] is None


def test_build_omits_long_or_keyword_descriptions() -> None:
    """Omit descriptions that are too long or contain secret keywords."""
    long_description = "A" * 161
    diagnostics = build_oauth_refresh_diagnostics(
        status=400,
        content_type="application/json",
        headers=None,
        raw_body=None,
        parsed={
            "error": "invalid_request",
            "error_description": long_description,
        },
    )
    keyword = build_oauth_refresh_diagnostics(
        status=400,
        content_type="application/json",
        headers=None,
        raw_body=None,
        parsed={
            "error": "invalid_request",
            "error_description": "refresh_token reused by client",
        },
    )

    assert diagnostics["oauth_error"] == "invalid_request"
    assert diagnostics["oauth_error_description"] is None
    assert keyword["oauth_error_description"] is None
    assert long_description not in str(diagnostics)


def test_build_rejects_unsafe_content_type() -> None:
    """Omit Content-Type values that do not look like MIME types."""
    diagnostics = build_oauth_refresh_diagnostics(
        status=200,
        content_type=f"Bearer {_SECRET_ACCESS_TOKEN}",
        headers=None,
        raw_body=None,
        parsed={},
    )

    assert diagnostics["content_type"] is None
    _assert_no_secrets(diagnostics)


@pytest.mark.asyncio
async def test_collect_reads_raw_body_and_status_override() -> None:
    """Collect diagnostics from a response double and status override."""

    class FakeResponse:
        """Response double with headers and a raw body."""

        status = 418
        content_type = "application/json"
        headers: ClassVar[dict[str, str]] = {"X-Request-Id": "req-collect"}
        raw_body = b'{"error":"invalid_grant"}'

    diagnostics = await collect_oauth_refresh_diagnostics(
        FakeResponse(),
        parsed={"error": "invalid_grant"},
        status_override=400,
    )

    assert diagnostics["status"] == 418
    assert diagnostics["request_ids"] == {"x-request-id": "req-collect"}
    assert diagnostics["oauth_error"] == "invalid_grant"
    assert diagnostics["body_sha256"] == sha256(b'{"error":"invalid_grant"}').hexdigest()


@pytest.mark.asyncio
async def test_collect_reads_async_read_and_none_response() -> None:
    """Use response.read when raw_body is absent, and tolerate a missing response."""

    class ReadableResponse:
        """Response double that exposes an async read method."""

        status = 200
        content_type = "text/plain"
        headers: ClassVar[dict[str, str]] = {}

        async def read(self) -> bytes:
            """Return a non-JSON body.

            Returns:
                Raw body bytes.
            """
            return b"not-json-body"

    readable = await collect_oauth_refresh_diagnostics(ReadableResponse(), parsed=None)
    missing = await collect_oauth_refresh_diagnostics(None, parsed=None, status_override=503)

    assert readable["body_class"] == "non_json"
    assert readable["body_length"] == len(b"not-json-body")
    assert missing["status"] == 503
    assert missing["body_class"] == "unavailable"


@pytest.mark.asyncio
async def test_collect_swallows_read_errors() -> None:
    """Ignore body-read failures so diagnostics cannot raise into refresh."""

    class FailingReadResponse:
        """Response double whose read method fails."""

        status = 500
        content_type = "application/json"
        headers: ClassVar[dict[str, str]] = {}

        async def read(self) -> bytes:
            """Raise a transport error.

            Raises:
                ClientError: Always raised for this double.
            """
            raise ClientError("read failed")

    diagnostics = await collect_oauth_refresh_diagnostics(FailingReadResponse(), parsed=None)

    assert diagnostics["status"] == 500
    assert diagnostics["body_class"] == "unavailable"
    assert diagnostics["body_length"] is None


def test_emit_and_log_do_not_include_secrets(caplog: pytest.LogCaptureFixture) -> None:
    """Render only allowlisted fields into the log record."""
    logger = getLogger("carrier_api.oauth_refresh_diagnostics")
    caplog.set_level(DEBUG, logger=logger.name)
    diagnostics = build_oauth_refresh_diagnostics(
        status=400,
        content_type="application/json",
        headers={"X-Request-Id": "req-emit"},
        raw_body=b'{"access_token":"secret-access-token-aaa"}',
        parsed={"error": "invalid_grant", "access_token": _SECRET_ACCESS_TOKEN},
    )

    emit_oauth_refresh_diagnostics(logger, diagnostics, level=WARNING)

    assert "status=400" in caplog.text
    assert "oauth_error=invalid_grant" in caplog.text
    assert "req-emit" in caplog.text
    _assert_no_secrets(caplog.text)


@pytest.mark.asyncio
async def test_log_oauth_refresh_response_swallows_collection_errors(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep refresh exception behavior when diagnostic collection fails."""
    logger = getLogger("carrier_api.oauth_refresh_diagnostics")
    caplog.set_level(WARNING, logger=logger.name)

    async def fail_collect(*args: Any, **kwargs: Any) -> OauthRefreshDiagnostics:
        """Raise a collection error.

        Raises:
            ValueError: Always raised for this test double.
        """
        raise ValueError(f"boom {_SECRET_PASSWORD}")

    monkeypatch.setattr(
        "carrier_api.oauth_refresh_diagnostics.collect_oauth_refresh_diagnostics",
        fail_collect,
    )

    await log_oauth_refresh_response(logger, object(), parsed={"error": "invalid_grant"})

    assert "diagnostics unavailable" in caplog.text
    _assert_no_secrets(caplog.text)


def test_diagnostic_record_has_only_allowlisted_keys() -> None:
    """Keep the public diagnostic record limited to the allowlisted fields."""
    diagnostics = build_oauth_refresh_diagnostics(
        status=200,
        content_type="application/json",
        headers=None,
        raw_body=b"{}",
        parsed={},
    )

    assert set(diagnostics) == {
        "status",
        "content_type",
        "body_class",
        "body_length",
        "body_sha256",
        "oauth_error",
        "oauth_error_description",
        "request_ids",
    }
