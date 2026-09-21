"""Secret-safe diagnostics for Carrier OAuth token refresh responses."""

from collections.abc import Mapping
from hashlib import sha256
from inspect import isawaitable
from logging import WARNING, Logger
import re
from typing import Any, TypedDict

from aiohttp import ClientError

_SAFE_OAUTH_ERRORS = frozenset(
    {
        "access_denied",
        "account_selection_required",
        "consent_required",
        "interaction_required",
        "invalid_client",
        "invalid_grant",
        "invalid_request",
        "invalid_scope",
        "invalid_token",
        "login_required",
        "server_error",
        "temporarily_unavailable",
        "unauthorized_client",
        "unsupported_grant_type",
    }
)
_REQUEST_ID_HEADERS = (
    "request-id",
    "x-correlation-id",
    "x-okta-request-id",
    "x-okta-requestid",
    "x-request-id",
)
_MAX_ERROR_DESCRIPTION_LEN = 160
_MAX_CONTENT_TYPE_LEN = 128
_SAFE_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SAFE_CONTENT_TYPE_RE = re.compile(
    r"(?i)^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*(?:\s*;\s*[\w.\"=+-]{1,40})*$"
)
_SAFE_ERROR_DESCRIPTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .,_:;()/'\"+-]{0,159}$")
_UNSAFE_DESCRIPTION_RE = re.compile(
    r"(?i)(\b(password|passwd|secret|cookie|access_token|refresh_token)\b"
    r"|authorization\s*[:=]"
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r"|bearer\s+[A-Za-z0-9._~+/=-]{8,})"
)
_JWT_LIKE_RE = re.compile(r"^[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}$")
_HTML_PREFIXES = (b"<", b"<!")


class OauthRefreshDiagnostics(TypedDict):
    """Allowlisted diagnostic fields for one OAuth refresh response."""

    status: int | None
    content_type: str | None
    body_class: str
    body_length: int | None
    body_sha256: str | None
    oauth_error: str | None
    oauth_error_description: str | None
    request_ids: dict[str, str]


def build_oauth_refresh_diagnostics(
    *,
    status: int | None,
    content_type: str | None,
    headers: Any | None,
    raw_body: bytes | None,
    parsed: object | None,
) -> OauthRefreshDiagnostics:
    """Build a secret-safe diagnostic record from refresh response parts.

    Args:
        status: HTTP status code when known.
        content_type: Response Content-Type when known.
        headers: Response headers object or mapping. Only allowlisted request
            ID headers are read.
        raw_body: Raw response body bytes when available. The bytes are hashed
            and classified but never copied into the returned record.
        parsed: JSON value decoded from the body, when decoding succeeded.

    Returns:
        Diagnostic fields that are safe to log.
    """
    safe_content_type = _safe_content_type(content_type)
    return {
        "status": status if isinstance(status, int) else None,
        "content_type": safe_content_type,
        "body_class": _classify_body(raw_body, safe_content_type, parsed),
        "body_length": len(raw_body) if isinstance(raw_body, bytes) else None,
        "body_sha256": sha256(raw_body).hexdigest() if isinstance(raw_body, bytes) else None,
        "oauth_error": _safe_oauth_error(_error_field(parsed, "error")),
        "oauth_error_description": _safe_oauth_error_description(
            _error_field(parsed, "error_description")
        ),
        "request_ids": _allowlisted_request_ids(headers),
    }


async def collect_oauth_refresh_diagnostics(
    response: Any,
    *,
    parsed: object | None = None,
    status_override: int | None = None,
) -> OauthRefreshDiagnostics:
    """Collect secret-safe diagnostics from an OAuth refresh response.

    Args:
        response: aiohttp-like response, or ``None`` when no response exists.
        parsed: JSON value already decoded from the response, when available.
        status_override: HTTP status from a raised ``ClientResponseError`` when
            the response object does not expose ``status``.

    Returns:
        Diagnostic fields that are safe to log.
    """
    status = status_override
    content_type = None
    headers = None
    raw_body = None
    if response is not None:
        response_status = getattr(response, "status", None)
        if isinstance(response_status, int):
            status = response_status
        content_type = _response_content_type(response)
        headers = getattr(response, "headers", None)
        raw_body = await _optional_raw_body(response)
    return build_oauth_refresh_diagnostics(
        status=status,
        content_type=content_type,
        headers=headers,
        raw_body=raw_body,
        parsed=parsed,
    )


def emit_oauth_refresh_diagnostics(
    logger: Logger,
    diagnostics: OauthRefreshDiagnostics,
    *,
    level: int = WARNING,
) -> None:
    """Emit one secret-safe OAuth refresh diagnostic log record.

    Args:
        logger: Logger to write the diagnostic record to.
        diagnostics: Allowlisted diagnostic fields.
        level: Logging level for the record.
    """
    logger.log(
        level,
        "Carrier OAuth token refresh response status=%s content_type=%s body_class=%s "
        "body_length=%s body_sha256=%s oauth_error=%s oauth_error_description=%s "
        "request_ids=%s",
        diagnostics["status"],
        diagnostics["content_type"],
        diagnostics["body_class"],
        diagnostics["body_length"],
        diagnostics["body_sha256"],
        diagnostics["oauth_error"],
        diagnostics["oauth_error_description"],
        diagnostics["request_ids"],
    )


async def log_oauth_refresh_response(
    logger: Logger,
    response: Any,
    *,
    parsed: object | None = None,
    status_override: int | None = None,
    level: int = WARNING,
) -> None:
    """Collect and log secret-safe OAuth refresh diagnostics.

    Collection or logging failures are swallowed so diagnostics cannot change
    token-refresh exception behavior.

    Args:
        logger: Logger to write the diagnostic record to.
        response: aiohttp-like response, or ``None`` when no response exists.
        parsed: JSON value already decoded from the response, when available.
        status_override: HTTP status from a raised client error.
        level: Logging level for the record.
    """
    try:
        diagnostics = await collect_oauth_refresh_diagnostics(
            response,
            parsed=parsed,
            status_override=status_override,
        )
        emit_oauth_refresh_diagnostics(logger, diagnostics, level=level)
    except TypeError, ValueError, AttributeError, OSError, RuntimeError, KeyError:
        try:
            logger.log(level, "Carrier OAuth token refresh response diagnostics unavailable")
        except TypeError, ValueError, OSError, RuntimeError:
            return


def _error_field(parsed: object, field_name: str) -> object:
    """Return one OAuth error field from a parsed JSON object.

    Args:
        parsed: Decoded JSON value.
        field_name: Field to read when ``parsed`` is a mapping.

    Returns:
        The field value, or ``None`` when it is absent or ``parsed`` is not a
        mapping.
    """
    if not isinstance(parsed, Mapping):
        return None
    try:
        return parsed.get(field_name)
    except TypeError, AttributeError:
        return None


def _safe_oauth_error(value: object) -> str | None:
    """Return an allowlisted OAuth error code.

    Args:
        value: Candidate error field from a JSON object.

    Returns:
        The canonical error code, or ``None`` when the value is not allowlisted.
    """
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in _SAFE_OAUTH_ERRORS:
        return normalized
    return None


def _safe_oauth_error_description(value: object) -> str | None:
    """Return a short OAuth error description when it does not look secret.

    Args:
        value: Candidate error_description field from a JSON object.

    Returns:
        The description, or ``None`` when it is missing or unsafe to log.
    """
    if not isinstance(value, str):
        return None
    description = value.strip()
    if not description or len(description) > _MAX_ERROR_DESCRIPTION_LEN:
        return None
    if _UNSAFE_DESCRIPTION_RE.search(description):
        return None
    if _SAFE_ERROR_DESCRIPTION_RE.fullmatch(description) is None:
        return None
    return description


def _safe_content_type(value: object) -> str | None:
    """Return a Content-Type value that looks like a MIME type.

    Args:
        value: Candidate Content-Type header.

    Returns:
        The trimmed Content-Type, or ``None`` when it is missing or unsafe.
    """
    if not isinstance(value, str):
        return None
    content_type = value.strip()
    if not content_type or len(content_type) > _MAX_CONTENT_TYPE_LEN:
        return None
    if _SAFE_CONTENT_TYPE_RE.fullmatch(content_type) is None:
        return None
    return content_type


def _classify_body(
    raw_body: bytes | None,
    content_type: str | None,
    parsed: object | None,
) -> str:
    """Classify a refresh response body without exposing its contents.

    Args:
        raw_body: Raw response body bytes when available.
        content_type: Safe Content-Type when available.
        parsed: Decoded JSON value when decoding succeeded.

    Returns:
        A body class label suitable for logs.
    """
    if isinstance(parsed, dict):
        return "json_object"
    if isinstance(parsed, list):
        return "json_array"
    if parsed is not None:
        return "json_other"
    if raw_body == b"":
        return "empty"
    if _looks_like_html(raw_body, content_type):
        return "html"
    if isinstance(raw_body, bytes):
        stripped = raw_body.lstrip()
        if stripped.startswith((b"{", b"[")):
            return "malformed"
        return "non_json"
    if content_type is not None and "html" in content_type.lower():
        return "html"
    return "unavailable"


def _looks_like_html(raw_body: bytes | None, content_type: str | None) -> bool:
    """Return whether the body or Content-Type looks like HTML.

    Args:
        raw_body: Raw response body bytes when available.
        content_type: Safe Content-Type when available.

    Returns:
        ``True`` when the response appears to be HTML.
    """
    if content_type is not None and "html" in content_type.lower():
        return True
    if not raw_body:
        return False
    stripped = raw_body.lstrip().lower()
    return stripped.startswith(_HTML_PREFIXES)


def _allowlisted_request_ids(headers: Any | None) -> dict[str, str]:
    """Return request IDs from allowlisted headers only.

    Args:
        headers: Response headers object or mapping.

    Returns:
        Mapping of allowlisted header names to safe ID values.
    """
    if headers is None:
        return {}
    request_ids: dict[str, str] = {}
    for name in _REQUEST_ID_HEADERS:
        value = _header_get(headers, name)
        if isinstance(value, str) and _is_safe_request_id(value):
            request_ids[name] = value
    return request_ids


def _is_safe_request_id(value: object) -> bool:
    """Return whether a header value looks like a request ID.

    Args:
        value: Candidate request ID.

    Returns:
        ``True`` when the value is a safe, non-token identifier.
    """
    if not isinstance(value, str):
        return False
    if _JWT_LIKE_RE.fullmatch(value) is not None:
        return False
    return _SAFE_REQUEST_ID_RE.fullmatch(value) is not None


def _header_get(headers: Any, name: str) -> str | None:
    """Read one header without iterating arbitrary header values.

    Args:
        headers: Response headers object or mapping.
        name: Header name to read.

    Returns:
        The header value when present and a string, otherwise ``None``.
    """
    getter = getattr(headers, "get", None)
    if callable(getter):
        for candidate in (name, name.lower(), name.title(), name.upper()):
            try:
                value = getter(candidate)
            except TypeError, ValueError, AttributeError, KeyError:
                continue
            if isinstance(value, str):
                return value
    items = getattr(headers, "items", None)
    if callable(items):
        try:
            pairs = items()
        except TypeError, ValueError, AttributeError:
            return None
        for key, value in pairs:
            if str(key).lower() == name and isinstance(value, str):
                return value
    return None


def _response_content_type(response: Any) -> str | None:
    """Read Content-Type from a response without scanning other headers.

    Args:
        response: aiohttp-like response object.

    Returns:
        The Content-Type string when available.
    """
    content_type = getattr(response, "content_type", None)
    if isinstance(content_type, str) and content_type:
        return content_type
    headers = getattr(response, "headers", None)
    return _header_get(headers, "content-type")


async def _optional_raw_body(response: Any) -> bytes | None:
    """Read raw response body bytes when the response exposes them.

    Args:
        response: aiohttp-like response object.

    Returns:
        Body bytes, or ``None`` when they are unavailable.
    """
    raw_body = getattr(response, "raw_body", None)
    if isinstance(raw_body, bytes):
        return raw_body
    read = getattr(response, "read", None)
    if not callable(read):
        return None
    try:
        result = read()
        if isawaitable(result):
            result = await result
    except ClientError, TimeoutError, OSError, TypeError, ValueError, RuntimeError:
        return None
    if isinstance(result, bytes):
        return result
    if isinstance(result, bytearray):
        return bytes(result)
    return None
