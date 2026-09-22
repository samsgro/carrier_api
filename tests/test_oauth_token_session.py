"""Deterministic oauthfix.1 token-session, recovery, and lock tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from logging import WARNING
from typing import Any, Self, cast

from aiohttp import ClientConnectionError, ClientResponseError, ClientSession
import pytest

from carrier_api.api_connection_graphql import ApiConnectionGraphql, TokenPair, TokenSessionState
from carrier_api.errors import CarrierApiAuthError, CarrierApiTokenRefreshError

from .test_api_connection_graphql import (
    _SECRET_ACCESS_TOKEN,
    _SECRET_AUTHORIZATION,
    _SECRET_COOKIE,
    _SECRET_NEW_REFRESH_TOKEN,
    _SECRET_PASSWORD,
    _SECRET_USERNAME,
    FakeResponse,
    FakeSession,
    _assert_logs_are_secret_safe,
    graphql_client_double,
)

_LOGIN_ACCESS = "login-access-token"
_LOGIN_REFRESH = "login-refresh-token"
_FIXED_NOW = datetime(2026, 9, 22, 6, 0, tzinfo=UTC)


class CountingSession(FakeSession):
    """Session that records every token-endpoint POST."""

    def __init__(self) -> None:
        """Initialize captured POST state."""
        super().__init__()
        self.posts: list[dict[str, Any]] = []

    async def post(self, url: str, data: dict[str, Any]) -> FakeResponse:
        """Capture each refresh POST and yield once so gather races can join.

        Args:
            url: Requested URL.
            data: Submitted form data.

        Returns:
            The configured fake response.
        """
        self.posts.append(data)
        self.post_url = url
        self.post_data = data
        await asyncio.sleep(0)
        return self.response


class DelayedSession(FakeSession):
    """Session that blocks the refresh POST until a test releases it."""

    def __init__(self) -> None:
        """Initialize delay coordination events."""
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.post_count = 0

    async def post(self, url: str, data: dict[str, Any]) -> FakeResponse:
        """Block until the test releases the in-flight POST.

        Args:
            url: Requested URL.
            data: Submitted form data.

        Returns:
            The configured fake response.
        """
        self.post_count += 1
        self.post_url = url
        self.post_data = data
        self.started.set()
        await self.release.wait()
        return self.response


class RecordingWebsocket:
    """Websocket manager double that records reconnect closes."""

    def __init__(self) -> None:
        """Initialize an open websocket stand-in."""
        self.websocket: object | None = object()
        self.close_calls = 0

    async def request_reconnect(self) -> None:
        """Record a reconnect close."""
        self.close_calls += 1
        self.websocket = None


class RecordingSleep:
    """Injected sleep that records delays without waiting."""

    def __init__(self) -> None:
        """Initialize the recorded delay list."""
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        """Record one recovery backoff delay.

        Args:
            delay: Requested sleep duration.
        """
        self.delays.append(delay)


def _login_payload(
    *,
    expires_in: int = 3600,
    access_token: str = _LOGIN_ACCESS,
    refresh_token: str = _LOGIN_REFRESH,
    success: bool = True,
    error_message: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an assistedLogin GraphQL payload.

    Args:
        expires_in: Access-token lifetime in seconds.
        access_token: Issued access token.
        refresh_token: Issued refresh token.
        success: Whether assistedLogin reports success.
        error_message: Optional Carrier error message.
        data: Optional successful-token object override.

    Returns:
        GraphQL-shaped assistedLogin payload.
    """
    if not success:
        return {
            "assistedLogin": {
                "success": False,
                "status": "FAILED",
                "errorMessage": error_message,
            }
        }
    return {
        "assistedLogin": {
            "success": True,
            "status": "OK",
            "errorMessage": None,
            "data": data
            or {
                "token_type": "Bearer",
                "expires_in": expires_in,
                "access_token": access_token,
                "scope": "offline_access",
                "refresh_token": refresh_token,
            },
        }
    }


def _auth_error(status: int) -> ClientResponseError:
    """Build a ClientResponseError for a token-endpoint status.

    Args:
        status: HTTP status to expose.

    Returns:
        aiohttp client response error.
    """
    return ClientResponseError(
        request_info=None,  # type: ignore[arg-type]
        history=(),
        status=status,
        message="token rejected",
    )


def _connection(
    session: FakeSession,
    *,
    now: datetime = _FIXED_NOW,
    sleep: RecordingSleep | None = None,
) -> ApiConnectionGraphql:
    """Create a connection with a deterministic clock and optional sleep.

    Args:
        session: Fake aiohttp session.
        now: Fixed clock reading.
        sleep: Optional injected recovery sleep.

    Returns:
        Configured API connection.
    """
    return ApiConnectionGraphql(
        username=_SECRET_USERNAME,
        password=_SECRET_PASSWORD,
        client_session=cast("ClientSession", session),
        time_fn=lambda: now,
        sleep_fn=sleep,
    )


def _install_pair(
    connection: ApiConnectionGraphql,
    *,
    access_token: str = _LOGIN_ACCESS,
    refresh_token: str = _LOGIN_REFRESH,
    expires_in: int = 3600,
    now: datetime = _FIXED_NOW,
    source: str = "login",
) -> TokenPair:
    """Commit a validated pair through the sole writer.

    Args:
        connection: Connection under test.
        access_token: Access token to install.
        refresh_token: Refresh token to install.
        expires_in: Lifetime in seconds.
        now: Clock reading used for expiry.
        source: TokenPair source label.

    Returns:
        The installed pair object.
    """
    connection._install_token_pair(
        TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="Bearer",
            expires_at=now + timedelta(seconds=expires_in),
            obtained_at=now,
            source=source,  # type: ignore[arg-type]
            generation=0,
        )
    )
    installed = connection._pair
    assert installed is not None
    return installed


def _spy_writer(connection: ApiConnectionGraphql) -> list[TokenPair]:
    """Wrap the sole token writer and record calls.

    Args:
        connection: Connection under test.

    Returns:
        List that receives every pair passed to the writer.
    """
    writer_calls: list[TokenPair] = []
    real_install = connection._install_token_pair

    def spy_install(pair: TokenPair) -> None:
        """Record writer calls.

        Args:
            pair: Pair that would be installed.
        """
        writer_calls.append(pair)
        real_install(pair)

    connection._install_token_pair = spy_install  # type: ignore[method-assign]
    return writer_calls


class ScriptedLogin:
    """assistedLogin double that yields scripted results in order."""

    def __init__(self, *results: dict[str, Any] | BaseException) -> None:
        """Store scripted login results.

        Args:
            results: Payloads or exceptions to yield in order.
        """
        self._results = list(results)
        self.calls = 0

    async def __call__(self) -> dict[str, Any]:
        """Return the next scripted payload or raise the next error.

        Returns:
            Next assistedLogin payload.

        Raises:
            BaseException: The next scripted login failure.
            AssertionError: If more logins are requested than scripted.
        """
        self.calls += 1
        if not self._results:
            raise AssertionError("unexpected extra assistedLogin attempt")
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.mark.asyncio
async def test_refresh_success_performs_no_assisted_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refresh success installs the new pair and never calls assistedLogin."""
    session = CountingSession()
    connection = _connection(session)
    _install_pair(connection, expires_in=-1)
    login = ScriptedLogin(_login_payload())
    monkeypatch.setattr(connection, "_execute_assisted_login", login)

    await connection.check_auth_expiration()

    assert login.calls == 0
    assert len(session.posts) == 1
    assert connection.access_token == "new-access"
    assert connection.refresh_token == "new-refresh"
    assert connection._pair is not None
    assert connection._pair.source == "refresh"
    assert connection._state is TokenSessionState.ACTIVE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "payload"),
    [
        (400, {"error": "invalid_grant"}),
        (401, {}),
        (403, {}),
    ],
)
async def test_invalid_grant_or_token_auth_recovers_once(
    status: int,
    payload: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """invalid_grant and token-endpoint 401/403 do one refresh then one login."""
    session = CountingSession()
    session.response = FakeResponse(payload, status_error=_auth_error(status))
    sleep = RecordingSleep()
    connection = _connection(session, sleep=sleep)
    _install_pair(connection, expires_in=-1)
    login = ScriptedLogin(
        _login_payload(access_token="recovered-access", refresh_token="recovered-refresh")
    )
    monkeypatch.setattr(connection, "_execute_assisted_login", login)

    await connection.check_auth_expiration()

    assert len(session.posts) == 1
    assert login.calls == 1
    assert sleep.delays == []
    assert connection.access_token == "recovered-access"
    assert connection.refresh_token == "recovered-refresh"
    assert connection._pair is not None
    assert connection._pair.source == "recovery"
    assert connection._state is TokenSessionState.ACTIVE


@pytest.mark.asyncio
async def test_transient_login_then_success_uses_one_second_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One transient login failure then success uses two attempts and 1s."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    sleep = RecordingSleep()
    connection = _connection(session, sleep=sleep)
    _install_pair(connection, expires_in=-1)
    login = ScriptedLogin(
        ClientConnectionError("login transport failed"),
        _login_payload(access_token="recovered-access", refresh_token="recovered-refresh"),
    )
    monkeypatch.setattr(connection, "_execute_assisted_login", login)

    await connection.check_auth_expiration()

    assert login.calls == 2
    assert sleep.delays == [1.0]
    assert connection.access_token == "recovered-access"


@pytest.mark.asyncio
async def test_two_transient_logins_then_success_use_one_and_three_second_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two transient login failures then success use three attempts and [1, 3]."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    sleep = RecordingSleep()
    connection = _connection(session, sleep=sleep)
    _install_pair(connection, expires_in=-1)
    login = ScriptedLogin(
        ClientConnectionError("login transport failed"),
        ClientConnectionError("login transport failed again"),
        _login_payload(access_token="recovered-access", refresh_token="recovered-refresh"),
    )
    monkeypatch.setattr(connection, "_execute_assisted_login", login)

    await connection.check_auth_expiration()

    assert login.calls == 3
    assert sleep.delays == [1.0, 3.0]
    assert connection.access_token == "recovered-access"


@pytest.mark.asyncio
async def test_three_transient_logins_are_retryable_and_skip_refresh_next_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three transient failures keep the pair and skip the next refresh POST."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    sleep = RecordingSleep()
    connection = _connection(session, sleep=sleep)
    original = _install_pair(connection, expires_in=-1)
    writer_calls = _spy_writer(connection)
    login = ScriptedLogin(
        ClientConnectionError("login 1"),
        ClientConnectionError("login 2"),
        ClientConnectionError("login 3"),
        _login_payload(access_token="later-access", refresh_token="later-refresh"),
    )
    monkeypatch.setattr(connection, "_execute_assisted_login", login)

    with pytest.raises(CarrierApiTokenRefreshError):
        await connection.check_auth_expiration()

    assert login.calls == 3
    assert sleep.delays == [1.0, 3.0]
    assert connection._pair is original
    assert writer_calls == []
    assert connection.access_token == _LOGIN_ACCESS
    assert connection.refresh_token == _LOGIN_REFRESH
    assert connection._state is not TokenSessionState.AUTH_FAILED

    sleep.delays.clear()
    await connection.check_auth_expiration()

    assert len(session.posts) == 1
    assert login.calls == 4
    assert sleep.delays == []
    assert connection.access_token == "later-access"
    assert connection._state is TokenSessionState.ACTIVE


@pytest.mark.asyncio
async def test_explicit_credential_rejection_stops_without_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AssistedLogin success=false raises AuthError after one attempt."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    sleep = RecordingSleep()
    connection = _connection(session, sleep=sleep)
    original = _install_pair(connection, expires_in=-1)
    login = ScriptedLogin(_login_payload(success=False, error_message="bad password"))
    monkeypatch.setattr(connection, "_execute_assisted_login", login)

    with pytest.raises(CarrierApiAuthError) as error:
        await connection.check_auth_expiration()

    assert error.value.reason == "login_failed"
    assert login.calls == 1
    assert sleep.delays == []
    assert connection._pair is original
    assert connection._state is TokenSessionState.AUTH_FAILED


@pytest.mark.asyncio
async def test_malformed_success_payload_is_retryable_then_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed successful-token payloads retry and do not start reauth."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    sleep = RecordingSleep()
    connection = _connection(session, sleep=sleep)
    _install_pair(connection, expires_in=-1)
    login = ScriptedLogin(
        _login_payload(data={"access_token": "only-access"}),
        _login_payload(access_token="recovered-access", refresh_token="recovered-refresh"),
    )
    monkeypatch.setattr(connection, "_execute_assisted_login", login)

    await connection.check_auth_expiration()

    assert login.calls == 2
    assert sleep.delays == [1.0]
    assert connection.access_token == "recovered-access"


@pytest.mark.asyncio
async def test_invalid_client_never_calls_assisted_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """invalid_client stays a retryable refresh error with no login fallback."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_client"}, status_error=_auth_error(400))
    sleep = RecordingSleep()
    connection = _connection(session, sleep=sleep)
    original = _install_pair(connection, expires_in=-1)
    login = ScriptedLogin(_login_payload())
    monkeypatch.setattr(connection, "_execute_assisted_login", login)

    with pytest.raises(CarrierApiTokenRefreshError):
        await connection.refresh_auth_token()

    assert login.calls == 0
    assert sleep.delays == []
    assert connection._pair is original
    assert connection._suppressed_refresh_fp is None


@pytest.mark.asyncio
async def test_unauthorized_client_never_calls_assisted_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """unauthorized_client is classified with invalid_client and never logs in."""
    session = CountingSession()
    session.response = FakeResponse({"error": "unauthorized_client"}, status_error=_auth_error(400))
    connection = _connection(session)
    _install_pair(connection, expires_in=-1)
    login = ScriptedLogin(_login_payload())
    monkeypatch.setattr(connection, "_execute_assisted_login", login)

    with pytest.raises(CarrierApiTokenRefreshError):
        await connection.refresh_auth_token()

    assert login.calls == 0


@pytest.mark.asyncio
async def test_concurrent_expiry_callers_share_one_refresh_and_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent expiry callers single-flight one refresh and one recovery."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    sleep = RecordingSleep()
    connection = _connection(session, sleep=sleep)
    _install_pair(connection, expires_in=-1)
    login = ScriptedLogin(
        _login_payload(access_token="recovered-access", refresh_token="recovered-refresh")
    )
    monkeypatch.setattr(connection, "_execute_assisted_login", login)

    await asyncio.gather(connection.check_auth_expiration(), connection.check_auth_expiration())

    assert len(session.posts) == 1
    assert login.calls == 1
    assert connection.access_token == "recovered-access"


@pytest.mark.asyncio
async def test_two_expiry_checks_share_one_successful_refresh() -> None:
    """Two expiry checks single-flight onto one successful token POST."""
    session = CountingSession()
    connection = _connection(session)
    _install_pair(connection, expires_in=-1)

    await asyncio.gather(connection.check_auth_expiration(), connection.check_auth_expiration())

    assert len(session.posts) == 1
    assert connection.access_token == "new-access"


@pytest.mark.asyncio
async def test_cleanup_generation_race_denies_late_refresh_commit() -> None:
    """An in-flight refresh POST cannot install after cleanup starts."""
    session = DelayedSession()
    connection = _connection(session)
    original = _install_pair(connection, expires_in=-1)
    writer_calls = _spy_writer(connection)
    task = asyncio.create_task(connection.check_auth_expiration())
    await session.started.wait()
    await connection.cleanup()
    session.release.set()
    await task

    assert writer_calls == []
    assert connection._pair is original
    assert connection.access_token == _LOGIN_ACCESS
    assert connection._closing is True


@pytest.mark.asyncio
async def test_successful_recovery_reconnects_websocket_only_after_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery reconnects the websocket only after the new pair is installed."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    connection = _connection(session)
    original = _install_pair(connection, expires_in=-1)
    websocket = RecordingWebsocket()
    connection.api_websocket = cast("Any", websocket)
    seen_none = False
    committed_before_reconnect: list[bool] = []

    class TrackingClient:
        """GraphQL client that watches published tokens during recovery login."""

        def __init__(self, **kwargs: Any) -> None:
            """Accept GraphQL client construction arguments."""

        async def __aenter__(self) -> Self:
            """Enter the fake session."""
            return self

        async def __aexit__(self, *args: object) -> None:
            """Exit the fake session."""

        async def execute(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            """Return a new pair while asserting the old token is still published.

            Returns:
                Successful assistedLogin payload.
            """
            nonlocal seen_none
            if connection.access_token is None:
                seen_none = True
            return _login_payload(
                access_token="recovered-access", refresh_token="recovered-refresh"
            )

    original_reconnect = websocket.request_reconnect

    async def track_reconnect() -> None:
        """Record whether the writer already committed before reconnect."""
        committed_before_reconnect.append(connection.access_token == "recovered-access")
        await original_reconnect()

    websocket.request_reconnect = track_reconnect  # type: ignore[method-assign]
    monkeypatch.setattr("carrier_api.api_connection_graphql.Client", TrackingClient)

    await connection.check_auth_expiration()

    assert seen_none is False
    assert committed_before_reconnect == [True]
    assert connection.access_token == "recovered-access"
    assert connection.ws_generation == original.generation + 1
    assert websocket.close_calls == 1


@pytest.mark.asyncio
async def test_failed_recovery_never_writes_tokens_or_reconnects_websocket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed recovery leaves the atomic pair and open websocket unchanged."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    sleep = RecordingSleep()
    connection = _connection(session, sleep=sleep)
    original = _install_pair(connection, expires_in=-1)
    writer_calls = _spy_writer(connection)
    websocket = RecordingWebsocket()
    connection.api_websocket = cast("Any", websocket)
    login = ScriptedLogin(
        ClientConnectionError("login 1"),
        ClientConnectionError("login 2"),
        ClientConnectionError("login 3"),
    )
    monkeypatch.setattr(connection, "_execute_assisted_login", login)

    with pytest.raises(CarrierApiTokenRefreshError):
        await connection.check_auth_expiration()

    assert writer_calls == []
    assert connection._pair is original
    assert connection.access_token == _LOGIN_ACCESS
    assert connection.refresh_token == _LOGIN_REFRESH
    assert connection.expires_at == original.expires_at
    assert connection.ws_generation == original.generation
    assert websocket.close_calls == 0
    assert websocket.websocket is not None


@pytest.mark.asyncio
async def test_credential_rejection_never_reconnects_websocket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credential rejection does not mutate tokens or disconnect websocket."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    connection = _connection(session)
    original = _install_pair(connection, expires_in=-1)
    websocket = RecordingWebsocket()
    connection.api_websocket = cast("Any", websocket)
    monkeypatch.setattr(
        connection,
        "_execute_assisted_login",
        ScriptedLogin(_login_payload(success=False)),
    )

    with pytest.raises(CarrierApiAuthError):
        await connection.check_auth_expiration()

    assert connection._pair is original
    assert websocket.close_calls == 0


@pytest.mark.asyncio
async def test_snapshot_uses_locked_locals_not_later_attribute() -> None:
    """Websocket connect uses snapshot locals, not a later attribute read."""
    session = CountingSession()
    connection = _connection(session)
    _install_pair(connection)
    snapshot = await connection.snapshot_websocket_auth()
    connection.access_token = "later-token"

    assert snapshot == (_LOGIN_ACCESS, 1)
    assert snapshot[0] != connection.access_token


@pytest.mark.asyncio
async def test_load_data_style_and_refresh_share_one_post() -> None:
    """Two locked auth checks at expiry share one refresh pair."""
    session = CountingSession()
    connection = _connection(session)
    _install_pair(connection, expires_in=-1)

    headers = await asyncio.gather(
        connection.check_auth_expiration(),
        connection.snapshot_websocket_auth(),
    )

    assert len(session.posts) == 1
    assert headers[0] == "Bearer new-access"
    assert headers[1] == ("new-access", connection.ws_generation)


@pytest.mark.asyncio
async def test_snapshot_after_recovery_uses_new_access_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After successful recovery, the next snapshot uses the new access token."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    connection = _connection(session)
    _install_pair(connection, expires_in=-1)
    monkeypatch.setattr(
        connection,
        "_execute_assisted_login",
        ScriptedLogin(
            _login_payload(access_token="recovered-access", refresh_token="recovered-refresh")
        ),
    )

    await connection.check_auth_expiration()
    access_token, generation = await connection.snapshot_websocket_auth()

    assert access_token == "recovered-access"
    assert generation == 2


@pytest.mark.asyncio
async def test_login_does_not_post_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful login does not POST a refresh grant."""
    session = CountingSession()
    connection = _connection(session)
    monkeypatch.setattr(
        "carrier_api.api_connection_graphql.Client",
        graphql_client_double(result=_login_payload()),
    )
    await connection.login()
    connection._now = lambda: _FIXED_NOW + timedelta(seconds=70)

    assert session.posts == []
    assert connection.access_token == _LOGIN_ACCESS


@pytest.mark.asyncio
async def test_temporarily_unavailable_stays_refresh_error() -> None:
    """Transient OAuth refresh errors stay TokenRefreshError with no login."""
    session = CountingSession()
    session.response = FakeResponse(
        {"error": "temporarily_unavailable"}, status_error=_auth_error(400)
    )
    connection = _connection(session)
    _install_pair(connection, expires_in=-1)

    with pytest.raises(CarrierApiTokenRefreshError):
        await connection.refresh_auth_token()

    assert len(session.posts) == 1


@pytest.mark.asyncio
async def test_invalid_grant_adversarial_payload_is_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Adversarial invalid_grant bodies never leak secrets into logs."""
    session = CountingSession()
    session.response = FakeResponse(
        {
            "error": "invalid_grant",
            "error_description": (
                f"password={_SECRET_PASSWORD} cookie={_SECRET_COOKIE} "
                f"Authorization={_SECRET_AUTHORIZATION} "
                f"access_token={_SECRET_ACCESS_TOKEN} "
                f"refresh_token={_SECRET_NEW_REFRESH_TOKEN}"
            ),
        },
        status_error=_auth_error(400),
        headers={"Authorization": _SECRET_AUTHORIZATION, "Cookie": _SECRET_COOKIE},
        raw_body=(
            f'{{"access_token":"{_SECRET_ACCESS_TOKEN}",'
            f'"refresh_token":"{_SECRET_NEW_REFRESH_TOKEN}"}}'
        ).encode(),
    )
    connection = _connection(session)
    _install_pair(connection, expires_in=-1)
    monkeypatch.setattr(
        connection,
        "_execute_assisted_login",
        ScriptedLogin(
            _login_payload(access_token="recovered-access", refresh_token="recovered-refresh")
        ),
    )
    caplog.set_level(WARNING, logger="carrier_api.api_connection_graphql")

    await connection.check_auth_expiration()

    _assert_logs_are_secret_safe(caplog)


@pytest.mark.asyncio
async def test_diagnostics_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Token-session diagnostics expose state and never include secrets."""
    session = CountingSession()
    connection = _connection(session)
    monkeypatch.setattr(
        "carrier_api.api_connection_graphql.Client",
        graphql_client_double(result=_login_payload()),
    )
    await connection.login()

    diagnostics = connection.token_session_diagnostics()

    assert diagnostics["state"] == "ACTIVE"
    assert diagnostics["generation"] == 1
    assert "early_refresh_canary" not in diagnostics
    assert "invalid_grant_recovery" not in diagnostics
    assert _SECRET_USERNAME not in str(diagnostics)
    assert _SECRET_PASSWORD not in str(diagnostics)
    assert _LOGIN_ACCESS not in str(diagnostics)
    assert _LOGIN_REFRESH not in str(diagnostics)


@pytest.mark.asyncio
async def test_recovery_transport_failure_keeps_no_invented_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single recovery transport failure raises ConnectionError and invents no token."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    sleep = RecordingSleep()
    connection = _connection(session, sleep=sleep)
    original = _install_pair(connection, expires_in=-1)
    monkeypatch.setattr(
        connection,
        "_execute_assisted_login",
        ScriptedLogin(
            ClientConnectionError("login 1"),
            ClientConnectionError("login 2"),
            ClientConnectionError("login 3"),
        ),
    )

    with pytest.raises(CarrierApiTokenRefreshError):
        await connection.check_auth_expiration()

    assert connection._pair is original
    assert connection.access_token == _LOGIN_ACCESS
    assert sleep.delays == [1.0, 3.0]


@pytest.mark.asyncio
async def test_auth_error_reason_defaults_to_none() -> None:
    """Optional AuthError.reason stays compatible when omitted."""
    error = CarrierApiAuthError("Carrier token refresh was rejected")

    assert error.reason is None


@pytest.mark.asyncio
async def test_unexpired_check_does_not_refresh() -> None:
    """An unexpired access token does not POST a refresh grant."""
    session = CountingSession()
    connection = _connection(session)
    _install_pair(connection)

    await connection.check_auth_expiration()

    assert session.posts == []
    assert connection.access_token == _LOGIN_ACCESS
