"""Deterministic oauthdiag.3 token-session, canary, recovery, and lock tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from logging import DEBUG, WARNING
from typing import Any, Self, cast
from unittest.mock import patch

from aiohttp import ClientConnectionError, ClientResponseError, ClientSession
import pytest

from carrier_api.api_connection_graphql import ApiConnectionGraphql, TokenPair, TokenSessionState
from carrier_api.errors import (
    CarrierApiAuthError,
    CarrierApiConnectionError,
    CarrierApiTokenRefreshError,
)

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
_FIXED_NOW = datetime(2026, 9, 21, 15, 0, tzinfo=UTC)


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


def _login_payload(
    *,
    expires_in: int = 3600,
    access_token: str = _LOGIN_ACCESS,
    refresh_token: str = _LOGIN_REFRESH,
    success: bool = True,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Build an assistedLogin GraphQL payload.

    Args:
        expires_in: Access-token lifetime in seconds.
        access_token: Issued access token.
        refresh_token: Issued refresh token.
        success: Whether assistedLogin reports success.
        error_message: Optional Carrier error message.

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
            "data": {
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
    early_refresh_canary: bool = False,
    invalid_grant_recovery: bool = False,
    canary_delay_seconds: float = 45.0,
    schedule_fn: Any = None,
    now: datetime = _FIXED_NOW,
) -> ApiConnectionGraphql:
    """Create a connection with a deterministic clock.

    Args:
        session: Fake aiohttp session.
        early_refresh_canary: Canary flag.
        invalid_grant_recovery: Recovery flag.
        canary_delay_seconds: Canary delay before clamping.
        schedule_fn: Optional background scheduler.
        now: Fixed clock reading.

    Returns:
        Configured API connection.
    """
    return ApiConnectionGraphql(
        username=_SECRET_USERNAME,
        password=_SECRET_PASSWORD,
        client_session=cast("ClientSession", session),
        early_refresh_canary=early_refresh_canary,
        invalid_grant_recovery=invalid_grant_recovery,
        canary_delay_seconds=canary_delay_seconds,
        schedule_fn=schedule_fn,
        time_fn=lambda: now,
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


async def _login(
    monkeypatch: pytest.MonkeyPatch,
    connection: ApiConnectionGraphql,
    payload: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> None:
    """Run login() against a GraphQL client double.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        connection: Connection under test.
        payload: Optional assistedLogin payload.
        error: Optional GraphQL execute error.
    """
    monkeypatch.setattr(
        "carrier_api.api_connection_graphql.Client",
        graphql_client_double(result=payload or _login_payload(), error=error),
    )
    await connection.login()


def _scheduler() -> tuple[Any, list[asyncio.Task[None]]]:
    """Return a schedule_fn that stores created tasks.

    Returns:
        Scheduler and the list of created tasks.
    """
    tasks: list[asyncio.Task[None]] = []

    def schedule(coro: Any) -> asyncio.Task[None]:
        """Create and record a background task.

        Args:
            coro: Coroutine to schedule.

        Returns:
            Created asyncio task.
        """
        task = asyncio.create_task(coro)
        tasks.append(task)
        return task

    return schedule, tasks


@pytest.mark.asyncio
async def test_f0_1_flags_off_login_does_not_post_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F0-1: flags off after login must not POST a canary refresh."""
    session = CountingSession()
    connection = _connection(session)
    await _login(monkeypatch, connection)
    connection._now = lambda: _FIXED_NOW + timedelta(seconds=70)

    assert connection._canary_task is None
    assert session.posts == []


@pytest.mark.asyncio
async def test_f0_2_flags_off_expiry_refreshes_once() -> None:
    """F0-2: flags off expiry refresh stays in the refresh bucket."""
    session = CountingSession()
    connection = _connection(session)
    _install_pair(connection, expires_in=-1)

    await connection.check_auth_expiration()

    assert len(session.posts) == 1
    assert connection.access_token == "new-access"
    assert connection._state is TokenSessionState.ACTIVE


@pytest.mark.asyncio
async def test_f0_3_flags_off_invalid_grant_raises_auth_error() -> None:
    """F0-3: flags off invalid_grant at expiry raises AuthError without a mixed pair."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    connection = _connection(session)
    original = _install_pair(connection, expires_in=-1)

    with pytest.raises(CarrierApiAuthError) as error:
        await connection.refresh_auth_token()

    assert error.value.reason == "invalid_grant"
    assert connection._pair is original
    assert connection.access_token == _LOGIN_ACCESS
    assert connection.refresh_token == _LOGIN_REFRESH


@pytest.mark.asyncio
async def test_f0_4_flags_off_temporarily_unavailable_is_refresh_error() -> None:
    """F0-4: flags off transient OAuth errors stay TokenRefreshError."""
    session = CountingSession()
    session.response = FakeResponse(
        {"error": "temporarily_unavailable"}, status_error=_auth_error(400)
    )
    connection = _connection(session)
    _install_pair(connection, expires_in=-1)

    with pytest.raises(CarrierApiTokenRefreshError):
        await connection.refresh_auth_token()


@pytest.mark.asyncio
async def test_c1_canary_success_installs_new_pair(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """C1: a successful canary commits a new pair and logs secret-safe success."""
    session = CountingSession()
    schedule, tasks = _scheduler()
    connection = _connection(session, early_refresh_canary=True, schedule_fn=schedule)
    caplog.set_level(DEBUG, logger="carrier_api.api_connection_graphql")

    async def instant_sleep(_delay: float) -> None:
        """Skip the canary delay.

        Args:
            _delay: Requested delay.
        """
        return

    with patch("carrier_api.api_connection_graphql.asyncio.sleep", instant_sleep):
        await _login(monkeypatch, connection)
        await tasks[0]

    assert len(session.posts) == 1
    assert connection.access_token == "new-access"
    assert connection.refresh_token == "new-refresh"
    assert connection._pair is not None
    assert connection._pair.source == "canary"
    assert connection.ws_generation == 2
    assert "event=canary_finished" in caplog.text
    assert "outcome=success" in caplog.text
    _assert_logs_are_secret_safe(caplog)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "payload", "expected_state", "expected_outcome"),
    [
        (400, {"error": "invalid_grant"}, TokenSessionState.ACTIVE_SUSPECT, "invalid_grant"),
        (401, {}, TokenSessionState.ACTIVE_SUSPECT, "unauthorized"),
        (
            400,
            {"error": "temporarily_unavailable"},
            TokenSessionState.ACTIVE,
            "transient",
        ),
        (400, {"error": "invalid_client"}, TokenSessionState.ACTIVE, "invalid_client"),
        (
            400,
            {"error": "unauthorized_client"},
            TokenSessionState.ACTIVE,
            "invalid_client",
        ),
    ],
)
async def test_canary_failures_do_not_write_tokens(
    status: int,
    payload: object,
    expected_state: TokenSessionState,
    expected_outcome: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """C2/C3/C4/C10/C11/SW1: failed canaries never call the writer."""
    session = CountingSession()
    session.response = FakeResponse(payload, status_error=_auth_error(status))
    connection = _connection(session, early_refresh_canary=True)
    original = _install_pair(connection)
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
    caplog.set_level(WARNING, logger="carrier_api.api_connection_graphql")

    await connection.refresh_auth_token(purpose="canary")

    assert connection._pair is original
    assert writer_calls == []
    assert connection.access_token == _LOGIN_ACCESS
    assert connection.refresh_token == _LOGIN_REFRESH
    assert connection.expires_at == original.expires_at
    assert connection._state is expected_state
    assert connection.ws_generation == original.generation
    assert connection.api_websocket is None
    assert f"outcome={expected_outcome}" in caplog.text
    if expected_outcome == "invalid_client":
        assert "recovery_eligible=False" in caplog.text
        assert connection._pre_expiry_task is None
        assert connection._suppressed_refresh_fp is None
    _assert_logs_are_secret_safe(caplog)


@pytest.mark.asyncio
async def test_c5_canary_transport_error_leaves_pair_identical() -> None:
    """C5: canary transport errors do not write tokens or mark suspect."""

    class FailingSession(CountingSession):
        """Session that fails the refresh POST."""

        async def post(self, url: str, data: dict[str, Any]) -> FakeResponse:
            """Raise a transport error.

            Args:
                url: Requested URL.
                data: Submitted form data.

            Raises:
                ClientConnectionError: Always raised.
            """
            self.posts.append(data)
            raise ClientConnectionError("canary transport failed")

    session = FailingSession()
    connection = _connection(session, early_refresh_canary=True)
    original = _install_pair(connection)

    await connection.refresh_auth_token(purpose="canary")

    assert connection._pair is original
    assert connection._state is TokenSessionState.ACTIVE


@pytest.mark.asyncio
async def test_c6_short_ttl_skips_canary(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """C6: login with a short TTL skips the canary instead of POSTing."""
    session = CountingSession()
    schedule, _tasks = _scheduler()
    connection = _connection(session, early_refresh_canary=True, schedule_fn=schedule)
    caplog.set_level(WARNING, logger="carrier_api.api_connection_graphql")
    await _login(monkeypatch, connection, _login_payload(expires_in=90))

    assert connection._canary_task is None
    assert session.posts == []
    assert "event=canary_skipped" in caplog.text
    assert "outcome=skipped_ttl" in caplog.text
    _assert_logs_are_secret_safe(caplog)


@pytest.mark.asyncio
async def test_c7_u1_cleanup_cancels_sleeping_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C7/U1: cleanup cancels and awaits a sleeping canary with no POST."""
    session = CountingSession()
    schedule, tasks = _scheduler()
    connection = _connection(session, early_refresh_canary=True, schedule_fn=schedule)
    started = asyncio.Event()
    blocked = asyncio.Event()

    async def hang_sleep(_delay: float) -> None:
        """Block the canary until cleanup cancels it.

        Args:
            _delay: Requested delay.

        Raises:
            CancelledError: When cleanup cancels the canary task.
        """
        started.set()
        await blocked.wait()

    with patch("carrier_api.api_connection_graphql.asyncio.sleep", hang_sleep):
        await _login(monkeypatch, connection)
        original = connection._pair
        await started.wait()
        assert tasks[0] is connection._canary_task
        await connection.cleanup()

    assert tasks[0].cancelled() or tasks[0].done()
    assert connection._closing is True
    assert connection._canary_task is None
    assert connection._pair is original
    assert session.posts == []
    assert session.closed is True


@pytest.mark.asyncio
async def test_c8_successful_canary_does_not_refresh_again() -> None:
    """C8: after a successful canary, an unexpired check does not refresh."""
    session = CountingSession()
    connection = _connection(session, early_refresh_canary=True)
    _install_pair(connection)

    await connection.refresh_auth_token(purpose="canary")
    await connection.check_auth_expiration()

    assert len(session.posts) == 1


@pytest.mark.asyncio
async def test_c9_canary_adversarial_payload_is_secret_safe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """C9: adversarial canary 400 bodies never leak secrets into logs."""
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
    connection = _connection(session, early_refresh_canary=True)
    _install_pair(connection)
    caplog.set_level(WARNING, logger="carrier_api.api_connection_graphql")

    await connection.refresh_auth_token(purpose="canary")

    _assert_logs_are_secret_safe(caplog)


@pytest.mark.asyncio
async def test_r1_pre_expiry_login_commits_new_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1: scheduled pre-expiry login commits a new pair and requests WS reconnect."""
    session = CountingSession()
    schedule, tasks = _scheduler()
    connection = _connection(session, invalid_grant_recovery=True, schedule_fn=schedule)
    _install_pair(connection)
    connection.api_websocket = cast("Any", RecordingWebsocket())
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    monkeypatch.setattr(
        "carrier_api.api_connection_graphql.Client",
        graphql_client_double(
            result=_login_payload(
                access_token="recovered-access", refresh_token="recovered-refresh"
            )
        ),
    )

    async def instant_sleep(_delay: float) -> None:
        """Skip the pre-expiry delay.

        Args:
            _delay: Requested delay.
        """
        return

    with patch("carrier_api.api_connection_graphql.asyncio.sleep", instant_sleep):
        await connection.refresh_auth_token(purpose="canary")
        assert connection._state is TokenSessionState.ACTIVE_SUSPECT
        await tasks[0]

    assert connection.access_token == "recovered-access"
    assert connection.refresh_token == "recovered-refresh"
    assert connection._state is TokenSessionState.ACTIVE
    assert connection.api_websocket.close_calls == 1


@pytest.mark.asyncio
async def test_r2_expired_invalid_grant_recovers_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2: expiry invalid_grant with recovery on logs in immediately."""
    session = CountingSession()
    connection = _connection(session, invalid_grant_recovery=True)
    _install_pair(connection, expires_in=-1)
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    monkeypatch.setattr(
        "carrier_api.api_connection_graphql.Client",
        graphql_client_double(
            result=_login_payload(
                access_token="recovered-access", refresh_token="recovered-refresh"
            )
        ),
    )

    await connection.check_auth_expiration()

    assert connection.access_token == "recovered-access"
    assert connection._state is TokenSessionState.ACTIVE


@pytest.mark.asyncio
async def test_r3_recovery_login_failure_is_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3: recovery assistedLogin success=false raises login_failed."""
    session = CountingSession()
    connection = _connection(session, invalid_grant_recovery=True)
    _install_pair(connection, expires_in=-1)
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    monkeypatch.setattr(
        "carrier_api.api_connection_graphql.Client",
        graphql_client_double(result=_login_payload(success=False, error_message="bad password")),
    )

    with pytest.raises(CarrierApiAuthError) as error:
        await connection.check_auth_expiration()

    assert error.value.reason == "login_failed"


@pytest.mark.asyncio
async def test_r4_recovery_login_transport_keeps_no_invented_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R4: recovery transport failure raises ConnectionError and invents no token."""
    session = CountingSession()
    connection = _connection(session, invalid_grant_recovery=True)
    original = _install_pair(connection, expires_in=-1)
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    monkeypatch.setattr(
        "carrier_api.api_connection_graphql.Client",
        graphql_client_double(error=ClientConnectionError("login transport failed")),
    )

    with pytest.raises(CarrierApiConnectionError):
        await connection.check_auth_expiration()

    assert connection._pair is original
    assert connection.access_token == _LOGIN_ACCESS


@pytest.mark.asyncio
async def test_r5_suppressed_refresh_fingerprint_does_not_post_again() -> None:
    """R5: a permanently rejected refresh fingerprint is not POSTed again."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    connection = _connection(session)
    _install_pair(connection, expires_in=-1)

    with pytest.raises(CarrierApiAuthError):
        await connection.refresh_auth_token()
    with pytest.raises(CarrierApiAuthError):
        await connection.check_auth_expiration()

    assert len(session.posts) == 1


@pytest.mark.asyncio
async def test_r6_ic1_invalid_client_does_not_recover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R6/IC1: expiry invalid_client raises TokenRefreshError and never logs in."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_client"}, status_error=_auth_error(400))
    connection = _connection(session, invalid_grant_recovery=True)
    _install_pair(connection, expires_in=-1)
    login_called = False

    async def fail_if_login() -> dict[str, Any]:
        """Fail if recovery tries assistedLogin.

        Raises:
            AssertionError: If login is attempted.
        """
        nonlocal login_called
        login_called = True
        raise AssertionError("recovery login must not run for invalid_client")

    monkeypatch.setattr(connection, "_execute_assisted_login", fail_if_login)

    with pytest.raises(CarrierApiTokenRefreshError):
        await connection.refresh_auth_token()

    assert login_called is False
    assert connection._state is not TokenSessionState.ACTIVE_SUSPECT


@pytest.mark.asyncio
async def test_r7_recovery_attempts_are_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R7: a third assistedLogin is not attempted after two recovery tries."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    connection = _connection(session, invalid_grant_recovery=True)
    _install_pair(connection, expires_in=-1)
    connection._recovery_attempts = 2
    login_calls = 0

    async def count_login() -> dict[str, Any]:
        """Count unexpected recovery logins.

        Returns:
            Empty payload; the attempt cap should prevent this call.
        """
        nonlocal login_calls
        login_calls += 1
        return _login_payload()

    monkeypatch.setattr(connection, "_execute_assisted_login", count_login)

    with pytest.raises(CarrierApiAuthError):
        await connection.check_auth_expiration()

    assert login_calls == 0


@pytest.mark.asyncio
async def test_x1_two_expiry_checks_share_one_refresh() -> None:
    """X1: two expiry checks single-flight onto one token POST."""
    session = CountingSession()
    connection = _connection(session)
    _install_pair(connection, expires_in=-1)

    await asyncio.gather(connection.check_auth_expiration(), connection.check_auth_expiration())

    assert len(session.posts) == 1
    assert connection.access_token == "new-access"


@pytest.mark.asyncio
async def test_x2_waiter_does_not_start_second_refresh_during_canary() -> None:
    """X2: a check_auth_expiration waiter does not start a second refresh."""
    session = DelayedSession()
    connection = _connection(session, early_refresh_canary=True)
    _install_pair(connection)
    canary = asyncio.create_task(connection.refresh_auth_token(purpose="canary"))
    await session.started.wait()
    waiter = asyncio.create_task(connection.check_auth_expiration())
    await asyncio.sleep(0)
    session.release.set()
    await asyncio.gather(canary, waiter)

    assert session.post_count == 1


@pytest.mark.asyncio
async def test_x3_canary_success_requests_ws_reconnect() -> None:
    """X3: a successful canary bumps generation and closes the open socket."""
    session = CountingSession()
    connection = _connection(session, early_refresh_canary=True)
    original = _install_pair(connection)
    websocket = RecordingWebsocket()
    connection.api_websocket = cast("Any", websocket)

    await connection.refresh_auth_token(purpose="canary")

    assert connection.ws_generation == original.generation + 1
    assert websocket.close_calls == 1
    assert connection.access_token == "new-access"


@pytest.mark.asyncio
async def test_x4_failed_canary_leaves_socket_open() -> None:
    """X4: canary invalid_grant leaves generation and the socket unchanged."""
    session = CountingSession()
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    connection = _connection(session, early_refresh_canary=True)
    original = _install_pair(connection)
    websocket = RecordingWebsocket()
    connection.api_websocket = cast("Any", websocket)

    await connection.refresh_auth_token(purpose="canary")

    assert connection.ws_generation == original.generation
    assert websocket.close_calls == 0
    assert websocket.websocket is not None
    assert connection.access_token == _LOGIN_ACCESS


@pytest.mark.asyncio
async def test_x5_ws1_snapshot_uses_locked_locals_not_later_attribute() -> None:
    """X5/WS1: websocket connect uses snapshot locals, not a later attribute read."""
    session = CountingSession()
    connection = _connection(session)
    _install_pair(connection)
    snapshot = await connection.snapshot_websocket_auth()
    connection.access_token = "later-token"

    assert snapshot == (_LOGIN_ACCESS, 1)
    assert snapshot[0] != connection.access_token


@pytest.mark.asyncio
async def test_x6_recovery_login_never_clears_access_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """X6: recovery login commits then reconnects without a None access token."""
    session = CountingSession()
    connection = _connection(session, invalid_grant_recovery=True)
    _install_pair(connection, expires_in=-1)
    websocket = RecordingWebsocket()
    connection.api_websocket = cast("Any", websocket)
    session.response = FakeResponse({"error": "invalid_grant"}, status_error=_auth_error(400))
    seen_none = False

    class TrackingClient:
        """GraphQL client that watches access_token during recovery login."""

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

    monkeypatch.setattr("carrier_api.api_connection_graphql.Client", TrackingClient)

    await connection.check_auth_expiration()

    assert seen_none is False
    assert connection.access_token == "recovered-access"
    assert websocket.close_calls == 1


@pytest.mark.asyncio
async def test_x8_load_data_style_and_refresh_share_one_post() -> None:
    """X8: two locked auth checks at expiry share one refresh pair."""
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
async def test_x9_snapshot_after_canary_uses_new_access_token() -> None:
    """X9: after a successful canary, the next snapshot uses the new access token."""
    session = CountingSession()
    connection = _connection(session, early_refresh_canary=True)
    _install_pair(connection)

    await connection.refresh_auth_token(purpose="canary")
    access_token, generation = await connection.snapshot_websocket_auth()

    assert access_token == "new-access"
    assert generation == 2


@pytest.mark.asyncio
async def test_u2_commit_gate_denies_install_after_closing() -> None:
    """U2: an in-flight canary POST cannot install after `_closing` is set."""
    session = DelayedSession()
    connection = _connection(session, early_refresh_canary=True)
    original = _install_pair(connection)
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
    task = asyncio.create_task(connection.refresh_auth_token(purpose="canary"))
    await session.started.wait()
    connection._closing = True
    session.release.set()
    await task

    assert writer_calls == []
    assert connection._pair is original


@pytest.mark.asyncio
async def test_u3_flags_off_cleanup_denies_late_expiry_commit() -> None:
    """U3: flags-off expiry POST cannot commit after cleanup starts."""
    session = DelayedSession()
    connection = _connection(session)
    original = _install_pair(connection, expires_in=-1)
    task = asyncio.create_task(connection.check_auth_expiration())
    await session.started.wait()
    await connection.cleanup()
    session.release.set()
    await task

    assert connection._pair is original
    assert connection.access_token == _LOGIN_ACCESS
    assert connection._closing is True


@pytest.mark.asyncio
async def test_auth_error_reason_defaults_to_none() -> None:
    """Optional AuthError.reason stays compatible when omitted."""
    error = CarrierApiAuthError("Carrier token refresh was rejected")

    assert error.reason is None
