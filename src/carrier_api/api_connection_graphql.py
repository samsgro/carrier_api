"""GraphQL client for Carrier authentication, queries, and config updates."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from logging import DEBUG, INFO, WARNING, getLogger
from typing import Any, Literal

from aiohttp import ClientError, ClientResponseError, ClientSession
from gql import Client, GraphQLRequest, gql
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.exceptions import (
    TransportError as GraphqlTransportError,
    TransportQueryError,
    TransportServerError,
)
from graphql import GraphQLError

from .api_websocket import ApiWebsocket
from .config import Config
from .const import ActivityTypes, FanModes, HeatSourceTypes, SystemModes
from .energy import Energy
from .entry_level import EntryLevelSystem
from .errors import (
    CarrierApiAuthError,
    CarrierApiConnectionError,
    CarrierApiGraphqlError,
    CarrierApiTokenRefreshError,
)
from .oauth_refresh_diagnostics import _optional_raw_body, log_oauth_refresh_response
from .profile import Profile
from .status import Status
from .system import System

_LOGGER = getLogger(__name__)
GRAPHQL_EXECUTE_TIMEOUT_SECONDS = 60
CANARY_DELAY_MIN = 30.0
CANARY_DELAY_MAX = 60.0
PRE_EXPIRY_LEAD_SECONDS = 120.0
MAX_RECOVERY_ATTEMPTS = 2

_CONNECTION_ERRORS = (GraphqlTransportError, ClientError, TimeoutError, OSError)
_AUTH_HTTP_STATUSES = {401, 403}
_SESSION_EVENTS = frozenset(
    {
        "login_committed",
        "canary_scheduled",
        "canary_skipped",
        "canary_started",
        "canary_finished",
        "refresh_started",
        "refresh_finished",
        "refresh_suppressed",
        "recovery_scheduled",
        "recovery_started",
        "recovery_finished",
        "ws_reconnect_requested",
    }
)
_SESSION_OUTCOMES = frozenset(
    {
        "success",
        "invalid_grant",
        "unauthorized",
        "invalid_client",
        "transient",
        "skipped_ttl",
        "skipped_flag_off",
        "cancelled",
        "login_failed",
        "login_transient",
    }
)
_SESSION_WS_ACTIONS = frozenset({"none", "reconnect_requested", "left_connected"})
_SESSION_PURPOSES = frozenset({"login", "canary", "refresh", "pre_expiry", "recovery"})
ScheduleFn = Callable[[Awaitable[None]], asyncio.Task[None]]
TimeFn = Callable[[], datetime]
TokenPairSource = Literal["login", "refresh", "canary", "recovery"]


def _is_auth_transport_error(error: BaseException) -> bool:
    """Return whether a transport error represents Carrier auth rejection.

    Args:
        error: Exception raised by the GraphQL transport.

    Returns:
        ``True`` when the GraphQL endpoint rejected the request with an
        authentication-related HTTP status.
    """
    return isinstance(error, TransportServerError) and error.code in _AUTH_HTTP_STATUSES


async def _consume_oauth_refresh_response(
    response: Any,
) -> tuple[bytes | None, object | None, BaseException | None]:
    """Read OAuth refresh body bytes and JSON before status handling.

    aiohttp's ``ClientResponse.raise_for_status`` calls ``release()`` before
    raising, which can make a later ``json()`` or ``read()`` fail. Capture both
    first so exception classification and secret-safe hashing still work.

    Args:
        response: aiohttp-like response object.

    Returns:
        Raw body bytes when available, the decoded JSON value or ``None``, and
        any JSON-read error to re-raise after a successful status check.
    """
    raw_body = await _optional_raw_body(response)
    try:
        return raw_body, await response.json(), None
    except (ClientError, TimeoutError, OSError, TypeError, ValueError) as error:
        return raw_body, None, error


class TokenSessionState(StrEnum):
    """Explicit token-session states used for logs and tests."""

    NO_TOKENS = "NO_TOKENS"
    ACTIVE = "ACTIVE"
    CANARY_IN_FLIGHT = "CANARY_IN_FLIGHT"
    REFRESH_IN_FLIGHT = "REFRESH_IN_FLIGHT"
    ACTIVE_SUSPECT = "ACTIVE_SUSPECT"
    PRE_EXPIRY_LOGIN = "PRE_EXPIRY_LOGIN"
    RECOVERY_LOGIN = "RECOVERY_LOGIN"
    AUTH_FAILED = "AUTH_FAILED"


@dataclass(frozen=True)
class TokenPair:
    """Immutable OAuth token pair installed as one commit."""

    access_token: str
    refresh_token: str
    token_type: str
    expires_at: datetime
    obtained_at: datetime
    source: TokenPairSource
    generation: int

    def authorization(self) -> str:
        """Return the Authorization header value for this pair.

        Returns:
            Combined token type and access token.
        """
        return f"{self.token_type} {self.access_token}"

    def seconds_until_expiry(self, now: datetime) -> float:
        """Return seconds remaining until this pair expires.

        Args:
            now: Clock reading used for the remaining-lifetime calculation.

        Returns:
            Signed seconds until ``expires_at``. Negative when already expired.
        """
        return (self.expires_at - now).total_seconds()


class ApiConnectionGraphql:
    """Async Carrier GraphQL API connection with token and websocket support."""

    def __init__(
        self,
        username: str,
        password: str,
        client_session: ClientSession | None = None,
        *,
        early_refresh_canary: bool = False,
        invalid_grant_recovery: bool = False,
        canary_delay_seconds: float = 45.0,
        schedule_fn: ScheduleFn | None = None,
        time_fn: TimeFn | None = None,
    ) -> None:
        """Create a Carrier GraphQL API connection.

        Args:
            username: Carrier account username.
            password: Carrier account password.
            client_session: Optional aiohttp session to reuse for token refresh
                and websocket operations. A new session is created when omitted.
            early_refresh_canary: When True, schedule one diagnostic refresh
                after a successful login. Default off.
            invalid_grant_recovery: When True, recover from a permanent refresh
                rejection by calling assistedLogin. Default off.
            canary_delay_seconds: Delay before the diagnostic canary. Clamped to
                ``[30, 60]``.
            schedule_fn: Optional Home Assistant (or test) scheduler. When
                omitted, no background canary or pre-expiry work is started.
            time_fn: Optional clock used by token lifetime checks and tests.
        """
        self.username = username
        self.password = password
        if client_session is None:
            self.api_session = ClientSession(raise_for_status=False)
        else:
            self.api_session = client_session
        self.expires_at: datetime | None = None
        self.refresh_token: str | None = None
        self.token_type: str | None = None
        self.access_token: str | None = None
        self.api_websocket: ApiWebsocket | None = None
        self._token_lock = asyncio.Lock()
        self._token_lock_owner: asyncio.Task[Any] | None = None
        self._pair: TokenPair | None = None
        self._state = TokenSessionState.NO_TOKENS
        self._suppressed_refresh_fp: str | None = None
        self._token_generation = 0
        self._closing = False
        self._canary_task: asyncio.Task[None] | None = None
        self._pre_expiry_task: asyncio.Task[None] | None = None
        self._recovery_attempts = 0
        self._canary_ran_for_generation: int | None = None
        self.reconnect_required = False
        self.early_refresh_canary = early_refresh_canary
        self.invalid_grant_recovery = invalid_grant_recovery
        self.canary_delay_seconds = min(
            CANARY_DELAY_MAX, max(CANARY_DELAY_MIN, canary_delay_seconds)
        )
        self._schedule = schedule_fn
        self._now = time_fn or (lambda: datetime.now(UTC))
        self._last_session_event: str | None = None
        self._last_session_outcome: str | None = None
        self._last_refresh_fp12: str | None = None

    @property
    def ws_generation(self) -> int:
        """Return the committed token generation used by websocket listeners.

        Returns:
            Installed pair generation, or ``0`` when no pair is committed.
        """
        return 0 if self._pair is None else self._pair.generation

    def token_session_diagnostics(self) -> dict[str, Any]:
        """Return a redacted token-session snapshot for Home Assistant diagnostics.

        Returns:
            Allowlisted flag, state, generation, and last-event fields. Tokens
            are never included.
        """
        now = self._now()
        seconds_until_expiry = None if self._pair is None else self._pair.seconds_until_expiry(now)
        return {
            "early_refresh_canary": self.early_refresh_canary,
            "invalid_grant_recovery": self.invalid_grant_recovery,
            "state": self._state.value,
            "generation": self.ws_generation,
            "seconds_until_expiry": seconds_until_expiry,
            "last_event": self._last_session_event,
            "last_outcome": self._last_session_outcome,
            "refresh_fp12": self._last_refresh_fp12,
        }

    async def cleanup(self) -> None:
        """Cancel connection-owned OAuth tasks and close the HTTP session."""
        self._closing = True
        tasks = [task for task in (self._canary_task, self._pre_expiry_task) if task is not None]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._canary_task = None
        self._pre_expiry_task = None
        try:
            await self.api_session.close()
        except (ClientError, TimeoutError, OSError) as error:
            raise CarrierApiConnectionError("Carrier API session cleanup failed") from error

    async def login(self) -> None:
        """Authenticate with Carrier and initialize websocket support.

        Raises:
            CarrierApiAuthError: If the assisted login mutation reports an unsuccessful
                authentication result.
        """
        async with self._held_token_lock():
            await self._login_locked(purpose="login")
        if self._token_lock_owner is not asyncio.current_task():
            await self._close_websocket_if_reconnect_required()

    async def check_auth_expiration(self) -> str | None:
        """Ensure the connection has a valid access token before API use.

        Returns:
            Authorization header for the committed pair when tokens exist.
        """
        async with self._held_token_lock():
            await self._ensure_tokens_locked()
            header = self._authorization_header()
        await self._close_websocket_if_reconnect_required()
        return header

    async def snapshot_websocket_auth(self) -> tuple[str, int]:
        """Return a locked access-token and generation snapshot for websocket connect.

        Returns:
            Access token and generation captured together under the token lock.

        Raises:
            CarrierApiTokenRefreshError: If the session is closing or has no pair
                after token ensure.
        """
        async with self._held_token_lock():
            await self._ensure_tokens_locked()
            if self._closing or self._pair is None:
                raise CarrierApiTokenRefreshError("Carrier websocket auth snapshot unavailable")
            return self._pair.access_token, self._pair.generation

    async def refresh_auth_token(self, *, purpose: str = "refresh") -> None:
        """Refresh the OAuth access token using the stored refresh token.

        Args:
            purpose: Why this refresh is running. ``canary`` swallows failures
                instead of raising into Home Assistant.

        Raises:
            CarrierApiAuthError: If Carrier rejects the refresh token as invalid
                or unauthorized and this is not a canary.
            CarrierApiTokenRefreshError: If token refresh fails before Carrier
                returns a valid OAuth response and this is not a canary.
        """
        async with self._held_token_lock():
            await self._refresh_locked(purpose=purpose)
        if self._token_lock_owner is not asyncio.current_task():
            await self._close_websocket_if_reconnect_required()

    @asynccontextmanager
    async def _held_token_lock(self) -> AsyncIterator[None]:
        """Acquire the token lock, reentrantly for the current task.

        Yields:
            Nothing. The lock is held until the context exits.
        """
        task = asyncio.current_task()
        if self._token_lock_owner is task:
            yield
            return
        await self._token_lock.acquire()
        self._token_lock_owner = task
        try:
            yield
        finally:
            self._token_lock_owner = None
            self._token_lock.release()

    def _authorization_header(self) -> str | None:
        """Return the published Authorization header, if tokens exist.

        Returns:
            Header value from the committed pair or compatibility attributes.
        """
        if self._pair is not None:
            return self._pair.authorization()
        if self.token_type is None or self.access_token is None:
            return None
        return f"{self.token_type} {self.access_token}"

    def _seconds_until_expiry(self, now: datetime) -> float:
        """Return remaining access-token lifetime from pair or compat attributes.

        Args:
            now: Clock reading used for the remaining-lifetime calculation.

        Returns:
            Signed seconds until expiry. Negative when expired or unset.
        """
        if self._pair is not None:
            return self._pair.seconds_until_expiry(now)
        if self.expires_at is None:
            return -1.0
        return (self.expires_at - now).total_seconds()

    def _refresh_fingerprint(self, refresh_token: str) -> str:
        """Return the SHA-256 fingerprint of a refresh token.

        Args:
            refresh_token: Refresh token to hash. The token itself is never
                logged.

        Returns:
            Hex digest of the refresh token.
        """
        return sha256(refresh_token.encode("utf-8")).hexdigest()

    def _commit_allowed(self, pair: TokenPair | None, generation_at_request: int) -> bool:
        """Return whether a post-HTTP token install may proceed.

        Args:
            pair: Pair object captured before the HTTP round-trip, or ``None``
                when only compatibility attributes were present.
            generation_at_request: Generation captured before the HTTP call.

        Returns:
            ``True`` when the session is still open and the captured pair is
            still the published pair.
        """
        if self._closing:
            return False
        if pair is None:
            return self._pair is None and self._token_generation == generation_at_request
        return self._pair is pair and pair.generation == generation_at_request

    def _install_token_pair(self, pair: TokenPair) -> None:
        """Install a validated token pair. This is the only token-field writer.

        Args:
            pair: Fully validated pair. ``generation`` is stamped here.
        """
        self._token_generation += 1
        installed = TokenPair(
            access_token=pair.access_token,
            refresh_token=pair.refresh_token,
            token_type=pair.token_type,
            expires_at=pair.expires_at,
            obtained_at=pair.obtained_at,
            source=pair.source,
            generation=self._token_generation,
        )
        self._pair = installed
        self.access_token = installed.access_token
        self.refresh_token = installed.refresh_token
        self.token_type = installed.token_type
        self.expires_at = installed.expires_at
        self._state = TokenSessionState.ACTIVE

    def _maybe_request_ws_reconnect(self, old_access: str | None) -> None:
        """Mark a websocket reconnect when a committed access token changed.

        Args:
            old_access: Access token published before this commit, if any.
        """
        if (
            old_access is not None
            and self._pair is not None
            and old_access != self._pair.access_token
        ):
            self.reconnect_required = True
            self._emit_session_log(
                event="ws_reconnect_requested",
                purpose=self._pair.source,
                outcome="success",
                ws_action="reconnect_requested",
            )

    async def _close_websocket_if_reconnect_required(self) -> None:
        """Close an open websocket after the token lock is released."""
        if not self.reconnect_required:
            return
        websocket_manager = self.api_websocket
        if websocket_manager is not None:
            await websocket_manager.request_reconnect()
        self.reconnect_required = False

    def _pair_from_oauth_data(self, data: object, *, source: TokenPairSource) -> TokenPair:
        """Build a TokenPair from a complete OAuth JSON object.

        Args:
            data: Decoded OAuth JSON object.
            source: Why this pair was obtained.

        Returns:
            Validated pair with placeholder generation ``0``.

        Raises:
            KeyError: If a required field is missing.
            TypeError: If ``data`` is not an object or a field has the wrong type.
        """
        if not isinstance(data, dict):
            raise TypeError("OAuth payload is not an object")
        access_token = data["access_token"]
        refresh_token = data["refresh_token"]
        token_type = data["token_type"]
        expires_in = data["expires_in"]
        if not isinstance(access_token, str) or not access_token:
            raise TypeError("access_token must be a non-empty string")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise TypeError("refresh_token must be a non-empty string")
        if not isinstance(token_type, str) or not token_type:
            raise TypeError("token_type must be a non-empty string")
        if (
            isinstance(expires_in, bool)
            or not isinstance(expires_in, int | float)
            or expires_in <= 0
        ):
            raise TypeError("expires_in must be a positive number")
        now = self._now()
        pair_source: TokenPairSource = "recovery" if source == "pre_expiry" else source
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type=token_type,
            expires_at=now + timedelta(seconds=float(expires_in)),
            obtained_at=now,
            source=pair_source,
            generation=0,
        )

    def _oauth_error_kind(self, error: ClientResponseError, parsed: object | None) -> str:
        """Map an OAuth HTTP error onto the reviewed classifier.

        Args:
            error: Raised ``ClientResponseError``.
            parsed: JSON value consumed before ``raise_for_status``.

        Returns:
            One of ``unauthorized``, ``invalid_grant``, ``invalid_client``, or
            ``transient``.
        """
        if error.status in {401, 403}:
            return "unauthorized"
        oauth_error = parsed.get("error") if isinstance(parsed, dict) else None
        if error.status == 400 and oauth_error == "invalid_grant":
            return "invalid_grant"
        if error.status == 400 and oauth_error in {"invalid_client", "unauthorized_client"}:
            return "invalid_client"
        return "transient"

    def _emit_session_log(
        self,
        *,
        event: str,
        purpose: str,
        outcome: str,
        access_valid: bool | None = None,
        seconds_until_expiry: float | None = None,
        delay_s: float | None = None,
        generation: int | None = None,
        refresh_fp12: str | None = None,
        ws_action: str = "none",
        recovery_eligible: bool | None = None,
        suppressed: bool | None = None,
    ) -> None:
        """Emit one allowlisted token-session orchestration log line.

        Args:
            event: Allowlisted session event name.
            purpose: Allowlisted purpose.
            outcome: Allowlisted outcome.
            access_valid: Whether the published access token is still valid.
            seconds_until_expiry: Remaining access-token lifetime.
            delay_s: Scheduled delay when relevant.
            generation: Token generation to record.
            refresh_fp12: First 12 hex characters of the refresh fingerprint.
            ws_action: Allowlisted websocket action.
            recovery_eligible: Whether recovery may run for this outcome.
            suppressed: Whether this refresh fingerprint is suppressed.
        """
        safe_event = event if event in _SESSION_EVENTS else "refresh_finished"
        safe_purpose = purpose if purpose in _SESSION_PURPOSES else "refresh"
        safe_outcome = outcome if outcome in _SESSION_OUTCOMES else "transient"
        safe_ws_action = ws_action if ws_action in _SESSION_WS_ACTIONS else "none"
        if access_valid is None:
            access_valid = self._seconds_until_expiry(self._now()) > 0
        if seconds_until_expiry is None:
            seconds_until_expiry = self._seconds_until_expiry(self._now())
        if generation is None:
            generation = self.ws_generation
        if recovery_eligible is None:
            recovery_eligible = self.invalid_grant_recovery
        if suppressed is None:
            suppressed = self._suppressed_refresh_fp is not None
        if refresh_fp12 is not None:
            self._last_refresh_fp12 = refresh_fp12
        self._last_session_event = safe_event
        self._last_session_outcome = safe_outcome
        if safe_outcome == "success" and safe_event not in {
            "canary_scheduled",
            "recovery_scheduled",
        }:
            level = DEBUG
        elif safe_event in {"canary_scheduled", "recovery_scheduled"}:
            level = INFO
        else:
            level = WARNING
        _LOGGER.log(
            level,
            "Carrier OAuth token session event=%s state=%s purpose=%s outcome=%s "
            "access_valid=%s seconds_until_expiry=%s delay_s=%s generation=%s "
            "refresh_fp12=%s ws_action=%s canary_flag=%s recovery_flag=%s "
            "recovery_eligible=%s suppressed=%s",
            safe_event,
            self._state.value,
            safe_purpose,
            safe_outcome,
            access_valid,
            seconds_until_expiry,
            delay_s,
            generation,
            refresh_fp12,
            safe_ws_action,
            self.early_refresh_canary,
            self.invalid_grant_recovery,
            recovery_eligible,
            suppressed,
        )

    async def _ensure_tokens_locked(self) -> None:
        """Ensure a usable access token while already holding the token lock."""
        if self._closing:
            raise CarrierApiTokenRefreshError("Carrier token session is closing")
        if self._state is TokenSessionState.AUTH_FAILED:
            raise CarrierApiAuthError("Carrier token session is unusable", reason="unauthorized")
        if self._pair is None and self.refresh_token is None:
            await self.login()
            return
        if self._seconds_until_expiry(self._now()) > 0:
            return
        refresh_token = self._pair.refresh_token if self._pair is not None else self.refresh_token
        if refresh_token is not None:
            fingerprint = self._refresh_fingerprint(refresh_token)
            if fingerprint == self._suppressed_refresh_fp:
                self._emit_session_log(
                    event="refresh_suppressed",
                    purpose="refresh",
                    outcome="invalid_grant",
                    refresh_fp12=fingerprint[:12],
                    suppressed=True,
                )
                if self.invalid_grant_recovery and self._recovery_attempts < MAX_RECOVERY_ATTEMPTS:
                    await self._login_locked(purpose="recovery")
                    return
                raise CarrierApiAuthError(
                    "Carrier token refresh was rejected", reason="invalid_grant"
                )
        await self.refresh_auth_token()

    async def _execute_assisted_login(self) -> dict[str, Any]:
        """Run the unauthenticated assistedLogin GraphQL mutation.

        Returns:
            Decoded assistedLogin GraphQL response.

        Raises:
            CarrierApiGraphqlError: If the GraphQL request fails.
            CarrierApiConnectionError: If the transport fails.
        """
        transport = AIOHTTPTransport(
            url="https://dataservice.infinity.iot.carrier.com/graphql-no-auth", ssl=True
        )
        async with Client(
            transport=transport,
            fetch_schema_from_transport=False,
        ) as session:
            query = gql(
                """
                mutation assistedLogin($input: AssistedLoginInput!) {
                    assistedLogin(input: $input) {
                        success
                        status
                        errorMessage
                        data {
                            token_type
                            expires_in
                            access_token
                            scope
                            refresh_token
                        }
                    }
                }
            """
            )
            return await session.execute(
                query,
                variable_values={"input": {"password": self.password, "username": self.username}},
                operation_name="assistedLogin",
            )

    async def _login_locked(self, *, purpose: str) -> None:
        """Run assistedLogin and commit a validated pair while holding the lock.

        Args:
            purpose: ``login``, ``recovery``, or ``pre_expiry``.

        Raises:
            CarrierApiAuthError: If credentials are rejected or the payload is
                unusable.
            CarrierApiGraphqlError: If the GraphQL request fails.
            CarrierApiConnectionError: If the transport fails.
        """
        if purpose in {"recovery", "pre_expiry"}:
            if self._recovery_attempts >= MAX_RECOVERY_ATTEMPTS:
                self._state = TokenSessionState.AUTH_FAILED
                raise CarrierApiAuthError(
                    "Carrier token refresh was rejected", reason="invalid_grant"
                )
            self._recovery_attempts += 1
            self._state = (
                TokenSessionState.PRE_EXPIRY_LOGIN
                if purpose == "pre_expiry"
                else TokenSessionState.RECOVERY_LOGIN
            )
            self._emit_session_log(
                event="recovery_started",
                purpose=purpose,
                outcome="success",
            )
        try:
            result = await self._execute_assisted_login()
        except TransportQueryError as error:
            if purpose in {"recovery", "pre_expiry"}:
                self._emit_session_log(
                    event="recovery_finished",
                    purpose=purpose,
                    outcome="login_transient",
                )
            raise CarrierApiGraphqlError("Carrier authentication GraphQL request failed") from error
        except _CONNECTION_ERRORS as error:
            if purpose in {"recovery", "pre_expiry"}:
                self._emit_session_log(
                    event="recovery_finished",
                    purpose=purpose,
                    outcome="login_transient",
                )
            raise CarrierApiConnectionError("Carrier authentication connection failed") from error
        except GraphQLError as error:
            if purpose in {"recovery", "pre_expiry"}:
                self._emit_session_log(
                    event="recovery_finished",
                    purpose=purpose,
                    outcome="login_transient",
                )
            raise CarrierApiGraphqlError("Carrier authentication GraphQL request failed") from error
        success = result["assistedLogin"]["success"]
        if not success:
            self._state = TokenSessionState.AUTH_FAILED
            error_message = result["assistedLogin"].get("errorMessage")
            if isinstance(error_message, str) and error_message:
                message = f"Carrier assistedLogin failed: {error_message}"
            else:
                message = "Carrier assistedLogin failed"
            self._emit_session_log(
                event="recovery_finished"
                if purpose in {"recovery", "pre_expiry"}
                else "login_committed",
                purpose=purpose,
                outcome="login_failed",
            )
            raise CarrierApiAuthError(message, payload=result, reason="login_failed")
        pair_source: TokenPairSource = (
            "recovery" if purpose in {"recovery", "pre_expiry"} else "login"
        )
        try:
            pair = self._pair_from_oauth_data(result["assistedLogin"]["data"], source=pair_source)
        except (KeyError, TypeError, ValueError) as error:
            self._state = TokenSessionState.AUTH_FAILED
            raise CarrierApiAuthError(
                "Carrier assistedLogin returned invalid token data",
                payload=result,
                reason="login_failed",
            ) from error
        if self._closing:
            self._emit_session_log(
                event="login_committed",
                purpose=purpose,
                outcome="cancelled",
            )
            return
        old_access = None if self._pair is None else self._pair.access_token
        self._install_token_pair(pair)
        self._suppressed_refresh_fp = None
        self._recovery_attempts = 0
        if self.api_websocket is None:
            self.api_websocket = ApiWebsocket(self)
        self._maybe_request_ws_reconnect(old_access)
        self._maybe_schedule_canary_locked()
        finished_event = (
            "recovery_finished" if purpose in {"recovery", "pre_expiry"} else "login_committed"
        )
        self._emit_session_log(
            event=finished_event,
            purpose=purpose,
            outcome="success",
            refresh_fp12=self._refresh_fingerprint(pair.refresh_token)[:12],
        )

    async def _refresh_locked(self, *, purpose: str) -> None:
        """Refresh or canary-refresh while holding the token lock.

        Args:
            purpose: ``refresh`` or ``canary``.
        """
        event_started = "canary_started" if purpose == "canary" else "refresh_started"
        event_finished = "canary_finished" if purpose == "canary" else "refresh_finished"
        if self._closing:
            self._emit_session_log(
                event=event_finished,
                purpose=purpose,
                outcome="cancelled",
            )
            return
        pair = self._pair
        refresh_token = pair.refresh_token if pair is not None else self.refresh_token
        if refresh_token is None:
            await self._login_locked(purpose="login")
            return
        generation_at_request = pair.generation if pair is not None else self._token_generation
        fingerprint = self._refresh_fingerprint(refresh_token)
        if fingerprint == self._suppressed_refresh_fp:
            self._emit_session_log(
                event="refresh_suppressed",
                purpose=purpose,
                outcome="invalid_grant",
                refresh_fp12=fingerprint[:12],
                suppressed=True,
            )
            if purpose == "canary":
                return
            if self.invalid_grant_recovery and self._recovery_attempts < MAX_RECOVERY_ATTEMPTS:
                await self._login_locked(purpose="recovery")
                return
            raise CarrierApiAuthError("Carrier token refresh was rejected", reason="invalid_grant")
        access_valid = self._seconds_until_expiry(self._now()) > 0
        self._state = (
            TokenSessionState.CANARY_IN_FLIGHT
            if purpose == "canary"
            else TokenSessionState.REFRESH_IN_FLIGHT
        )
        self._emit_session_log(
            event=event_started,
            purpose=purpose,
            outcome="success",
            access_valid=access_valid,
            refresh_fp12=fingerprint[:12],
        )
        url = "https://sso.carrier.com/oauth2/default/v1/token"
        json_body = {
            "client_id": "0oa1ce7hwjuZbfOMB4x7",
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "scope": "offline_access",
        }
        response: Any | None = None
        raw_body: bytes | None = None
        parsed: object | None = None
        try:
            response = await self.api_session.post(url=url, data=json_body)
            raw_body, parsed, parse_error = await _consume_oauth_refresh_response(response)
            response.raise_for_status()
            if parse_error is not None:
                raise parse_error
            pair_source: TokenPairSource = "canary" if purpose == "canary" else "refresh"
            new_pair = self._pair_from_oauth_data(parsed, source=pair_source)
        except asyncio.CancelledError:
            self._state = TokenSessionState.ACTIVE if access_valid else self._state
            self._emit_session_log(
                event=event_finished,
                purpose=purpose,
                outcome="cancelled",
                access_valid=access_valid,
            )
            raise
        except ClientResponseError as error:
            await log_oauth_refresh_response(
                _LOGGER,
                response,
                parsed=parsed,
                raw_body=raw_body,
                status_override=error.status,
                level=WARNING,
            )
            kind = self._oauth_error_kind(error, parsed)
            if purpose == "canary":
                self._handle_canary_failure(kind, fingerprint)
                return
            if kind == "invalid_client":
                self._state = TokenSessionState.ACTIVE if access_valid else self._state
                self._emit_session_log(
                    event=event_finished,
                    purpose=purpose,
                    outcome="invalid_client",
                    access_valid=access_valid,
                    recovery_eligible=False,
                    refresh_fp12=fingerprint[:12],
                )
                raise CarrierApiTokenRefreshError("Carrier token refresh failed") from error
            if kind in {"invalid_grant", "unauthorized"}:
                self._suppressed_refresh_fp = fingerprint
                if access_valid:
                    self._state = TokenSessionState.ACTIVE_SUSPECT
                    self._maybe_schedule_pre_expiry_locked()
                    if purpose != "refresh":
                        self._emit_session_log(
                            event=event_finished,
                            purpose=purpose,
                            outcome=kind,
                            access_valid=True,
                            ws_action="left_connected",
                            refresh_fp12=fingerprint[:12],
                        )
                        return
                if self.invalid_grant_recovery and self._recovery_attempts < MAX_RECOVERY_ATTEMPTS:
                    await self._login_locked(purpose="recovery")
                    return
                self._state = TokenSessionState.AUTH_FAILED
                self._emit_session_log(
                    event=event_finished,
                    purpose=purpose,
                    outcome=kind,
                    access_valid=access_valid,
                    refresh_fp12=fingerprint[:12],
                    suppressed=True,
                )
                auth_reason: Literal["invalid_grant", "unauthorized"] = (
                    "invalid_grant" if kind == "invalid_grant" else "unauthorized"
                )
                raise CarrierApiAuthError(
                    "Carrier token refresh was rejected", reason=auth_reason
                ) from error
            self._state = TokenSessionState.ACTIVE if access_valid else self._state
            self._emit_session_log(
                event=event_finished,
                purpose=purpose,
                outcome="transient",
                access_valid=access_valid,
                refresh_fp12=fingerprint[:12],
            )
            raise CarrierApiTokenRefreshError("Carrier token refresh failed") from error
        except (ClientError, TimeoutError, OSError, TypeError, ValueError, KeyError) as error:
            await log_oauth_refresh_response(
                _LOGGER,
                response,
                parsed=parsed,
                raw_body=raw_body,
                level=WARNING,
            )
            if purpose == "canary":
                self._state = TokenSessionState.ACTIVE
                self._emit_session_log(
                    event=event_finished,
                    purpose=purpose,
                    outcome="transient",
                    access_valid=True,
                    ws_action="left_connected",
                    refresh_fp12=fingerprint[:12],
                )
                return
            self._state = TokenSessionState.ACTIVE if access_valid else self._state
            self._emit_session_log(
                event=event_finished,
                purpose=purpose,
                outcome="transient",
                access_valid=access_valid,
                refresh_fp12=fingerprint[:12],
            )
            raise CarrierApiTokenRefreshError("Carrier token refresh failed") from error
        if not self._commit_allowed(pair, generation_at_request):
            self._state = TokenSessionState.ACTIVE if access_valid else self._state
            self._emit_session_log(
                event=event_finished,
                purpose=purpose,
                outcome="cancelled",
                access_valid=access_valid,
                refresh_fp12=fingerprint[:12],
            )
            return
        old_access = None if pair is None else pair.access_token
        if old_access is None:
            old_access = self.access_token
        self._install_token_pair(new_pair)
        self._maybe_request_ws_reconnect(old_access)
        await log_oauth_refresh_response(
            _LOGGER,
            response,
            parsed=parsed,
            raw_body=raw_body,
            level=DEBUG,
        )
        ws_action = "reconnect_requested" if self.reconnect_required else "none"
        self._emit_session_log(
            event=event_finished,
            purpose=purpose,
            outcome="success",
            access_valid=True,
            ws_action=ws_action,
            refresh_fp12=self._refresh_fingerprint(new_pair.refresh_token)[:12],
            suppressed=False,
        )

    def _handle_canary_failure(self, kind: str, fingerprint: str) -> None:
        """Restore state after a failed canary without writing token fields.

        Args:
            kind: Classified OAuth error kind.
            fingerprint: Refresh-token fingerprint from the request generation.
        """
        if kind in {"invalid_grant", "unauthorized"}:
            self._suppressed_refresh_fp = fingerprint
            self._state = TokenSessionState.ACTIVE_SUSPECT
            self._maybe_schedule_pre_expiry_locked()
            self._emit_session_log(
                event="canary_finished",
                purpose="canary",
                outcome=kind,
                access_valid=True,
                ws_action="left_connected",
                recovery_eligible=self.invalid_grant_recovery,
                refresh_fp12=fingerprint[:12],
                suppressed=True,
            )
            return
        if kind == "invalid_client":
            self._state = TokenSessionState.ACTIVE
            self._emit_session_log(
                event="canary_finished",
                purpose="canary",
                outcome="invalid_client",
                access_valid=True,
                ws_action="left_connected",
                recovery_eligible=False,
                refresh_fp12=fingerprint[:12],
            )
            return
        self._state = TokenSessionState.ACTIVE
        self._emit_session_log(
            event="canary_finished",
            purpose="canary",
            outcome="transient",
            access_valid=True,
            ws_action="left_connected",
            refresh_fp12=fingerprint[:12],
        )

    def _maybe_schedule_canary_locked(self) -> None:
        """Schedule one diagnostic canary for the current login generation."""
        if not self.early_refresh_canary or self._schedule is None or self._pair is None:
            return
        if self._canary_ran_for_generation == self._pair.generation:
            return
        ttl = self._pair.seconds_until_expiry(self._now())
        if ttl <= self.canary_delay_seconds + PRE_EXPIRY_LEAD_SECONDS:
            self._emit_session_log(
                event="canary_skipped",
                purpose="canary",
                outcome="skipped_ttl",
                delay_s=self.canary_delay_seconds,
            )
            return
        self._canary_ran_for_generation = self._pair.generation
        self._canary_task = self._schedule(self._run_canary())
        self._emit_session_log(
            event="canary_scheduled",
            purpose="canary",
            outcome="success",
            delay_s=self.canary_delay_seconds,
        )

    async def _run_canary(self) -> None:
        """Sleep outside the lock, then run one canary refresh."""
        try:
            await asyncio.sleep(self.canary_delay_seconds)
            if self._closing:
                self._emit_session_log(
                    event="canary_finished",
                    purpose="canary",
                    outcome="cancelled",
                )
                return
            await self.refresh_auth_token(purpose="canary")
        except asyncio.CancelledError:
            self._emit_session_log(
                event="canary_finished",
                purpose="canary",
                outcome="cancelled",
            )
            raise
        except (
            CarrierApiAuthError,
            CarrierApiTokenRefreshError,
            CarrierApiConnectionError,
            CarrierApiGraphqlError,
        ):
            self._emit_session_log(
                event="canary_finished",
                purpose="canary",
                outcome="transient",
                access_valid=True,
                ws_action="left_connected",
            )

    def _maybe_schedule_pre_expiry_locked(self) -> None:
        """Schedule a pre-expiry recovery login when recovery is enabled."""
        if not self.invalid_grant_recovery or self._schedule is None or self._pair is None:
            return
        ttl = self._pair.seconds_until_expiry(self._now())
        if ttl <= 0:
            return
        delay = max(0.0, ttl - PRE_EXPIRY_LEAD_SECONDS)
        self._pre_expiry_task = self._schedule(self._run_pre_expiry_login(delay))
        self._emit_session_log(
            event="recovery_scheduled",
            purpose="pre_expiry",
            outcome="success",
            delay_s=delay,
            recovery_eligible=True,
        )

    async def _run_pre_expiry_login(self, delay: float) -> None:
        """Sleep until the pre-expiry window, then login if still suspect.

        Args:
            delay: Seconds to wait outside the lock.
        """
        try:
            await asyncio.sleep(delay)
            if self._closing:
                self._emit_session_log(
                    event="recovery_finished",
                    purpose="pre_expiry",
                    outcome="cancelled",
                )
                return
            async with self._held_token_lock():
                if self._state is not TokenSessionState.ACTIVE_SUSPECT:
                    return
                await self._login_locked(purpose="pre_expiry")
            await self._close_websocket_if_reconnect_required()
        except asyncio.CancelledError:
            self._emit_session_log(
                event="recovery_finished",
                purpose="pre_expiry",
                outcome="cancelled",
            )
            raise
        except (
            CarrierApiAuthError,
            CarrierApiTokenRefreshError,
            CarrierApiConnectionError,
            CarrierApiGraphqlError,
        ):
            return

    async def authed_query(
        self, operation_name: str, query: GraphQLRequest, variable_values: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute an authenticated Carrier GraphQL operation.

        Args:
            operation_name: GraphQL operation name to execute.
            query: Parsed GraphQL request.
            variable_values: Variables to send with the operation.

        Returns:
            The decoded GraphQL response data.
        """
        auth_header = await self.check_auth_expiration()
        if auth_header is None:
            raise CarrierApiTokenRefreshError("Carrier authorization header is unavailable")
        transport = AIOHTTPTransport(
            url="https://dataservice.infinity.iot.carrier.com/graphql",
            headers={"Authorization": auth_header},
            ssl=True,
        )
        try:
            async with Client(
                transport=transport,
                fetch_schema_from_transport=False,
                execute_timeout=GRAPHQL_EXECUTE_TIMEOUT_SECONDS,
            ) as session:
                return await session.execute(
                    query, variable_values=variable_values, operation_name=operation_name
                )
        except TransportQueryError as error:
            raise CarrierApiGraphqlError(
                f"Carrier GraphQL operation failed: {operation_name}"
            ) from error
        except _CONNECTION_ERRORS as error:
            if _is_auth_transport_error(error):
                raise CarrierApiAuthError(
                    f"Carrier authorization failed during GraphQL operation: {operation_name}"
                ) from error
            raise CarrierApiConnectionError(
                f"Carrier connection failed during GraphQL operation: {operation_name}"
            ) from error
        except GraphQLError as error:
            raise CarrierApiGraphqlError(
                f"Carrier GraphQL operation failed: {operation_name}"
            ) from error

    async def get_user_info(self) -> dict[str, Any]:
        """Fetch Carrier account profile, location, and device metadata.

        Returns:
            The decoded ``getUser`` GraphQL response data.
        """
        operation_name = "getUser"
        query = gql(
            """
            query getUser(
                $userName: String!,
                $appVersion: String,
                $brand: String,
                $os: String,
                $osVersion: String
            ) {
                user(
                    userName: $userName
                    appVersion: $appVersion
                    brand: $brand
                    os: $os
                    osVersion: $osVersion
                ) {
                    username
                    identityId
                    first
                    last
                    email
                    emailVerified
                    postal
                    locations {
                        locationId
                        name
                        systems {
                            config {
                                zones {
                                    id
                                    enabled
                                }
                            }
                            profile {
                                serial
                                name
                            }
                            status {
                                isDisconnected
                            }
                        }
                        devices {
                            deviceId
                            type
                            thingName
                            name
                            connectionStatus
                        }
                    }
                }
            }
            """
        )
        variable_values = {"userName": self.username}
        return await self.authed_query(
            operation_name=operation_name, query=query, variable_values=variable_values
        )

    async def get_systems(self) -> dict[str, Any]:
        """Fetch configured Carrier Infinity systems for the current user.

        Returns:
            The decoded ``getInfinitySystems`` GraphQL response data containing
            profile, status, and config payloads.
        """
        operation_name = "getInfinitySystems"
        query = gql(
            """
            query getInfinitySystems($userName: String!) {
              infinitySystems(userName: $userName) {
                profile {
                  serial
                  name
                  firmware
                  model
                  brand
                  indoorModel
                  indoorSerial
                  idutype
                  idusource
                  outdoorModel
                  outdoorSerial
                  odutype
                }
                status {
                  localTime
                  localTimeOffset
                  utcTime
                  wcTime
                  isDisconnected
                  cfgem
                  mode
                  vacatrunning
                  oat
                  odu {
                    type
                    opstat
                    iducfm
                  }
                  filtrlvl
                  idu {
                    type
                    opstat
                    cfm
                    statpress
                    blwrpm
                  }
                  vent
                  ventlvl
                  humid
                  humlvl
                  uvlvl
                  zones {
                    id
                    rt
                    rh
                    fan
                    htsp
                    clsp
                    hold
                    enabled
                    currentActivity
                    zoneconditioning
                  }
                }
                config {
                  etag
                  mode
                  cfgem
                  cfgdead
                  cfgvent
                  cfghumid
                  cfguv
                  cfgfan
                  heatsource
                  vacat
                  vacstart
                  vacend
                  vacmint
                  vacmaxt
                  vacfan
                  fueltype
                  gasunit
                  vacat
                  filtertype
                  filterinterval
                  humidityVacation {
                    rclgovercool
                    ventspdclg
                    ventclg
                    rhtg
                    humidifier
                    humid
                    venthtg
                    rclg
                    ventspdhtg
                  }
                  zones {
                    id
                    name
                    enabled
                    hold
                    holdActivity
                    otmr
                    occEnabled
                    program {
                      id
                      day {
                        id
                        zoneId
                        period {
                          id
                          zoneId
                          dayId
                          activity
                          time
                          enabled
                        }
                      }
                    }
                    activities {
                      id
                      zoneId
                      type
                      fan
                      htsp
                      clsp
                    }
                  }
                  humidityAway {
                    humid
                    humidifier
                    rhtg
                    rclg
                    rclgovercool
                  }
                  humidityHome {
                    humid
                    humidifier
                    rhtg
                    rclg
                    rclgovercool
                  }
                }
              }
            }
            """
        )
        variable_values = {"userName": self.username}
        return await self.authed_query(
            operation_name=operation_name, query=query, variable_values=variable_values
        )

    async def get_energy(self, system_serial: str) -> dict[str, Any]:
        """Fetch energy configuration and usage for a Carrier system.

        Args:
            system_serial: Serial number of the Carrier system to query.

        Returns:
            The decoded ``getInfinityEnergy`` GraphQL response data.
        """
        operation_name = "getInfinityEnergy"
        query = gql(
            """
            query getInfinityEnergy($serial: String!) {
              infinityEnergy(serial: $serial) {
                energyConfig {
                  cooling {
                    display
                    enabled
                  }
                  eheat {
                    display
                    enabled
                  }
                  fan {
                    display
                    enabled
                  }
                  fangas {
                    display
                    enabled
                  }
                  gas {
                    display
                    enabled
                  }
                  hpheat {
                    display
                    enabled
                  }
                  looppump {
                    display
                    enabled
                  }
                  reheat {
                    display
                    enabled
                  }
                  hspf
                  seer
                }
                energyPeriods {
                  energyPeriodType
                  eHeatKwh
                  coolingKwh
                  fanGasKwh
                  fanKwh
                  hPHeatKwh
                  loopPumpKwh
                  gasKwh
                  reheatKwh
                }
              }
            }
            """
        )
        variable_values = {"serial": system_serial}
        return await self.authed_query(
            operation_name=operation_name, query=query, variable_values=variable_values
        )

    async def load_data(self) -> list[System]:
        """Load all Carrier systems with status, config, and energy models.

        Returns:
            A list of fully constructed system aggregates for the account.
        """
        systems_response = await self.get_systems()
        systems = []
        for system_response in systems_response["infinitySystems"]:
            profile = Profile(raw=system_response["profile"])
            status = Status(raw=system_response["status"])
            config = Config(raw=system_response["config"])
            energy_response = await self.get_energy(profile.serial)
            energy = Energy(raw=energy_response["infinityEnergy"])
            systems.append(System(profile=profile, status=status, config=config, energy=energy))
        return systems

    async def get_entry_level_systems(self) -> dict[str, Any]:
        """Fetch entry-level (Smart Thermostat) systems for the current user.

        These are the non-Infinity Carrier Smart Thermostat devices, exposed by
        a separate query from ``infinitySystems``.

        Returns:
            The decoded ``getEntryLevelSystems`` GraphQL response data.
        """
        operation_name = "getEntryLevelSystems"
        query = gql(
            """
            query getEntryLevelSystems($username: String!) {
              entryLevelSystems(username: $username) {
                serial
                name
                location_id
                model
                firmware
                temp_unit_format
                connection {
                  isConnected
                  deviceId
                }
                zones {
                  index
                  mode
                  rt
                  rh
                  clsp { current min }
                  htsp { current max }
                  fan_mode
                  schedule_enabled
                  hold_end_time
                  hold_countdown
                  stage_status
                  outside_temp
                }
              }
            }
            """
        )
        variable_values = {"username": self.username}
        return await self.authed_query(
            operation_name=operation_name, query=query, variable_values=variable_values
        )

    async def load_entry_level_data(self) -> list[EntryLevelSystem]:
        """Load all entry-level systems for the account.

        Returns:
            A list of entry-level system models for the account.
        """
        response = await self.get_entry_level_systems()
        return [
            EntryLevelSystem(raw=system_response)
            for system_response in (response.get("entryLevelSystems") or [])
        ]

    async def update_entry_level_zone(
        self,
        serial: str,
        index: int = 0,
        mode: str | None = None,
        cool_set_point: float | None = None,
        heat_set_point: float | None = None,
        schedule_enabled: bool | None = None,
        hold_end_time: int | None = None,
        fan_mode: str | None = None,
    ) -> dict[str, Any]:
        """Update an entry-level zone's mode, set points, hold, or fan.

        Only the provided fields are sent. Carrier expects the cool and heat set
        points together, so pass both when changing a set point.

        Args:
            serial: Serial number of the entry-level system to update.
            index: Zone index to update (entry-level systems are single-zone).
            mode: Requested HVAC mode (``cool``/``heat``/``off``/``auto``).
            cool_set_point: Requested cool set point.
            heat_set_point: Requested heat set point.
            schedule_enabled: ``False`` holds the zone, ``True`` resumes the
                programmed schedule.
            hold_end_time: Optional Carrier hold-until value.
            fan_mode: Requested fan mode.

        Returns:
            The decoded mutation response.
        """
        query = gql(
            """
            mutation updateEntryLevelZone($input: EntryLevelZoneInput!) {
              updateEntryLevelZone(input: $input) {
                success
              }
            }
            """
        )
        zone_input: dict[str, Any] = {"serial": serial, "index": index}
        if mode is not None:
            zone_input["mode"] = mode
        if cool_set_point is not None:
            zone_input["clsp"] = {"current": cool_set_point}
        if heat_set_point is not None:
            zone_input["htsp"] = {"current": heat_set_point}
        if schedule_enabled is not None:
            zone_input["schedule_enabled"] = schedule_enabled
        if hold_end_time is not None:
            zone_input["hold_end_time"] = hold_end_time
        if fan_mode is not None:
            zone_input["fan_mode"] = fan_mode
        _LOGGER.debug("updateEntryLevelZone: %s", zone_input)
        return await self.authed_query(
            operation_name="updateEntryLevelZone",
            query=query,
            variable_values={"input": zone_input},
        )

    async def hold_entry_level_zone(
        self,
        serial: str,
        index: int = 0,
        cool_set_point: float | None = None,
        heat_set_point: float | None = None,
        hold_end_time: int | None = None,
    ) -> dict[str, Any]:
        """Hold an entry-level zone off its schedule at the given set points.

        Args:
            serial: Serial number of the entry-level system to update.
            index: Zone index to update.
            cool_set_point: Optional cool set point to apply with the hold.
            heat_set_point: Optional heat set point to apply with the hold.
            hold_end_time: Optional Carrier hold-until value.

        Returns:
            The decoded mutation response.
        """
        return await self.update_entry_level_zone(
            serial,
            index,
            cool_set_point=cool_set_point,
            heat_set_point=heat_set_point,
            schedule_enabled=False,
            hold_end_time=hold_end_time,
        )

    async def resume_entry_level_schedule(self, serial: str, index: int = 0) -> dict[str, Any]:
        """Clear an entry-level zone hold and resume its programmed schedule.

        Args:
            serial: Serial number of the entry-level system to update.
            index: Zone index to update.

        Returns:
            The decoded mutation response.
        """
        return await self.update_entry_level_zone(serial, index, schedule_enabled=True)

    async def _update_infinity_config(self, variables: dict[str, Any]) -> dict[str, Any]:
        """Run the Carrier system-level configuration mutation.

        Args:
            variables: GraphQL variables containing an ``InfinityConfigInput``.

        Returns:
            The decoded mutation response.
        """
        query = gql(
            """
            mutation updateInfinityConfig($input: InfinityConfigInput!) {
                updateInfinityConfig(input: $input) {
                    etag
                }
            }
            """
        )
        _LOGGER.debug("updateInfinityConfig: %s", variables)
        response = await self.authed_query(
            operation_name="updateInfinityConfig", query=query, variable_values=variables
        )
        if self.api_websocket is not None:
            await self.api_websocket.send_reconcile()
        else:
            _LOGGER.warning("No API websocket connection")
        return response

    async def _update_infinity_zone_activity(self, variables: dict[str, Any]) -> dict[str, Any]:
        """Run the Carrier zone activity configuration mutation.

        Args:
            variables: GraphQL variables containing an
                ``InfinityZoneActivityInput``.

        Returns:
            The decoded mutation response.
        """
        query = gql(
            """
            mutation updateInfinityZoneActivity($input: InfinityZoneActivityInput!) {
                updateInfinityZoneActivity(input: $input) {
                    etag
                }
            }
            """
        )
        _LOGGER.debug("updateInfinityZoneActivity: %s", variables)
        response = await self.authed_query(
            operation_name="updateInfinityZoneActivity", query=query, variable_values=variables
        )
        if self.api_websocket is not None:
            await self.api_websocket.send_reconcile()
        else:
            _LOGGER.warning("No API websocket connection")
        return response

    async def _update_infinity_zone_config(self, variables: dict[str, Any]) -> dict[str, Any]:
        """Run the Carrier zone configuration mutation.

        Args:
            variables: GraphQL variables containing an ``InfinityZoneConfigInput``.

        Returns:
            The decoded mutation response.
        """
        query = gql(
            """
            mutation updateInfinityZoneConfig($input: InfinityZoneConfigInput!) {
                updateInfinityZoneConfig(input: $input) {
                    etag
                }
            }
            """
        )
        _LOGGER.debug("updateInfinityZoneConfig: %s", variables)
        response = await self.authed_query(
            operation_name="updateInfinityZoneConfig", query=query, variable_values=variables
        )
        if self.api_websocket is not None:
            await self.api_websocket.send_reconcile()
        else:
            _LOGGER.warning("No API websocket connection")
        return response

    async def set_config_mode(self, system_serial: str, mode: SystemModes) -> dict[str, Any]:
        """Update a Carrier system's operating mode.

        Args:
            system_serial: Serial number of the system to update.
            mode: Requested system operating mode.

        Returns:
            The decoded mutation response.

        Raises:
            ValueError: If ``mode`` is not a ``SystemModes`` member.
        """
        if mode not in SystemModes:
            raise ValueError(f"{mode} is not a valid system mode")
        variables = {"input": {"serial": system_serial, "mode": mode.value}}
        return await self._update_infinity_config(variables)

    async def set_config_heat_humidity(
        self, system_serial: str, humidity_target: int
    ) -> dict[str, Any]:
        """Update the heating humidifier target for home mode.

        Args:
            system_serial: Serial number of the system to update.
            humidity_target: Target relative humidity percentage. Carrier
                accepts zero or five-percent increments from 5 through 45.

        Returns:
            The decoded mutation response.

        Raises:
            ValueError: If ``humidity_target`` is outside Carrier's accepted
                values.
        """
        if humidity_target not in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]:
            raise ValueError(f"{humidity_target} is not a valid humidity target")
        variables = {"input": {"serial": system_serial, "humidityHome": {}}}
        if humidity_target == 0:
            variables["input"]["humidityHome"] = {"humidifier": "off"}
        else:
            variables["input"]["humidityHome"] = {"humidifier": "on", "rhtg": humidity_target / 5}
        return await self._update_infinity_config(variables)

    async def set_heat_source(
        self, system_serial: str, heat_source: HeatSourceTypes
    ) -> dict[str, Any]:
        """Update which equipment source should provide heat.

        Args:
            system_serial: Serial number of the system to update.
            heat_source: Requested heat source routing mode.

        Returns:
            The decoded mutation response.

        Raises:
            ValueError: If ``heat_source`` is not a ``HeatSourceTypes`` member.
        """
        if heat_source not in HeatSourceTypes:
            raise ValueError(f"{heat_source} is not a valid heat source")
        variables = {"input": {"serial": system_serial, "heatsource": heat_source.value}}
        return await self._update_infinity_config(variables)

    async def set_humidifier(
        self,
        system_serial: str,
        humidifier_on: bool | None = None,
        over_cooling: bool | None = None,
        cooling_percent: Literal[5, 10, 15, 20, 25, 30, 35, 40, 45] | None = None,
        heating_percent: Literal[5, 10, 15, 20, 25, 30, 35, 40, 45] | None = None,
    ) -> dict[str, Any]:
        """Update home-mode humidifier and dehumidification settings.

        Args:
            system_serial: Serial number of the system to update.
            humidifier_on: When ``False``, disable humidification. ``None``
                leaves the default manual-on mutation payload in place.
            over_cooling: Optional over-cooling setting for dehumidification.
            cooling_percent: Optional cooling humidity target in five-percent
                increments accepted by Carrier.
            heating_percent: Optional heating humidity target in five-percent
                increments accepted by Carrier.

        Returns:
            The decoded mutation response.
        """
        variables: dict[str, Any] = {
            "input": {
                "serial": system_serial,
                "humidityHome": {
                    "humid": "manual",
                    "humidifier": "on",
                },
            }
        }

        if humidifier_on is not None and humidifier_on is False:
            variables["input"]["humidityHome"] = {
                "humid": "off",
                "humidifier": "off",
            }
        if over_cooling is not None:
            variables["input"]["humidityHome"]["rclgovercool"] = "on" if over_cooling else "off"
        if cooling_percent is not None:
            variables["input"]["humidityHome"]["rclg"] = cooling_percent / 5
        if heating_percent is not None:
            variables["input"]["humidityHome"]["rhtg"] = heating_percent / 5
        return await self._update_infinity_config(variables)

    async def update_fan(
        self, system_serial: str, zone_id: str, activity_type: ActivityTypes, fan_mode: FanModes
    ) -> dict[str, Any]:
        """Update the fan mode for a zone activity.

        Args:
            system_serial: Serial number of the system to update.
            zone_id: Carrier zone identifier.
            activity_type: Activity whose fan mode should be changed.
            fan_mode: Requested fan mode.

        Returns:
            The decoded mutation response.

        Raises:
            ValueError: If ``fan_mode`` or ``activity_type`` is not a valid enum
                member.
        """
        if fan_mode not in FanModes:
            raise ValueError(f"{fan_mode} is not a valid fan mode")
        if activity_type not in ActivityTypes:
            raise ValueError(f"{activity_type} is not a valid activity type")
        variables = {
            "input": {
                "serial": system_serial,
                "zoneId": zone_id,
                "activityType": activity_type.value,
                "fan": fan_mode.value,
            }
        }
        return await self._update_infinity_zone_activity(variables=variables)

    async def set_config_hold(
        self,
        system_serial: str,
        zone_id: str,
        activity_type: ActivityTypes,
        hold_until: str | None = None,
    ) -> dict[str, Any]:
        """Place a zone on hold for a selected activity.

        Args:
            system_serial: Serial number of the system to update.
            zone_id: Carrier zone identifier.
            activity_type: Activity to hold.
            hold_until: Optional Carrier hold-until time string. ``None`` keeps
                the hold indefinite according to Carrier's API behavior.

        Returns:
            The decoded mutation response.

        Raises:
            ValueError: If ``activity_type`` is not a valid enum member.
        """
        if activity_type not in ActivityTypes:
            raise ValueError(f"{activity_type} is not a valid activity type")
        variables = {
            "input": {
                "serial": system_serial,
                "zoneId": zone_id,
                "hold": "on",
                "holdActivity": activity_type.value,
                "otmr": hold_until,
            }
        }
        return await self._update_infinity_zone_config(variables=variables)

    async def resume_schedule(self, system_serial: str, zone_id: str) -> dict[str, Any]:
        """Clear a zone hold and resume its programmed schedule.

        Args:
            system_serial: Serial number of the system to update.
            zone_id: Carrier zone identifier.

        Returns:
            The decoded mutation response.
        """
        variables = {
            "input": {
                "serial": system_serial,
                "zoneId": zone_id,
                "hold": "off",
                "holdActivity": None,
                "otmr": None,
            }
        }
        return await self._update_infinity_zone_config(variables=variables)

    async def set_config_activity(
        self,
        system_serial: str,
        zone_id: str,
        activity_type: ActivityTypes,
        heat_set_point: str,
        cool_set_point: str,
        fan_mode: FanModes | None = None,
    ) -> dict[str, Any]:
        """Update a zone's activity set points and optional fan mode.

        Args:
            system_serial: Serial number of the system to update.
            zone_id: Carrier zone identifier.
            activity_type: Activity to update.
            heat_set_point: Optional requested heat set point as Carrier expects it.
            cool_set_point: Optional requested cool set point as Carrier expects it.
            fan_mode: Optional fan mode to include in the activity update.

        Returns:
            The decoded mutation response.

        Raises:
            ValueError: If ``fan_mode`` or ``activity_type`` is not a valid enum
                member.
        """
        if activity_type not in ActivityTypes:
            raise ValueError(f"{activity_type} is not a valid activity type")
        variables = {
            "input": {
                "serial": system_serial,
                "zoneId": zone_id,
                "activityType": activity_type.value,
                "htsp": heat_set_point,
                "clsp": cool_set_point,
            }
        }
        if fan_mode is not None:
            if fan_mode not in FanModes:
                raise ValueError(f"{fan_mode} is not a valid fan mode")
            variables["input"]["fan"] = fan_mode.value
        return await self._update_infinity_zone_activity(variables=variables)

    async def set_config_manual_activity(
        self,
        system_serial: str,
        zone_id: str,
        heat_set_point: str,
        cool_set_point: str,
        fan_mode: FanModes | None = None,
    ) -> dict[str, Any]:
        """Update a zone's manual activity set points and optional fan mode.

        Args:
            system_serial: Serial number of the system to update.
            zone_id: Carrier zone identifier.
            heat_set_point: Requested heat set point as Carrier expects it.
            cool_set_point: Requested cool set point as Carrier expects it.
            fan_mode: Optional fan mode to include in the manual activity update.

        Returns:
            The decoded mutation response.

        Raises:
            ValueError: If ``fan_mode`` is supplied and is not a valid enum
                member.
        """
        return await self.set_config_activity(
            system_serial=system_serial,
            zone_id=zone_id,
            activity_type=ActivityTypes.MANUAL,
            heat_set_point=heat_set_point,
            cool_set_point=cool_set_point,
            fan_mode=fan_mode,
        )
