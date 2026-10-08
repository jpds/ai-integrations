"""Worker-side ownership of the TypeSafe HTTP client.

The plugin constructs and holds the :class:`AsyncTypeSafeClient`, so
credentials and model defaults stay out of workflow history.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

import httpx2
from typesafe_sdk import (
    AsyncTypeSafeClient,
    RetryPolicy,
    TypeSafeAPITimeoutError,
    TypeSafeError,
)
from typesafe_sdk.constants import DEFAULT_TIMEOUT

from temporalio.activity import Info

TypeSafeClientFactory = Callable[..., AsyncTypeSafeClient]
"""Worker-side client factory. Its parameters are unpacked per call."""

DEFAULT_TIMEOUT_RESERVE = 1.0
"""Seconds held back from the Activity budget for receiving and decoding.

Capped at half the remaining budget, so a short attempt still sends a
request; the SDK rejects a zero timeout.
"""


def accepted_client_kwargs(
    client_factory: TypeSafeClientFactory,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """The settings the factory's signature can hold, like the mcp factory convention.

    A factory declaring ``**kwargs`` receives everything; otherwise each
    keyword-only or keyword-capable parameter selects its namesake, and a
    factory setting left out of the signature raises here unless the plugin
    caller left it unset. A factory declaring no parameters cannot hold the
    plugin's retry policy, which is always set, so it also rejects here.
    """
    try:
        parameters = list(inspect.signature(client_factory).parameters.values())
    except (TypeError, ValueError):
        # Callables without a signature (C builtins, say) get everything and
        # report a real mismatch at the call.
        return dict(settings)
    if parameters and parameters[-1].kind is inspect.Parameter.VAR_KEYWORD:
        return dict(settings)
    kinds = inspect.Parameter
    names = {
        parameter.name
        for parameter in parameters
        if parameter.kind in (kinds.POSITIONAL_OR_KEYWORD, kinds.KEYWORD_ONLY)
    }
    unholdable = {
        name: value
        for name, value in settings.items()
        if name not in names and value is not None
    }
    if unholdable:
        raise TypeSafeError(
            "The client factory cannot hold: "
            + ", ".join(sorted(unholdable))
            + "; declare matching keyword-only parameters or leave them unset"
        )
    return {name: value for name, value in settings.items() if name in names}


def attempt_timeout(
    info: Info | None,
    timeout: float | httpx2.Timeout | None,
    reserve: float | None = DEFAULT_TIMEOUT_RESERVE,
) -> float | httpx2.Timeout:
    """This attempt's HTTP timeout: the deadline fits the Activity budget.

    The deadline is ``start_to_close_timeout`` from the attempt's start, or
    the earlier ``schedule_to_close_timeout`` from its scheduling. The HTTP
    wait is what remains after ``reserve`` seconds for receiving and
    decoding, with the reserve capped at half the remaining budget so a
    short attempt still sends a request. The plugin's ``timeout`` (SDK
    default 10 s) stays a ceiling, and each ``httpx2.Timeout`` phase clamps
    independently to min(configured, budget); when nothing shrinks, the
    object is returned unchanged.

    Raises:
        TypeSafeAPITimeoutError: The deadline passed before the request
            started. Retryable, so the next attempt gets a fresh budget.
    """
    if info is None:
        return timeout if timeout is not None else DEFAULT_TIMEOUT
    deadline: datetime | None = None
    if info.start_to_close_timeout is not None:
        deadline = info.started_time + info.start_to_close_timeout
    if info.schedule_to_close_timeout is not None:
        overall = info.scheduled_time + info.schedule_to_close_timeout
        deadline = overall if deadline is None else min(deadline, overall)
    if deadline is None:
        return timeout if timeout is not None else DEFAULT_TIMEOUT
    ceiling = timeout if timeout is not None else DEFAULT_TIMEOUT
    left = (deadline - datetime.now(timezone.utc)).total_seconds()
    if left <= 0:
        # The deadline passed before the request started; the SDK error carries
        # the configured timeout and stays retryable, so the next attempt gets a
        # fresh budget.
        raise TypeSafeAPITimeoutError(ceiling)
    held_back = min(DEFAULT_TIMEOUT_RESERVE if reserve is None else reserve, left / 2)
    remaining = left - held_back
    if isinstance(ceiling, (int, float)):
        return min(remaining, float(ceiling))
    # Clamp each phase to min(configured, budget); a phase already set to
    # None stays unbounded.
    phases = {
        "connect": ceiling.connect,
        "read": ceiling.read,
        "write": ceiling.write,
        "pool": ceiling.pool,
    }
    clamped = {
        name: None if phase is None else min(phase, remaining)
        for name, phase in phases.items()
    }
    if clamped == phases:
        return ceiling
    return httpx2.Timeout(**clamped)


def factory(
    *,
    base_url: str | None = None,
    model: str | None = None,
    retry: RetryPolicy | None = None,
    api_key: str | None = None,
    timeout: float | httpx2.Timeout | None = None,
    headers: Mapping[str, str] | None = None,
) -> AsyncTypeSafeClient:
    """Build the default :class:`AsyncTypeSafeClient`.

    ``None`` options defer to the SDK's environment variables
    (``TYPESAFE_API_KEY``, ``TYPESAFE_BASE_URL``, ``TYPESAFE_DEFAULT_MODEL``).
    A missing API key raises here, at plugin construction.
    """
    return AsyncTypeSafeClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        retry=retry,
        timeout=timeout,
        headers=headers,
    )


class ClientBackend:
    """Per-loop ownership of the TypeSafe HTTP client.

    The ``AsyncTypeSafeClient`` HTTP pool attaches to the loop that built
    it, so the backend keeps one client per loop and closes it at that
    loop's last lease drop.
    """

    def __init__(
        self,
        client_factory: TypeSafeClientFactory,
        client_kwargs: dict[str, Any],
        *,
        owned: bool = True,
    ) -> None:
        """Store the factory; each loop's client is built when a run takes a lease.

        ``owned=False`` serves one injected client to every loop and leaves
        closing it to the caller that constructed it.
        """
        self._client_factory = client_factory
        self._client_kwargs = client_kwargs
        self._owned = owned
        self._clients: dict[asyncio.AbstractEventLoop, AsyncTypeSafeClient] = {}
        self._leases: dict[asyncio.AbstractEventLoop, int] = {}

    def client(self) -> AsyncTypeSafeClient:
        """Return the running loop's client, building it if absent."""
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None:
            self._clients[loop] = client = self._client_factory(**self._client_kwargs)
        return client

    def hold(self, loop: asyncio.AbstractEventLoop) -> AsyncTypeSafeClient:
        """Take a lease for ``loop``, building that loop's client if absent."""
        client = self._clients.get(loop)
        if client is None:
            self._clients[loop] = client = self._client_factory(**self._client_kwargs)
        self._leases[loop] = self._leases.get(loop, 0) + 1
        return client

    async def drop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Release a lease for ``loop``; the last lease closes its client."""
        remaining = self._leases.get(loop, 0) - 1
        if remaining:
            self._leases[loop] = remaining
            return
        self._leases.pop(loop, None)
        if not self._owned:
            self._clients.pop(loop, None)
            return
        client = self._clients.pop(loop, None)
        if client is not None:
            await client.aclose()
