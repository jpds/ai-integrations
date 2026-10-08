"""TypeSafe Activities: one execution is one ``POST /v1/systemone``.

Every supplied question runs in that one request against one state. Fan-out
over many states lives on the workflow side, where gathering and deduping are
durable. No heartbeats: calls are short, and a hung one is visible in the UI.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeError

from temporalio import activity
from temporalio.typesafe._client_backend import attempt_timeout
from temporalio.typesafe._errors import translate_exception
from temporalio.typesafe._types import AskInput, lookup_response_model

_STANDARD_REQUEST_ID_HEADER = "x-request-id"
"""De-facto standard response header carrying a server-side request ID."""


def _response_request_id(response: Any) -> str | None:
    """The backend's request ID from the response headers, or ``None``.

    Prefers the protocol's ``x-typesafe-request-id`` (via the SDK accessor),
    then falls back to ``x-request-id``.
    """
    try:
        return response.request_id
    except (AttributeError, TypeSafeError):
        pass
    try:
        headers = response.raw_http_response.headers
    except (AttributeError, TypeSafeError):
        return None
    return headers.get(_STANDARD_REQUEST_ID_HEADER)


class TypeSafeActivities:
    """The Activities registered by :class:`temporalio.typesafe.TypeSafePlugin`.

    Args:
        ask: Callable that returns the TypeSafe client for this loop.
    """

    def __init__(
        self,
        ask: Any,
        *,
        timeout: float | httpx2.Timeout | None = None,
        timeout_reserve: float | None = None,
    ) -> None:
        """Construct the holder around one client provider.

        ``timeout`` is the plugin's HTTP ceiling; ``timeout_reserve`` holds
        that much time back from every attempt's budget before the ceiling
        applies, so receiving and decoding always have room.
        """
        self._ask = ask
        self._timeout = timeout
        self._timeout_reserve = timeout_reserve
        self.activities = [
            self.ask_activity,
        ]

    @activity.defn(name="temporalio.typesafe.ask")
    async def ask_activity(self, payload: AskInput) -> dict[str, Any]:
        """Answer every question in one TypeSafe request against one state.

        ``payload.model_name`` names the request's model when the workflow
        named one; ``None`` defers to the client's configured default.
        Either way the response carries the model that actually answered.
        ``payload.response_model`` names a plugin-registered
        ``SystemOneResponse`` subclass to validate against, or ``None`` for
        the SDK's default parsing. Unregistered names fail without retrying.

        Returns plain JSON dicts; the workflow side decodes them into typed
        answers. ``request_id`` carries the backend's request ID from the
        response headers, when it reported one. Any failure, from building the
        provider's client through a 200 whose body fails the response schema,
        is translated by the errors module.
        """
        state_wire: Any = payload.state
        if isinstance(state_wire, Mapping):
            state_wire = dict(state_wire)
        try:
            # Inside the block so an unregistered name fails non-retryably.
            response_model = lookup_response_model(payload.response_model)
            client: AsyncTypeSafeClient = self._ask()
            response = await client.system_one(
                state=state_wire,
                questions=dict(payload.questions),
                model=payload.model_name,
                response_model=response_model,
                timeout=attempt_timeout(
                    activity.info() if activity.in_activity() else None,
                    self._timeout,
                    self._timeout_reserve,
                ),
            )
        except Exception as err:
            translate_exception(err)  # always raises, chaining the original
        # Full serialized model so registered subclass fields survive.
        return {
            **response.model_dump(mode="json"),
            "request_id": _response_request_id(response),
        }
