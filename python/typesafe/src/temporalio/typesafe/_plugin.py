"""Worker plugin registering TypeSafe Activities.

:class:`TypeSafePlugin` holds the HTTP client worker-side, so credentials stay out of
workflow history, and appends the ``ask`` Activity to every Worker built.
Workflow code talks to it through
:class:`temporalio.typesafe.TemporalTypeSafe`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx2
from typesafe_sdk import (
    AsyncTypeSafeClient,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeError,
)

from temporalio.contrib.pydantic import PydanticPayloadConverter
from temporalio.converter import DataConverter, DefaultPayloadConverter
from temporalio.plugin import SimplePlugin
from temporalio.typesafe._activities import TypeSafeActivities
from temporalio.typesafe._client_backend import (
    ClientBackend,
    accepted_client_kwargs,
    factory,
)
from temporalio.typesafe._types import register_response_model
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

_SDK_RETRY_MESSAGE = (
    "The SDK must not retry inside one Activity attempt; Temporal owns "
    "retries between attempts and the two loops would compound. Leave the "
    "SDK's retry at max_retries=0 and configure Temporal's policy through "
    "TemporalTypeSafe(activity_config={'retry_policy': ...})."
)

_UNBOUNDED = sys.maxsize
"""Attempt-count stance of a tenacity stop node that caps by time, not count."""


def _reject_sdk_retries(retry: RetryPolicy) -> None:
    """Turn an SDK retry policy that retries into a construction error."""
    if retry.max_retries > 0:
        raise TypeSafeError(_SDK_RETRY_MESSAGE)


def _client_retries_enabled(client: AsyncTypeSafeClient) -> bool:
    """Whether the client's built retry loop performs more than one attempt.

    The SDK collapses the configured :class:`~typesafe_sdk.RetryPolicy` into a
    tenacity stop policy at construction, so the policy itself is gone by the
    time a client is injected.
    """
    return _attempt_cap(client._retry.stop) > 1


def _attempt_cap(stop: Any) -> int:
    """How many attempts one tenacity stop node allows; ``_UNBOUNDED`` is no cap.

    Combined stops trigger when any member fires, so choose the smallest cap.
    """
    stops = getattr(stop, "stops", None)
    if stops is not None:
        return min(_attempt_cap(one) for one in stops)
    attempts = getattr(stop, "max_attempt_number", None)
    if attempts is not None:
        return attempts
    # stop_before_delay and anything unfamiliar cap by time, not by count.
    return _UNBOUNDED


def _data_converter(converter: DataConverter | None) -> DataConverter:
    """Upgrade the default payload converter to the Pydantic one, keeping the rest."""
    if converter is None:
        return DataConverter(payload_converter_class=PydanticPayloadConverter)
    if converter.payload_converter_class is DefaultPayloadConverter:
        return dataclasses.replace(
            converter, payload_converter_class=PydanticPayloadConverter
        )
    return converter


class TypeSafePlugin(SimplePlugin):
    """Register the TypeSafe ``ask`` Activity on a Worker.

    Args:
        base_url: TypeSafe endpoint. ``None`` defers to the SDK's
            ``TYPESAFE_BASE_URL`` environment override, then its canonical
            endpoint.
        model: Model alias or pinned version to send when a call names none.
            ``None`` leaves the default to the SDK's environment variables.
            A plugin pin holds only over the request body; to name the model
            in the Activity input too, set ``TemporalTypeSafe(model=...)``
            workflow-side.
        api_key: API key for requests. ``None`` defers to
            ``TYPESAFE_API_KEY``.
        timeout: Ceiling on one HTTP operation. The ask Activity clamps each
            call under this ceiling to the attempt's remaining
            ``start_to_close_timeout`` budget, less ``timeout_reserve``, so a
            wider Activity budget lets HTTP waits grow and the SDK default
            ceiling (10 s) applies but never overruns it. Pass an
            ``httpx2.Timeout`` to tune connect/read/write/pool separately;
            the clamp replaces it when the budget cannot hold its widest
            phase.
        timeout_reserve: Seconds the clamp holds back from the Activity budget
            for receiving and decoding the answer. Defaults to 1.
        headers: Headers sent with every request.
        client: A pre-configured ``typesafe_sdk.AsyncTypeSafeClient`` to use
            instead of building one from the settings above. Its settings and
            authentication are the caller's, so the other client settings
            (``base_url``, ``model``, ``api_key``, ``headers``, ``retry``,
            ``client_factory``) must be left unset. The plugin shares this one
            instance across loops and Workers and never closes it; close it
            yourself.
        client_factory: Replacement for the built-in
            :func:`temporalio.typesafe._client_backend.factory`, for a
            backend whose endpoint or auth differs. The factory receives the
            constructor settings its signature declares (like the mcp
            plugin's server factories); an explicitly set setting that the
            signature cannot hold rejects here. Each Worker loop builds and
            closes its own client through the factory.
        retry: ``typesafe_sdk.RetryPolicy`` forwarded to the client. Only
            ``max_retries=0`` (the default) is accepted: Temporal owns
            retries between attempts, so an SDK retry loop inside one attempt
            is rejected at construction.
        response_models: Registry for ``TemporalTypeSafe(...,
            response_model="name")`` calls: names mapped to subclasses of
            ``typesafe_sdk.SystemOneResponse``, built once per Worker and
            validated there.

    Raises:
        TypeSafeError: The default client factory rejects the settings, for
            example a missing ``TYPESAFE_API_KEY``, at startup.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | httpx2.Timeout | None = None,
        timeout_reserve: float | None = None,
        headers: Mapping[str, str] | None = None,
        client: AsyncTypeSafeClient | None = None,
        client_factory: Any | None = None,
        retry: RetryPolicy | None = None,
        response_models: Mapping[str, type[SystemOneResponse]] | None = None,
    ) -> None:
        """Configure client options and construct the backing Activity."""
        if retry is not None:
            _reject_sdk_retries(retry)
        if client is not None:
            pinned = {
                "base_url": base_url,
                "model": model,
                "api_key": api_key,
                "headers": headers,
                "retry": retry,
                "client_factory": client_factory,
            }
            if any(value is not None for value in pinned.values()):
                set_name = next(
                    name for name, value in pinned.items() if value is not None
                )
                raise TypeSafeError(
                    f"client is fully configured; leave {set_name} unset or drop client"
                )
            if _client_retries_enabled(client):
                raise TypeSafeError(_SDK_RETRY_MESSAGE)
            backend = ClientBackend(lambda **_kwargs: client, {}, owned=False)
        else:
            if client_factory is None:
                client_factory = factory
            client_kwargs = accepted_client_kwargs(
                client_factory,
                {
                    # None defers base_url to TYPESAFE_BASE_URL env, then the
                    # SDK's canonical endpoint; an explicit kwarg pins it.
                    "base_url": base_url,
                    # model=None defers to TYPESAFE_DEFAULT_MODEL env, then the
                    # SDK's jev-latest; a plugin kwarg pins it outright.
                    "model": model,
                    "api_key": api_key,
                    "timeout": timeout,
                    "headers": headers,
                    "retry": retry if retry is not None else RetryPolicy(max_retries=0),
                },
            )
            # Settings probe: bad settings fail here, before any Worker runs.
            # Real clients are built per loop by ClientBackend.
            client_factory(**client_kwargs)
            backend = ClientBackend(client_factory, client_kwargs)
        self._backend = backend
        self._register_response_models(response_models)
        support = TypeSafeActivities(
            backend.client,
            timeout=timeout,
            timeout_reserve=timeout_reserve,
        )
        self._support = support

        @asynccontextmanager
        async def run_context() -> AsyncGenerator[None, None]:
            """Hold this loop's lease on the backend's client for the run."""
            loop = asyncio.get_running_loop()
            backend.hold(loop)
            try:
                yield
            finally:
                await backend.drop(loop)

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
            if runner is None:
                raise ValueError("No WorkflowRunner provided to the TypeSafe plugin")
            if isinstance(runner, SandboxedWorkflowRunner):
                # httpx2 subclasses urllib.request.Request at import time,
                # which the sandbox's __mro_entries__ rule rejects. The SDK's
                # pydantic question/response models run in the workflow too;
                # annotated_types is outside the SDK's default passthrough,
                # and pydantic's compiled core is not in any default
                # passthrough, so the whole chain passes through.
                return dataclasses.replace(
                    runner,
                    restrictions=runner.restrictions.with_passthrough_modules(
                        "typesafe_sdk",
                        "pydantic",
                        "pydantic_core",
                        "httpx2",
                        "annotated_types",
                    ),
                )
            return runner

        super().__init__(
            name="TypeSafePlugin",
            data_converter=_data_converter,
            activities=support.activities,
            workflow_runner=workflow_runner,
            run_context=run_context,
        )

    @staticmethod
    def _register_response_models(
        response_models: Mapping[str, type[SystemOneResponse]] | None,
    ) -> None:
        """Validate and register into the module-level registry.

        Validation runs at Worker startup; the Activities and workflow code
        then decode by ``lookup_response_model``. A re-registered class
        overwrites nothing usable.
        """
        if response_models is None:
            return
        for name, model in response_models.items():
            if not (isinstance(model, type) and issubclass(model, SystemOneResponse)):
                raise TypeSafeError(
                    f"response_models[{name!r}] must subclass typesafe_sdk.SystemOneResponse"
                )
            register_response_model(name, model)
