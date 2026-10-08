"""Plugin construction: settings are validated at Worker startup.

The backend builds the SDK client eagerly when the plugin is constructed, so
a missing key or a bad default model rejects the worker up front, the way the
OpenAI integration's eager ``AsyncOpenAI()`` does. An injected client is
checked in its place: no settings probe, but the SDK retry policy is audited.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import httpx2
import pytest
from typesafe_sdk import (
    RetryPolicy,
    SystemOneResponse,
    TypeSafeAPITimeoutError,
    TypeSafeError,
)
from typesafe_sdk.constants import DEFAULT_TIMEOUT

from temporalio.contrib.pydantic import PydanticPayloadConverter
from temporalio.converter import DataConverter, DefaultPayloadConverter
from temporalio.exceptions import ApplicationError
from temporalio.typesafe._client_backend import attempt_timeout
from temporalio.typesafe._errors import translate_exception
from temporalio.typesafe._plugin import TypeSafePlugin, _data_converter
from temporalio.typesafe._types import lookup_response_model
from tests.helpers.activity_info import fake_info
from tests.helpers.fake_typesafe import canned_client


def test_missing_api_key_fails_at_plugin_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(TypeSafeError, match="TYPESAFE_API_KEY"):
        TypeSafePlugin()


def test_env_api_key_satisfies_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake")
    TypeSafePlugin()


async def test_plugin_model_kwarg_outranks_env_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-from-env")
    plugin = TypeSafePlugin(model="jev-pinned")
    client = plugin._backend.client()
    assert client._config.default_model == "jev-pinned"


async def test_env_model_reaches_client_without_plugin_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-from-env")
    plugin = TypeSafePlugin()
    assert plugin._backend.client()._config.default_model == "jev-from-env"


async def test_unset_sources_fall_back_to_jev_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake")
    monkeypatch.delenv("TYPESAFE_DEFAULT_MODEL", raising=False)
    plugin = TypeSafePlugin()
    assert plugin._backend.client()._config.default_model == "jev-latest"


async def test_env_base_url_reaches_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://localhost:8765")
    plugin = TypeSafePlugin()
    assert plugin._backend.client()._config.base_url == "http://localhost:8765"


async def test_plugin_base_url_kwarg_outranks_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://localhost:8765")
    plugin = TypeSafePlugin(base_url="https://endpoint.example")
    assert plugin._backend.client()._config.base_url == "https://endpoint.example"


class _FakeClient:
    """Stands in for the SDK client while lifetime tests track closes."""

    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


def _fake_client_factory() -> Callable[..., _FakeClient]:
    """A factory returning a fresh client per build, so identity shows rebuilds."""

    def factory(**_kwargs: Any) -> _FakeClient:
        return _FakeClient()

    return factory


async def test_one_run_exit_keeps_client_for_surviving_run() -> None:
    """A shared plugin's client outlives one stopped Worker or done Replayer."""
    plugin = TypeSafePlugin(client_factory=_fake_client_factory())
    enter_run = plugin.run_context
    assert enter_run is not None

    async with enter_run():
        shared = plugin._backend.client()
        assert isinstance(shared, _FakeClient)
        nested = enter_run()
        await nested.__aenter__()
        await nested.__aexit__(None, None, None)
        assert plugin._backend.client() is shared and shared.closed == 0

    assert shared.closed == 1


async def test_backend_leaves_no_state_for_closing_loop() -> None:
    """After the last lease, the next run builds its own client."""
    plugin = TypeSafePlugin(client_factory=_fake_client_factory())
    enter_run = plugin.run_context
    assert enter_run is not None

    async with enter_run():
        first = plugin._backend.client()
        assert isinstance(first, _FakeClient)
    assert first.closed == 1

    async with enter_run():
        rebuilt = plugin._backend.client()
        assert isinstance(rebuilt, _FakeClient)
    assert rebuilt is not first and rebuilt.closed == 1


def _capturing_factory(
    **accepted: Any,
) -> tuple[Any, dict[str, Any]]:
    """A factory recording its call, holding only the parameters it declares."""
    received: dict[str, Any] = {}

    def factory(**kwargs: Any) -> _FakeClient:
        received.update(kwargs)
        return _FakeClient()

    factory.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=[
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
            for name in accepted
        ]
    )
    return factory, received


async def test_factory_receives_only_declared_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake")
    factory, received = _capturing_factory(base_url=None, model=None, retry=None)
    plugin = TypeSafePlugin(base_url="https://endpoint.example", client_factory=factory)
    plugin._backend.client()
    assert set(received) >= {"base_url", "model", "retry"}
    assert received["base_url"] == "https://endpoint.example"


def test_factory_unholdable_setting_rejects_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake")
    factory, _received = _capturing_factory(base_url=None, model=None)
    with pytest.raises(TypeSafeError, match="cannot hold.*timeout"):
        TypeSafePlugin(timeout=60, client_factory=factory)


async def test_factory_parameterless_rejects_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake")
    factory, _received = _empty_factory()
    # The plugin always holds a retry policy, so a parameterless factory can
    # never receive it honestly.
    with pytest.raises(TypeSafeError, match="cannot hold: retry"):
        TypeSafePlugin(client_factory=factory)


def test_attempt_timeout_outside_activity_is_the_configured_ceiling() -> None:
    assert attempt_timeout(None, None) == DEFAULT_TIMEOUT
    assert attempt_timeout(None, 5.0) == 5.0


def test_attempt_timeout_clamps_to_remaining_budget() -> None:
    info = fake_info(timedelta(seconds=45))
    # Ceiling 300 s stays under the far-away budget; a 1 s reserve applies.
    clamped = attempt_timeout(info, 300.0)
    assert isinstance(clamped, float) and 43 < clamped < 45


def test_attempt_timeout_short_attempt_keeps_positive_budget() -> None:
    """A one-second attempt still sends a request; the SDK rejects a zero."""
    clamped = attempt_timeout(fake_info(timedelta(seconds=1)), None)
    assert isinstance(clamped, float) and 0 < clamped <= 1


def test_attempt_timeout_exhausted_budget_is_retryable() -> None:
    """A deadline that already passed fails retryably."""
    info = fake_info(timedelta(seconds=5), elapsed=timedelta(seconds=5.5))
    with pytest.raises(TypeSafeAPITimeoutError) as raised:
        attempt_timeout(info, None)
    with pytest.raises(ApplicationError) as err:
        translate_exception(raised.value)
    assert err.value.non_retryable is False


def test_attempt_timeout_keeps_ceiling_under_wide_budget() -> None:
    info = fake_info(timedelta(seconds=300))
    assert attempt_timeout(info, 30.0) == 30.0


def test_attempt_timeout_uses_earlier_schedule_to_close() -> None:
    info = fake_info(timedelta(seconds=600), schedule_to_close=timedelta(seconds=10))
    # 10 s overall budget minus the 1 s reserve beats the wider budget.
    clamped = attempt_timeout(info, 300.0)
    assert isinstance(clamped, float) and 8 < clamped <= 10


def test_attempt_timeout_shrinks_fine_tuned_timeout_object() -> None:
    info = fake_info(timedelta(seconds=45))
    fine = httpx2.Timeout(300.0, connect=60.0)
    clamped = attempt_timeout(info, fine)
    assert isinstance(clamped, httpx2.Timeout)
    assert (
        clamped.connect is not None
        and clamped.read is not None
        and 43 < clamped.connect < 45
        and 43 < clamped.read < 45
    )
    # When the budget still holds the widest phase, the object survives.
    fine_narrow = httpx2.Timeout(30.0, connect=60.0)
    info_wide = fake_info(timedelta(seconds=90))
    assert attempt_timeout(info_wide, fine_narrow) is fine_narrow


def test_attempt_timeout_preserves_shorter_configured_phase() -> None:
    """A connect bound tighter than the budget must not loosen to it."""
    info = fake_info(timedelta(seconds=5))
    fine = httpx2.Timeout(30.0, connect=0.1)
    clamped = attempt_timeout(info, fine)
    assert isinstance(clamped, httpx2.Timeout)
    assert clamped.connect == 0.1
    assert clamped.read is not None and 3 < clamped.read < 5
    assert clamped.write is not None and 3 < clamped.write < 5
    assert clamped.pool is not None and 3 < clamped.pool < 5


def test_attempt_timeout_clamps_finite_phases_when_one_is_disabled() -> None:
    """A disabled phase stays disabled, but its siblings still clamp."""
    info = fake_info(timedelta(seconds=5))
    fine = httpx2.Timeout(30.0, read=None)
    clamped = attempt_timeout(info, fine)
    assert isinstance(clamped, httpx2.Timeout)
    assert clamped.read is None
    assert clamped.connect is not None and 3 < clamped.connect < 5
    assert clamped.write is not None and 3 < clamped.write < 5
    assert clamped.pool is not None and 3 < clamped.pool < 5


def test_attempt_timeout_keeps_object_with_disabled_phase_under_wide_budget() -> None:
    wide = fake_info(timedelta(seconds=90))
    fine = httpx2.Timeout(30.0, read=None)
    assert attempt_timeout(wide, fine) is fine


def test_attempt_timeout_zero_length_attempt_is_a_timeout() -> None:
    """A zero-length attempt has nothing to spend, so it raises."""
    with pytest.raises(TypeSafeAPITimeoutError):
        attempt_timeout(fake_info(timedelta(seconds=0)), 30.0)


def _empty_factory() -> tuple[Any, dict[str, Any]]:
    received: dict[str, Any] = {}

    def factory(**received_kwargs: Any) -> _FakeClient:
        received.update(received_kwargs)
        return _FakeClient()

    factory.__signature__ = inspect.Signature()  # type: ignore[attr-defined]
    return factory, received


def _monkeypatched_fake_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake")


class _CustomResponse(SystemOneResponse):
    pass


def test_plugin_registers_response_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _monkeypatched_fake_api_key(monkeypatch)
    TypeSafePlugin(
        client_factory=_fake_client_factory(),
        response_models={"plugin-registered": _CustomResponse},
    )
    assert lookup_response_model("plugin-registered") is _CustomResponse


def test_plugin_rejects_non_response_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _monkeypatched_fake_api_key(monkeypatch)
    with pytest.raises(TypeSafeError, match="must subclass"):
        TypeSafePlugin(
            client_factory=_fake_client_factory(),
            response_models={"bad": int},  # type: ignore[dict-item]
        )
    with pytest.raises(TypeSafeError, match="not registered"):
        lookup_response_model("bad")


def test_plugin_duplicate_name_with_other_class_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _monkeypatched_fake_api_key(monkeypatch)
    TypeSafePlugin(
        client_factory=_fake_client_factory(),
        response_models={"plugin-dup": _CustomResponse},
    )

    class Other(SystemOneResponse):
        pass

    with pytest.raises(TypeSafeError, match="already registered"):
        TypeSafePlugin(
            client_factory=_fake_client_factory(),
            response_models={"plugin-dup": Other},
        )


def test_sdk_retries_reject_plugin_retry_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _monkeypatched_fake_api_key(monkeypatch)
    with pytest.raises(TypeSafeError, match="retry_policy"):
        TypeSafePlugin(retry=RetryPolicy(max_retries=1))
    # max_retries=0 stays accepted.
    TypeSafePlugin(retry=RetryPolicy(max_retries=0))


def test_sdk_retries_reject_on_injected_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client built without a retry policy gets the SDK's two-retry default."""
    _monkeypatched_fake_api_key(monkeypatch)
    with pytest.raises(TypeSafeError, match="retry_policy"):
        TypeSafePlugin(client=canned_client())


def test_injected_client_rejects_client_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _monkeypatched_fake_api_key(monkeypatch)
    client = canned_client(retry=RetryPolicy(max_retries=0))
    with pytest.raises(TypeSafeError, match="api_key"):
        TypeSafePlugin(client=client, api_key="key")
    with pytest.raises(TypeSafeError, match="base_url"):
        TypeSafePlugin(client=client, base_url="https://endpoint.example")


async def test_plugin_never_closes_an_injected_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _monkeypatched_fake_api_key(monkeypatch)
    injected = canned_client(retry=RetryPolicy(max_retries=0))
    closes: list[None] = []

    async def spy() -> None:
        closes.append(None)

    monkeypatch.setattr(injected, "aclose", spy)
    plugin = TypeSafePlugin(client=injected)
    enter_run = plugin.run_context
    assert enter_run is not None

    async with enter_run():
        assert plugin._backend.client() is injected
    # The next run reaches the same instance; the plugin closed nothing.
    assert plugin._backend.client() is injected and closes == []


def test_public_api_exposes_workflow_symbols() -> None:
    from temporalio import typesafe as root
    from temporalio.typesafe import workflow as workflow_module

    assert root.TemporalTypeSafe is workflow_module.TemporalTypeSafe
    with pytest.raises(AttributeError):
        root.AskInput  # type: ignore[attr-defined]  # noqa: B018
    with pytest.raises(AttributeError):
        root.DEFAULT_MODEL  # type: ignore[attr-defined]  # noqa: B018


def test_data_converter_upgrades_default() -> None:
    """Composition swaps only the payload converter, and an explicit one wins."""
    assert type(_data_converter(None).payload_converter) is PydanticPayloadConverter
    plain = DataConverter()
    assert type(plain.payload_converter) is DefaultPayloadConverter
    upgraded = _data_converter(plain)
    assert upgraded is not plain
    assert isinstance(upgraded.payload_converter, PydanticPayloadConverter)
    assert type(upgraded.failure_converter) is type(plain.failure_converter)
    # A caller's own payload converter survives untouched.
    custom = DataConverter(payload_converter_class=PydanticPayloadConverter)
    assert _data_converter(custom) is custom
