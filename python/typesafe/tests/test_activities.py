"""The ``ask`` Activity against a fake TypeSafe HTTP transport."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from typesafe_sdk import AsyncTypeSafeClient

from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.typesafe._activities import TypeSafeActivities
from temporalio.typesafe._types import AskInput, register_response_model
from temporalio.typesafe._workflow import ACTIVITY_ASK
from tests.helpers.activity_info import fake_info
from tests.helpers.fake_typesafe import (
    BillingResponse,
    canned_client,
    fake_response,
    fake_typesafe_responder,
)


def _activities() -> tuple[TypeSafeActivities, AsyncTypeSafeClient]:
    client = canned_client()
    support = TypeSafeActivities(lambda: client)
    return support, client


def _activity(support: TypeSafeActivities) -> Any:
    """Return the registered ``ask`` function by its Temporal definition name."""
    definitions = [
        (activity._Definition.from_callable(fn), fn) for fn in support.activities
    ]
    [(definition, function)] = definitions
    assert definition is not None
    assert definition.name == "temporalio.typesafe.ask"
    return function


@pytest.mark.asyncio
async def test_ask_sends_all_questions_in_one_request_and_returns_typed_dicts() -> None:
    captured: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> Any:
        captured.append(body)
        return fake_response(
            200,
            {
                "model": "jev-1.13.0",
                "answers": {
                    "spam": {"type": "noul", "noul": 0.1},
                    "tone": {
                        "type": "choice",
                        "choice": "calm",
                        "confidence": 0.95,
                        "probabilities": {"calm": 0.95, "angry": 0.05},
                    },
                },
                "usage": {"input_tokens": 44, "output_tokens": 3},
            },
        )

    support = TypeSafeActivities(lambda: fake_typesafe_responder(handler))
    ask = _activity(support)
    result = await ask(
        AskInput(
            model_name=None,
            state={"message": "Hello"},
            questions={
                "spam": {"type": "noul", "instructions": "Is this spam?"},
                "tone": {"type": "choice", "criteria": {"calm": None, "angry": None}},
            },
        )
    )
    assert result["model"] == "jev-1.13.0"
    assert result["answers"]["spam"] == {"type": "noul", "noul": 0.1}
    assert result["answers"]["tone"]["choice"] == "calm"
    assert result["usage"] == {"input_tokens": 44, "output_tokens": 3}
    [(sent)] = captured
    assert set(sent["questions"]) == {"spam", "tone"}


async def test_activity_name_matches_workflow_proxy_constant() -> None:
    support, _ = _activities()
    ask = _activity(support)
    definition = activity._Definition.from_callable(ask)
    assert definition is not None
    assert definition.name == ACTIVITY_ASK


@pytest.mark.asyncio
async def test_http_500_is_retryable_through_activity() -> None:
    support = TypeSafeActivities(lambda: canned_client(status=500))
    ask = _activity(support)
    with pytest.raises(ApplicationError) as err:
        await ask(
            AskInput(
                model_name=None,
                state="state",
                questions={"s": {"type": "noul", "instructions": "yes?"}},
            )
        )
    assert err.value.non_retryable is False


@pytest.mark.asyncio
async def test_http_422_is_non_retryable_through_activity() -> None:
    support = TypeSafeActivities(lambda: canned_client(status=422))
    ask = _activity(support)
    with pytest.raises(ApplicationError) as err:
        await ask(
            AskInput(model_name=None, state="state", questions={"s": {"type": "noul"}})
        )
    assert err.value.non_retryable is True


@pytest.mark.asyncio
async def test_state_passes_through_untouched() -> None:
    captured: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> Any:
        captured.append(body)
        return fake_response(
            200,
            {
                "model": "m",
                "answers": {"n": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    support = TypeSafeActivities(lambda: fake_typesafe_responder(handler))
    ask = _activity(support)
    state = {"nested": {"a": 1}, "items": [1, "two", None]}
    await ask(AskInput(model_name=None, state=state, questions={"n": {"type": "noul"}}))
    assert captured[0]["state"] == state


@pytest.mark.asyncio
async def test_unnamed_model_input_lets_client_default_answer() -> None:
    """A ``None`` model_name reaches the client, which fills its default.

    The recorded input stays exactly what the workflow named: nothing. The
    request body still carries the client's pin, so the served model shows
    up in the Activity result either way.
    """
    captured: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> Any:
        captured.append(body)
        return fake_response(
            200,
            {
                "model": "jev-pin",
                "answers": {"n": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    support = TypeSafeActivities(
        lambda: fake_typesafe_responder(handler, model="jev-pin")
    )
    ask = _activity(support)
    await ask(
        AskInput(model_name=None, state="state", questions={"n": {"type": "noul"}})
    )
    assert captured[0]["model"] == "jev-pin"


@pytest.mark.asyncio
async def test_named_model_input_is_sent_verbatim() -> None:
    """A named model overrides the client default on the request body."""
    captured: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> Any:
        captured.append(body)
        return fake_response(
            200,
            {
                "model": "jev-1.13.0",
                "answers": {"n": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    support = TypeSafeActivities(
        lambda: fake_typesafe_responder(handler, model="jev-pin")
    )
    ask = _activity(support)
    await ask(
        AskInput(
            state="state", questions={"n": {"type": "noul"}}, model_name="jev-1.13.0"
        )
    )
    assert captured[0]["model"] == "jev-1.13.0"


async def _ask_id(client: AsyncTypeSafeClient) -> str | None:
    """Return the captured ``request_id`` for one call through a client."""
    ask = _activity(TypeSafeActivities(lambda: client))
    result = await ask(
        AskInput(model_name=None, state="state", questions={"n": {"type": "noul"}})
    )
    return result["request_id"]


@pytest.mark.asyncio
async def test_registered_subclass_fields_survive_the_result_payload() -> None:
    """The Activity returns every field the registered subclass validates."""
    register_response_model("billing-activity", BillingResponse)
    support = TypeSafeActivities(lambda: canned_client())
    ask = _activity(support)
    result = await ask(
        AskInput(
            state="state",
            questions={"billing": {"type": "noul", "instructions": "yes?"}},
            response_model="billing-activity",
        )
    )
    assert result["billing"] == {"type": "noul", "noul": 0.8}
    assert result["model"] == "jev-1.13.0"


@pytest.mark.asyncio
async def test_unregistered_response_model_name_is_non_retryable() -> None:
    """The registry lookup sits inside the translation block on purpose."""
    support, _ = _activities()
    ask = _activity(support)
    with pytest.raises(ApplicationError) as err:
        await ask(
            AskInput(
                state="state",
                questions={"n": {"type": "noul"}},
                response_model="never-registered",
            )
        )
    assert err.value.non_retryable is True
    assert err.value.type == "TypeSafeClientError"


@pytest.mark.asyncio
async def test_exhausted_attempt_budget_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deadline that already passed leaves a retryable failure, not a zero timeout."""
    monkeypatch.setattr("temporalio.activity.in_activity", lambda: True)
    monkeypatch.setattr(
        "temporalio.activity.info",
        lambda: fake_info(timedelta(seconds=5), elapsed=timedelta(seconds=5.5)),
    )
    support = TypeSafeActivities(lambda: canned_client())
    ask = _activity(support)
    with pytest.raises(ApplicationError) as err:
        await ask(
            AskInput(model_name=None, state="state", questions={"n": {"type": "noul"}})
        )
    assert err.value.non_retryable is False


@pytest.mark.asyncio
async def test_request_id_captured_from_protocol_header() -> None:
    client = canned_client(headers={"x-typesafe-request-id": "req_protocol"})
    assert await _ask_id(client) == "req_protocol"


@pytest.mark.asyncio
async def test_request_id_falls_back_to_standard_header() -> None:
    client = canned_client(headers={"x-request-id": "req_standard"})
    assert await _ask_id(client) == "req_standard"


@pytest.mark.asyncio
async def test_request_id_prefers_protocol_over_standard_header() -> None:
    client = canned_client(
        headers={
            "x-typesafe-request-id": "req_protocol",
            "x-request-id": "req_standard",
        }
    )
    assert await _ask_id(client) == "req_protocol"


@pytest.mark.asyncio
async def test_request_id_absent_is_none() -> None:
    assert await _ask_id(canned_client()) is None
