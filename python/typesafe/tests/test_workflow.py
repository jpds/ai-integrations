"""End-to-end durable calls: workflow to Activity to a fake TypeSafe backend."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any

import pytest
from typesafe_sdk import Choice, Noul, NoulAnswer, RetryPolicy, Score, ScoreAnswer

from temporalio import workflow
from temporalio.client import Client, WorkflowFailureError
from temporalio.exceptions import ApplicationError, TemporalError
from temporalio.typesafe import TypeSafePlugin
from temporalio.typesafe.workflow import AskResult, TemporalTypeSafe
from temporalio.worker import Replayer
from tests.helpers import new_worker
from tests.helpers.fake_typesafe import (
    MODEL,
    BillingResponse,
    canned_client,
    fake_response,
    fake_typesafe_responder,
)

with workflow.unsafe.imports_passed_through():
    # typesafe_sdk reaches the sandbox through the workflow module's import
    # graph; passing it through keeps urllib.request out of module validation.
    import typesafe_sdk  # noqa: F401  # pyright: ignore[reportUnusedImport]


@workflow.defn
class AskOne:
    """Ask three questions of one state and return the raw typed result."""

    @workflow.run
    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        result = await TemporalTypeSafe().ask(
            state,
            {
                "spam": Noul(instructions="Is this spam?"),
                "priority": Score(criteria=["low", "medium", "high"]),
                "target": Choice(criteria={"a": None, "b": None}),
            },
        )
        response = result.response
        return {
            "answers": {
                name: vars(answer) for name, answer in response.answers.items()
            },
            "model": response.model,
            "usage": vars(response.usage),
            "request_id": result.request_id,
        }


@workflow.defn
class AskMany:
    """Fan one score question out over several states."""

    @workflow.run
    async def run(self, states: list[dict[str, Any]]) -> list[dict[str, Any]]:
        answers = await TemporalTypeSafe().ask_all(
            states,
            {"priority": Score(criteria=["low", "high"])},
        )
        return [
            {name: vars(answer) for name, answer in one.response.answers.items()}
            for one in answers
        ]


@workflow.defn
class AskPinned:
    """Ask under an instance default and a per-call override back to back.

    The captured request bodies pin the recorded model: the instance's
    default when the call names none, and the per-call name when it does.
    """

    @workflow.run
    async def run(
        self,
        per_call: str,
    ) -> dict[str, Any]:
        typesafe = TemporalTypeSafe(model="jev-instance-pin")
        default_named = await typesafe.ask(
            "state", {"n": Noul(instructions="Is this spam?")}
        )
        override_named = await typesafe.ask(
            "state",
            {"n": Noul(instructions="Is this spam?")},
            model=per_call,
        )
        return {
            "instance": vars(default_named),
            "per_call": vars(override_named),
        }


@workflow.defn
class AskFailing:
    """Trigger a non-retryable backend rejection through the durable path."""

    @workflow.run
    async def run(self, _mode: str) -> dict[str, Any]:
        result = await TemporalTypeSafe(
            activity_config={"start_to_close_timeout": timedelta(seconds=5)}
        ).ask("state", {"spam": Noul(instructions="Is this spam?")})
        return {name: vars(answer) for name, answer in result.response.answers.items()}


def test_activity_config_merges_per_call_over_instance() -> None:
    """Per-call entries win; the instance's other entries survive the merge."""
    proxy = TemporalTypeSafe(
        activity_config={
            "start_to_close_timeout": timedelta(seconds=30),
            "summary": "instance",
        }
    )
    merged = proxy._merged_config({"summary": "call", "heartbeat_timeout": None})
    assert merged.get("start_to_close_timeout") == timedelta(seconds=30)
    assert merged.get("summary") == "call"
    assert "heartbeat_timeout" in merged


def test_activity_config_adds_the_default_attempt_budget() -> None:
    """The 5 s default applies only when the config names neither timeout."""
    default = TemporalTypeSafe()._merged_config(None)
    assert default.get("start_to_close_timeout") == timedelta(seconds=5)
    scheduled = TemporalTypeSafe(
        activity_config={"schedule_to_close_timeout": timedelta(minutes=1)}
    )._merged_config(None)
    assert scheduled.get("start_to_close_timeout") is None


def _plugin_with_status(status: int) -> TypeSafePlugin:
    """A plugin whose client answers everything with a canned HTTP status."""
    return TypeSafePlugin(client_factory=lambda **kwargs: canned_client(status=status))


async def test_workflow_ask_returns_typed_answers(client: Client) -> None:
    plugin = TypeSafePlugin(client_factory=lambda **kwargs: canned_client())
    async with new_worker(client, AskOne, plugins=[plugin]) as worker:
        handle = await client.start_workflow(
            AskOne.run,
            {"message": "Hello"},
            id=f"ask-one-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        result = await handle.result()

    # Cross-boundary payloads are JSON: int dict keys stringify and the `type`
    # tag stays in the activity payload, so these asserts read the plain JSON
    # result. Typed decoding is pinned in tests/test_types.py.
    assert result["answers"]["spam"]["noul"] == pytest.approx(0.8)
    assert result["answers"]["priority"]["score"] == pytest.approx(1.0)
    assert result["answers"]["priority"]["legend"] == {
        "0": "low",
        "1": "medium",
        "2": "high",
    }
    assert result["answers"]["target"]["choice"] == "a"
    assert result["model"] == "jev-1.13.0"
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 2}
    assert result["request_id"] is None

    assert "temporalio.typesafe.ask" in await _activity_names(handle)
    history = await handle.fetch_history()
    await Replayer(workflows=[AskOne], plugins=[plugin]).replay_workflow(history)


@workflow.defn
class AskRawResult:
    """Return the plugin's own result object across the workflow boundary."""

    @workflow.run
    async def run(self, state: dict[str, Any]) -> AskResult:
        return await TemporalTypeSafe().ask(
            state,
            {
                "spam": Noul(instructions="Is this spam?"),
                "priority": Score(criteria=["low", "high"]),
            },
        )


@workflow.defn
class AskBilling:
    """Decode through a registered response-model subclass end to end."""

    @workflow.run
    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        result = await TemporalTypeSafe().ask(
            state,
            {"billing": Noul(instructions="Is this about billing?")},
            response_model="billing-boundary",
        )
        answer = result.response.answers["billing"]
        assert isinstance(answer, NoulAnswer)
        return {"billing": answer.noul}


async def test_registered_response_model_survives_the_activity_boundary(
    client: Client,
) -> None:
    plugin = TypeSafePlugin(
        client_factory=lambda **kwargs: canned_client(),
        response_models={"billing-boundary": BillingResponse},
    )
    async with new_worker(client, AskBilling, plugins=[plugin]) as worker:
        handle = await client.start_workflow(
            AskBilling.run,
            {"message": "Invoice"},
            id=f"ask-billing-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        result = await handle.result()
        history = await handle.fetch_history()

    assert result["billing"] == pytest.approx(0.8)
    await Replayer(workflows=[AskBilling], plugins=[plugin]).replay_workflow(history)


async def test_workflow_returns_typed_ask_result(client: Client) -> None:
    """The plugin's Pydantic converter round-trips the result object itself.

    The plugin reconfigures the Client, so both the workflow's encoding and the
    client's decoding of the return annotation go through the Pydantic
    converter, restoring native score-map integer keys.
    """
    plugin = TypeSafePlugin(client_factory=lambda **kwargs: canned_client())
    config = client.config()
    config["plugins"] = [plugin]
    typed_client = Client(**config)
    async with new_worker(typed_client, AskRawResult) as worker:
        handle = await typed_client.start_workflow(
            AskRawResult.run,
            {"message": "Hello"},
            id=f"ask-raw-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        result = await handle.result()
        history = await handle.fetch_history()
    await Replayer(workflows=[AskRawResult], plugins=[plugin]).replay_workflow(history)

    assert isinstance(result, AskResult)
    assert result.response.model == MODEL
    spam = result.response.answers["spam"]
    assert isinstance(spam, NoulAnswer) and spam.noul == pytest.approx(0.8)
    priority = result.response.answers["priority"]
    assert isinstance(priority, ScoreAnswer)
    assert priority.probabilities.keys() == {0, 1}
    assert priority.legend.keys() == {0, 1}


async def test_workflow_with_injected_client_survives_a_stopped_worker(
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plugin keeps calling through one injected client and never closes it."""
    injected = canned_client(retry=RetryPolicy(max_retries=0))
    closes: list[None] = []

    async def spy() -> None:
        closes.append(None)

    monkeypatch.setattr(injected, "aclose", spy)
    plugin = TypeSafePlugin(client=injected)

    async with new_worker(client, AskOne, plugins=[plugin]) as worker:
        handle = await client.start_workflow(
            AskOne.run,
            {"message": "First"},
            id=f"ask-injected-first-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        await handle.result()

    async with new_worker(client, AskOne, plugins=[plugin]) as worker:
        handle = await client.start_workflow(
            AskOne.run,
            {"message": "Second"},
            id=f"ask-injected-second-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        result = await handle.result()

    assert closes == []
    assert result["model"] == MODEL


async def test_workflow_records_backend_request_id(client: Client) -> None:
    plugin = TypeSafePlugin(
        client_factory=lambda **kwargs: canned_client(
            headers={"x-typesafe-request-id": "req_durable"}
        )
    )
    async with new_worker(client, AskOne, plugins=[plugin]) as worker:
        handle = await client.start_workflow(
            AskOne.run,
            {"message": "Hello"},
            id=f"ask-request-id-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        result = await handle.result()

    assert result["request_id"] == "req_durable"
    history = await handle.fetch_history()
    await Replayer(workflows=[AskOne], plugins=[plugin]).replay_workflow(history)


async def test_ask_records_named_models_into_requests(client: Client) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> Any:
        bodies.append(body)
        return fake_response(
            200,
            {
                "model": "jev-pin",
                "answers": {"n": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    captured_client = fake_typesafe_responder(handler, model="jev-pin")
    plugin = TypeSafePlugin(client_factory=lambda **kwargs: captured_client)
    async with new_worker(client, AskPinned, plugins=[plugin]) as worker:
        handle = await client.start_workflow(
            AskPinned.run,
            "jev-per-call",
            id=f"ask-pinned-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        await handle.result()

    assert bodies[0]["model"] == "jev-instance-pin"
    assert bodies[1]["model"] == "jev-per-call"


async def test_ask_all_fans_out_and_deduplicates_states(client: Client) -> None:
    plugin = TypeSafePlugin(client_factory=lambda **kwargs: canned_client())
    states = [{"n": 1}, {"n": 2}, {"n": 2}]
    async with new_worker(client, AskMany, plugins=[plugin]) as worker:
        handle = await client.start_workflow(
            AskMany.run,
            states,
            id=f"ask-many-{uuid.uuid4()}",
            task_queue=worker.task_queue,
        )
        result = await handle.result()

    assert [one["priority"]["score"] for one in result] == [
        pytest.approx(0.5),
        pytest.approx(0.5),
        pytest.approx(0.5),
    ]


async def test_non_retryable_backend_rejection_fails_workflow(
    client: Client,
) -> None:
    plugin = _plugin_with_status(422)
    async with new_worker(client, AskFailing, plugins=[plugin]) as worker:
        with pytest.raises(WorkflowFailureError) as failure:
            await client.execute_workflow(
                AskFailing.run,
                "422",
                id=f"ask-fail-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
    root = failure.value
    found: list[ApplicationError] = []
    while isinstance(root, TemporalError):
        if isinstance(root, ApplicationError):
            found.append(root)
        root = root.cause  # type: ignore[assignment]
    [ours] = [e for e in found if e.type == "TypeSafeRequestValidationError"]
    # The SDK's own TypeSafeUnprocessableEntityError rides the chain as a
    # reconstructed ApplicationError whose message carries the request label;
    # MockTransport never opens a socket, so that label is fixture text.
    assert ours.message == "TypeSafe API returned status 422"
    assert ours.non_retryable


async def _activity_names(handle: Any) -> list[str]:
    names: list[str] = []
    async for event in handle.fetch_history_events():
        if event.HasField("activity_task_scheduled_event_attributes"):
            names.append(
                event.activity_task_scheduled_event_attributes.activity_type.name
            )
    return names
