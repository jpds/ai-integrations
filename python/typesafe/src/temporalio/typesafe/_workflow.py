"""Workflow-side durable TypeSafe proxy.

:class:`TemporalTypeSafe` schedules the plugin's ``ask`` Activity by name and
rebuilds the SDK's ``SystemOneResponse`` from the recorded payload.
:meth:`ask_all` deduplicates states and fans one durable execution per state,
gathered in input order.

``_errors`` is deliberately not imported here: its ``email.utils`` import
trips the workflow sandbox on a stdlib module workflow code never uses.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Collection, Mapping, Sequence
from datetime import timedelta
from typing import Any

from typesafe_sdk import Choice, Noul, Score, SystemOneResponse

from temporalio import workflow
from temporalio.typesafe._types import (
    AskInput,
    AskResult,
    lookup_response_model,
)
from temporalio.workflow import ActivityConfig

ACTIVITY_ASK = "temporalio.typesafe.ask"
"""Activity name registered by the plugin."""

DEFAULT_START_TO_CLOSE_TIMEOUT = timedelta(seconds=5)
"""Attempt budget the proxy adds when the config names neither timeout.

The Activity clamps its HTTP waits to this budget, so large states need a
longer one."""


def _decode_result(raw: dict[str, Any], response_model_name: str | None) -> AskResult:
    """Rebuild the native TypeSafe response from its recorded payload.

    Validation goes through the JSON form: the Activity records the response
    as ``model_dump(mode="json")``, which stringifies the numeric keys of
    score maps (``probabilities``, ``legend``), and strict ``model_validate``
    rejects those strings while ``model_validate_json`` coerces them exactly
    like the SDK's own decode path.
    """
    model = lookup_response_model(response_model_name) or SystemOneResponse
    return AskResult(
        # The correlation ID rides outside the response's own fields.
        response=model.model_validate_json(
            json.dumps(
                {key: value for key, value in raw.items() if key != "request_id"}
            )
        ),
        # Absent on histories recorded before request-ID capture.
        request_id=raw.get("request_id"),
    )


def _questions_payload(
    questions: Mapping[str, Choice | Noul | Score | dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Convert questions into their wire dictionaries.

    The SDK's ``Choice``/``Noul``/``Score`` models and raw ``dict`` questions
    serialize without local re-validation, so the server's own per-question
    validation stays the source of a 422 instead of a local error.
    """
    payload: dict[str, dict[str, Any]] = {}
    for name, question in questions.items():
        if isinstance(question, dict):
            payload[name] = dict(question)
        elif isinstance(question, (Choice, Noul, Score)):
            # The SDK's omit-none serializer drops unset optionals.
            payload[name] = question.model_dump()
        else:
            raise TypeError(f"unsupported question {name!r}: {type(question).__name__}")
    return payload


class TemporalTypeSafe:
    """A workflow-side view of the ``ask`` Activity.

    Every question in one :meth:`ask` call goes out in a single request
    against one state, and each is answered independently.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        activity_config: ActivityConfig | None = None,
        response_model: str | None = None,
    ) -> None:
        """Set durable-call defaults; per-call keyword overrides win.

        Args:
            model: Model sent with every call, recorded in the Activity input.
                ``None`` records no name, so the client serves its configured
                default (plugin kwarg, ``TYPESAFE_DEFAULT_MODEL`` env, or
                ``jev-latest``).
            activity_config: Temporal options for every backing Activity, such
                as ``start_to_close_timeout``, ``retry_policy``, and
                ``summary``. A 5 s ``start_to_close_timeout`` is added when the
                config names neither timeout. Each call merges its own config
                over these.
            response_model: Registry name from
                ``TypeSafePlugin(response_models=...)`` whose
                ``SystemOneResponse`` subclass validates the response.
                ``None`` uses the SDK's default parsing.
        """
        self._model = model
        config = (
            ActivityConfig(**activity_config) if activity_config else ActivityConfig()
        )
        if (
            config.get("start_to_close_timeout") is None
            and config.get("schedule_to_close_timeout") is None
        ):
            config["start_to_close_timeout"] = DEFAULT_START_TO_CLOSE_TIMEOUT
        self._activity_config: ActivityConfig = config
        self._response_model = response_model

    def _merged_config(self, override: ActivityConfig | None) -> ActivityConfig:
        """The instance's Activity options, with per-call entries over them."""
        if override is None:
            return self._activity_config
        merged = ActivityConfig(**self._activity_config)
        merged.update(override)
        return merged

    async def ask(
        self,
        state: Mapping[str, Any] | Sequence[Any] | str,
        questions: Mapping[str, Choice | Noul | Score | dict[str, Any]],
        *,
        model: str | None = None,
        activity_config: ActivityConfig | None = None,
        response_model: str | None = None,
    ) -> AskResult:
        """Ask all questions about one state in one TypeSafe request.

        Args:
            state: The content all questions refer to (text, dict, or array).
            questions: Named questions, each answered independently against
                the same state.
            model: Per-call model override; supersedes the instance default.
            activity_config: Per-call Activity options, merged over the
                instance's; entries set here win.
            response_model: Per-call registry name override.

        Returns:
            Typed answers keyed by the supplied question names, plus the
            model, per-request token usage, and backend request ID reported
            by the backend.
        """
        raw = await self._execute(
            state=state,
            questions=_questions_payload(questions),
            model=model or self._model,
            activity_config=self._merged_config(activity_config),
            response_model=response_model or self._response_model,
        )
        return _decode_result(raw, response_model or self._response_model)

    async def ask_all(
        self,
        states: Collection[Mapping[str, Any] | Sequence[Any] | str],
        questions: Mapping[str, Choice | Noul | Score | dict[str, Any]],
        *,
        model: str | None = None,
        activity_config: ActivityConfig | None = None,
        response_model: str | None = None,
    ) -> list[AskResult]:
        """Ask the same questions of many states, durably fanned out.

        Identical states share one Activity execution, so a duplicate is not
        billed twice. Deduplication keys on each state's canonical JSON, so
        only JSON-encodable states can be deduped (and passed at all).
        Every unique state is scheduled up front; how many run at once on a
        worker is that Worker's ``max_concurrent_activities`` setting.

        Args:
            states: One state per return slot; duplicates share executions.
            questions: Sent with every state.
            model: Per-call model override.
            activity_config: Per-call Activity options, merged over the
                instance's; entries set here win.
            response_model: Per-call registry name override.

        Returns:
            One result per input state, in input order; each carries that
            state's answers plus the model, token usage, and backend request
            ID of the request that answered it.
        """
        payload = _questions_payload(questions)
        # States are dicts and lists, so dedupe on canonical JSON instead of
        # hashing them.
        keys = [
            json.dumps(state, sort_keys=True, separators=(",", ":")) for state in states
        ]
        unique: dict[str, Any] = {}
        for state, key in zip(states, keys, strict=True):
            unique.setdefault(key, state)
        raw_groups = await self._execute_all(
            states=list(unique.values()),
            questions=payload,
            model=model or self._model,
            activity_config=self._merged_config(activity_config),
            response_model=response_model or self._response_model,
        )
        raw_by_key = dict(zip(unique, raw_groups, strict=True))
        model_name = response_model or self._response_model
        return [_decode_result(raw_by_key[key], model_name) for key in keys]

    async def _execute(
        self,
        *,
        state: Any,
        questions: dict[str, Any],
        model: str | None,
        activity_config: ActivityConfig,
        response_model: str | None,
    ) -> dict[str, Any]:
        """Schedule one durable ask Activity, returning its raw dict."""
        return await workflow.execute_activity(
            ACTIVITY_ASK,
            args=[
                AskInput(
                    state=state,
                    questions=questions,
                    model_name=model,
                    response_model=response_model,
                )
            ],
            **activity_config,
        )

    async def _execute_all(
        self,
        *,
        states: Sequence[Any],
        questions: dict[str, Any],
        model: str | None,
        activity_config: ActivityConfig,
        response_model: str | None,
    ) -> list[dict[str, Any]]:
        """Fan one durable Activity per unique state."""

        async def one(state: Any) -> dict[str, Any]:
            return await self._execute(
                state=state,
                questions=questions,
                model=model,
                activity_config=activity_config,
                response_model=response_model,
            )

        return await asyncio.gather(*(one(s) for s in states))
