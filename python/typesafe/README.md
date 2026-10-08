# Temporal TypeSafe integration

> Release stage: [Pre-release](https://docs.temporal.io/develop/python/integrations/typesafe).

Temporal integration for [TypeSafe](https://docs.typesafe.ai/concepts/system-one)
decision calls, published as [`temporalio-typesafe`](https://pypi.org/project/temporalio-typesafe/)
and imported as `temporalio.typesafe`.

One Activity is one `POST /v1/systemone`: a state plus any number of questions
(choice, score, noul), answered in a single request. Per-call thresholds and
confidence routing stay in workflow code.

## Install

```bash
uv add temporalio-typesafe
```

## Usage

Register `TypeSafePlugin` on the worker. The HTTP client and its credentials stay
there, out of workflow history:

```python
from temporalio.client import Client
from temporalio.typesafe import TypeSafePlugin

client = await Client.connect(
    "localhost:7233",
    plugins=[
        TypeSafePlugin(
            # base_url="http://localhost:8000",  # endpoint override
            # model="jev-1.13.0",  # pin an exact version
            # api_key=os.environ["BACKEND_API_KEY"],  # credential sent to base_url, instead of TYPESAFE_API_KEY
            # headers={"X-Request-Source": "triage"},
            # timeout=httpx2.Timeout(30, connect=5),  # one network operation
        )
    ],
)
```

Workflow code asks questions through `TemporalTypeSafe`:

```python
from typing import Any

from temporalio import workflow
from temporalio.typesafe.workflow import TemporalTypeSafe
from typesafe_sdk import Noul, NoulAnswer


@workflow.defn
class Triage:
    """Rank items by how likely each needs attention today."""

    @workflow.run
    async def run(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = await TemporalTypeSafe().ask_all(
            items,
            questions={
                "needs_attention": Noul(
                    instructions="Does this item need attention today?",
                )
            },
        )
        ranked = []
        for item, result in zip(items, results):
            answer = result.response.answers.get("needs_attention")
            if not isinstance(answer, NoulAnswer):
                continue
            ranked.append({"id": item["id"], "urgency": answer.noul})
        return sorted(ranked, key=lambda d: -d["urgency"])
```

The example uses `ask_all`, which fans one durable execution per state. The
API has two entry points:

- `ask(state, questions)`: several questions over one state in a single
  request. Add more questions as more keys in the mapping; they are
  evaluated in parallel.
- `ask_all(states, questions)`: one execution per state, reusing executions
  for duplicate states (compared as canonical JSON, so states must be
  JSON-encodable). Every unique state is scheduled up front.

The number of ask Activities running at once on a worker is capped by
Temporal's `Worker(max_concurrent_activities=...)` setting. It caps simultaneous
Activity execution across that worker's workflows; the queued batch itself
stays unbounded, so a slow run queues Activities instead of dropping them.

Each result is an `AskResult`: `.response` is the SDK's own response, with
`.answers` keyed by question name, the served `.model`, and the request's token
`.usage`. A `NoulAnswer` carries a single `.noul`: the probability (0-1) that
the answer is yes, with 0.5 meaning undecided. Noul answers have no separate
confidence field; `.noul` is both the answer and the certainty. When the yes/no
boundary is subtle, add `criteria` with `true` and `false` descriptions of what
each outcome means.

Retries ride Temporal's RetryPolicy. The client is built (or validated) with
`typesafe-sdk max_retries=0`, so no retry loop runs inside the SDK; the
server's `retry-after` hint becomes `next_retry_delay`, and the caller's policy
owns the timing. The plugin rejects a client or client factory that enables the
SDK's own retries, and points callers at
`TemporalTypeSafe(activity_config={"retry_policy": ...})` instead.

Payloads at the Workflow/Activity boundary go through the Pydantic payload
converter: the plugin upgrades a default payload converter and leaves an
explicitly configured custom one alone, so a registered response instance
and score maps with integer keys round-trip in both directions.

The ask Activity clamps every HTTP call to the attempt's own budget: one
attempt's HTTP wait fits inside `started_time + start_to_close_timeout`
(plus the earlier `scheduled_time + schedule_to_close_timeout` when set),
less one second of reserve for receiving and decoding. The reserve is
capped at half the remaining budget, so a short attempt still sends a
request, and an attempt whose deadline has already passed fails retryably.
`TypeSafePlugin(timeout=...)` stays a ceiling for one HTTP operation and
defaults to the SDK's 10 s. Raise `start_to_close_timeout` and the HTTP
waits widen into that room; the cap keeps configured
`timeout=`/`httpx2.Timeout` bounds (a `Timeout` object's phases each clamp
independently to the remaining budget, never widening a tighter one).

Tune these layers:

| Layer | What it bounds | Default | Knob |
| --- | --- | --- | --- |
| HTTP operation | One connect, read, write, or pooled-connection acquire | 10 s ceiling | `TypeSafePlugin(timeout=...)` |
| Activity attempt | One whole Activity Task Execution, including all HTTP waits | None in Temporal; 5 s in `TemporalTypeSafe` | `start_to_close_timeout` in `activity_config`, on `TemporalTypeSafe(...)` or on a call |
| Total duration | Queueing, every attempt, retries, and backoff | unbounded | `schedule_to_close_timeout` in `activity_config`, on `TemporalTypeSafe(...)` or on a call |

```python
# Worker side: tune the ceiling for fine-grained bounds
TypeSafePlugin(timeout=httpx2.Timeout(30, connect=5), timeout_reserve=2.0)

# Workflow side: widen the budget and bound the overall deadline even
# when it retries.
s1 = TemporalTypeSafe(
    activity_config={
        "start_to_close_timeout": timedelta(seconds=45),
        "schedule_to_close_timeout": timedelta(minutes=5),
    }
)
await s1.ask_all(items, questions)
```

Three notes tie this to the retry behavior above. First, the HTTP ceiling
and each `httpx2` kind are inactivity bounds for one operation, not a total
request deadline; raising the ceiling alone does not lengthen a slow read
if the attempt budget stays fixed. Second, `start_to_close_timeout`
restarts per attempt, so a retried ask's HTTP waits do not accumulate
inside it; `schedule_to_close_timeout` is the only knob that covers
attempts end to end, which matters when the server's `retry-after` hints
stretch the total. Third, a timed-out attempt does not stop the Worker's
request: without heartbeats no cancellation gets delivered, so the SDK's
HTTP wait runs to its own cap and the provider may process the ask even
though the attempt is recorded as timed out.

## Bring your own client

For a backend whose settings come from elsewhere in your program, pass a
constructed `typesafe_sdk.AsyncTypeSafeClient`:

```python
plugin = TypeSafePlugin(client=client)  # any explicit client setting rejects here
```

The plugin shares that instance across loops and Workers and never closes it;
call `await client.aclose()` yourself. `client_factory=` instead builds a
fresh client per Worker loop and closes it when that loop's last Worker
stops. SDK-level retries are rejected on both paths.

## Typed responses

Questions are the SDK's native `Noul`, `Score`, and `Choice` models (or
plain dicts with the same wire shape). To get SDK-validated
answers, register the response class on the plugin and name it on the call.
The workflow receives answers on the `AskResult` envelope and decodes them
as usual.

```python
from typesafe_sdk import NoulAnswer, SystemOneResponse


class BillingResponse(SystemOneResponse):
    billing: NoulAnswer


# Worker side:
TypeSafePlugin(response_models={"billing": BillingResponse})

# Workflow side:
result = await TemporalTypeSafe().ask(
    state,
    {"billing": Noul(instructions="Is this about billing?")},
    response_model="billing",
)

answer = result.response.answers["billing"]
assert isinstance(answer, NoulAnswer)
```

Naming is a `str` because the model class itself cannot cross the
Workflow/Activity boundary durably; the registry lives worker-side. Names must
map to `SystemOneResponse` subclasses; the plugin rejects other classes at
construction, and an unregistered name fails the Activity without retrying.

## Configuration

The `typesafe` SDK reads these environment variables; a `TypeSafePlugin(...)`
constructor kwarg with the same meaning takes precedence. The plugin builds
the SDK client when it is constructed, or validates the one you inject, so
bad settings fail at worker startup with the SDK's error.

- `TYPESAFE_API_KEY`: key for requests to the API. Required unless the
  plugin kwarg sets one.
- `TYPESAFE_BASE_URL`: endpoint override.
- `TYPESAFE_DEFAULT_MODEL`: model used when the call doesn't name one. The
  call can name one per question set (`TemporalTypeSafe(model=...)` or a
  per-call `model=`), which the Activity input also records.
  Pin an exact version (`jev-1.13.0`, not `jev-latest`) once you have tuned
  thresholds: calibration can change between releases.
- `TYPESAFE_LOG_LEVEL`: SDK logging level.

A `None` plugin kwarg always defers to these variables; only an explicit
kwarg overrides them.

## Develop

```bash
make sync   # install (non-editable) into .venv
make lint
make test
```
