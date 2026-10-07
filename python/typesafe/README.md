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

Register `Plugin` on the worker. The HTTP client and its credentials stay
there, out of workflow history:

```python
from temporalio.client import Client
from temporalio.typesafe import Plugin

client = await Client.connect(
    "localhost:7233",
    plugins=[
        Plugin(
            # base_url="http://localhost:8000",  # self-hosted server
            # model="jev-1.13.0",                # pin instead of "jev-latest"
        )
    ],
)
```

Workflow code asks questions through `TemporalTypeSafe`:

```python
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from temporalio.typesafe import (
        NoulQuestion,
        ScoreQuestion,
        TemporalTypeSafe,
    )


@workflow.defn
class Prioritize:
    @workflow.run
    async def run(self, directors: list[dict[str, int]]) -> dict[str, float]:
        answers = await TemporalTypeSafe().ask_all(
            directors,
            {
                "priority": ScoreQuestion(
                    criteria=["sparse", "nearly complete", "complete"]
                )
            },
        )
        return {
            d["director"]: one["priority"].score for d, one in zip(directors, answers)
        }
```

`ask` batches several questions over one state in a single request; `ask_all`
fans one durable execution per state, reusing executions for duplicate states
and bounding in-flight calls only when you pass `max_concurrent`. Retries ride
Temporal's RetryPolicy: the client sets `typesafe-sdk max_retries=0`, so the
server's `retry-after` hint becomes `next_retry_delay` rather than a hidden
SDK retry loop.

## Configuration

Environment variables come from the typesafe SDK: `TYPESAFE_API_KEY`
(required; missing keys fail at startup), `TYPESAFE_BASE_URL`,
`TYPESAFE_DEFAULT_MODEL`. Plugin constructor kwargs override them.

Confidence is model-specific. Route with a workflow-side threshold and pin the
exact model version once it is tuned: calibration can change between releases.

## Develop

```bash
make sync   # install (non-editable) into .venv
make lint
make test
```
