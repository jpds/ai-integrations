"""Workflow-side durable TypeSafe calls.

Workflow code imports :class:`TemporalTypeSafe` and :class:`AskResult` from
here. The module imports only workflow-side code, so a sandboxed workflow
never executes the plugin's worker-only imports.
"""

from __future__ import annotations

from temporalio.typesafe._types import AskResult
from temporalio.typesafe._workflow import TemporalTypeSafe

__all__ = [
    "AskResult",
    "TemporalTypeSafe",
]
