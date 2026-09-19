"""Tests for attribution headers on the default model activity OpenAI client."""

import importlib.metadata

import pytest
from agents import (
    OpenAIProvider,
)
from agents import __version__ as agents_version
from openai import AsyncOpenAI

from temporalio import service as temporalio_service
from temporalio.openai_agents._invoke_model_activity import (
    ModelActivity,
    _default_openai_headers,
)


# ModelActivity constructs AsyncOpenAI eagerly, which requires an API key.
# Tests must not require real provider credentials.
@pytest.fixture(autouse=True)
def openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


class TestAttributionHeaders:
    def _default_client(self) -> AsyncOpenAI:
        model_activity = ModelActivity()
        provider = model_activity._model_provider  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        assert isinstance(provider, OpenAIProvider)
        client = provider._client  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        assert isinstance(client, AsyncOpenAI)
        return client

    async def test_default_client_attribution_headers(self):
        client = self._default_client()

        headers = client.default_headers
        plugin_version = importlib.metadata.version("temporalio-openai-agents")
        assert headers["User-Agent"] == (
            f"Temporal/{temporalio_service.__version__} "
            f"openai-agents/{agents_version} "
            f"temporalio-openai-agents/{plugin_version}"
        )
        assert headers["HTTP-Referer"] == "https://temporal.io/"
        assert headers["X-Title"] == "Temporal OpenAI Agents"

    async def test_user_provided_provider_not_modified(self):
        own_client = AsyncOpenAI(api_key="test-key", max_retries=0)
        model_activity = ModelActivity(
            model_provider=OpenAIProvider(openai_client=own_client)
        )
        assert model_activity._model_provider._client is own_client  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
        assert own_client.default_headers.get("HTTP-Referer") is None
        assert own_client.default_headers.get("X-Title") is None
        assert (
            own_client.default_headers.get("User-Agent")
            != _default_openai_headers["User-Agent"]
        )
