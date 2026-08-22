import pytest

from app.config import get_settings
from app.llm_gateway import factory
from app.llm_gateway.anthropic_adapter import AnthropicAdapter
from app.llm_gateway.gemini_adapter import GeminiAdapter


@pytest.fixture(autouse=True)
def _reset_factory_singleton_and_settings(monkeypatch):
    # get_llm_adapter() caches its result in a module-level global;
    # reset it around each test so tests don't leak the adapter picked
    # by an earlier test/provider setting.
    factory._adapter = None
    get_settings.cache_clear()
    yield
    factory._adapter = None
    get_settings.cache_clear()


def test_anthropic_provider_returns_anthropic_adapter(monkeypatch):
    monkeypatch.setenv("TARGET_LLM_PROVIDER", "anthropic")

    adapter = factory.get_llm_adapter()

    assert isinstance(adapter, AnthropicAdapter)


def test_gemini_provider_returns_gemini_adapter(monkeypatch):
    # genai.Client validates its api_key eagerly at construction (unlike
    # AsyncAnthropic, which only fails on an actual call) — a dummy
    # non-empty key is enough since this test never calls generate().
    monkeypatch.setenv("TARGET_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-construction-only")

    adapter = factory.get_llm_adapter()

    assert isinstance(adapter, GeminiAdapter)


def test_provider_name_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("TARGET_LLM_PROVIDER", "GEMINI")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-construction-only")

    adapter = factory.get_llm_adapter()

    assert isinstance(adapter, GeminiAdapter)


def test_unknown_provider_raises_value_error(monkeypatch):
    monkeypatch.setenv("TARGET_LLM_PROVIDER", "not-a-real-provider")

    with pytest.raises(ValueError, match="Unknown TARGET_LLM_PROVIDER"):
        factory.get_llm_adapter()


def test_result_is_a_singleton_across_calls(monkeypatch):
    monkeypatch.setenv("TARGET_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key-for-construction-only")

    first = factory.get_llm_adapter()
    second = factory.get_llm_adapter()

    assert first is second
