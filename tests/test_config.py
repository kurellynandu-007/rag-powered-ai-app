import sys
from types import ModuleType

import pytest

from config import (
    GroqConfigurationError,
    create_groq_client,
    get_groq_model,
)


def test_groq_model_requires_explicit_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    with pytest.raises(
        GroqConfigurationError, match="currently available chat model"
    ):
        get_groq_model()


def test_groq_model_uses_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_MODEL", "configured-chat-model")

    assert get_groq_model() == "configured-chat-model"


def test_groq_client_creation_requires_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-placeholder")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    with pytest.raises(GroqConfigurationError, match="GROQ_MODEL"):
        create_groq_client()


def test_groq_client_creation_requires_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_MODEL", "configured-chat-model")

    with pytest.raises(GroqConfigurationError, match="GROQ_API_KEY is not configured"):
        create_groq_client()


def test_groq_client_receives_configured_key_without_logging(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    supplied: dict[str, str] = {}

    class FakeGroq:
        def __init__(self, *, api_key: str) -> None:
            supplied["api_key"] = api_key

    fake_groq = ModuleType("groq")
    fake_groq.Groq = FakeGroq  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "groq", fake_groq)
    monkeypatch.setenv("GROQ_API_KEY", "test-placeholder")
    monkeypatch.setenv("GROQ_MODEL", "configured-chat-model")

    client = create_groq_client()

    assert isinstance(client, FakeGroq)
    assert supplied == {"api_key": "test-placeholder"}
    assert capsys.readouterr() == ("", "")
