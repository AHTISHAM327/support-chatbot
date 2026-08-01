from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from google.genai import errors as genai_errors

import main


def make_client(*stream_results):
    """Build a mock Gemini client whose stream call yields the given results.

    Each item in stream_results is either an exception (raised on that call)
    or a list of chunk texts (streamed on that call).
    """
    client = MagicMock()
    effects = []
    for result in stream_results:
        if isinstance(result, Exception):
            effects.append(result)
        else:
            effects.append(iter([SimpleNamespace(text=t) for t in result]))
    client.models.generate_content_stream.side_effect = effects
    return client


def drain(stream):
    """Consume a stream_reply generator and return its final value.

    stream_reply yields text chunks and returns the full joined text (or
    None) via StopIteration, so tests capture that return value here.
    """
    try:
        while True:
            next(stream)
    except StopIteration as done:
        return done.value


def test_build_messages_appends_user_turn():
    history = [{"role": "user", "parts": [{"text": "hi"}]}]
    result = main.build_messages(history, "hello")
    assert result[-1] == {"role": "user", "parts": [{"text": "hello"}]}
    assert len(result) == 2


def test_build_messages_does_not_mutate_history():
    history = [{"role": "user", "parts": [{"text": "hi"}]}]
    main.build_messages(history, "hello")
    assert len(history) == 1


def test_trim_history_keeps_last_pairs():
    history = [{"role": "user", "parts": [{"text": str(i)}]} for i in range(30)]
    trimmed = main.trim_history(history, max_turns=10)
    assert len(trimmed) == 20
    assert trimmed[-1] is history[-1]


def test_trim_history_short_history_unchanged():
    history = [{"role": "user", "parts": [{"text": "hi"}]}]
    assert main.trim_history(history) == history


def test_stream_reply_returns_joined_text():
    client = make_client(["Hello", " there", "."])
    assert drain(main.stream_reply(client, [])) == "Hello there."


def test_stream_reply_empty_response_returns_none():
    client = make_client([])
    assert drain(main.stream_reply(client, [])) is None


def test_stream_reply_whitespace_only_response_returns_none():
    client = make_client([" ", "\n"])
    assert drain(main.stream_reply(client, [])) is None


def test_stream_reply_retries_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr(main.time, "sleep", lambda _: None)
    rate_limit = genai_errors.ClientError(
        429, {"error": {"message": "rate limited", "code": 429}}
    )
    client = make_client(rate_limit, ["ok"])
    assert drain(main.stream_reply(client, [])) == "ok"
    assert client.models.generate_content_stream.call_count == 2


def test_stream_reply_non_retryable_client_error_returns_none():
    bad_request = genai_errors.ClientError(
        400, {"error": {"message": "bad request", "code": 400}}
    )
    client = make_client(bad_request)
    assert drain(main.stream_reply(client, [])) is None
    assert client.models.generate_content_stream.call_count == 1


def test_stream_reply_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(main.time, "sleep", lambda _: None)
    total_calls = main.MAX_RETRIES * len(main.MODEL_NAMES)
    errors = [httpx.ConnectError("down") for _ in range(total_calls)]
    client = make_client(*errors)
    assert drain(main.stream_reply(client, [])) is None
    assert client.models.generate_content_stream.call_count == total_calls
