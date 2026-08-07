import logging
import os
import sys
import threading
import time
from collections.abc import Generator

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

from prompts import SYSTEM_PROMPT

load_dotenv()

# Tried in order: if a model is unavailable, rate-limited, or the server
# is busy, the next one is tried before backing off and retrying.
MODEL_NAMES = ["gemini-flash-latest", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
MAX_HISTORY_TURNS = 10
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0

logger = logging.getLogger(__name__)


def build_messages(history: list[dict], user_input: str) -> list[dict]:
    """Build a new messages list with the user input appended.

    Args:
        history: List of role/parts dicts, not mutated.
        user_input: String from input().

    Returns:
        New list with user turn appended — does NOT mutate history in place.
    """
    messages = list(history)
    messages.append({"role": "user", "parts": [{"text": user_input}]})
    return messages


def trim_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """Return the last max_turns conversation pairs from history.

    Args:
        history: Full list of role/parts dicts.
        max_turns: Number of user/model pairs to keep.

    Returns:
        Trimmed list keeping only the last (max_turns * 2) items; shorter
        histories are returned unchanged.
    """
    return history[-(max_turns * 2):]


def _spin(stop: threading.Event) -> None:
    """Animate a thinking indicator until stop is set, then clear the line."""
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while not stop.is_set():
        print(f"\r🤖 {frames[i % len(frames)]} thinking…", end="", flush=True)
        i += 1
        stop.wait(0.1)
    print("\r\033[K", end="", flush=True)


def _start_spinner() -> tuple[threading.Event, threading.Thread]:
    """Start the thinking spinner in a background thread."""
    stop = threading.Event()
    thread = threading.Thread(target=_spin, args=(stop,), daemon=True)
    thread.start()
    return stop, thread


def _stop_spinner(stop: threading.Event, thread: threading.Thread) -> None:
    """Stop the spinner and wait for it to clear its line. Safe to call twice."""
    stop.set()
    thread.join()


def _is_retryable(exc: Exception) -> bool:
    """Return True for errors worth a model fallback or retry
    (model unavailable, 429, 5xx, network)."""
    if isinstance(exc, genai_errors.ClientError):
        return exc.code in (404, 429)
    return isinstance(exc, (genai_errors.ServerError, httpx.RequestError))


def stream_reply(
    client: genai.Client,
    messages: list[dict],
    *,
    _use_spinner: bool = False,
) -> Generator[str, None, str | None]:
    """Yield a Gemini reply chunk by chunk and return the full text.

    Tries each model in MODEL_NAMES in order, falling back to the next
    when one is unavailable, rate-limited, or the server is busy. If
    every model fails, retries the whole list with exponential backoff.
    A failure after output has started is not retried, to avoid
    yielding the reply twice.

    Args:
        client: Configured Gemini client.
        messages: Full conversation history including the current turn.
        _use_spinner: Internal flag — True only from run_chat() (CLI path).
            The terminal spinner must NOT run from the Streamlit path because
            Thread.start() + Thread.join() add blocking overhead on every model
            attempt with no visible effect in the browser.

    Yields:
        Each text chunk as it arrives from the model.

    Returns:
        The complete response text, or None if the request failed.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        for model_name in MODEL_NAMES:
            chunks: list[str] = []
            stop = spinner = None
            if _use_spinner:
                stop, spinner = _start_spinner()
            try:
                stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=messages,
                    config={"system_instruction": SYSTEM_PROMPT},
                )
                for chunk in stream:
                    if chunk.text:
                        if not chunks and _use_spinner and stop and spinner:
                            _stop_spinner(stop, spinner)
                            stop = spinner = None
                        chunks.append(chunk.text)
                        yield chunk.text
                if _use_spinner and stop and spinner:
                    _stop_spinner(stop, spinner)
                    stop = spinner = None
                if chunks and _use_spinner:
                    print()
                text = "".join(chunks).strip()
                if not text:
                    logger.error("Empty response from Gemini.")
                    return None
                return text
            except (genai_errors.APIError, httpx.RequestError) as exc:
                if _use_spinner and stop and spinner:
                    _stop_spinner(stop, spinner)
                    stop = spinner = None
                if chunks:
                    if _use_spinner:
                        print()
                    logger.error("Connection lost mid-response: %s", exc)
                    return None
                if not _is_retryable(exc):
                    logger.error("Gemini request failed: %s", exc)
                    return None
                logger.warning("%s failed (%s), trying next model…", model_name, exc)
            finally:
                # Guarantee spinner is cleaned up even on unexpected exceptions.
                if _use_spinner and stop and spinner:
                    _stop_spinner(stop, spinner)
        if attempt == MAX_RETRIES:
            break
        delay = RETRY_BASE_DELAY * 2 ** (attempt - 1)
        logger.warning("All models failed, retrying in %.0fs…", delay)
        time.sleep(delay)
    logger.error("Gemini request failed after trying all models.")
    return None


def run_chat(client: genai.Client) -> None:
    """Run the main REPL loop for the support chatbot.

    Args:
        client: Configured Gemini client, created once at startup.
    """
    history: list[dict] = []
    print("🤖 Nexus Analytics Support — type 'exit' to quit")
    print("─" * 46)
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            print("Please type a message.")
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("👋 Goodbye! Have a great day.")
            return
        messages = build_messages(history, user_input)
        first = True
        response: str | None = None
        stream = stream_reply(client, messages, _use_spinner=True)
        while True:
            try:
                chunk = next(stream)
            except StopIteration as done:
                response = done.value
                break
            if first:
                print("🤖 ", end="", flush=True)
                first = False
            print(chunk, end="", flush=True)
        if response is None:
            print("❌ Could not get a response. Try again.")
            continue
        history.append({"role": "user", "parts": [{"text": user_input}]})
        history.append({"role": "model", "parts": [{"text": response}]})
        history = trim_history(history)


def main() -> None:
    """Entry point — validate environment then start the chat loop."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s", stream=sys.stderr)
    logger.setLevel(logging.INFO)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)
    client = genai.Client(api_key=api_key)
    try:
        run_chat(client)
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Goodbye! Have a great day.")


if __name__ == "__main__":
    main()
