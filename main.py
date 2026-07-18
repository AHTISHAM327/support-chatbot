import logging
import os
import sys
import threading
import time

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

from prompts import SYSTEM_PROMPT

load_dotenv()

MODEL_NAME = "gemini-flash-latest"
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


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient errors worth retrying (429, 5xx, network)."""
    if isinstance(exc, genai_errors.ClientError):
        return exc.code == 429
    return isinstance(exc, (genai_errors.ServerError, httpx.RequestError))


def stream_reply(client: genai.Client, messages: list[dict]) -> str | None:
    """Stream a Gemini reply to stdout and return the full text.

    Retries transient failures (rate limit, server, network) with
    exponential backoff. A failure after output has started is not
    retried, to avoid printing the reply twice.

    Args:
        client: Configured Gemini client.
        messages: Full conversation history including the current turn.

    Returns:
        The complete response text, or None if the request failed.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        chunks: list[str] = []
        try:
            stream = client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=messages,
                config={"system_instruction": SYSTEM_PROMPT},
            )
            for chunk in stream:
                if chunk.text:
                    if not chunks:
                        print("🤖 ", end="", flush=True)
                    print(chunk.text, end="", flush=True)
                    chunks.append(chunk.text)
            if not chunks:
                logger.error("Empty response from Gemini.")
                return None
            print()
            return "".join(chunks).strip()
        except (genai_errors.APIError, httpx.RequestError) as exc:
            if chunks:
                print()
                logger.error("Connection lost mid-response: %s", exc)
                return None
            if not _is_retryable(exc) or attempt == MAX_RETRIES:
                logger.error("Gemini request failed: %s", exc)
                return None
            delay = RETRY_BASE_DELAY * 2 ** (attempt - 1)
            logger.warning("Transient error (%s), retrying in %.0fs…", exc, delay)
            time.sleep(delay)
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
        response = stream_reply(client, messages)
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
