# support-chatbot

> AI-powered customer support chatbot with full conversation memory. Asks follow-up questions, remembers what was said earlier, and stays in character as a professional support agent. Built on Gemini free tier — $0.00 cost.

## What It Does

Simulates a real customer support conversation in your terminal. The bot remembers everything said earlier in the conversation — ask "what was my first message?" and it knows. It's powered by a configurable system prompt, so the persona, allowed topics, and tone can all be customized per client.

**Built for:**
- Freelance demos of AI chatbot capabilities
- Prototyping customer support bots before full deployment
- Understanding how multi-turn LLM conversations work in production

## Features

| Feature | Description |
|---------|-------------|
| Conversation memory | Remembers full conversation history across all turns |
| Streaming replies | Responses print word-by-word as Gemini generates them — no long silent waits |
| Thinking indicator | Animated `🤖 ⠋ thinking…` spinner while waiting for the first token, so the app never looks frozen |
| Auto-retry with backoff | Rate limits (429), server errors (5xx), and network drops are retried up to 3 times with exponential delays (2s → 4s) before giving up |
| System prompt | Configurable persona in `prompts.py` — change one file to rebrand |
| Context trimming | Keeps last 10 turns to prevent API errors from long histories |
| Graceful exits | Type `exit`, `quit`, or `bye` — or press `Ctrl+C` / `Ctrl+D` — and the session ends cleanly, never with a traceback |
| Error-safe | Missing API key, empty responses, and mid-stream connection drops are all caught and reported clearly on stderr |
| Tested | 10 pytest unit tests covering history helpers and every API error path (mocked — no real API calls needed) |

## Setup

**Requirements:** Python 3.10+, a free [Google AI Studio API key](https://aistudio.google.com/apikey)

```bash
git clone https://github.com/YOUR_USERNAME/support-chatbot.git
cd support-chatbot
python3 -m pip install -r requirements.txt
cp .env.example .env
# Open .env and add your Gemini API key
```

## Usage

```bash
python3 main.py
```

To end the session: type `exit`, `quit`, or `bye` (or press `Ctrl+C`).

## Example Conversation

```
🤖 Nexus Analytics Support — type 'exit' to quit
──────────────────────────────────────────────
You: Hi, I can't log in to my account
🤖 I'm sorry to hear that! Could you let me know if you're seeing a specific
   error message, or is the page just not loading?

You: It says invalid credentials
🤖 That usually means the password doesn't match. Try resetting it via the
   "Forgot Password" link on the login page — it only takes a minute.

You: what was my first message?
🤖 Your first message was that you couldn't log in to your account.
   Is there anything else I can help you with?

You: exit
👋 Goodbye! Have a great day.
```

While the bot is composing a reply you'll see an animated spinner:

```
You: why is my dashboard empty?
🤖 ⠼ thinking…
```

The spinner clears itself the instant the first word arrives, and the reply streams in live.

## How Memory Works

Every message you send — and every bot reply — is stored in a Python list called `history`. When you type a new message, the entire history plus your new message is sent to the Gemini API in one request. The model reads all previous turns and responds with full context. After each turn, the history is trimmed to the last 10 exchanges to prevent API errors from overly long conversations.

## Error Handling

All failures are handled without crashing the chat loop:

| Failure | Behavior |
|---------|----------|
| `GEMINI_API_KEY` missing | Clear startup error, exits with code 1 |
| Rate limit (HTTP 429) | Retried up to 3× with exponential backoff, then a friendly message |
| Server error (5xx) | Same retry-with-backoff treatment |
| Network drop before reply | Retried; if it keeps failing, "Could not get a response. Try again." |
| Network drop mid-reply | Not retried (avoids printing the reply twice) — reported, loop continues |
| Empty / whitespace-only response | Reported as an error, nothing saved to history |

Errors and retry notices go to **stderr** via Python's `logging` module (SDK/library noise is suppressed); normal chat output stays on stdout.

## Running Tests

```bash
python3 -m pip install pytest
python3 -m pytest
```

10 tests cover message building (no mutation of history), context trimming, streaming assembly, retry-then-succeed on rate limits, immediate failure on non-retryable errors, and giving up after max retries. The Gemini client is mocked, so tests run offline in under a second.

## Project Structure

```
support-chatbot/
├── main.py          # REPL loop, streaming API calls, retries, spinner, history management
├── prompts.py       # SYSTEM_PROMPT — bot persona and behavior rules
├── test_main.py     # Pytest suite — 10 tests, fully mocked, no API key needed
├── requirements.txt # google-genai, python-dotenv, httpx
├── .env.example     # Copy to .env and add your Gemini API key
├── .gitignore       # .env blocked from commits
└── README.md        # This file
```

## Tech Stack

- **LLM:** Google Gemini (`gemini-flash-latest`) via `google-genai` SDK
- **Memory:** Python list passed as Gemini's `contents` parameter on every call
- **System prompt:** Passed via `config={"system_instruction": ...}` — constant across all turns
- **Streaming:** `generate_content_stream` with chunk-by-chunk printing; spinner runs in a background thread until the first token lands
- **Resilience:** Exponential-backoff retry loop around transient API/network errors
- **Cost:** Free — Gemini API free tier (1500+ requests/day)

## License

MIT
