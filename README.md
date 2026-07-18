# support-chatbot

AI-powered customer support chatbot for Nexus Analytics, built on the Gemini free tier. Runs as a multi-turn CLI: it remembers the conversation across turns, streams replies as they're generated, and retries transient API failures automatically.

## Features

- **Conversation memory** — keeps the last 10 turns, so follow-up questions work naturally
- **Streaming replies** — responses print token-by-token instead of after a long pause
- **Resilient** — automatic retry with exponential backoff on rate limits, server errors, and network failures
- **Clean exits** — `exit`/`quit`/`bye`, Ctrl+C, and Ctrl+D all end the session gracefully

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste your Gemini API key into .env
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com/apikey).

## Usage

```bash
python3 main.py
```

```
🤖 Nexus Analytics Support — type 'exit' to quit
──────────────────────────────────────────────
You: I can't log in to my dashboard
🤖 I'm sorry to hear that! Are you seeing an error message, or ...
```

## Running tests

```bash
pip install pytest
python3 -m pytest
```

## Project structure

| File | Purpose |
|---|---|
| `main.py` | CLI loop, Gemini streaming client, retry logic |
| `prompts.py` | System prompt defining the support-agent persona |
| `test_main.py` | Unit tests (pure helpers + mocked API calls) |
