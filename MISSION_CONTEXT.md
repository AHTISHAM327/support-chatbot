# support-chatbot — MISSION_CONTEXT

## What this project is

Terminal chatbot using Gemini, with conversation memory (plain Python list, no vector DB, no Chroma).

## Current state (Day 12)

- main.py, prompts.py, test_main.py — all working, 10 passing tests
- No app.py yet — still terminal-only

## H3 findings (before building the UI)

- build_messages() and trim_history() — clean, no I/O, reusable as-is
- stream_reply() — mixes retry logic with print() calls, needs refactor before Streamlit can use it
- Plan: turn stream_reply() into a generator (yield chunks instead of print), so main.py and app.py can both wrap it differently

## Next

H4: refactor stream_reply() to yield, keep main.py working the same way
H5: build app.py using st.write_stream() on the same generator
