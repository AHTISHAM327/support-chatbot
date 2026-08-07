"""Streamlit chat UI for the Nexus Analytics support-chatbot.

This is a thin web front-end over the exact same conversation logic the
terminal app uses. It imports build_messages(), trim_history(), and the
stream_reply() generator from main.py and wraps them for the browser —
none of that logic is reimplemented here.

Messages render via st.chat_message(), Streamlit's native chat component,
rather than hand-built flexbox containers. That native component sizes
its own box correctly by construction (no fit-content/shrink-to-fit CSS
required), which is what a hand-rolled version kept getting wrong at the
edges — this rewrite trades a from-scratch layout for a boring, tested
one instead of patching the symptom again.

Run with:  streamlit run app.py
"""

import base64
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

from main import build_messages, stream_reply, trim_history

load_dotenv()

# API key: .env locally, st.secrets when deployed — same pattern as
# document-intelligence/app.py. st.stop() halts before any other st command
# runs, so set_page_config() below still counts as the first Streamlit call.
if not os.environ.get("GEMINI_API_KEY"):
    try:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except Exception:
        st.error(
            "GEMINI_API_KEY not found. Add it to .env locally, "
            "or to Streamlit secrets when deployed."
        )
        st.stop()

ASSISTANT_NAME = "Aira"
COMPANY_NAME = "Nexus Analytics"

EXAMPLE_QUESTIONS = [
    "Reset account credentials",
    "Billing & invoice inquiry",
    "Dashboard configuration",
    "Report a platform issue",
]


def _monogram_avatar(letter: str, bg: str) -> str:
    """Builds a small branded letter-badge avatar as an inline SVG data URI.

    Used instead of an emoji (renders inconsistently across OSes) or a
    stock "robot" icon (reads as a generic AI-chatbot cliché rather than
    a support agent).
    """
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
        f'<rect width="64" height="64" rx="16" fill="{bg}"/>'
        f'<text x="32" y="33" font-family="Arial, sans-serif" font-size="26" '
        f'font-weight="700" fill="#ffffff" text-anchor="middle" '
        f'dominant-baseline="middle">{letter}</text></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


ASSISTANT_AVATAR = _monogram_avatar("A", "#f97316")
USER_AVATAR = _monogram_avatar("U", "#334155")

# Colors/fonts live here because config.toml's [theme] can't express
# per-role message styling or the hero/card layout.
#
# Chat messages are styled via [data-testid="stChatMessage"]:has(...) on
# the aria-label Streamlit puts on stChatMessageContent ("Chat message
# from user"/"from assistant") — a stable, semantic selector, rather than
# targeting emotion-generated classes that change across Streamlit
# versions. Streamlit's own component owns sizing/wrapping for the
# message box, so none of the width/fit-content overrides a hand-rolled
# bubble needs are necessary here.
#
# Passed to st.html() through _iter_css_chunks() rather than in one call —
# empirically, st.html() silently drops an entire <style> tag past a certain
# content size, and the cutoff isn't a clean byte count (confirmed by
# bisecting in a real browser: identical bytes rendered fine standalone but
# vanished as part of a larger block, and vice versa). Slicing into small,
# fixed-size windows sidesteps it reliably regardless of the exact cause.
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── Hero section ──────────────────────────────── */
.hero-wrap {
    text-align: center;
    padding: 2.75rem 1rem 1.5rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.hero-wordmark {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #f97316;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0 0 0.65rem;
    color: #f4f5f7;
    letter-spacing: -0.01em;
}
.hero-subtitle {
    font-size: 0.95rem;
    line-height: 1.6;
    color: rgba(229, 231, 235, 0.65);
    max-width: 32rem;
    margin: 0 auto;
}
/* ── Quick-action cards ────────────────────────── */
.suggest-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin: 0.25rem 0 0.6rem;
}
[class*="st-key-suggest_card_"] button {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    height: 3rem;
    text-align: left;
    font-size: 0.87rem;
    font-weight: 500;
    color: #d1d5db;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}
[class*="st-key-suggest_card_"] button:hover {
    border-color: rgba(249, 115, 22, 0.5);
    background-color: rgba(249, 115, 22, 0.06);
    color: #f4f5f7;
}
/* ── Chat messages ─────────────────────────────── */
[data-testid="stChatMessage"] {
    background-color: transparent;
    border-radius: 12px;
    padding: 0.15rem 0;
    margin-bottom: 0.4rem;
    gap: 0.75rem;
}
[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) [data-testid="stChatMessageContent"] {
    background-color: #161b2c;
    border: 1px solid #232a42;
    border-left: 3px solid #f97316;
    border-radius: 4px 12px 12px 12px;
    padding: 0.75rem 1.1rem;
}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"] {
    background-color: #0f1420;
    border: 1px solid #1e2536;
    border-radius: 4px 12px 12px 12px;
    padding: 0.75rem 1.1rem;
}
[data-testid="stChatMessageContent"] p {
    color: #e2e5ec;
    line-height: 1.6;
    margin: 0 0 0.5em 0;
    font-size: 0.93rem;
}
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }
[data-testid="stChatMessage"] img {
    border-radius: 10px;
}
/* ── Thinking indicator ────────────────────────── */
.thinking-label {
    color: rgba(214, 217, 224, 0.6);
    font-style: italic;
    font-size: 0.9rem;
    animation: thinking-pulse 1.4s ease-in-out infinite;
}
@keyframes thinking-pulse {
    0%, 100% { opacity: 0.45; }
    50%      { opacity: 1; }
}
/* ── Sidebar ───────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #080c14;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
    padding-left: 1.25rem;
    padding-right: 1.25rem;
}
.sidebar-brand-name {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f4f5f7;
    letter-spacing: -0.01em;
    margin: 0;
}
.sidebar-product-label {
    font-size: 0.72rem;
    color: #f97316;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.1rem;
    margin-bottom: 1rem;
    display: block;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 1rem 0;
}
.sidebar-about {
    font-size: 0.87rem;
    line-height: 1.6;
    color: #94a3b8;
    margin-bottom: 1rem;
}
/* ── Global buttons ────────────────────────────── */
.stButton > button {
    border-radius: 7px;
    font-size: 0.87rem;
    font-weight: 500;
    transition: all 0.15s ease;
}
</style>
"""


def _iter_css_chunks(css: str, max_size: int = 400):
    """Yield css split into small chunks, cutting only at top-level rule
    boundaries (brace-depth back to 0) so no selector/declaration/@keyframes
    block is ever split mid-way — see CUSTOM_CSS comment for why chunking
    is needed at all.
    """
    rules = []
    depth = 0
    start = 0
    for i, ch in enumerate(css):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                rules.append(css[start : i + 1])
                start = i + 1
    if css[start:].strip():
        rules.append(css[start:])

    chunk = ""
    for rule in rules:
        if chunk and len(chunk) + len(rule) > max_size:
            yield chunk
            chunk = rule
        else:
            chunk += rule
    if chunk:
        yield chunk


@st.cache_resource
def get_client() -> genai.Client:
    """Create the Gemini client once and reuse it across reruns."""
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def render_history() -> None:
    """Replay stored turns as native chat messages.

    History is kept in the Gemini format that build_messages()/stream_reply()
    expect (role "user"/"model", parts→text), so "model" maps to "assistant".
    """
    for turn in st.session_state.history:
        if turn["role"] == "user":
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(turn["parts"][0]["text"])
        else:
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                st.markdown(turn["parts"][0]["text"])


def render_empty_state() -> None:
    """Centered hero + example-question cards, shown before the first message."""
    st.html(
        '<div class="hero-wrap">'
        f'<span class="hero-wordmark">{COMPANY_NAME}</span>'
        '<h1 class="hero-title">Customer Support Portal</h1>'
        '<p class="hero-subtitle">Get precise answers about your account, billing, '
        "and dashboards — 24 / 7. Select a category below or type your inquiry directly.</p>"
        "</div>"
    )
    st.html('<div class="suggest-section-label">Common Inquiries</div>')
    col_a, col_b = st.columns(2, gap="small")
    for i, label in enumerate(EXAMPLE_QUESTIONS):
        col = col_a if i % 2 == 0 else col_b
        with col:
            if st.button(label, key=f"suggest_card_{i}", use_container_width=True):
                st.session_state.pending_prompt = label
                st.rerun()


def render_sidebar() -> None:
    """Sidebar: brand, About text, live session stats, and a reset button."""
    with st.sidebar:
        st.html(
            f'<p class="sidebar-brand-name">{ASSISTANT_NAME}</p>'
            f'<span class="sidebar-product-label">{COMPANY_NAME} Support</span>'
        )
        st.html('<hr class="sidebar-divider">')
        st.markdown(
            f'<p class="sidebar-about">'
            f"I'm <strong>{ASSISTANT_NAME}</strong>, an automated support specialist for "
            f"<strong>{COMPANY_NAME}</strong>. I handle account, billing, and dashboard "
            "inquiries and stream replies in real time via <strong>Google Gemini</strong>."
            "</p>",
            unsafe_allow_html=True,
        )
        st.html('<hr class="sidebar-divider">')
        st.metric("Messages sent", st.session_state.message_count)
        if st.button(
            "New Conversation", key="new_chat_btn", type="primary", use_container_width=True,
        ):
            st.session_state.history = []
            st.session_state.message_count = 0
            st.session_state.pop("pending_prompt", None)
            st.rerun()


st.set_page_config(
    page_title=f"{COMPANY_NAME} | Support",
    page_icon=None,
    layout="centered",
)
_css_inner = CUSTOM_CSS.strip().removeprefix("<style>").removesuffix("</style>")
for _chunk in _iter_css_chunks(_css_inner):
    st.html(f"<style>{_chunk}</style>")

if "history" not in st.session_state:
    # Kept in Streamlit's session_state because the whole script reruns on
    # every interaction; a plain variable would reset each time.
    st.session_state.history = []
if "message_count" not in st.session_state:
    # Tracked separately from history, which trim_history() may shorten —
    # this keeps counting every message sent this session, not just the
    # ones still retained in the trimmed window.
    st.session_state.message_count = 0

client = get_client()

# Read the input first (Streamlit pins chat_input to the bottom regardless of
# where it's called), so the empty state can be suppressed the moment a
# message is in flight rather than lingering beside the first reply.
prompt = st.chat_input("Type your message…")
if not prompt and "pending_prompt" in st.session_state:
    # Set by an example-question card, which reruns immediately after
    # stashing it here — see render_empty_state(). Routing it through a
    # rerun (rather than using it in-place) keeps this identical to the
    # chat_input path, so the empty state cleanly disappears on this run
    # instead of briefly rendering alongside the new exchange.
    prompt = st.session_state.pop("pending_prompt")

if not st.session_state.history and not prompt:
    render_empty_state()

render_history()

if prompt:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # Reuse main.py's builder to assemble the payload for this turn.
    messages = build_messages(st.session_state.history, prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        # Show a placeholder the instant the bubble appears, so it's
        # never empty while stream_reply() is retrying models behind
        # the scenes.
        placeholder = st.empty()
        placeholder.markdown(
            f'<span class="thinking-label">{ASSISTANT_NAME} is thinking…</span>',
            unsafe_allow_html=True,
        )
        reply = ""
        try:
            stream = stream_reply(client, messages)
            first_chunk = next(stream, None)
            if first_chunk is not None:
                # Real text has arrived — swap the placeholder for the
                # live stream, re-attaching the chunk already pulled off.
                placeholder.empty()

                def _resume():
                    yield first_chunk
                    yield from stream

                result = st.write_stream(_resume())
                reply = result.strip() if isinstance(result, str) else ""
        except Exception:
            reply = ""
        if not reply:
            placeholder.empty()
            st.error("Sorry — I couldn't get a response just now. Please try again.")

    # Only remember successful exchanges, then trim with main.py's helper.
    if reply:
        st.session_state.history.append({"role": "user", "parts": [{"text": prompt}]})
        st.session_state.history.append({"role": "model", "parts": [{"text": reply}]})
        st.session_state.history = trim_history(st.session_state.history)
        st.session_state.message_count += 1

# Rendered last so the "Messages sent" metric reflects this run's count
# immediately — st.sidebar content is positioned by the frontend regardless
# of where in the script it's called, so this doesn't affect layout.
render_sidebar()
