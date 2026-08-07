"""Streamlit chat UI for the Nexus Analytics support-chatbot.

This is a thin web front-end over the exact same conversation logic the
terminal app uses. It imports build_messages(), trim_history(), and the
stream_reply() generator from main.py and wraps them for the browser —
none of that logic is reimplemented here.

Run with:  streamlit run app.py
"""

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

EXAMPLE_QUESTIONS = [
    ("", "Reset account credentials"),
    ("", "Billing & invoice inquiry"),
    ("", "Dashboard configuration"),
    ("", "Report a platform issue"),
]

# Colors/gradients/animation live here because config.toml's [theme] can't
# express them (per-role bubble backgrounds, keyframes).
# Selectors use [class*="st-key-<prefix>"] because each st.container(key=...)
# gets its own unique "st-key-<key>" class — the substring match lets one
# rule style every bubble sharing a prefix (e.g. bub_b0, bub_b1, bub_b_live)
# without listing every message index individually.
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
    padding: 3rem 1rem 1.75rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.hero-wordmark {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #f97316;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.75rem;
    color: #f4f5f7;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 0.97rem;
    line-height: 1.65;
    color: rgba(229, 231, 235, 0.65);
    max-width: 32rem;
    margin: 0 auto;
}
/* ── Quick-action cards ────────────────────────── */
.suggest-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
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
    font-size: 0.88rem;
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
/* ── Chat bubbles ──────────────────────────────── */
/* width/height/flex-shrink overrides fix a flex-shrink bug where Streamlit's
   horizontal containers collapse text below its natural width. */
[class*="st-key-bub_u"] {
    background-color: #ea580c;
    color: #ffffff;
    border-radius: 16px 16px 4px 16px;
    padding: 0.65rem 1.1rem;
    max-width: min(76%, 33rem);
    width: fit-content !important;
    min-width: 0 !important;
    height: fit-content !important;
    min-height: fit-content !important;
    flex-shrink: 0 !important;
    line-height: 1.55;
    word-break: break-word;
    overflow-wrap: anywhere;
}
[class*="st-key-bub_b"] {
    background-color: #1c2942;
    color: #e2e5ec;
    border-radius: 16px 16px 16px 4px;
    padding: 0.65rem 1.1rem 0.9rem;
    max-width: min(76%, 33rem);
    width: fit-content !important;
    min-width: 0 !important;
    height: fit-content !important;
    min-height: fit-content !important;
    flex-shrink: 0 !important;
    line-height: 1.55;
    border: 1px solid #2c3b5a;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
    word-break: break-word;
    overflow-wrap: anywhere;
}
[class*="st-key-row_u"] [data-testid="stLayoutWrapper"] {
    width: fit-content !important;
    max-width: min(76%, 33rem) !important;
    flex: 0 1 auto !important;
}
[class*="st-key-bub_u"] p,
[class*="st-key-bub_b"] p { margin: 0 0 0.45em 0; }
[class*="st-key-bub_u"] p:last-child,
[class*="st-key-bub_b"] p:last-child { margin-bottom: 0; }
/* ── Bot avatar ────────────────────────────────── */
/* Rendered via st.markdown(":material/...") inside a keyed container rather
   than raw SVG markup through st.html() — st.html()'s sanitizer strips any
   tag content containing the literal substring "svg" (a guard against a
   known SVG-based mutation-XSS technique), which is why an inline icon
   rendered as an empty box. Material Symbols go through the normal
   markdown path instead, so nothing gets stripped. */
/* !important throughout: Streamlit's own emotion-generated CSS for these
   elements is injected later in the document than our <style> block and
   ties on specificity, so it wins the cascade unless forced — confirmed by
   inspecting computed styles in a real browser (background/border-radius/
   icon color were all silently no-ops without it). */
[class*="st-key-row_b"] {
    align-items: flex-start !important;
    flex-wrap: nowrap !important;
}
/* The avatar+bubble wrapping divs (Streamlit's own stLayoutWrapper, one
   level above the st-key-tagged element) report full row width for
   wrapping purposes even though their content hugs the bubble/icon — that
   mismatch is what pushed the avatar onto its own line above the bubble
   instead of beside it. Constraining the wrapper itself (not just the
   st-key element inside it) is what actually fixes it.
   max-width has to be repeated here too, not just on the bubble inside:
   shrink-to-fit sizing on this wrapper is computed from its content's
   unwrapped (single-line) preferred width, which ignores the bubble's own
   max-width deeper in the tree — for any reply long enough to wrap to a
   second line, that preferred width exceeds the row and shrink-to-fit
   falls back to the full row width instead of the bubble's max-width. */
[class*="st-key-row_b"] [data-testid="stLayoutWrapper"] {
    width: fit-content !important;
    max-width: min(76%, 33rem) !important;
    flex: 0 1 auto !important;
}
[class*="st-key-avatar_b"] {
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    border-radius: 6px !important;
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-top: 0.65rem;
    flex-shrink: 0 !important;
}
[class*="st-key-avatar_b"] [data-testid="stMarkdownContainer"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[class*="st-key-avatar_b"] span[role="img"] {
    color: #f97316 !important;
    font-size: 16px !important;
}
/* ── Thinking indicator ────────────────────────── */
.thinking-label {
    color: rgba(214, 217, 224, 0.6);
    font-style: italic;
    font-size: 0.9rem;
    margin-right: 0.5rem;
}
.dot-flashing {
    position: relative;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background-color: #f97316;
    display: inline-block;
    vertical-align: middle;
    animation: dot-flashing 1s infinite linear alternate;
    animation-delay: 0.5s;
}
.dot-flashing::before, .dot-flashing::after {
    content: "";
    position: absolute;
    top: 0;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background-color: #f97316;
    animation: dot-flashing 1s infinite alternate;
}
.dot-flashing::before { left: -9px; animation-delay: 0s; }
.dot-flashing::after  { left:  9px; animation-delay: 1s; }
@keyframes dot-flashing {
    0%         { background-color: #f97316; }
    50%, 100%  { background-color: rgba(249, 115, 22, 0.2); }
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
    font-size: 1.25rem;
    font-weight: 700;
    color: #f4f5f7;
    letter-spacing: -0.01em;
    margin: 0;
}
.sidebar-product-label {
    font-size: 0.75rem;
    color: #f97316;
    font-weight: 600;
    letter-spacing: 0.07em;
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
    font-size: 0.875rem;
    line-height: 1.65;
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


def render_bubble(role: str, text: str, index: int) -> None:
    """Render one completed message as a custom-styled bubble row.

    role is "user" or "bot". index only needs to be unique within this
    script run (it seeds the container keys) — it doesn't need to stay
    stable across reruns, since bubbles hold no widget state of their own
    and are rebuilt fresh from history every run.
    """
    if role == "user":
        with st.container(key=f"row_u{index}", horizontal_alignment="right"):
            with st.container(key=f"bub_u{index}", width="content"):
                st.markdown(text)
    else:
        with st.container(
            key=f"row_b{index}", horizontal=True, horizontal_alignment="left",
            gap="small", vertical_alignment="top",
        ):
            with st.container(key=f"avatar_b{index}", width="content"):
                st.markdown(":material/smart_toy:")
            with st.container(key=f"bub_b{index}", width="content"):
                st.markdown(text)


def render_history() -> None:
    """Replay stored turns as bubbles.

    History is kept in the Gemini format that build_messages()/stream_reply()
    expect (role "user"/"model", parts→text), so "model" maps to "bot" here.
    """
    for i, turn in enumerate(st.session_state.history):
        role = "user" if turn["role"] == "user" else "bot"
        render_bubble(role, turn["parts"][0]["text"], i)


def render_empty_state() -> None:
    """Centered hero + example-question cards, shown before the first message."""
    st.html(
        '<div class="hero-wrap">'
        '<span class="hero-wordmark">Nexus Analytics</span>'
        '<h1 class="hero-title">Customer Support Portal</h1>'
        '<p class="hero-subtitle">Get precise answers about your account, billing, '
        "and dashboards — 24 / 7. Select a category below or type your inquiry directly.</p>"
        "</div>"
    )
    st.html('<div class="suggest-section-label">Common Inquiries</div>')
    col_a, col_b = st.columns(2, gap="small")
    for i, (icon, label) in enumerate(EXAMPLE_QUESTIONS):
        col = col_a if i % 2 == 0 else col_b
        with col:
            if st.button(
                f"{label}", key=f"suggest_card_{i}", use_container_width=True,
            ):
                st.session_state.pending_prompt = label
                st.rerun()


def render_sidebar() -> None:
    """Sidebar: brand, About text, live session stats, and a reset button."""
    with st.sidebar:
        st.html(
            f'<p class="sidebar-brand-name">{ASSISTANT_NAME}</p>'
            '<span class="sidebar-product-label">Nexus Analytics Support</span>'
        )
        st.html('<hr class="sidebar-divider">')
        st.markdown(
            f'<p class="sidebar-about">'
            f"I'm <strong>{ASSISTANT_NAME}</strong>, an automated support specialist for "
            "<strong>Nexus Analytics</strong>. I handle account, billing, and dashboard "
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
    page_title="Nexus Analytics | Support",
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
    render_bubble("user", prompt, index=len(st.session_state.history))

    # Reuse main.py's builder to assemble the payload for this turn.
    messages = build_messages(st.session_state.history, prompt)

    with st.container(
        key="row_b_live", horizontal=True, horizontal_alignment="left",
        gap="small", vertical_alignment="top",
    ):
        with st.container(key="avatar_b_live", width="content"):
            st.markdown(":material/smart_toy:")
        with st.container(key="bub_b_live", width="content"):
            # Show a placeholder the instant the bubble appears, so it's
            # never empty while stream_reply() is retrying models behind
            # the scenes.
            placeholder = st.empty()
            placeholder.markdown(
                f'<span class="thinking-label">{ASSISTANT_NAME} is '
                f'thinking</span><span class="dot-flashing"></span>',
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
