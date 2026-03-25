import tempfile
import os
import logging
import time
import streamlit as st
from dotenv import load_dotenv

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),                          # terminal
        logging.FileHandler("docmind.log", mode="a"),    # persistent log file
    ],
)
log = logging.getLogger("docmind")

load_dotenv()
log.info("Environment loaded")

from backend.document_loader import load_and_split_document
from backend.vector_store import create_vector_store
from backend.qa_chain import create_qa_chain

log.info("All modules imported successfully")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — PDF Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #080810;
    color: #e2e0f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0c0c18 !important;
    border-right: 1px solid rgba(120,100,220,0.15) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: #b8b5d0 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar brand ── */
.sb-brand {
    padding: 1.8rem 1.2rem 1.2rem;
    border-bottom: 1px solid rgba(120,100,220,0.15);
    margin-bottom: 1.2rem;
}
.sb-brand-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #a78bfa !important;
    letter-spacing: 0.05em;
}
.sb-brand-sub {
    font-size: 0.7rem;
    color: #4a4868 !important;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 3px;
}

/* ── Upload area ── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(120,100,220,0.04) !important;
    border: 1.5px dashed rgba(120,100,220,0.25) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(120,100,220,0.08) !important;
    border-color: rgba(167,139,250,0.5) !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #6b6888 !important; }
[data-testid="stFileUploader"] section > button {
    background: rgba(120,100,220,0.12) !important;
    color: #a78bfa !important;
    border: 1px solid rgba(167,139,250,0.3) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
}

/* ── Status pill ── */
.status-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(34,197,94,0.06);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 10px 0 6px;
}
.status-dot {
    width: 6px; height: 6px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(34,197,94,0.6);
    flex-shrink: 0;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.status-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.73rem;
    color: #86efac !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Stat grid ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin: 10px 0 16px;
}
.stat-box {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 10px 6px 8px;
    text-align: center;
}
.stat-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.15rem;
    font-weight: 600;
    color: #a78bfa !important;
    line-height: 1;
}
.stat-lbl {
    font-size: 0.65rem;
    color: #4a4868 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* ── Stack info ── */
.stack-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 0;
    font-size: 0.78rem;
    color: #5a5778 !important;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.stack-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    background: rgba(120,100,220,0.1);
    color: #7c6fa8 !important;
    padding: 2px 7px;
    border-radius: 4px;
    margin-left: auto;
}

/* ── Clear btn ── */
.stButton > button {
    width: 100%;
    background: rgba(239,68,68,0.06) !important;
    color: #f87171 !important;
    border: 1px solid rgba(239,68,68,0.2) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 8px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: rgba(239,68,68,0.12) !important;
    border-color: rgba(239,68,68,0.4) !important;
}

/* ── Main header ── */
.main-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 2rem 0 1.4rem;
    border-bottom: 1px solid rgba(120,100,220,0.12);
    margin-bottom: 2rem;
}
.main-logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #a78bfa;
    letter-spacing: -0.02em;
}
.main-logo span { color: #6d4ddc; }
.main-sub {
    font-size: 0.78rem;
    color: #3d3b58;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding-bottom: 4px;
}

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 6rem 2rem;
    text-align: center;
}
.empty-icon {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.8rem;
    color: #1e1c34;
    margin-bottom: 1.5rem;
    line-height: 1;
}
.empty-title {
    font-size: 1.1rem;
    font-weight: 500;
    color: #2a2840;
    margin-bottom: 0.5rem;
}
.empty-hint {
    font-size: 0.83rem;
    color: #1e1c30;
    max-width: 320px;
    line-height: 1.6;
}

/* ── Chat messages ── */
.chat-wrap { display: flex; flex-direction: column; gap: 1.4rem; padding-bottom: 0.5rem; }

.msg-user { display: flex; justify-content: flex-end; }
.msg-user .bubble {
    background: linear-gradient(135deg, #3b2c8a 0%, #6d3fd6 100%);
    color: #ede9ff;
    border-radius: 16px 16px 3px 16px;
    padding: 13px 18px;
    max-width: 68%;
    font-size: 0.92rem;
    line-height: 1.65;
    box-shadow: 0 4px 24px rgba(109,63,214,0.18);
}

.msg-ai { display: flex; align-items: flex-start; gap: 10px; }
.ai-avatar {
    width: 30px; height: 30px;
    background: linear-gradient(135deg, #1a1530 0%, #2a1f50 100%);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #a78bfa;
    flex-shrink: 0;
    margin-top: 3px;
}
.msg-ai .bubble {
    background: #0f0e1e;
    border: 1px solid rgba(120,100,220,0.12);
    color: #d8d5ee;
    border-radius: 3px 16px 16px 16px;
    padding: 14px 18px;
    max-width: 80%;
    font-size: 0.92rem;
    line-height: 1.75;
}

/* ── Source badges ── */
.sources-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 9px; }
.src-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    background: rgba(120,100,220,0.08);
    border: 1px solid rgba(120,100,220,0.2);
    color: #7c6fa8;
    padding: 3px 9px;
    border-radius: 5px;
}

/* ── Thinking dots ── */
.thinking {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #0f0e1e;
    border: 1px solid rgba(120,100,220,0.12);
    border-radius: 3px 16px 16px 16px;
    padding: 14px 20px;
}
.thinking span {
    width: 6px; height: 6px;
    background: #7c5fc8;
    border-radius: 50%;
    animation: tdot 1.3s ease-in-out infinite;
}
.thinking span:nth-child(2) { animation-delay: 0.18s; }
.thinking span:nth-child(3) { animation-delay: 0.36s; }
@keyframes tdot {
    0%,70%,100% { transform: scale(0.55); opacity: 0.3; }
    35% { transform: scale(1.1); opacity: 1; }
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: transparent !important;
}
[data-testid="stChatInput"] > div {
    background: #0f0e1e !important;
    border: 1.5px solid rgba(120,100,220,0.25) !important;
    border-radius: 14px !important;
    padding: 6px 6px 6px 6px !important;
    box-shadow: 0 0 0 0 transparent;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(167,139,250,0.6) !important;
    box-shadow: 0 0 0 3px rgba(109,63,214,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e2e0f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 10px 14px !important;
    line-height: 1.6 !important;
    caret-color: #a78bfa !important;
    resize: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #3a3858 !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background: rgba(109,63,214,0.2) !important;
    border-radius: 9px !important;
    border: 1px solid rgba(167,139,250,0.2) !important;
    margin: 4px !important;
    transition: background 0.15s !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
    background: rgba(109,63,214,0.4) !important;
}
[data-testid="stChatInputSubmitButton"] svg { fill: #a78bfa !important; }

/* ── Log panel ── */
.log-panel {
    background: #060610;
    border: 1px solid rgba(120,100,220,0.12);
    border-radius: 10px;
    padding: 14px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    line-height: 1.8;
    max-height: 260px;
    overflow-y: auto;
}
.log-DEBUG   { color: #4a4870; }
.log-INFO    { color: #6b9e6b; }
.log-WARNING { color: #b8860b; }
.log-ERROR   { color: #c0392b; }
.log-ts      { color: #2e2c48; margin-right: 6px; }
.log-lvl     { margin-right: 8px; min-width: 52px; display: inline-block; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.01) !important;
    border: 1px solid rgba(120,100,220,0.1) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.8rem !important;
    color: #5a5778 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(120,100,220,0.2); border-radius: 3px; }

/* ── Divider ── */
hr { border-color: rgba(120,100,220,0.1) !important; margin: 0.8rem 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #7c5fc8 !important; }
</style>
""", unsafe_allow_html=True)

# ── In-memory log handler ─────────────────────────────────────────────────────
class MemoryLogHandler(logging.Handler):
    """Keeps the last N log records in memory for display in the UI."""
    def __init__(self, capacity=120):
        super().__init__()
        self.records = []
        self.capacity = capacity

    def emit(self, record):
        self.records.append(self.format(record))
        if len(self.records) > self.capacity:
            self.records.pop(0)

if "mem_log_handler" not in st.session_state:
    handler = MemoryLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(handler)
    st.session_state.mem_log_handler = handler

mem_handler = st.session_state.mem_log_handler

# ── Session state init ────────────────────────────────────────────────────────
for key, default in {
    "qa_chain": None,
    "retriever": None,
    "chat_history": [],
    "doc_name": None,
    "doc_chunks": 0,
    "doc_pages": 0,
    "show_logs": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sb-brand'>
        <div class='sb-brand-name'>◈ DocMind</div>
        <div class='sb-brand-sub'>PDF Intelligence Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:0 0.2rem;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.78rem;color:#4a4868;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;'>Upload Document</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file and uploaded_file.name != st.session_state.doc_name:
        log.info(f"New file uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        with st.spinner("Indexing…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
                log.debug(f"Saved to temp: {tmp_path}")
            try:
                t0 = time.time()
                log.info("Loading and splitting document…")
                chunks = load_and_split_document(tmp_path)
                log.info(f"Split into {len(chunks)} chunks in {time.time()-t0:.2f}s")

                t1 = time.time()
                log.info("Building FAISS vector store…")
                vs = create_vector_store(chunks)
                log.info(f"Vector store built in {time.time()-t1:.2f}s")

                t2 = time.time()
                log.info("Creating QA chain…")
                st.session_state.qa_chain = create_qa_chain(vs)
                st.session_state.retriever = vs.as_retriever(search_kwargs={"k": 4})
                log.info(f"QA chain ready in {time.time()-t2:.2f}s")

                pages = max((c.metadata.get("page", 0) for c in chunks), default=0) + 1
                st.session_state.doc_name   = uploaded_file.name
                st.session_state.doc_chunks = len(chunks)
                st.session_state.doc_pages  = pages
                st.session_state.chat_history = []
                log.info(f"Document ready — {pages} pages, {len(chunks)} chunks, total {time.time()-t0:.2f}s")

            except Exception as e:
                log.error(f"Failed to process document: {e}", exc_info=True)
                st.error(f"Error: {e}")
            finally:
                os.unlink(tmp_path)
                log.debug(f"Temp file removed: {tmp_path}")

    if st.session_state.doc_name:
        st.markdown(f"""
        <div class='status-pill'>
            <span class='status-dot'></span>
            <span class='status-name'>{st.session_state.doc_name}</span>
        </div>
        <div class='stat-grid'>
            <div class='stat-box'>
                <div class='stat-num'>{st.session_state.doc_pages}</div>
                <div class='stat-lbl'>Pages</div>
            </div>
            <div class='stat-box'>
                <div class='stat-num'>{st.session_state.doc_chunks}</div>
                <div class='stat-lbl'>Chunks</div>
            </div>
            <div class='stat-box'>
                <div class='stat-num'>{len(st.session_state.chat_history) // 2}</div>
                <div class='stat-lbl'>Q&amp;As</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:2px 0 8px;'>
        <div class='stack-item'>Embeddings <span class='stack-tag'>MiniLM-L6</span></div>
        <div class='stack-item'>LLM <span class='stack-tag'>flan-t5-base (local)</span></div>
        <div class='stack-item'>Vector DB <span class='stack-tag'>FAISS</span></div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("🗑  Clear conversation"):
            log.info("Conversation cleared by user")
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    show_logs = st.toggle("🪵  Show debug logs", value=st.session_state.show_logs)
    st.session_state.show_logs = show_logs
    st.markdown("</div>", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
main_col, log_col = st.columns([3, 1]) if st.session_state.show_logs else (st.container(), None)

with (main_col if st.session_state.show_logs else main_col):
    st.markdown("""
    <div class='main-header'>
        <div>
            <div class='main-logo'>◈ Doc<span>Mind</span></div>
        </div>
        <div class='main-sub'>PDF Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.qa_chain is None:
        st.markdown("""
        <div class='empty-state'>
            <div class='empty-icon'>◈</div>
            <div class='empty-title'>No document loaded</div>
            <div class='empty-hint'>Upload a PDF in the sidebar — research paper, book, ticket, or manual — and start asking questions.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Chat history ──
        st.markdown("<div class='chat-wrap'>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class='msg-user'>
                    <div class='bubble'>{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)
            else:
                source_html = ""
                if msg.get("sources"):
                    pages = sorted(set(d.metadata.get("page", 0) + 1 for d in msg["sources"]))
                    badges = "".join(f"<span class='src-badge'>pg {p}</span>" for p in pages)
                    source_html = f"<div class='sources-row'>{badges}</div>"

                st.markdown(f"""
                <div class='msg-ai'>
                    <div class='ai-avatar'>AI</div>
                    <div>
                        <div class='bubble'>{msg["content"]}</div>
                        {source_html}
                    </div>
                </div>""", unsafe_allow_html=True)

                if msg.get("sources"):
                    with st.expander("source excerpts"):
                        for doc in msg["sources"]:
                            page = doc.metadata.get("page", 0) + 1
                            st.markdown(f"<div style='font-size:0.72rem;color:#a78bfa;font-family:IBM Plex Mono,monospace;margin-bottom:4px;'>page {page}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-size:0.82rem;color:#6b6888;line-height:1.7;background:#07070f;border-left:2px solid rgba(120,100,220,0.3);padding:10px 14px;border-radius:0 6px 6px 0;margin-bottom:10px;'>{doc.page_content[:400]}…</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

        # ── Chat input ────────────────────────────────────────────────────────
        question = st.chat_input(f"Ask about  {st.session_state.doc_name[:35]}…")

        if question:
            log.info(f"User question: {question!r}")
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.spinner(""):
                placeholder = st.empty()
                placeholder.markdown("""
                <div class='msg-ai'>
                    <div class='ai-avatar'>AI</div>
                    <div class='thinking'><span></span><span></span><span></span></div>
                </div>""", unsafe_allow_html=True)

                try:
                    t0 = time.time()
                    log.debug("Invoking QA chain…")
                    answer = st.session_state.qa_chain.invoke(question)
                    if isinstance(answer, dict):
                        answer = answer.get("result", str(answer))
                    answer = answer.strip()
                    log.info(f"Chain answered in {time.time()-t0:.2f}s — {len(answer)} chars")

                    log.debug("Fetching source docs…")
                    sources = st.session_state.retriever.invoke(question)
                    log.debug(f"Retrieved {len(sources)} source chunks")

                except Exception as e:
                    log.error(f"Chain error: {e}", exc_info=True)
                    import html as _html
                    answer = f"Something went wrong: {_html.escape(str(e))}"
                    sources = []

            placeholder.empty()
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
            st.rerun()

# ── Log panel (right column) ──────────────────────────────────────────────────
if st.session_state.show_logs and log_col is not None:
    with log_col:
        st.markdown("<div style='padding-top:2rem;'>", unsafe_allow_html=True)
        st.markdown("<p style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#4a4868;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;'>Debug logs</p>", unsafe_allow_html=True)

        if st.button("⟳ Refresh", key="refresh_logs"):
            st.rerun()

        records = mem_handler.records[-60:]
        if records:
            def colorize(line):
                if "|DEBUG|"   in line: lvl = "DEBUG"
                elif "|INFO|"  in line: lvl = "INFO"
                elif "|WARNING|" in line: lvl = "WARNING"
                elif "|ERROR|" in line: lvl = "ERROR"
                else: lvl = "DEBUG"
                parts = line.split("|", 2)
                ts  = parts[0] if len(parts) > 0 else ""
                msg = parts[2] if len(parts) > 2 else line
                return f"<span class='log-ts'>{ts}</span><span class='log-{lvl} log-lvl'>{lvl}</span><span class='log-{lvl}'>{msg}</span>"

            rows = "<br>".join(colorize(r) for r in records)
            st.markdown(f"<div class='log-panel'>{rows}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='log-panel' style='color:#2a2848;'>No logs yet.</div>", unsafe_allow_html=True)

        st.markdown("<p style='font-size:0.68rem;color:#2a2848;margin-top:6px;'>Full log → docmind.log</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)