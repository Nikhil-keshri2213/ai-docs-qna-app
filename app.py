import streamlit as st
import requests

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="DocMind", page_icon="◈", layout="wide")

# ── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("◈ DocMind")

    # Server status check
    st.subheader("Server Status")
    try:
        res = requests.get(API_BASE_URL, timeout=2)
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "ready":
                st.success("🟢 Server Online — Document Ready")
                st.session_state.doc_processed = True
            else:
                st.success("🟢 Server Online")
        else:
            st.warning("🟡 Server Responding (Check API)")
    except requests.exceptions.ConnectionError:
        st.error("🔴 Server Offline — Start the FastAPI server first")
    except Exception:
        st.error("🔴 Cannot reach server")

    st.divider()

    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document... (first run downloads model ~900MB)"):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }
            try:
                res = requests.post(
                    f"{API_BASE_URL}/upload",
                    files=files,
                    timeout=300   # model download can take a while on first run
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.doc_processed = True
                    st.success(
                        f"✅ Processed {data.get('chunks_indexed', '?')} chunks"
                    )
                else:
                    detail = res.json().get("detail", "Unknown error")
                    st.error(f"Upload failed ❌ — {detail}")
            except requests.exceptions.Timeout:
                st.warning("⏳ Still processing (model loading) — please wait and try asking a question")
                st.session_state.doc_processed = True
            except Exception as e:
                st.error(f"Backend not reachable ❌ — {e}")

    if st.session_state.doc_processed:
        st.divider()
        if st.button("🗑 Reset / Upload New Doc"):
            try:
                requests.delete(f"{API_BASE_URL}/reset", timeout=5)
            except Exception:
                pass
            st.session_state.chat_history = []
            st.session_state.doc_processed = False
            st.rerun()

# ── Chat History ──────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📄 Source pages: {', '.join(map(str, msg['sources']))}")

# ── Chat Input ────────────────────────────────────────────────────────────────
if question := st.chat_input(
    "Ask a question about your document...",
    disabled=not st.session_state.doc_processed
):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={"question": question},
                    timeout=120
                )
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    sources = data.get("source_pages", [])

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                    st.markdown(answer)
                    if sources:
                        st.caption(f"📄 Source pages: {', '.join(map(str, sources))}")
                else:
                    detail = res.json().get("detail", "Unknown error")
                    st.error(f"Backend error: {detail}")
            except requests.exceptions.Timeout:
                st.error("⏳ Request timed out — the model may be loading, try again")
            except Exception as e:
                st.error(f"Connection failed: {e}")