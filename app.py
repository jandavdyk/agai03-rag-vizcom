"""
app.py

Streamlit UI for the Vizcom RAG chatbot.

Features:
  - Chat interface with multi-turn memory
  - Source citations under each answer
  - Mode badge showing whether the answer came from Q/A, vector search, or
    was declined by the guardrail
  - Sidebar with project info, scraping statistics, and sample Q/A pairs
  - Clear Chat button

Run from project root:
    streamlit run app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatbot import VizcomChatbot  # noqa: E402


# ---- Page configuration ----
st.set_page_config(
    page_title="Vizcom RAG Chatbot",
    page_icon=":speech_balloon:",
    layout="wide",
)


# ---- Cached resources (load once, reuse across interactions) ----
@st.cache_resource(
    show_spinner="Loading Vizcom chatbot (first time takes ~30 seconds)..."
)
def load_chatbot():
    return VizcomChatbot()


@st.cache_data
def load_stats():
    """Load project statistics for the sidebar."""
    stats = {"pages": 0, "qa_pairs": 0, "doc_chunks": 0}

    index_file = Path("data/raw/_index.json")
    if index_file.exists():
        with open(index_file) as f:
            stats["pages"] = len(json.load(f))

    qa_file = Path("data/qa_dataset.csv")
    if qa_file.exists():
        stats["qa_pairs"] = len(pd.read_csv(qa_file))

    try:
        import chromadb
        client = chromadb.PersistentClient(path="chroma_db")
        stats["doc_chunks"] = client.get_collection("vizcom_docs").count()
    except Exception:
        pass

    return stats


@st.cache_data
def load_sample_qa(n=8):
    """Load a random sample of Q/A pairs for the sidebar."""
    qa_file = Path("data/qa_dataset.csv")
    if qa_file.exists():
        df = pd.read_csv(qa_file)
        return df.sample(min(n, len(df)), random_state=42)
    return pd.DataFrame()


# ---- Sidebar ----
with st.sidebar:
    st.title("Vizcom RAG Chatbot")
    st.markdown(
        "A document-grounded chatbot built on **vizcom.com** content."
    )

    st.markdown("**Pipeline**")
    st.markdown("""
- Web scraping (BeautifulSoup)
- Synthetic Q/A generation (Llama 3.3 70B via Groq)
- Hybrid retrieval (Q/A semantic search, then vector fallback)
- LLM-powered answer generation with guardrails
""")

    st.markdown("**Website:** [vizcom.com](https://vizcom.com)")

    st.divider()

    st.subheader("Statistics")
    stats = load_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pages Scraped", stats["pages"])
        st.metric("Doc Chunks", stats["doc_chunks"])
    with col2:
        st.metric("Q/A Pairs", stats["qa_pairs"])

    st.divider()

    with st.expander("Sample Q/A pairs"):
        samples = load_sample_qa(n=8)
        if not samples.empty:
            for _, row in samples.iterrows():
                st.markdown(f"**Q:** {row['question']}")
                st.caption(row['answer'][:120] + "...")
                st.markdown("---")
        else:
            st.warning("No Q/A dataset found.")

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if "bot" in st.session_state:
            st.session_state.bot.reset_history()
        st.rerun()

    st.divider()
    st.caption("Course: AGAI-03  |  RAG Chatbot Assignment")
    st.caption("Built with Python, Streamlit, Chroma, Groq")


# ---- Main chat area ----
st.title("Vizcom Assistant")
st.caption(
    "Ask me anything about Vizcom - features, design workflows, blog content."
)

# Initialize chatbot and session state
if "bot" not in st.session_state:
    st.session_state.bot = load_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_assistant_metadata(msg: dict):
    """Render the mode badge and source citations below an assistant message."""
    mode = msg.get("mode", "unknown")
    confidence = msg.get("confidence")

    mode_labels = {
        "qa": "Direct Q/A match",
        "vector": "Generated from documents",
        "unknown": "Out of knowledge base",
    }
    label = mode_labels.get(mode, "Unknown")

    if confidence is not None:
        st.caption(
            f"**Mode:** {label}  |  **Confidence (lower is better):** "
            f"{confidence:.3f}"
        )
    else:
        st.caption(f"**Mode:** {label}")

    if msg.get("matched_question"):
        st.caption(f"Matched question: _{msg['matched_question']}_")

    sources = msg.get("sources", [])
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"- [{s['title']}]({s['url']})")


# Display existing message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_assistant_metadata(msg)


# Handle new user input
if user_query := st.chat_input("Ask about Vizcom..."):
    # Append + render user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Get and render bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.bot.ask(user_query)

        st.markdown(result["answer"])
        render_assistant_metadata(result)

    # Save assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "mode": result["mode"],
        "confidence": result["confidence"],
        "sources": result.get("sources", []),
        "matched_question": result.get("matched_question"),
    })