import os
from pathlib import Path
import streamlit as st
from app.hybrid_search import hybrid_search, load_raw_items
from transformers import pipeline
import textwrap
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="AstroNews Explorer", page_icon="🛰️", layout="wide")

@st.cache_resource(show_spinner=False)
def get_pipelines():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return summarizer, qa

@st.cache_resource(show_spinner=False)
def get_raw_items():
    return load_raw_items()

def make_context(indices, raw_items, max_chars=1800):
    # concatenate titles + summaries of the selected indices
    chunks = []
    for idx in indices:
        if idx < len(raw_items):
            it = raw_items[idx]
            title = it.get("title", "")
            summary = it.get("summary", "")
            url = it.get("url", "")
            text = f"{title}\n\n{summary}\n\nSource: {url}\n---\n"
            chunks.append(text)
    ctx = "".join(chunks)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "…"
    return ctx

st.title("🛰️ AstroNews Explorer")

# Sidebar controls
st.sidebar.header("Search")
query = st.sidebar.text_input("Query", value="cargo to the station")
k = st.sidebar.slider("Results", 1, 15, 8)
st.sidebar.caption("Hybrid = semantic + keyword + recency")

# Add AI settings in sidebar
st.sidebar.header("AI Settings")
max_context_chars = st.sidebar.slider("Context length", 1000, 5000, 3000, step=500, 
                                     help="Maximum characters for AI context")
summary_length = st.sidebar.select_slider("Summary length", 
                                         options=["Short", "Medium", "Long"], 
                                         value="Medium",
                                         help="Control output length for summaries")
eli5_length = st.sidebar.select_slider("ELI5 length", 
                                      options=["Brief", "Normal", "Detailed"], 
                                      value="Normal",
                                      help="Control output length for ELI5 explanations")

# Convert length selections to actual values
length_mapping = {
    "Short": {"max": 80, "min": 30},
    "Medium": {"max": 130, "min": 40}, 
    "Long": {"max": 200, "min": 60}
}

eli5_mapping = {
    "Brief": {"max": 100, "min": 40},
    "Normal": {"max": 150, "min": 60},
    "Detailed": {"max": 250, "min": 80}
}

# Run search
if st.sidebar.button("Search", type="primary"):
    st.session_state["hits"] = hybrid_search(query, k=k)

hits = st.session_state.get("hits", [])
raw_items = get_raw_items()
summarizer, qa = get_pipelines()

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.subheader("Results")
    if not hits:
        st.info("Run a search from the sidebar.")
    else:
        # show hits with checkboxes to include in context
        selected_idxs = []
        for i, h in enumerate(hits, 1):
            date = (h.get("published_at") or "")[:10]
            title = h["title"]
            url = h["url"]
            sem = h["score_semantic"]; kw = h["score_keyword"]; rec = h["score_recency"]; final = h["score_final"]
            with st.container(border=True):
                checked = st.checkbox(f"[{i}] {date} — {title}", value=(i <= 3), key=f"sel_{i}")
                st.write(url)
                st.caption(f"score: {final:.3f} | sem {sem:.2f} · kw {kw:.2f} · rec {rec:.2f}")
                if checked:
                    # We need the index that produced this hit; hybrid_search doesn't return it.
                    # Quick heuristic: match by URL back to raw_items index.
                    # (For a perfect mapping, include idx in your hybrid_search results next time.)
                    for idx, it in enumerate(raw_items):
                        if it.get("url") == url:
                            selected_idxs.append(idx)
                            break

        st.markdown("---")
        st.subheader("AI Tools")

        # Build context with dynamic max chars
        ctx = make_context(selected_idxs, raw_items, max_chars=max_context_chars)
        st.text_area("Context (top selected articles)", value=ctx, height=180)

        # Summarize with dynamic length
        st.markdown("**Summarize selected**")
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Generate bullets"):
                if ctx.strip():
                    # Use dynamic length settings
                    max_len = length_mapping[summary_length]["max"]
                    min_len = length_mapping[summary_length]["min"]
                    
                    # short abstractive summary; split if too long
                    parts = [ctx[i:i+900] for i in range(0, len(ctx), 900)]
                    bullets = []
                    for p in parts[:3]:  # keep it fast
                        out = summarizer(p, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
                        bullets.append(out.strip())
                    st.success("• " + "\n• ".join(bullets))
                else:
                    st.warning("Select at least one article.")
        with col2:
            st.caption(f"Length: {summary_length}")

        # ELI5 with dynamic length
        st.markdown("**Explain like I'm 12**")
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("ELI5"):
                if ctx.strip():
                    max_len = eli5_mapping[eli5_length]["max"]
                    min_len = eli5_mapping[eli5_length]["min"]
                    
                    prompt = "Explain this to a 12-year-old:\n\n" + ctx
                    out = summarizer(prompt, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
                    st.info(out.strip())
                else:
                    st.warning("Select at least one article.")
        with col2:
            st.caption(f"Length: {eli5_length}")

        # Q&A (length already controlled by model)
        st.markdown("**Ask a question about selected articles**")
        q_user = st.text_input("Your question", value="ex: What cargo did Dragon deliver and to where?")
        if st.button("Answer"):
            if ctx.strip():
                ans = qa(question=q_user, context=ctx)
                st.write(f"**Answer:** {ans.get('answer')}  \n(confidence: {ans.get('score'):.2f})")
            else:
                st.warning("Select at least one article.")

with col_right:
    st.subheader("Tips")
    st.write("- **Adjust AI settings** in the sidebar to control output length and context size.")
    st.write("- Select the most relevant 2–4 articles to focus the AI.")
    st.write("- Use **Summarize** for quick updates, **ELI5** for simple explanations, and **Q&A** for specifics.")
    st.write("- Results are CPU-only; first run may be slower while models load.")
    st.write("- For best results, use recent articles (past few months).")
    
    # Show current settings
    st.subheader("Current Settings")
    st.write(f"📄 Context: {max_context_chars:,} chars")
    st.write(f"📝 Summary: {summary_length}")
    st.write(f"🎯 ELI5: {eli5_length}")