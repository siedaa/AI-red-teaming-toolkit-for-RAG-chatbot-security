"""
app.py — Streamlit frontend for the multimodal RAG pipeline.

Usage:
    streamlit run app.py

This imports the existing retrieve() and answer_query() from the pipeline
scripts so we don't duplicate the retrieval or generation logic — the app
is purely a UI layer on top of the same backend.
"""

import importlib
from pathlib import Path

import streamlit as st

# ── Import pipeline logic ──────────────────────────────────────────
# Names starting with a digit aren't valid Python identifiers, so we
# use importlib instead of a plain "from 07_retrieve import ...".
_retrieve_mod = importlib.import_module("07_retrieve")
retrieve      = _retrieve_mod.retrieve

_answer_mod   = importlib.import_module("08_generate_answer")
answer_query  = _answer_mod.answer_query

# ═══════════════════════════════════════════════════════════════════
#  Page configuration
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Multimodal RAG — Attention Is All You Need",
    page_icon="📄",
    layout="wide",
)


# ── Sidebar ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### About")
    st.markdown("**Built by Maha Fatima**")
    st.markdown(
        "A multimodal RAG system built as a learning project, "
        "combining text, tables, and figures from the "
        "'Attention Is All You Need' paper into one retrieval "
        "pipeline using the Gemini API."
    )

st.markdown("""
# Multimodal RAG:  "Attention Is All You Need"

A **retrieval-augmented generation** system over the Transformer paper.
The pipeline chunks the PDF into **text**, **tables**, and **figures**,
embeds everything with **Gemini**, stores it in a **Chroma** vector store,
and answers questions using **Gemini 2.5 Flash**  with the ability to
re-attach the original figure images at generation time for richer answers.

---
""")


# ═══════════════════════════════════════════════════════════════════
#  Session state  persist the latest results across re-runs
# ═══════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results = None    # full retrieval results (for display)
    st.session_state.answer  = None    # generated answer string
    st.session_state.query   = ""      # last query text


# ═══════════════════════════════════════════════════════════════════
#  Input area
# ═══════════════════════════════════════════════════════════════════

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "Ask a question about the Transformer paper:",
        placeholder="e.g. What BLEU score did the Transformer achieve on English-to-French translation?",
        label_visibility="collapsed",
    )
with col2:
    ask_clicked = st.button("Ask", type="primary", use_container_width=True)

# Example query buttons — one per modality so users can try with one click.
st.markdown("**Try an example:**")
ex_cols = st.columns(4)
example_queries = [
    (" Table", "What BLEU score did the Transformer achieve on English-to-French translation, and how does it compare to the training cost of previous models?"),
    (" Architecture", "Describe the encoder-decoder architecture shown in the Transformer diagram."),
    (" Attention", "What is multi-head attention and why is it used instead of single-head attention?"),
    (" Training", "What regularization techniques does the Transformer use during training?"),
]

for col, (label, ex_q) in zip(ex_cols, example_queries):
    with col:
        if st.button(label, use_container_width=True):
            query = ex_q
            ask_clicked = True


# ═══════════════════════════════════════════════════════════════════
#  Run the pipeline when the user asks a question
# ═══════════════════════════════════════════════════════════════════

if ask_clicked and query.strip():
    with st.spinner("🔍 Retrieving relevant chunks and generating answer..."):
        try:
            # Call answer_query once — it returns both the answer text
            # and the full source results (via the "full_sources" key),
            # so we don't need a separate retrieve() call.
            ans = answer_query(query.strip(), top_k=3)
            st.session_state.results = ans["full_sources"]
            st.session_state.answer  = ans["answer"]
            st.session_state.query   = query.strip()
        except RuntimeError as e:
            st.session_state.answer = None
            st.error(f"Quota error: {e}")
        except Exception as e:
            st.session_state.answer = None
            st.error(f"An unexpected error occurred:\n\n{e}")


# ═══════════════════════════════════════════════════════════════════
#  Source-card helper (defined here so it's available when called below)
# ═══════════════════════════════════════════════════════════════════

def _draw_source_card(r: dict):
    """Render a single retrieved source as a bordered card inside a column.

    Handles all three types differently:
      - text  → section title, page, text preview
      - table → caption + formatted table content (rendered as markdown)
      - figure → caption + the actual image via st.image()
    """
    etype = r["type"]
    sim   = r.get("similarity", 0)
    title = r.get("title", "")
    page  = r.get("page", "")
    content = r.get("display_content", "")

    with st.container(border=True):
        # Badge row: type + similarity
        badge_cols = st.columns([1, 1])
        badge_cols[0].markdown(f"**{etype.upper()}**  ȯ page {page}")
        badge_cols[1].markdown(f"<div style='text-align:right'>sim={sim:.3f}</div>",
                               unsafe_allow_html=True)

        if etype == "figure":
            # Show the caption and the actual image.
            st.markdown(f"*{title}*")
            img_path = r.get("image_path")
            if img_path and Path(img_path).is_file():
                st.image(str(img_path), use_container_width=True)
            else:
                st.caption("(image file not found)")

        elif etype == "table":
            # Show caption, then try to render the content as a markdown table.
            st.markdown(f"*{title}*")
            if content:
                st.text(content[:600])
            else:
                st.caption("(empty table)")

        else:  # text
            st.markdown(f"*{title}*")
            if content:
                st.markdown(content[:500])
            else:
                st.caption("(empty chunk)")


# ═══════════════════════════════════════════════════════════════════
#  Display results (if we have any)
# ═══════════════════════════════════════════════════════════════════

if st.session_state.answer:
    st.markdown("---")

    # ── Answer ─────────────────────────────────────────────────────
    st.markdown("### ✨ Answer")
    st.info(st.session_state.answer)

    st.markdown("")

    # ── Sources used ───────────────────────────────────────────────
    with st.expander("📚 Sources used", expanded=True):
        src_results = st.session_state.results
        if src_results:
            # Lay out source cards in columns (2 per row).
            for i in range(0, len(src_results), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(src_results):
                        break
                    r = src_results[idx]
                    with col:
                        _draw_source_card(r)

    st.markdown("")
    st.caption(f"Query: _{st.session_state.query}_")


# ═══════════════════════════════════════════════════════════════════
#  Footer
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "*Learning project  multimodal RAG over the "
    "[Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper "
    "using the Gemini API and ChromaDB.*"
)



