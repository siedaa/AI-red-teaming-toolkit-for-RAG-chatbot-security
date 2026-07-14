"""
08_generate_answer.py — Final grounded-generation step of the RAG pipeline.

Usage:
    python 08_generate_answer.py

What it does:
    1. Retrieves relevant chunks for a query using 07_retrieve.py's retrieve().
    2. Builds a context string from the retrieved chunks (type, title, page,
       display_content).
    3. If any chunk is a figure with an image_path, opens that image with PIL
       and sends it to Gemini alongside the text — so the model can look at
       the actual figure, not just its text description.
    4. Calls Gemini 2.5 Flash to generate an answer grounded *only* in the
       provided context.
    5. Runs three test queries (one per modality: table, text, figure) and
       saves the results to demo_results.json and demo_results.md.

Why re-attach figure images at generation time?
    During retrieval we search using the *text* embedding (caption + vision
    description) because vector databases can't index pixels.  But once
    the figure is retrieved, we can show the actual image to the LLM so it
    can read the diagram directly — this gives richer answers than relying
    on the text description alone.
"""

import importlib
import json
import os
import sys
import time as _time
from pathlib import Path

import PIL.Image
from dotenv import load_dotenv
from google import genai

# Import retrieve() from 07_retrieve.py via importlib since names starting
# with a digit are not valid Python identifiers for a regular import.
_retrieve_mod = importlib.import_module("07_retrieve")
retrieve = _retrieve_mod.retrieve

# Reuse retry helpers from 07_retrieve (no need to duplicate the logic).
_interruptible_sleep = _retrieve_mod._interruptible_sleep
_api_retry          = _retrieve_mod._api_retry
_MAX_RETRY_SECONDS  = _retrieve_mod._MAX_RETRY_SECONDS

# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

GENERATION_MODEL = "gemini-2.5-flash"
OUTPUT_DIR       = Path("extracted")
DEMO_JSON        = OUTPUT_DIR / "demo_results.json"
DEMO_MD          = OUTPUT_DIR / "demo_results.md"

# The top-k setting for retrieval.
TOP_K = 3


def _safe(msg: str):
    """Print via UTF-8 bytes so Windows cp1252 can't choke on Unicode."""
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


# ═══════════════════════════════════════════════════════════════════
#  Context building
# ═══════════════════════════════════════════════════════════════════

def build_context(results: list[dict]) -> tuple[str, list[PIL.Image.Image]]:
    """
    Turn a list of retrieval results into:

    1. A single text string where each chunk is clearly labelled with its
       source (type, title, page, content).

    2. A list of PIL.Image objects for any figure-type results that have
       an image_path on disk.

    This context will be given to the LLM so it can answer the question
    based *only* on what we provide — no outside knowledge.
    """
    context_parts: list[str] = []
    figure_images: list[PIL.Image.Image] = []

    for i, r in enumerate(results, start=1):
        src_label = f"[Source {i}: {r['type'].title()} — \"{r['title']}\" (page {r['page']})]"
        content   = r["display_content"]
        block     = f"{src_label}\n{content}\n---"
        context_parts.append(block)

        # If this is a figure with an actual image file, load it so Gemini
        # can see the original diagram (not just its text description).
        if r["type"] == "figure" and r["image_path"]:
            img_path = Path(r["image_path"])
            if img_path.is_file():
                figure_images.append(PIL.Image.open(img_path))

    return "\n".join(context_parts), figure_images


# ═══════════════════════════════════════════════════════════════════
#  Answer generation
# ═══════════════════════════════════════════════════════════════════

def answer_query(query: str, top_k: int = TOP_K, pre_retrieved_results: list | None = None) -> dict:
    """
    Full RAG pipeline for a single query:

    retrieve → build_context → generate_content → return result dict.

    Args:
        query: The user's question.
        top_k: Number of chunks to retrieve.
        pre_retrieved_results: Optional pre-fetched results from retrieve().
            When provided, the internal retrieve() call is skipped so the
            embedding API isn't called twice (useful when the caller already
            called retrieve() for display purposes).
    """
    gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # 1. Retrieve relevant chunks (or reuse already-retrieved results).
    results = pre_retrieved_results if pre_retrieved_results is not None else retrieve(query, top_k)

    # 2. Build the grounded context text + collect any figure images.
    context_text, figure_images = build_context(results)

    # 3. Build the system prompt.
    #
    # IMPORTANT: We explicitly forbid citation markers like [Source 1] or
    # [1] because they are meaningless to the end user — we already print
    # a separate SOURCES list with real labels.  Instead, Gemini should
    # refer to sources naturally (e.g. "as shown in Table 2").
    prompt = f"""\
You are a helpful research assistant answering questions about the paper
"Attention Is All You Need".  Use ONLY the context below to answer the
question.  If the context does not contain enough information, say so
explicitly — do not guess or use outside knowledge.

IMPORTANT: Do NOT insert citation markers like [Source 1] or [1] or
[Source N] into your answer.  Write the answer as clean, natural prose.
If you need to reference where a fact came from, refer to it naturally
by name (e.g. "as shown in Table 2" or "as described in Figure 1"),
using the actual title/caption from the provided context.

---
CONTEXT:
{context_text}
---
QUESTION: {query}

ANSWER:"""

    # 4. Assemble the contents list: prompt text + any figure images.
    #    The prompt goes first, followed by the images (if any).
    contents: list = [prompt]
    contents.extend(figure_images)

    # 5. Call Gemini for generation (with retry on 429 quota errors).
    #    The _api_retry helper handles exponential backoff, full error
    #    messages, interruptible sleep (for clean Ctrl+C), and a hard
    #    timeout of _MAX_RETRY_SECONDS.
    _safe(f"  └─ Generating answer ({len(figure_images)} figure(s) attached)...")
    response = _api_retry(
        "generation",
        gemini.models.generate_content,
        model=GENERATION_MODEL,
        contents=contents,
    )
    answer = response.text.strip()

    return {
        "query": query,
        "sources": [
            {
                "id": r["id"],
                "type": r["type"],
                "page": r["page"],
                "title": r["title"],
                "similarity": r["similarity"],
            }
            for r in results
        ],
        "full_sources": results,  # includes display_content, image_path for UI
        "answer": answer,
    }


# ═══════════════════════════════════════════════════════════════════
#  Logging / saving
# ═══════════════════════════════════════════════════════════════════

def run_and_log(query: str) -> dict:
    """Run answer_query and print the results to the terminal."""
    _safe("")
    _safe("=" * 70)
    _safe(f"QUERY: {query}")
    _safe("=" * 70)

    record = answer_query(query)

    _safe("\nSOURCES:")
    for s in record["sources"]:
        _safe(f"  [{s['type']}] {s['id']}  page {s['page']}  sim={s['similarity']}  — {s['title'][:80]}")

    _safe(f"\nANSWER:\n{record['answer']}\n")

    return record


def save_results(records: list[dict]):
    """Save results as JSON and as a readable Markdown file."""

    # JSON
    with open(DEMO_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    _safe(f"Saved {DEMO_JSON}")

    # Markdown
    lines: list[str] = [
        "# RAG Demo Results — Attention Is All You Need\n",
        f"_{len(records)} queries run using Gemini 2.5 Flash for generation._\n",
        "---\n",
    ]
    for i, rec in enumerate(records, start=1):
        lines.append(f"## Query {i}: {rec['query']}\n")
        lines.append("### Retrieved Sources\n")
        for s in rec["sources"]:
            lines.append(f"- **[{s['type']}]** `{s['id']}` (page {s['page']})  "
                         f"sim={s['similarity']}  \n  _{s['title']}_")
        lines.append("")
        lines.append("### Generated Answer\n")
        lines.append(rec["answer"])
        lines.append("")
        lines.append("---\n")

    with open(DEMO_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    _safe(f"Saved {DEMO_MD}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        _safe("ERROR: GEMINI_API_KEY not found in .env file.")
        return

    queries = [
        "What BLEU score did the Transformer achieve on English-to-French "
        "translation, and how does it compare to the training cost of "
        "previous models?",

        "Describe the encoder-decoder architecture shown in the Transformer "
        "diagram.",

        "What is multi-head attention and why is it used instead of "
        "single-head attention?",
    ]

    all_records: list[dict] = []
    for i, query in enumerate(queries):
        try:
            record = run_and_log(query)
            all_records.append(record)
            # Save incrementally after each successful query so we don't
            # lose results to quota exhaustion on later queries.
            save_results(all_records)
        except Exception as exc:
            _safe(f"  !! Query failed after retries: {exc}")
        if i < len(queries) - 1:
            time.sleep(5)  # generous delay between generation calls

    if all_records:
        _safe(f"\nFinal.  Demo results written to {DEMO_JSON} and {DEMO_MD}")
    else:
        _safe("No records to save — all queries failed.")

    _safe(f"\nDone.  Demo results written to {DEMO_JSON} and {DEMO_MD}")


if __name__ == "__main__":
    main()
