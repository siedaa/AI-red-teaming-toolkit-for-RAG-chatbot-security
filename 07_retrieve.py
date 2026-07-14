"""
07_retrieve.py — Reusable retrieval function against the existing Chroma
vector store built by 06_build_vectorstore.py.

Usage (import):
    from 07_retrieve import retrieve

    results = retrieve("What BLEU score did the Transformer achieve?")

Usage (CLI):
    python 07_retrieve.py

What it does:
    1. Loads GEMINI_API_KEY from .env, creates a genai.Client.
    2. Connects to the persistent Chroma store at chroma_db/ and gets the
       "attention_paper" collection.
    3. Defines retrieve(query, top_k=4) which:
       a. Embeds the query with Gemini.
       b. Queries Chroma for the top_k nearest neighbours.
       c. Returns a clean list of result dicts with the distance score
          included.
    4. When run as a script, tests retrieval on three sample queries and
       prints the top results so you can manually inspect relevance.

Understanding the distance metric:
    We created the collection with "hnsw:space": "cosine", so Chroma uses
    cosine distance internally.  Cosine distance = 1 - cosine_similarity.
    Lower values mean the vectors are more similar:
        0.0  → identical direction (perfect match)
        0.5  → somewhat similar
        1.0  → orthogonal (no relation)
        2.0  → opposite directions
    We convert this to a cosine_similarity score (= 1 - distance) so that
    higher = better, which is more intuitive for display.
"""

import os
import sys
import time as _time
from typing import Any

import chromadb
from dotenv import load_dotenv
from google import genai


def _safe(msg: str):
    """Print via UTF-8 bytes so Windows cp1252 can't choke on Unicode."""
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


_MAX_RETRY_SECONDS = 180  # hard cap on total retry time


def _interruptible_sleep(seconds: float):
    """Sleep in small increments so KeyboardInterrupt fires promptly."""
    deadline = _time.time() + seconds
    while _time.time() < deadline:
        remaining = deadline - _time.time()
        _time.sleep(min(remaining, 0.2))


def _api_retry(label: str, fn, *args, **kwargs):
    """
    Call *fn(*args, **kwargs)* and retry on 429 / RESOURCE_EXHAUSTED with
    exponential backoff.  Prints the full error message on each failure.
    Raises RuntimeError if quota is still exhausted after *MAX_RETRY_SECONDS*.
    """
    max_attempts = 10  # upper bound; the time cap is what stops us
    start = _time.time()
    last_err = ""

    for attempt in range(max_attempts):
        if attempt > 0:
            elapsed = _time.time() - start
            if elapsed >= _MAX_RETRY_SECONDS:
                raise RuntimeError(
                    f"[{label}] Gemini API quota still exhausted after "
                    f"{elapsed:.0f}s of retries.\n  Last error: {last_err}\n"
                    "  Try again later."
                )
            # Exponential backoff capped at 60s per wait.
            wait = min(30 * (2 ** (attempt - 1)), 60)
            # Don't exceed the remaining time budget.
            remaining_budget = _MAX_RETRY_SECONDS - elapsed
            wait = min(wait, remaining_budget, 60)
            _safe(
                f"     429 — [{label}] retrying in {wait:.0f}s "
                f"(attempt {attempt + 1}, {elapsed:.0f}s elapsed)...\n"
                f"           Full error: {last_err}"
            )
            _interruptible_sleep(wait)

        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            err_str = str(exc)
            if "429" not in err_str and "RESOURCE_EXHAUSTED" not in err_str:
                raise  # non-quota error → fail immediately
            last_err = err_str
            # Fall through to retry.

    raise RuntimeError(
        f"[{label}] Exceeded max attempts ({max_attempts}) for quota error.\n"
        f"  Last error: {last_err}"
    )

# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

CHROMA_DB_DIR   = "chroma_db"
COLLECTION_NAME = "attention_paper"
EMBEDDING_MODEL = "gemini-embedding-001"

# ═══════════════════════════════════════════════════════════════════
#  Globals (lazy-initialised once on first call to retrieve())
# ═══════════════════════════════════════════════════════════════════

_gemini_client: genai.Client | None = None
_chroma_collection = None


def _ensure_initialised():
    """Load the API key and connect to Chroma, if not already done."""
    global _gemini_client, _chroma_collection

    if _chroma_collection is not None:
        return

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env file.")

    _gemini_client = genai.Client(api_key=api_key)

    chroma = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    _chroma_collection = chroma.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,  # we provide embeddings via Gemini
    )


# ═══════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════

def retrieve(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    """
    Embed *query* with Gemini and return the *top_k* most similar chunks
    from the Chroma vector store.

    Each result dict has:
        id               — e.g. "text_3", "table_1", "figure_2"
        type             — "text" | "table" | "figure"
        page             — page number in the PDF
        title            — section name / caption
        display_content  — human-readable content to show when retrieved
        image_path       — path to image file (None for text/table)
        distance         — cosine distance (0 = perfect, lower = better)
        similarity       — 1 - distance (higher = better, for convenience)
    """
    _ensure_initialised()

    # 1. Embed the query string using Gemini (with retry on 429).
    response = _api_retry(
        "embedding",
        _gemini_client.models.embed_content,
        model=EMBEDDING_MODEL,
        contents=query,
    )
    query_vector: list[float] = response.embeddings[0].values

    # 2. Query Chroma for the nearest neighbours.
    #    Chroma returns parallel lists inside a dict:
    #      ids[0]        → list of result IDs
    #      distances[0]  → list of distances (cosine)
    #      documents[0]  → list of original embedding_text strings
    #      metadatas[0]  → list of metadata dicts
    raw = _chroma_collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )

    # 3. Turn the raw Chroma output into clean result dicts.
    results: list[dict[str, Any]] = []
    for i in range(len(raw["ids"][0])):
        meta = raw["metadatas"][0][i]
        distance = raw["distances"][0][i]  # cosine distance (lower = better)
        results.append({
            "id": raw["ids"][0][i],
            "type": meta["type"],
            "page": meta["page"],
            "title": meta["title"],
            "display_content": meta["display_content"],
            "image_path": meta["image_path"] or None,  # "" back to None
            "distance": round(distance, 4),
            "similarity": round(1.0 - distance, 4),
        })

    return results


# ═══════════════════════════════════════════════════════════════════
#  CLI test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_queries = [
        "What BLEU score did the Transformer achieve on English-to-French translation?",
        "Describe the encoder-decoder architecture of the Transformer",
        "What is multi-head attention?",
    ]

    for q in test_queries:
        _safe("")
        _safe("=" * 70)
        _safe(f"QUERY: {q}")
        _safe("=" * 70)

        results = retrieve(q, top_k=4)

        for rank, r in enumerate(results, start=1):
            preview = r["display_content"][:150].replace("\n", " ")
            _safe("")
            _safe(f"  [{rank}]  {r['id']}  ({r['type']}, page {r['page']})")
            _safe(f"         distance={r['distance']}  similarity={r['similarity']}")
            _safe(f"         title: {r['title'][:100]}")
            _safe(f"         {preview}...")

        _safe("")
