"""
06_build_vectorstore.py — Embed every unified element and store the vectors
in a persistent Chroma vector database.

Usage:
    python 06_build_vectorstore.py

What it does:
    1. Loads GEMINI_API_KEY from .env and creates a genai.Client.
    2. Loads extracted/unified_elements.json (produced by 05_unify_elements.py).
    3. For each element, calls Gemini's embedding API to convert the text
       into a dense vector (a list of floats).
    4. Stores the vector + its metadata in a local ChromaDB collection so
       we can later retrieve relevant chunks by semantic similarity.

Background — what is an embedding?
    An embedding is a numerical vector (list of floats) that captures the
    "meaning" of a piece of text.  Semantically similar texts have vectors
    that are close together (high cosine similarity).  We use the Gemini
    embedding model which produces 3072-dimensional vectors.

Why store metadata alongside the vector?
    ChromaDB stores three things per entry:
      - embedding  : the vector (used for similarity search)
      - document   : the original text (so the search can return it)
      - metadata   : a dict of extra fields (type, page, title, etc.)
    When we query "give me the top-5 most similar chunks", we want to know
    *which* chunk it is (its id, its type, its page number) and what to
    *show* the user (display_content, image_path).  Metadata makes this
    possible — without it we'd only get back a vector and a doc string.
"""

import json
import os
import sys
import time
from pathlib import Path


def safe_print(text: str):
    """Print to stdout using UTF-8 bytes, avoiding Windows cp1252 errors."""
    sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()

import chromadb
from dotenv import load_dotenv
from google import genai

# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

EXTRACTED_DIR        = Path("extracted")
UNIFIED_PATH        = EXTRACTED_DIR / "unified_elements.json"
CHROMA_DB_DIR       = "chroma_db"
COLLECTION_NAME     = "attention_paper"
EMBEDDING_MODEL     = "gemini-embedding-001"
API_DELAY_SECONDS   = 1  # delay between embedding calls


# ═══════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    # 1. Load the API key and create a Gemini client.
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        safe_print("ERROR: GEMINI_API_KEY not found in .env file.")
        return

    gemini = genai.Client(api_key=api_key)

    # 2. Load the unified elements.
    with open(UNIFIED_PATH, "r", encoding="utf-8") as f:
        elements: list[dict] = json.load(f)

    safe_print(f"Loaded {len(elements)} elements from {UNIFIED_PATH}\n")

    # 3. Set up a persistent ChromaDB client and (re)create the collection.
    #    Using cosine distance because we'll query with semantic similarity.
    chroma = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Delete any existing collection so we start fresh each run.
    # Get existing collection names to avoid a spurious error on first run.
    existing = [c.name for c in chroma.list_collections()]
    if COLLECTION_NAME in existing:
        chroma.delete_collection(COLLECTION_NAME)

    # embedding_function=None tells Chroma we are providing our own
    # pre-computed embeddings (via Gemini), so it should NOT try to
    # download or use its default all-MiniLM-L6-v2 ONNX model.
    collection = chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )

    # 4. Process every element: embed → store.
    for i, elem in enumerate(elements, start=1):
        eid      = elem["id"]
        etype    = elem["type"]
        title    = elem["title"]
        text     = elem["embedding_text"]
        display  = elem["display_content"]
        img_path = elem["image_path"]

        # Print progress.
        preview = text[:80].replace("\n", " ")
        safe_print(f"[{i}/{len(elements)}]  {eid}  ({etype}, page {elem['page']})")
        safe_print(f"        {preview}...")

        # 4a. Call Gemini's embedding API.
        #     gemini-embedding-001 produces a 3072-dimensional vector for
        #     any input text.  The result object has an `.embeddings` list
        #     (one entry per input in `contents`), and each entry has a
        #     `.values` attribute containing the float list.
        response = gemini.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )
        vector: list[float] = response.embeddings[0].values

        # 4b. Prepare metadata.
        #     Chroma's metadata dict must contain only string → str/int/float
        #     values — it cannot store Python None.  We convert None to "".
        metadata = {
            "type": etype,
            "page": elem["page"],
            "title": title,
            "display_content": display,
            "image_path": img_path if img_path is not None else "",
        }

        # 4c. Store in Chroma.
        collection.add(
            ids=[eid],
            embeddings=[vector],
            documents=[text],
            metadatas=[metadata],
        )

        # 4d. Small delay to stay under free-tier rate limits.
        if i < len(elements):
            time.sleep(API_DELAY_SECONDS)

    # 5. Summary.
    final_count = collection.count()
    safe_print(f"\n{'='*60}")
    safe_print(f"Done.  Stored {final_count} elements in Chroma collection '{COLLECTION_NAME}'")
    safe_print(f"Database location: {CHROMA_DB_DIR}/")
    safe_print(f"Embedding model:   {EMBEDDING_MODEL} (3072-dimensional vectors)")
    safe_print(f"Distance metric:   cosine")

    # Quick sanity query to confirm retrieval works.
    # Since we set embedding_function=None, Chroma cannot auto-embed raw
    # text — we must embed the query ourselves using Gemini, then pass the
    # resulting vector as query_embeddings.
    query_text = "transformer architecture"
    query_response = gemini.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query_text,
    )
    query_vector = query_response.embeddings[0].values
    sample = collection.query(query_embeddings=[query_vector], n_results=2)
    safe_print(f"\nSanity query — \"{query_text}\":")
    for j in range(len(sample["ids"][0])):
        sid = sample["ids"][0][j]
        stype = sample["metadatas"][0][j]["type"]
        spreview = sample["documents"][0][j][:100]
        safe_print(f"  {j+1}. {sid}  ({stype})  {spreview}...")


if __name__ == "__main__":
    main()
