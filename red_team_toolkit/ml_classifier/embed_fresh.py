"""
embed_fresh.py — Step 5 of the ML classifier sub-project (PART C: fresh embeddings).

WHAT this script does:
    Embeds the 24 texts in fresh_eval_set.json with the SAME embedding model and
    the SAME safety measures used to build the original training data, and saves
    embeddings_fresh.npy + labels_fresh.npy for evaluate_fresh.py.

WHY MUST THESE SETTINGS MATCH embed_dataset.py EXACTLY?
    Same reason as before: TRAIN/SERVE SKEW. The MLP and linear-probe weights
    were trained on vectors produced by gemini-embedding-001 with no task_type
    and no dimension override (default 3072-dim). If we embedded the fresh set
    with any other model or option, the new vectors would live in a DIFFERENT
    space and the classifiers' scores would be meaningless — the code would run
    fine, the predictions would just be garbage. We therefore:
      - read the model name from embedding_meta.json (the recorded source of
        truth), and
      - reuse the exact helpers/constants from embed_dataset.py so the timeout,
        delay, retry, and atomic-write behaviour are identical by construction.

*** THE FRESH SET IS NEVER USED FOR TRAINING OR TUNING ***
    fresh_eval_set.json / embeddings_fresh.npy are a FINAL, untouched exam —
    like the test set, but written by us after the models were frozen. No weight
    update, no threshold change, no hyperparameter choice may be made because
    of anything these 24 examples reveal. If a model misses a fresh attack, we
    REPORT it; we do not go retrain. This is what makes the fresh score a real
    measurement of generalization instead of a self-congratulatory number.

SAFETY MEASURES (identical to embed_dataset.py, reused from it):
    1. IPv4-first DNS patch        (avoids Windows IPv6 TLS breakage)
    2. explicit 60s HTTP timeout   (in MILLISECONDS for the SDK)
    3. 1s delay between calls      (stay under per-minute quota)
    4. bounded 3-attempt retry     (stops on 429/RESOURCE_EXHAUSTED)
    5. atomic tmp-to-rename writes (never leave a half-written .npy)

HOW to run it (from the ml_classifier/ folder):
    python embed_fresh.py

WHAT it produces:
    embeddings_fresh.npy   float32, shape (24, 3072)
    labels_fresh.npy       int64,   shape (24,)
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse embed_dataset.py's helpers + constants directly. This is deliberate:
# importing them guarantees the fresh embeddings are produced under the exact
# same code path as the training embeddings (no drift from a copy-paste).
from embed_dataset import (
    EMBEDDING_MODEL,          # "gemini-embedding-001"
    HTTP_TIMEOUT_MS,          # 60 * 1000
    API_DELAY_SECONDS,        # 1.0
    MAX_ATTEMPTS_PER_TEXT,    # 3
    apply_ipv4_first_patch,
    setup_environment,
    embed_all,
    save_npy_atomic,
    save_json_atomic,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FRESH_JSON = SCRIPT_DIR / "fresh_eval_set.json"
EMB_FRESH_NPY = SCRIPT_DIR / "embeddings_fresh.npy"
LABELS_FRESH_NPY = SCRIPT_DIR / "labels_fresh.npy"
META_JSON = SCRIPT_DIR / "embedding_meta.json"


def main() -> None:
    # (a) Network + environment safety BEFORE any client exists.
    apply_ipv4_first_patch()
    setup_environment()

    # (b) Read the embedding model from embedding_meta.json — the recorded
    #     source of truth for what the trained weights expect. We do NOT
    #     hardcode it here so a future re-embed can never silently diverge.
    meta = json.loads(META_JSON.read_text(encoding="utf-8"))
    model_name = meta["model"]
    if model_name != EMBEDDING_MODEL:
        # Defensive: if they ever disagree, refuse to run rather than produce
        # vectors in the wrong space.
        raise RuntimeError(
            f"embedding_meta.json says {model_name!r} but embed_dataset.py uses "
            f"{EMBEDDING_MODEL!r}. Fix the mismatch before embedding the fresh set."
        )
    print(f"embedding model (from embedding_meta.json): {model_name}")

    # (c) Build the client with the same explicit millisecond timeout.
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS),  # milliseconds!
    )

    # (d) Load the 24 fresh texts + labels.
    records = json.loads(FRESH_JSON.read_text(encoding="utf-8"))
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    print(f"fresh set: {len(texts)} texts  "
          f"({sum(1 for l in labels if l == 0)} legit / "
          f"{sum(1 for l in labels if l == 1)} attack)")

    # (e) Embed everything with the exact same helpers + safety measures.
    print(f"Embedding with {model_name} (delay {API_DELAY_SECONDS}s, "
          f"timeout {HTTP_TIMEOUT_MS // 1000}s, "
          f"max {MAX_ATTEMPTS_PER_TEXT} attempts/text)...")
    t0 = time.time()
    vectors = embed_all(client, texts)
    print(f"Done in {time.time() - t0:.1f}s")

    # (f) Convert to numpy with explicit dtypes + shape sanity checks.
    import numpy as np

    emb = np.array(vectors, dtype=np.float32)
    lab = np.array(labels, dtype=np.int64)
    n_dim = emb.shape[1]
    assert emb.shape == (len(texts), n_dim), f"shape mismatch: {emb.shape}"
    assert lab.shape == (len(texts),), f"shape mismatch: {lab.shape}"
    # The dimension must equal what the trained weights expect (3072), or the
    # Linear layers would reject the input at scoring time.
    assert n_dim == meta["dimension"], (
        f"embedded dim {n_dim} != recorded {meta['dimension']}"
    )

    # (g) Save atomically (tmp + rename — nothing half-written).
    save_npy_atomic(EMB_FRESH_NPY, emb)
    save_npy_atomic(LABELS_FRESH_NPY, lab)

    # (h) Report shapes.
    print()
    print("=" * 60)
    print("FRESH EMBEDDING OUTPUT")
    print("=" * 60)
    print(f"embeddings_fresh.npy : shape={emb.shape}  dtype={emb.dtype}")
    print(f"labels_fresh.npy     : shape={lab.shape}  dtype={lab.dtype}")
    print(f"model                : {model_name}  dim={n_dim}")
    print(f"created_utc          : {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
