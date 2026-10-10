"""
embed_stress.py — Stress-test embeddings for the ML classifier sub-project.

WHAT this script does:
    Embeds the 18 stress-test texts in stress_set.json using the SAME embedding
    model and the SAME safety measures used for the training, fresh, and (now)
    stress sets, then saves embeddings_stress.npy + labels_stress.npy so
    evaluate_stress.py can score them with the frozen classifiers.

WHY REUSE embed_fresh.py's HELPERS?
    Exactly as before: TRAIN/SERVE SKEW. The frozen MLP and linear-probe
    weights were trained on vectors produced by gemini-embedding-001 with no
    task_type and no dimension override (default 3072-dim). If we embedded the
    stress set with any other model or option, the new vectors would live in a
    DIFFERENT space and the classifiers' scores would be meaningless — the
    code would run fine, the predictions would just be garbage. We therefore
    import the exact helpers and constants from embed_fresh.py (which itself
    imports them from embed_dataset.py) so the timeout, delay, retry, IPv4
    patch, and atomic-write behaviour are identical by construction.

*** THE STRESS SET IS NEVER USED FOR TRAINING OR TUNING ***
    stress_set.json / embeddings_stress.npy are a FINAL, untouched exam —
    like the test and fresh sets, but focused on input TYPES the training data
    never covered (very short attacks, disguised paper-question attacks, very
    long legitimate questions, off-topic questions, small talk, non-English).
    No weight update, no threshold change, no hyperparameter choice may be
    made because of anything these 18 examples reveal. If a model misses a
    stress attack or wrongly blocks a stress legitimate prompt, we REPORT it;
    we do NOT go retrain. This is what makes the stress score a real
    measurement of generalization instead of a self-congratulatory number.

HONESTY NOTE (also stored inside stress_set.json):
    This set was drafted by the AI assistant, NOT by the user. So it is NOT
    fully independent of the training data's author style. Its value is the
    BREADTH of input types it covers, not authorship independence. See the
    "authorship" field in stress_set.json.

SAFETY MEASURES (identical to embed_fresh.py / embed_dataset.py, reused):
    1. IPv4-first DNS patch        (avoids Windows IPv6 TLS breakage)
    2. explicit 60s HTTP timeout   (in MILLISECONDS for the SDK)
    3. 1s delay between calls      (stay under per-minute quota)
    4. bounded 3-attempt retry     (stops on 429/RESOURCE_EXHAUSTED)
    5. atomic tmp-to-rename writes (never leave a half-written .npy)

WINDOWS CONSOLE SAFETY:
    The stress set contains non-ASCII text (e.g. the Spanish prompt S-17 with
    '¿', 'é', 'á'). Printing those directly can raise UnicodeEncodeError on a
    Windows console whose default encoding is cp1252. We therefore print via a
    _safe() helper that writes UTF-8 bytes to sys.stdout.buffer with
    errors="replace", exactly like 07_retrieve.py's _safe().

HOW to run it (from the ml_classifier/ folder):
    python embed_stress.py

WHAT it produces:
    embeddings_stress.npy   float32, shape (18, 3072)
    labels_stress.npy       int64,   shape (18,)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse embed_fresh.py's helpers + constants directly. This is deliberate:
# importing them guarantees the stress embeddings are produced under the exact
# same code path as every earlier embedding run (no drift from a copy-paste).
from embed_fresh import (
    EMBEDDING_MODEL,          # "gemini-embedding-001"
    HTTP_TIMEOUT_MS,          # 60 * 1000
    API_DELAY_SECONDS,        # 1.0
    MAX_ATTEMPTS_PER_TEXT,    # 3
    apply_ipv4_first_patch,
    setup_environment,
    embed_all,
    save_npy_atomic,
)

SCRIPT_DIR = Path(__file__).resolve().parent
STRESS_JSON = SCRIPT_DIR / "stress_set.json"
EMB_STRESS_NPY = SCRIPT_DIR / "embeddings_stress.npy"
LABELS_STRESS_NPY = SCRIPT_DIR / "labels_stress.npy"
META_JSON = SCRIPT_DIR / "embedding_meta.json"


def _safe(msg: str) -> None:
    """
    Print via UTF-8 bytes so Windows cp1252 can't choke on Unicode.

    WHY: the stress set includes a Spanish prompt containing '¿' and accented
    letters. On a cp1252 console, print() of those characters raises
    UnicodeEncodeError and kills the run. Writing UTF-8 bytes with
    errors="replace" guarantees the script always finishes.
    """
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


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
            f"embedding_meta.json says {model_name!r} but embed_fresh.py uses "
            f"{EMBEDDING_MODEL!r}. Fix the mismatch before embedding the stress set."
        )
    _safe(f"embedding model (from embedding_meta.json): {model_name}")

    # (c) Build the client with the same explicit millisecond timeout.
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS),  # milliseconds!
    )

    # (d) Load the 18 stress texts + labels from the top-level "records" list.
    payload = json.loads(STRESS_JSON.read_text(encoding="utf-8"))
    records = payload["records"]
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    _safe(f"stress set: {len(texts)} texts  "
          f"({sum(1 for l in labels if l == 0)} legit / "
          f"{sum(1 for l in labels if l == 1)} attack)  "
          f"[authorship: {payload.get('authorship', 'unknown')}]")

    # (e) Embed everything with the exact same helpers + safety measures.
    _safe(f"Embedding with {model_name} (delay {API_DELAY_SECONDS}s, "
          f"timeout {HTTP_TIMEOUT_MS // 1000}s, "
          f"max {MAX_ATTEMPTS_PER_TEXT} attempts/text)...")
    t0 = time.time()
    vectors = embed_all(client, texts)
    _safe(f"Done in {time.time() - t0:.1f}s")

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
    save_npy_atomic(EMB_STRESS_NPY, emb)
    save_npy_atomic(LABELS_STRESS_NPY, lab)

    # (h) Report shapes.
    _safe("")
    _safe("=" * 60)
    _safe("STRESS EMBEDDING OUTPUT")
    _safe("=" * 60)
    _safe(f"embeddings_stress.npy : shape={emb.shape}  dtype={emb.dtype}")
    _safe(f"labels_stress.npy     : shape={lab.shape}  dtype={lab.dtype}")
    _safe(f"model                 : {model_name}  dim={n_dim}")
    _safe(f"created_utc           : {datetime.now(timezone.utc).isoformat()}")
    _safe("=" * 60)


if __name__ == "__main__":
    main()
