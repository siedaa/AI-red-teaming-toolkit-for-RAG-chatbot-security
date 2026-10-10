"""
embed_aug.py — Augmentation-set embeddings for the ML classifier v2 sub-project.

WHAT this script does:
    Embeds the 62 texts in aug_set.json using the SAME embedding model and the
    SAME safety measures used for every earlier embedding run, and saves
    embeddings_aug.npy + labels_aug.npy so train_v2.py can concatenate them
    with the original training pool.

WHY REUSE embed_stress.py's HELPERS?
    TRAIN/SERVE SKEW, same as every earlier step. The frozen classifiers (and
    the v2 models about to be trained) all expect vectors from
    gemini-embedding-001 with no task_type and no dimension override (default
    3072-dim). We import the exact helpers and constants from embed_stress.py
    (which imports them from embed_fresh.py, which imports them from
    embed_dataset.py) so the timeout, delay, retry, IPv4 patch, and
    atomic-write behaviour are identical by construction — no copy-paste drift.

WHY AN AUGMENTATION SET AT ALL?
    The v1 classifier flagged off-topic and small-talk prompts as attacks
    because its legitimate class contained only on-topic paper questions, so
    it learned a spurious rule: "on-topic vs not-on-topic" instead of
    "attack vs legitimate". That is DISTRIBUTION SHIFT — the training data's
    input distribution did not match real traffic. aug_set.json adds the
    missing legitimate types (off-topic, small talk, multilingual, short paper
    questions) plus more attack styles (short, multilingual, format-disguised,
    long social-engineered) so v2 sees a broader, more realistic spread.

*** THE AUG SET IS TRAINING DATA — NOT AN EVALUATION SET ***
    These 62 texts are concatenated into the v2 training pool (see
    train_v2.py). They are therefore NOT a valid way to measure v2's quality:
    scoring v2 on data it trained on would be memorization, not
    generalization. v2 is measured on the held-out test set and the fresh set
    (both untouched by training), and reported on the stress set as a
    DEVELOPMENT set (its earlier failures motivated this change, so it is no
    longer an independent estimate).

*** NO HYPERPARAMETER TUNING ON ANY EVALUATION SET ***
    We do not tune any hyperparameter, threshold, or architecture based on
    results from the held-out test set, the fresh set, or the stress set.
    v2 uses exactly train.py's defaults.

HONESTY NOTE (also stored in aug_set.json):
    This set was drafted by the AI assistant, NOT by the user, so it is not
    fully independent of the training data's author style.

SAFETY MEASURES (identical to earlier steps, reused from embed_stress.py):
    1. IPv4-first DNS patch        (avoids Windows IPv6 TLS breakage)
    2. explicit 60s HTTP timeout   (in MILLISECONDS for the SDK)
    3. 1s delay between calls      (stay under per-minute quota)
    4. bounded 3-attempt retry     (stops on 429/RESOURCE_EXHAUSTED)
    5. atomic tmp-to-rename writes (never leave a half-written .npy)

WINDOWS CONSOLE SAFETY:
    The aug set contains non-ASCII text (Spanish/French/German/Urdu prompts).
    Printing those directly can raise UnicodeEncodeError on a Windows console
    whose default encoding is cp1252. We print via a _safe() helper that
    writes UTF-8 bytes to sys.stdout.buffer with errors="replace", exactly
    like the earlier scripts.

HOW to run it (from the ml_classifier/ folder):
    python embed_aug.py

WHAT it produces:
    embeddings_aug.npy   float32, shape (62, 3072)
    labels_aug.npy       int64,   shape (62,)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse embed_stress.py's helpers + constants directly. This is deliberate:
# importing them guarantees the aug embeddings are produced under the exact
# same code path as every earlier embedding run (no drift from a copy-paste).
from embed_stress import (
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
AUG_JSON = SCRIPT_DIR / "aug_set.json"
EMB_AUG_NPY = SCRIPT_DIR / "embeddings_aug.npy"
LABELS_AUG_NPY = SCRIPT_DIR / "labels_aug.npy"
META_JSON = SCRIPT_DIR / "embedding_meta.json"

# The four existing sets we must have NO text overlap with (checked loudly
# before spending any API quota — a duplicate would silently leak evaluation
# data into training).
EXISTING_SETS = (
    "dataset_train.json",
    "dataset_test.json",
    "fresh_eval_set.json",
    "stress_set.json",
)


def _safe(msg: str) -> None:
    """
    Print via UTF-8 bytes so Windows cp1252 can't choke on Unicode.

    WHY: the aug set includes Spanish/French/German/Urdu prompts with '¿',
    'é', 'ü', etc. On a cp1252 console, print() of those raises
    UnicodeEncodeError and kills the run. Writing UTF-8 bytes with
    errors="replace" guarantees the script always finishes.
    """
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def norm(text: str) -> str:
    """Normalize a text for the duplicate check: lowercase + strip whitespace.

    WHY: two texts that differ only in capitalization or leading/trailing
    spaces are effectively the same prompt. Comparing raw strings would miss
    those near-duplicates, so we normalize before checking.
    """
    return text.lower().strip()


def load_texts_from(path: Path) -> list[str]:
    """Read the 'text' field from any of our dataset JSON shapes.

    fresh_eval_set.json is a bare LIST of records; the others are OBJECTS with
    a "records" list. We handle both so this works for all four existing sets.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [r["text"] for r in data]
    return [r["text"] for r in data["records"]]


def assert_no_overlap(aug_texts: list[str]) -> None:
    """
    LOUDLY fail if any aug text duplicates a text in any existing set.

    WHY this matters: if an evaluation-set text leaked into the augmentation
    set, it would become training data — and then scoring v2 on that
    evaluation set would measure memorization, not generalization. We check
    before embedding (no wasted quota) and raise RuntimeError on any overlap.
    """
    aug_norm = {norm(t): t for t in aug_texts}
    # A duplicate WITHIN the aug set itself would also be a problem.
    if len(aug_norm) != len(aug_texts):
        dupes = [t for t in aug_texts if list(aug_norm.values()).count(aug_norm[norm(t)]) > 1]
        raise RuntimeError(f"Duplicate texts within aug_set.json itself: {dupes}")

    overlap_found = False
    for name in EXISTING_SETS:
        other_norm = {norm(t) for t in load_texts_from(SCRIPT_DIR / name)}
        hits = aug_norm.keys() & other_norm
        if hits:
            overlap_found = True
            _safe(f"OVERLAP with {name}:")
            for h in hits:
                _safe(f"    {aug_norm[h]!r}")
        else:
            _safe(f"  {name}: no overlap ({len(other_norm)} texts checked)")
    if overlap_found:
        raise RuntimeError(
            "aug_set.json duplicates text from an existing evaluation/training "
            "set. Remove the duplicate before embedding — allowing it would "
            "leak evaluation data into the v2 training pool."
        )


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
            f"embedding_meta.json says {model_name!r} but embed_stress.py uses "
            f"{EMBEDDING_MODEL!r}. Fix the mismatch before embedding the aug set."
        )
    _safe(f"embedding model (from embedding_meta.json): {model_name}")

    # (c) Load the 62 aug texts + labels, then run the loud dedup check
    #     BEFORE spending any API quota.
    payload = json.loads(AUG_JSON.read_text(encoding="utf-8"))
    records = payload["records"]
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    _safe(f"aug set: {len(texts)} texts  "
          f"({sum(1 for l in labels if l == 0)} legit / "
          f"{sum(1 for l in labels if l == 1)} attack)  "
          f"[authorship: {payload.get('authorship', 'unknown')}]")

    _safe("checking for duplicate texts against existing sets...")
    assert_no_overlap(texts)
    _safe("  OK: all aug texts unique vs dataset_train/test, fresh, stress.\n")

    # (d) Build the client with the same explicit millisecond timeout.
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS),  # milliseconds!
    )

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
    save_npy_atomic(EMB_AUG_NPY, emb)
    save_npy_atomic(LABELS_AUG_NPY, lab)

    # (h) Report shapes.
    _safe("")
    _safe("=" * 60)
    _safe("AUG EMBEDDING OUTPUT")
    _safe("=" * 60)
    _safe(f"embeddings_aug.npy : shape={emb.shape}  dtype={emb.dtype}")
    _safe(f"labels_aug.npy     : shape={lab.shape}  dtype={lab.dtype}")
    _safe(f"model              : {model_name}  dim={n_dim}")
    _safe(f"created_utc        : {datetime.now(timezone.utc).isoformat()}")
    _safe("=" * 60)


if __name__ == "__main__":
    main()
