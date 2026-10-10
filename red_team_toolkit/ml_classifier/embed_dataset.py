"""
embed_dataset.py — Step 2 of the ML classifier sub-project (PART A: embeddings).

WHAT this script does:
    Converts every query text in dataset_train.json / dataset_test.json into a
    numerical vector ("embedding") using Google's Gemini embedding API, and
    saves the vectors + labels as .npy files for the PyTorch training step.

WHY embeddings?
    Machine learning models only understand numbers, not text. An embedding
    model maps each sentence to a point in a high-dimensional space where
    SEMANTICALLY SIMILAR sentences land close together. Our classifier will
    learn "attack territory" vs "legitimate territory" inside that space.

WHY MIRROR 07_retrieve.py's settings (model, task_type, dimension)?
    This is the single most important consistency rule in ML systems:
    training and serving MUST use the exact same embedding model and options.
    07_retrieve.py embeds chat queries with `gemini-embedding-001`, no
    task_type, no dimension override (so the default 3072-dim output).
    If we trained on, say, a 768-dim model but served with gemini-embedding-001,
    the live vectors would be meaningless compared to what the model learned —
    this failure mode is called TRAIN/SERVE SKEW, and it is silent: the code
    runs fine, predictions are just garbage.

SAFETY MEASURES (each learned from earlier incidents in this project):
    1. Explicit 60s HTTP timeout  — a hung socket once stalled a run forever.
    2. IPv4-first DNS patch       — some Windows networks break TLS to Google
                                    APIs via IPv6; the patch (same as the test
                                    harness uses) prefers IPv4 records.
    3. 1s delay between calls     — avoids hammering the API's per-minute quota.
    4. Bounded retry (max 3)      — on 429/RESOURCE_EXHAUSTED we retry a couple
                                    of times with backoff, then STOP with a
                                    clear message. We never loop forever, and
                                    we never silently skip texts — a dataset
                                    with missing rows would be worse than no
                                    dataset at all.

HOW to run it:
    python red_team_toolkit/ml_classifier/embed_dataset.py

WHAT it produces (in this folder):
    embeddings_train.npy   float32, shape (57, 3072)
    embeddings_test.npy    float32, shape (14, 3072)
    labels_train.npy       int64,   shape (57,)
    labels_test.npy        int64,   shape (14,)
    embedding_meta.json    model name, settings, counts, timestamp

SAFETY OF OUTPUTS (atomic writes):
    Files are first written under temporary names (*.tmp) and only then
    renamed to their final names. os.replace() is near-atomic, so an
    interrupted run can never leave a half-written file that LOOKS valid —
    the next person (or script) either sees the old file or the complete new
    one, never a corrupt in-between.
"""

import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. ENVIRONMENT SETUP — same patterns as test_harness.py
# ---------------------------------------------------------------------------

# Resolve paths: this script lives in <project>/red_team_toolkit/ml_classifier/.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # <project>/
ML_DIR = SCRIPT_DIR


def apply_ipv4_first_patch() -> None:
    """
    Prefer IPv4 when resolving hostnames (same patch as test_harness.py).

    WHY: on some Windows networks the AAAA (IPv6) records for Google APIs
    resolve to addresses that silently drop the TLS handshake, producing
    confusing 'connection reset' errors. Reordering getaddrinfo's results so
    IPv4 addresses come first avoids the problem. We install the patch BEFORE
    creating any HTTP client, exactly like the harness does.
    """
    _orig_gai = socket.getaddrinfo

    def _ipv4_first(*args, **kwargs):
        results = _orig_gai(*args, **kwargs)
        v4 = [r for r in results if r[0] == socket.AF_INET]
        v6 = [r for r in results if r[0] == socket.AF_INET6]
        return v4 + v6  # IPv4 first, IPv6 still available as fallback

    socket.getaddrinfo = _ipv4_first


def setup_environment() -> None:
    """
    Mirror test_harness.py's _setup_environment():

    1. Put the project root on sys.path (so any project module is importable).
    2. chdir to the project root (relative paths like venv/.env resolve there).
    3. Load GEMINI_API_KEY: real environment variable first, then venv/.env,
       then .env — same fallback order as the harness.
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)

    if not os.environ.get("GEMINI_API_KEY"):
        from dotenv import load_dotenv  # python-dotenv, already a dependency

        fallback_env = PROJECT_ROOT / "venv" / ".env"
        if fallback_env.is_file():
            load_dotenv(dotenv_path=fallback_env)
        else:
            load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY not found. Set it in the environment or add "
            "GEMINI_API_KEY=... to venv/.env (or .env)."
        )


# ---------------------------------------------------------------------------
# 1. CONFIGURATION — mirrored from 07_retrieve.py (READ THE COMMENTS!)
# ---------------------------------------------------------------------------

# EXACTLY the model 07_retrieve.py uses (line 115) — do not change one side only.
EMBEDDING_MODEL = "gemini-embedding-001"

# 07_retrieve.py calls embed_content(model=..., contents=...) and nothing else:
# no task_type, no output_dimensionality. That means the API's defaults apply
# (a 3072-dimensional vector). We mirror that call signature exactly so live
# chat queries and training queries are embedded identically. Record the
# choices in embedding_meta.json so nobody has to guess later.
TASK_TYPE = None           # not passed — mirrors 07_retrieve.py
OUTPUT_DIMENSIONALITY = None  # not passed — default 3072

HTTP_TIMEOUT_SECONDS = 60  # fail fast instead of hanging forever
# NOTE: google-genai's HttpOptions.timeout is measured in MILLISECONDS
# (see get_timeout_in_seconds() in the SDK source: it divides by 1000).
# Passing 60 here would mean 60 MILLISECONDS — every call would fail instantly
# with a bogus "handshake operation timed out". So we convert seconds -> ms.
HTTP_TIMEOUT_MS = HTTP_TIMEOUT_SECONDS * 1000
API_DELAY_SECONDS = 1.0    # polite gap between consecutive API calls
MAX_ATTEMPTS_PER_TEXT = 3  # bounded retry: never loop forever
BACKOFF_SECONDS = (5.0, 15.0)  # sleeps after the 1st and 2nd quota failure

# Output files (final names; written via .tmp + rename, see save helpers).
TRAIN_JSON = ML_DIR / "dataset_train.json"
TEST_JSON = ML_DIR / "dataset_test.json"
EMB_TRAIN_NPY = ML_DIR / "embeddings_train.npy"
EMB_TEST_NPY = ML_DIR / "embeddings_test.npy"
LABELS_TRAIN_NPY = ML_DIR / "labels_train.npy"
LABELS_TEST_NPY = ML_DIR / "labels_test.npy"
META_JSON = ML_DIR / "embedding_meta.json"


# ---------------------------------------------------------------------------
# 2. LOAD THE LABELED DATASET
# ---------------------------------------------------------------------------
def load_split(path: Path) -> tuple[list[str], list[int]]:
    """Return (texts, labels) from one dataset_*.json split, in file order."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]  # 0 = legitimate, 1 = attack
    return texts, labels


# ---------------------------------------------------------------------------
# 3. CALL THE EMBEDDING API (with bounded retry)
# ---------------------------------------------------------------------------
def is_quota_error(err: Exception) -> bool:
    """True if the exception looks like a rate/quota limit (429-style)."""
    text = str(err)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def embed_one(client, text: str, index: int, total: int) -> list[float]:
    """
    Embed a single text, retrying up to MAX_ATTEMPTS_PER_TEXT times on quota
    errors. Raises RuntimeError with a clear message if the quota is truly
    exhausted — better to stop loudly than to save an incomplete dataset.
    """
    for attempt in range(1, MAX_ATTEMPTS_PER_TEXT + 1):
        try:
            # Same call shape as 07_retrieve.py lines 168-173.
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
            )
            vector = response.embeddings[0].values
            if attempt > 1:
                print(f"  [ok] text {index + 1}/{total} succeeded on attempt {attempt}")
            return vector

        except Exception as err:  # noqa: BLE001 — we re-raise with context anyway
            quota = is_quota_error(err)
            if not quota:
                # Non-quota errors (bad key, network down, ...): stop immediately.
                raise RuntimeError(
                    f"Embedding failed for text {index + 1}/{total} "
                    f"(non-quota error): {err}"
                ) from err

            if attempt == MAX_ATTEMPTS_PER_TEXT:
                # Bounded retry exhausted: STOP, do not skip the text.
                raise RuntimeError(
                    f"QUOTA EXHAUSTED while embedding text {index + 1}/{total} "
                    f"after {MAX_ATTEMPTS_PER_TEXT} attempts.\n"
                    f"Last error: {err}\n"
                    "Free-tier embedding quota resets daily — re-run this "
                    "script tomorrow. No output files were finalized."
                ) from err

            # Quota error but attempts remain: wait, then try again.
            wait = BACKOFF_SECONDS[attempt - 1]
            print(f"  [429] text {index + 1}/{total}: quota hit, "
                  f"waiting {wait:.0f}s (attempt {attempt}/{MAX_ATTEMPTS_PER_TEXT})")
            time.sleep(wait)

    raise AssertionError("unreachable — loop always returns or raises")


def embed_all(client, texts: list[str]) -> "list[list[float]]":
    """Embed every text one at a time (mirrors 07_retrieve.py's per-query call)."""
    vectors = []
    total = len(texts)
    for i, text in enumerate(texts):
        vectors.append(embed_one(client, text, i, total))
        # Progress line every 10 texts + always the last one.
        if (i + 1) % 10 == 0 or i + 1 == total:
            print(f"  embedded {i + 1}/{total}")
        time.sleep(API_DELAY_SECONDS)  # stay under per-minute rate limits
    return vectors


# ---------------------------------------------------------------------------
# 4. SAVE OUTPUTS SAFELY (temp file + atomic-ish rename)
# ---------------------------------------------------------------------------
def save_npy_atomic(path: Path, array) -> None:
    """
    Write a numpy array to path via a temporary file.

    WHY: np.save() writing directly to the final name could leave a partial
    file if the process is killed mid-write — and a partial .npy file still
    has a valid-looking name. Writing to *.tmp first and renaming afterwards
    means the final file only ever appears complete.
    """
    import numpy as np

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)  # file object -> numpy uses the exact name given
    os.replace(tmp, path)  # near-atomic rename on the same filesystem


def save_json_atomic(path: Path, obj: dict) -> None:
    """Same temp-file + rename pattern, for the metadata JSON."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    # (a) Network + environment safety first — BEFORE any API client exists.
    apply_ipv4_first_patch()
    setup_environment()

    # (b) Build the client with an explicit timeout (new safety measure).
    #     http_options is how the google-genai SDK accepts HTTP settings.
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS),  # milliseconds!
    )

    # (c) Load both splits.
    print("Loading dataset files...")
    train_texts, train_labels = load_split(TRAIN_JSON)
    test_texts, test_labels = load_split(TEST_JSON)
    print(f"  train: {len(train_texts)} texts | test: {len(test_texts)} texts")

    # (d) Embed everything (train first, then test).
    print(f"Embedding with {EMBEDDING_MODEL} (delay {API_DELAY_SECONDS}s, "
          f"timeout {HTTP_TIMEOUT_SECONDS}s, max {MAX_ATTEMPTS_PER_TEXT} attempts/text)...")
    t0 = time.time()
    train_vectors = embed_all(client, train_texts)
    test_vectors = embed_all(client, test_texts)
    print(f"Done in {time.time() - t0:.1f}s")

    # (e) Convert to numpy arrays with explicit dtypes.
    #     float32: the standard float width for ML features — half the memory
    #     of float64 and exactly what PyTorch expects by default.
    #     int64 labels: integer class ids (we convert to float later, in the
    #     training step, where the loss function's expectations matter).
    import numpy as np

    emb_train = np.array(train_vectors, dtype=np.float32)
    emb_test = np.array(test_vectors, dtype=np.float32)
    lab_train = np.array(train_labels, dtype=np.int64)
    lab_test = np.array(test_labels, dtype=np.int64)

    # Sanity checks: shapes must line up, or training would silently misalign.
    n_dim = emb_train.shape[1]
    assert emb_train.shape == (len(train_texts), n_dim)
    assert emb_test.shape == (len(test_texts), n_dim)
    assert lab_train.shape == (len(train_texts),)
    assert lab_test.shape == (len(test_texts),)

    # (f) Write outputs atomically (tmp + rename — nothing half-written).
    save_npy_atomic(EMB_TRAIN_NPY, emb_train)
    save_npy_atomic(EMB_TEST_NPY, emb_test)
    save_npy_atomic(LABELS_TRAIN_NPY, lab_train)
    save_npy_atomic(LABELS_TEST_NPY, lab_test)

    # (g) Metadata: everything needed to reproduce/serve this embedding setup.
    meta = {
        "model": EMBEDDING_MODEL,
        "task_type": TASK_TYPE,                 # null — not passed (see header)
        "output_dimensionality": OUTPUT_DIMENSIONALITY,  # null — default used
        "dimension": int(n_dim),                # actual vector length
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "n_total": len(train_texts) + len(test_texts),
        "delay_seconds": API_DELAY_SECONDS,
        "timeout_seconds": HTTP_TIMEOUT_SECONDS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": ("Settings mirror 07_retrieve.py's embed_content() call so "
                 "training vectors and live chat-query vectors match exactly "
                 "(prevents train/serve skew)."),
    }
    save_json_atomic(META_JSON, meta)

    # (h) Final report: shapes of all four arrays.
    print()
    print("=" * 60)
    print("EMBEDDING OUTPUT")
    print("=" * 60)
    print(f"embeddings_train.npy : shape={emb_train.shape}  dtype={emb_train.dtype}")
    print(f"embeddings_test.npy  : shape={emb_test.shape}  dtype={emb_test.dtype}")
    print(f"labels_train.npy     : shape={lab_train.shape}  dtype={lab_train.dtype}")
    print(f"labels_test.npy      : shape={lab_test.shape}  dtype={lab_test.dtype}")
    print(f"embedding_meta.json  : model={meta['model']}  dim={meta['dimension']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
