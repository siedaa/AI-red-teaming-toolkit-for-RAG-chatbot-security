"""
leakage_check.py — Nearest-neighbour leakage check (NO API calls, pure numpy).

WHAT is the leakage idea here?
    If an evaluation prompt has a near-paraphrase sitting in the TRAINING pool,
    then the classifier scoring it correctly is weak evidence of generalization —
    it may simply be recognizing a text it has essentially already seen. This
    is called TRAINING-DATA LEAKAGE (or train/test contamination): the exam
    question was never really "new".

    We measure how close each evaluation prompt is to its nearest training
    neighbour using COSINE SIMILARITY (how aligned two embedding vectors are,
    from -1 to +1; 1.0 means "same direction"). If an evaluation item has a
    nearest neighbour at similarity >= 0.95, treat that item's score as
    "recognition", not "generalization".

WHY TWO POOLS?
    v1 was trained on the ORIGINAL training embeddings only (embeddings_train,
    57 items). v2 was trained on the AUGMENTED pool (train + aug, 119 items).
    Adding 62 NEW assistant-drafted texts could make some evaluation prompts
    CLOSER to the pool (if an aug text paraphrases an eval item). This script
    reports the nearest-neighbour similarity of every eval item under BOTH
    pools so we can see exactly what changed when aug was added.

WHY IS COSINE SIMILARITY A DOT PRODUCT HERE?
    cosine(a, b) = dot(a, b) / (|a| * |b|). If every vector has unit length
    (|v| = 1.0), the denominator is 1.0 * 1.0 = 1.0, so cosine == dot product
    and we can compute the whole similarity matrix with one matrix multiply
    (fast). We VERIFY the unit-norm property below; if any norm deviates, we
    normalize and say so loudly.

HOW to run it (from the ml_classifier/ folder):
    python leakage_check.py

WHAT it prints:
    - per eval set: mean/median/min/max nearest-neighbour similarity under the
      v1 pool and the v2 pool, plus counts of items >= 0.90 and >= 0.95
    - for stress and fresh sets: the 10 items with the highest v2-pool
      similarity (eval text, nearest training text, similarity, label match)
    - a leave-one-out reference scale INSIDE the v1 pool (what "ordinary"
      similarity looks like between two real training texts)
    - a plain-language summary of which sets got closer to the pool when aug
      was added, and how many near-duplicates (>= 0.95) each set has
"""

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Training-pool embeddings (v1 = train only; v2 = train + aug).
TRAIN_NPY = SCRIPT_DIR / "embeddings_train.npy"
AUG_NPY = SCRIPT_DIR / "embeddings_aug.npy"
# Evaluation-set embeddings.
TEST_NPY = SCRIPT_DIR / "embeddings_test.npy"
FRESH_NPY = SCRIPT_DIR / "embeddings_fresh.npy"
STRESS_NPY = SCRIPT_DIR / "embeddings_stress.npy"

# Matching text sources (read in the SAME order the embeddings were built).
TRAIN_JSON = SCRIPT_DIR / "dataset_train.json"   # bare LIST of records
AUG_JSON = SCRIPT_DIR / "aug_set.json"           # OBJECT with "records"
TEST_JSON = SCRIPT_DIR / "dataset_test.json"     # bare LIST
FRESH_JSON = SCRIPT_DIR / "fresh_eval_set.json"  # bare LIST
STRESS_JSON = SCRIPT_DIR / "stress_set.json"     # OBJECT with "records"

# Similarity thresholds for "near-duplicate" callouts.
NEAR_DUP = 0.95
CLOSE = 0.90

# How many top-similar items to print for the stress and fresh sets.
TOP_N = 10


def _safe(msg: str) -> None:
    """UTF-8-safe print for Windows cp1252 consoles (stress/aug sets have
    non-ASCII text)."""
    import sys
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def load_texts(path: Path) -> list[str]:
    """
    Load texts from one JSON file, handling BOTH shapes used in this project:
        - a bare LIST of records          (dataset_train/test, fresh_eval_set)
        - an OBJECT with a "records" key  (aug_set, stress_set)
    Returns the texts in FILE order (the same order the .npy rows were built).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        records = data["records"]
    else:
        records = data
    return [r["text"] for r in records]


def ensure_unit_norm(name: str, emb: np.ndarray) -> np.ndarray:
    """
    Check that every row of `emb` has unit L2 norm; if not, normalize it.

    WHY: with unit-norm vectors, cosine similarity == plain dot product, so we
    can compute all pairwise similarities as one matrix multiply. A deviation
    here would silently corrupt every number this script prints, so we check
    loudly.
    """
    norms = np.linalg.norm(emb, axis=1)
    _safe(f"  {name:<12} norms: min={norms.min():.6f}  max={norms.max():.6f}")
    if np.allclose(norms, 1.0, atol=1e-4):
        return emb
    _safe(f"    -> NOT unit length; normalizing in place (cosine would otherwise "
          f"be wrong).")
    return emb / norms[:, None]


def nn_similarities(eval_emb: np.ndarray, pool_emb: np.ndarray):
    """
    For every evaluation vector, find its MAXIMUM cosine similarity to the
    pool and the INDEX of that nearest neighbour.

    Both inputs are assumed unit-length (see ensure_unit_norm), so a single
    matrix multiply (eval @ pool.T) gives all pairwise cosines at once.
    np.argmax picks the best column per row; np.max picks the value.
    """
    sims = eval_emb @ pool_emb.T          # (n_eval, n_pool) all cosines
    nn_idx = np.argmax(sims, axis=1)      # nearest neighbour per eval item
    nn_sim = sims[np.arange(len(sims)), nn_idx]
    return nn_sim, nn_idx


def fmt_set(name: str, nn_sim: np.ndarray) -> None:
    """Print summary statistics of nearest-neighbour similarity for one set."""
    ge90 = int((nn_sim >= CLOSE).sum())
    ge95 = int((nn_sim >= NEAR_DUP).sum())
    _safe(f"  {name:<8} n={len(nn_sim):>2}  mean={nn_sim.mean():.4f}  "
          f"median={np.median(nn_sim):.4f}  min={nn_sim.min():.4f}  "
          f"max={nn_sim.max():.4f}  >=0.90: {ge90}  >=0.95: {ge95}")


def main() -> None:
    _safe("=" * 78)
    _safe("NEAREST-NEIGHBOUR LEAKAGE CHECK (no API calls)")
    _safe("=" * 78)
    _safe("Idea: if an eval prompt has a near-paraphrase in the training pool, a")
    _safe("correct prediction is weak evidence of generalization. Nearest-neighbour")
    _safe("cosine similarity measures how close the closest training example is.")
    _safe(f"v1 pool = embeddings_train (57).  v2 pool = train + aug (119).")
    _safe(f"Callout thresholds: CLOSE={CLOSE}, NEAR_DUP={NEAR_DUP}\n")

    # ---- 1. load embeddings ----
    emb_train = np.load(TRAIN_NPY)
    emb_aug = np.load(AUG_NPY)
    emb_test = np.load(TEST_NPY)
    emb_fresh = np.load(FRESH_NPY)
    emb_stress = np.load(STRESS_NPY)

    # ---- load texts + labels; assert counts match embedding rows ----
    train_texts = load_texts(TRAIN_JSON)
    aug_texts = load_texts(AUG_JSON)
    test_texts = load_texts(TEST_JSON)
    fresh_texts = load_texts(FRESH_JSON)
    stress_texts = load_texts(STRESS_JSON)

    lab_train = np.load(SCRIPT_DIR / "labels_train.npy")
    lab_aug = np.load(SCRIPT_DIR / "labels_aug.npy")
    lab_test = np.load(SCRIPT_DIR / "labels_test.npy")
    lab_fresh = np.load(SCRIPT_DIR / "labels_fresh.npy")
    lab_stress = np.load(SCRIPT_DIR / "labels_stress.npy")

    for name, emb, texts in [
        ("train", emb_train, train_texts), ("aug", emb_aug, aug_texts),
        ("test", emb_test, test_texts), ("fresh", emb_fresh, fresh_texts),
        ("stress", emb_stress, stress_texts),
    ]:
        assert emb.shape[0] == len(texts) == len(
            {"train": lab_train, "aug": lab_aug, "test": lab_test,
             "fresh": lab_fresh, "stress": lab_stress}[name]
        ), f"{name}: embedding rows / text count / label count mismatch"
    _safe("count check: every embedding row count matches its text and label "
          "count [OK]\n")

    # ---- 2. verify unit norms so cosine == dot product ----
    _safe("vector norms (should be 1.0 for cosine == dot product):")
    emb_train = ensure_unit_norm("train", emb_train)
    emb_aug = ensure_unit_norm("aug", emb_aug)
    emb_test = ensure_unit_norm("test", emb_test)
    emb_fresh = ensure_unit_norm("fresh", emb_fresh)
    emb_stress = ensure_unit_norm("stress", emb_stress)
    _safe("")

    # ---- build the two pools (v1 = train only; v2 = train + aug) ----
    pool_v1_emb = emb_train
    pool_v1_texts = train_texts
    pool_v1_labels = lab_train
    pool_v2_emb = np.concatenate([emb_train, emb_aug], axis=0)
    pool_v2_texts = train_texts + aug_texts
    pool_v2_labels = np.concatenate([lab_train, lab_aug], axis=0)
    _safe(f"v1 pool: {pool_v1_emb.shape[0]} items   "
          f"v2 pool: {pool_v2_emb.shape[0]} items\n")

    # ---- 3+4. per eval set: nearest-neighbour stats under BOTH pools ----
    eval_sets = {
        "test": (emb_test, test_texts, lab_test),
        "fresh": (emb_fresh, fresh_texts, lab_fresh),
        "stress": (emb_stress, stress_texts, lab_stress),
    }
    nn_v2 = {}   # stash for the top-N listings in step 5
    _safe("-" * 78)
    _safe("PER-SET nearest-neighbour similarity (mean / median / min / max, "
          "and counts >= thresholds)")
    _safe("-" * 78)
    for name, (emb, texts, labels) in eval_sets.items():
        _safe(f"\n[{name}]")
        sim_v1, idx_v1 = nn_similarities(emb, pool_v1_emb)
        sim_v2, idx_v2 = nn_similarities(emb, pool_v2_emb)
        fmt_set("v1 pool", sim_v1)
        fmt_set("v2 pool", sim_v2)
        # label agreement under the v2 pool
        agree = int((pool_v2_labels[idx_v2] == labels).sum())
        _safe(f"          label agreement with v2-pool nearest neighbour: "
              f"{agree}/{len(labels)}")
        nn_v2[name] = (sim_v2, idx_v2, texts, labels)

    # ---- 5. top-10 highest v2-pool similarity for stress + fresh ----
    for name in ("stress", "fresh"):
        sim, idx, texts, labels = nn_v2[name]
        _safe("\n" + "=" * 78)
        _safe(f"TOP {TOP_N} {name.upper()} items by v2-pool nearest-neighbour "
              f"similarity")
        _safe("=" * 78)
        order = np.argsort(-sim)[:TOP_N]
        for rank, i in enumerate(order, start=1):
            nn_i = idx[i]
            label_ok = "MATCH" if pool_v2_labels[nn_i] == labels[i] else "MISMATCH"
            _safe(f"{rank:>2}. sim={sim[i]:.4f}  label {int(labels[i])} vs "
                  f"nn {int(pool_v2_labels[nn_i])}  ({label_ok})")
            _safe(f"    eval : \"{texts[i]}\"")
            _safe(f"    train: \"{pool_v2_texts[nn_i]}\"")

    # ---- 6. reference scale: leave-one-out inside the v1 pool ----
    _safe("\n" + "=" * 78)
    _safe("REFERENCE SCALE — leave-one-out nearest neighbour INSIDE the v1 pool")
    _safe("(each training item vs all OTHER training items; what 'ordinary' "
          "similarity looks like)")
    _safe("=" * 78)
    sims_in = pool_v1_emb @ pool_v1_emb.T
    np.fill_diagonal(sims_in, -np.inf)          # exclude self-match
    loo_nn = np.max(sims_in, axis=1)
    _safe(f"v1 pool internal: mean={loo_nn.mean():.4f}  "
          f"median={np.median(loo_nn):.4f}  min={loo_nn.min():.4f}  "
          f"max={loo_nn.max():.4f}")

    # ---- 7. plain-language summary ----
    _safe("\n" + "=" * 78)
    _safe("SUMMARY")
    _safe("=" * 78)
    for name in ("test", "fresh", "stress"):
        sim, idx, texts, labels = nn_v2[name]
        # recompute v1 stats for the comparison sentence
        emb, _, _ = eval_sets[name]
        sim_v1, _ = nn_similarities(emb, pool_v1_emb)
        closer = int((sim > sim_v1 + 1e-9).sum())
        nd = int((sim >= NEAR_DUP).sum())
        _safe(f"  {name:<7}: {closer}/{len(sim)} items got CLOSER to the pool "
              f"when aug was added; {nd} item(s) have a near-duplicate "
              f"(>= {NEAR_DUP}) in the v2 pool.")
    _safe("\nReminder: a high nearest-neighbour similarity does not prove the")
    _safe("classifier merely memorized — but a correct prediction on an item")
    _safe(f"with similarity >= {NEAR_DUP} is weak evidence of generalization.")
    _safe("=" * 78)


if __name__ == "__main__":
    main()
