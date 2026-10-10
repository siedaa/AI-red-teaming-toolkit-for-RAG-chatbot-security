"""
evaluate_fresh.py — Step 5 of the ML classifier sub-project (PART C: fresh scoring).

WHAT this script does:
    Scores the 24-example FRESH evaluation set (fresh_eval_set.json +
    embeddings_fresh.npy) with FIVE different methods, prints a per-example
    table, a method-comparison table, and drills into each method's misses
    and false positives.

THE FIVE METHODS:
    (a) MLP        — the trained neural classifier (mlp_best.pt), threshold 0.5
    (b) linear     — the trained linear probe (linear_best.pt), threshold 0.5
    (c) regex      — the EXISTING input_guard() regex filter from
                     08_generate_answer.py (imported, NOT copied)
    (d) combined   — flag if EITHER the regex OR the MLP flags
    (e) length     — the character-length baseline, threshold learned on the
                     TRAIN set only (never on this fresh set)

*** THE FRESH SET IS A FINAL, UNTOUCHED EXAM ***
    These 24 examples were written after the models were frozen. We score them
    ONCE to measure generalization. If a method misses a fresh attack or
    wrongly blocks a legitimate prompt, we REPORT that honestly — we do NOT
    retrain, re-threshold, or re-tune in response. Doing so would turn a real
    measurement into a self-fulfilling score (the same data-leakage mistake
    the whole sub-project has been careful to avoid).

WHY IMPORT input_guard INSTEAD OF COPYING THE REGEX?
    Because a copy drifts. If someone later updates the patterns in
    08_generate_answer.py, a copied regex here would silently go stale and the
    comparison would be against an out-of-date guard. Importing means we always
    test the REAL guard the pipeline actually uses. We import via importlib
    (not a normal import) because the filename starts with a digit, which is
    not a valid Python identifier. We call ONLY input_guard() — never
    answer_query() — so NO generation quota is spent.

HOW to run it (from the ml_classifier/ folder):
    python evaluate_fresh.py
"""

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from evaluate import (
    compute_metrics,
    fit_length_baseline,
    load_checkpoint,
    predict_proba,
    print_metrics,
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent   # <project>/
FRESH_JSON = SCRIPT_DIR / "fresh_eval_set.json"
EMB_FRESH_NPY = SCRIPT_DIR / "embeddings_fresh.npy"
LABELS_FRESH_NPY = SCRIPT_DIR / "labels_fresh.npy"
TRAIN_JSON = SCRIPT_DIR / "dataset_train.json"   # for the length baseline threshold
LABELS_TRAIN_NPY = SCRIPT_DIR / "labels_train.npy"


# ---------------------------------------------------------------------------
# IMPORT THE REAL input_guard() FROM 08_generate_answer.py (no copy)
# ---------------------------------------------------------------------------
def load_input_guard():
    """
    Import 08_generate_answer.py via importlib and return its input_guard.

    We must put the project root on sys.path AND chdir there first, because
    08_generate_answer.py imports 07_retrieve.py (also a digit-prefixed name)
    at module load time, and 07_retrieve resolves paths relative to the
    project root. Importing does NOT connect to Chroma or call any API —
    07_retrieve initialises its client lazily, only inside retrieve(), which
    we never call. We only take input_guard, the pure-regex function.
    """
    # Put project root on sys.path so "08_generate_answer" is findable.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    # chdir to project root so 07_retrieve's relative paths (e.g. chroma_db/)
    # resolve correctly IF anything ever touched them (nothing does here).
    os.chdir(PROJECT_ROOT)

    try:
        mod = importlib.import_module("08_generate_answer")
    except Exception as err:   # noqa: BLE001
        # If the import fails (missing dep, syntax error, ...), STOP loudly.
        # We do NOT fall back to a copied regex — that would silently test a
        # stale guard instead of the real one.
        raise RuntimeError(
            "Could not import input_guard from 08_generate_answer.py. "
            f"Exact error: {err}"
        ) from err

    return mod.input_guard


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 68)
    print("PART C — FRESH EVALUATION SET (24 examples, written after models froze)")
    print("=" * 68)
    print("No retraining / re-thresholding will be done because of these scores.\n")

    # ---- load the real regex guard (import, don't copy) ----
    input_guard = load_input_guard()
    print("input_guard imported from 08_generate_answer.py (regex only, no API calls).\n")

    # ---- load fresh data ----
    records = json.loads(FRESH_JSON.read_text(encoding="utf-8"))
    feats = np.load(EMB_FRESH_NPY)        # (24, 3072)
    labels = np.load(LABELS_FRESH_NPY)    # (24,)
    texts = [r["text"] for r in records]
    ids = [r["id"] for r in records]
    hard_neg = [bool(r.get("hard_negative", False)) for r in records]
    n = len(records)

    # ---- neural model probabilities (threshold 0.5) ----
    mlp_model, _ = load_checkpoint("mlp")
    lin_model, _ = load_checkpoint("linear")
    mlp_probs = predict_proba(mlp_model, feats)   # P(attack)
    lin_probs = predict_proba(lin_model, feats)
    mlp_flags = (mlp_probs > 0.5).astype(int)
    lin_flags = (lin_probs > 0.5).astype(int)

    # ---- regex flags (the real input_guard) ----
    regex_flags = np.zeros(n, dtype=int)
    regex_reasons = []
    for i, t in enumerate(texts):
        blocked, reason = input_guard(t)
        regex_flags[i] = 1 if blocked else 0
        regex_reasons.append(reason)

    # ---- combined: flag if regex OR MLP flags ----
    combined_flags = np.maximum(regex_flags, mlp_flags)

    # ---- length baseline (threshold learned on the TRAIN set only) ----
    train_records = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    train_texts = [r["text"] for r in train_records]
    train_labels = np.load(LABELS_TRAIN_NPY)
    len_train_acc, len_thr, len_dir = fit_length_baseline(
        train_texts, train_labels.tolist()
    )
    lengths = np.array([len(t) for t in texts])
    if len_dir == "long":
        length_flags = (lengths > len_thr).astype(int)
    else:
        length_flags = (lengths < len_thr).astype(int)
    print(f"LENGTH baseline threshold (fit on train only): "
          f"attack if len(text) {'>' if len_dir == 'long' else '<'} {len_thr} "
          f"(train acc {len_train_acc:.3f})\n")

    # =====================================================================
    # 1. PER-EXAMPLE TABLE
    # =====================================================================
    print("-" * 68)
    print("PER-EXAMPLE TABLE")
    print("  id     true  regex  mlp_p  lin_p  | text")
    for i in range(n):
        prefix = texts[i][:38] + ("..." if len(texts[i]) > 38 else "")
        hn = " [HARD-NEG]" if hard_neg[i] else ""
        print(f"  {ids[i]:<6} {labels[i]:>4}  {regex_flags[i]:>5}  "
              f"{mlp_probs[i]:.3f}  {lin_probs[i]:.3f}  | {prefix}{hn}")
    print()

    # =====================================================================
    # 2. METHOD COMPARISON TABLE
    # =====================================================================
    methods = {
        "mlp": mlp_flags,
        "linear": lin_flags,
        "regex": regex_flags,
        "combined": combined_flags,
        "length": length_flags,
    }
    results = {name: compute_metrics(labels, flags) for name, flags in methods.items()}

    print("=" * 68)
    print("METHOD COMPARISON (positive class = attack)")
    print(f"{'method':<10} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}   TP/FP/TN/FN")
    print("-" * 68)
    for name in ("mlp", "linear", "regex", "combined", "length"):
        m = results[name]
        print(f"{name:<10} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
              f"{m['recall']:>6.3f} {m['f1']:>6.3f}   "
              f"{m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}")
    print("=" * 68)
    print()

    # =====================================================================
    # 3. MISSED ATTACKS (false negatives) per method
    # =====================================================================
    print("-" * 68)
    print("FRESH ATTACKS EACH METHOD MISSED (false negatives):")
    for name in ("mlp", "linear", "regex", "combined", "length"):
        flags = methods[name]
        missed = [i for i in range(n) if labels[i] == 1 and flags[i] == 0]
        if not missed:
            print(f"  {name:<10}: none — caught all 12 fresh attacks")
            continue
        print(f"  {name:<10}: missed {len(missed)}/12")
        for i in missed:
            print(f"      {ids[i]} [{records[i]['category']}]: \"{texts[i]}\"")
    print()

    # =====================================================================
    # 4. FALSE POSITIVES (legitimate wrongly blocked) per method
    # =====================================================================
    print("-" * 68)
    print("LEGITIMATE PROMPTS WRONGLY BLOCKED (false positives):")
    for name in ("mlp", "linear", "regex", "combined", "length"):
        flags = methods[name]
        fps = [i for i in range(n) if labels[i] == 0 and flags[i] == 1]
        if not fps:
            print(f"  {name:<10}: none — blocked no legitimate prompt")
            continue
        print(f"  {name:<10}: {len(fps)} false positive(s)")
        for i in fps:
            tag = " (HARD NEGATIVE)" if hard_neg[i] else ""
            extra = f" [regex reason: {regex_reasons[i]}]" if (name in ("regex", "combined") and regex_reasons[i]) else ""
            print(f"      {ids[i]}{tag}: \"{texts[i]}\"{extra}")
    print()

    # hard-negative summary line
    hn_ids = [ids[i] for i in range(n) if hard_neg[i]]
    print(f"  (Hard negatives are the {len(hn_ids)} legitimate prompts that LOOK "
          f"attack-like: {', '.join(hn_ids)}.)")
    print(f"  A method that flags a hard negative is a FALSE POSITIVE — it is the "
          f"kind of\n  legitimate question most likely to be wrongly blocked in "
          f"production.")


if __name__ == "__main__":
    main()
