"""
evaluate_v2.py — Compare v1 vs v2 classifiers on test, fresh, and stress sets.

WHAT this script does:
    Loads BOTH generations of frozen checkpoints — v1 (checkpoints/) and v2
    (checkpoints_v2/) — plus the regex input_guard(), and scores them side by
    side on three evaluation sets:
        1. the held-out TEST set (14)   — untouched by any training
        2. the FRESH set (24)           — user-authored, untouched by training
        3. the STRESS set (18)          — DEVELOPMENT set (see below)
    For each set it prints a side-by-side table (v1-mlp, v2-mlp, v1-linear,
    v2-linear, regex, and regex-OR-model combinations) with TP/FP/TN/FN,
    accuracy, precision, recall, and F1 — all computed by hand with a
    divide-by-zero guard (metrics imported from evaluate.py, no sklearn).

*** THE STRESS SET IS NOW A DEVELOPMENT SET — NOT AN INDEPENDENT ESTIMATE ***
    The stress set's earlier failures (off-topic / small-talk false positives
    in evaluate_stress.py) are exactly what MOTIVATED v2. Because its failures
    drove the design change, it can no longer serve as an unbiased estimate of
    generalization — it is a DEVELOPMENT set. Every table row and the final
    summary label it as such. The honest generalization numbers come from the
    held-out test set and the fresh set, neither of which influenced v2.

*** EVALUATION DISCIPLINE ***
    We do NOT tune any hyperparameter, threshold, or architecture based on any
    of these results. Both generations use the fixed threshold 0.5 and their
    saved configs. This script only SCORES frozen checkpoints — it contains no
    training, no optimizer, no torch.save, and no threshold logic beyond the
    constant THRESHOLD = 0.5. If v2 is worse somewhere, we report it honestly.

WHY IMPORT input_guard INSTEAD OF COPYING THE REGEX?
    Same as before: a copy drifts. We import the REAL input_guard from
    08_generate_answer.py via importlib (reusing evaluate_fresh.py's loader,
    because the filename starts with a digit) and call ONLY input_guard() —
    never answer_query() — so NO generation quota is spent.

HOW to run it (from the ml_classifier/ folder):
    python evaluate_v2.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Reuse the by-hand metrics and the frozen-checkpoint probability helper.
from evaluate import compute_metrics, load_checkpoint, predict_proba, build_model
# Reuse evaluate_fresh.py's importlib-based loader for the REAL input_guard.
from evaluate_fresh import load_input_guard

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
V1_CKPT_DIR = SCRIPT_DIR / "checkpoints"      # v1 checkpoints
V2_CKPT_DIR = SCRIPT_DIR / "checkpoints_v2"   # v2 checkpoints

TEST_NPY = SCRIPT_DIR / "embeddings_test.npy"
LABELS_TEST_NPY = SCRIPT_DIR / "labels_test.npy"
FRESH_NPY = SCRIPT_DIR / "embeddings_fresh.npy"
LABELS_FRESH_NPY = SCRIPT_DIR / "labels_fresh.npy"
STRESS_NPY = SCRIPT_DIR / "embeddings_stress.npy"
LABELS_STRESS_NPY = SCRIPT_DIR / "labels_stress.npy"

STRESS_JSON = SCRIPT_DIR / "stress_set.json"   # for per-bucket breakdown

THRESHOLD = 0.5   # fixed for BOTH generations — never tuned on any result

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe(msg: str) -> None:
    """UTF-8-safe print for Windows cp1252 consoles (the stress set has
    non-ASCII text)."""
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def load_ckpt(name: str, ckpt_dir: Path):
    """
    Build a FRESH model from a checkpoint's saved config and load its weights.

    We re-implement evaluate.py's load_checkpoint here so we can point it at
    EITHER checkpoints/ (v1) or checkpoints_v2/ (v2) without modifying
    evaluate.py. Same logic: build from config, load_state_dict, model.eval().
    """
    path = ckpt_dir / f"{name}_best.pt"
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model_type"], input_dim=cfg["input_dim"],
                        hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()   # Dropout off; we're scoring, not training.
    return model


def score_set(feats, labels, models, regex_flags, input_guard, texts=None):
    """
    Score every method on one set and return {method_name: metrics_dict}.

    models: dict of name -> loaded model (e.g. v1-mlp, v2-mlp, ...).
    The regex-OR-model combinations are derived here: flag if EITHER the regex
    OR the model flags (logical OR of the two boolean flag arrays).
    """
    results = {}
    model_flags = {}
    for name, model in models.items():
        probs = predict_proba(model, feats)
        flags = (probs > THRESHOLD).astype(int)
        model_flags[name] = flags
        results[name] = compute_metrics(labels, flags)

    # regex alone
    results["regex"] = compute_metrics(labels, regex_flags)

    # regex OR each model
    for name, flags in model_flags.items():
        combo = np.maximum(regex_flags, flags)
        results[f"regex+{name}"] = compute_metrics(labels, combo)

    return results, model_flags


def print_table(results, title):
    """Print the side-by-side comparison table for one set."""
    _safe("=" * 78)
    _safe(title)
    _safe(f"  {'method':<16} {'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3}   "
          f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}")
    _safe("-" * 78)
    order = ["v1-mlp", "v2-mlp", "v1-linear", "v2-linear", "regex",
             "regex+v1-mlp", "regex+v2-mlp", "regex+v1-linear", "regex+v2-linear"]
    for name in order:
        m = results[name]
        _safe(f"  {name:<16} {m['tp']:>3} {m['fp']:>3} {m['tn']:>3} {m['fn']:>3}   "
              f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} "
              f"{m['recall']:>6.3f} {m['f1']:>6.3f}")
    _safe("=" * 78)


def main() -> None:
    _safe("=" * 78)
    _safe("v1 vs v2 CLASSIFIER COMPARISON")
    _safe("=" * 78)
    _safe("No hyperparameter, threshold, or architecture is changed by this script.")
    _safe("Both generations use the fixed threshold 0.5 and their saved configs.")
    _safe("The STRESS set is a DEVELOPMENT set (its failures motivated v2), so it")
    _safe("is NOT an independent estimate of generalization.\n")
    _safe(f"device: {DEVICE.type}\n")

    # ---- load the REAL regex guard (import, don't copy) ----
    input_guard = load_input_guard()
    _safe("input_guard imported from 08_generate_answer.py (regex only, no API calls).\n")

    # ---- load all four frozen models (2 v1 + 2 v2) ----
    models = {
        "v1-mlp": load_ckpt("mlp", V1_CKPT_DIR),
        "v2-mlp": load_ckpt("mlp", V2_CKPT_DIR),
        "v1-linear": load_ckpt("linear", V1_CKPT_DIR),
        "v2-linear": load_ckpt("linear", V2_CKPT_DIR),
    }
    _safe("loaded 4 frozen checkpoints (v1-mlp, v2-mlp, v1-linear, v2-linear).\n")

    # =====================================================================
    # 1. HELD-OUT TEST SET (14) — honest generalization
    # =====================================================================
    test_feats = np.load(TEST_NPY)
    test_labels = np.load(LABELS_TEST_NPY)
    test_texts = [r["text"] for r in json.loads(
        (SCRIPT_DIR / "dataset_test.json").read_text(encoding="utf-8"))]
    test_regex = np.array([1 if input_guard(t)[0] else 0 for t in test_texts], dtype=int)

    test_results, _ = score_set(test_feats, test_labels, models, test_regex, input_guard)
    print_table(test_results, "HELD-OUT TEST SET (14) — untouched by any training")
    _safe("")

    # =====================================================================
    # 2. FRESH SET (24) — user-authored, untouched by training
    # =====================================================================
    fresh_feats = np.load(FRESH_NPY)
    fresh_labels = np.load(LABELS_FRESH_NPY)
    fresh_texts = [r["text"] for r in json.loads(
        (SCRIPT_DIR / "fresh_eval_set.json").read_text(encoding="utf-8"))]
    fresh_regex = np.array([1 if input_guard(t)[0] else 0 for t in fresh_texts], dtype=int)

    fresh_results, _ = score_set(fresh_feats, fresh_labels, models, fresh_regex, input_guard)
    print_table(fresh_results, "FRESH SET (24) — user-authored, untouched by training")
    _safe("")

    # =====================================================================
    # 3. STRESS SET (18) — DEVELOPMENT set (not independent)
    # =====================================================================
    stress_payload = json.loads(STRESS_JSON.read_text(encoding="utf-8"))
    stress_records = stress_payload["records"]
    stress_feats = np.load(STRESS_NPY)
    stress_labels = np.load(LABELS_STRESS_NPY)
    stress_texts = [r["text"] for r in stress_records]
    stress_buckets = [r["bucket"] for r in stress_records]
    stress_regex = np.array([1 if input_guard(t)[0] else 0 for t in stress_texts], dtype=int)

    stress_results, stress_flags = score_set(
        stress_feats, stress_labels, models, stress_regex, input_guard)
    print_table(stress_results,
                "STRESS SET (18) — DEVELOPMENT set (failures observed before v2; "
                "not an independent estimate)")

    # ---- per-bucket accuracy for the stress set ----
    _safe("")
    _safe("STRESS SET per-bucket accuracy (DEVELOPMENT set):")
    methods = ["v1-mlp", "v2-mlp", "v1-linear", "v2-linear", "regex"]
    _safe(f"  {'bucket':<20} {'n':>2} " + " ".join(f"{m:>10}" for m in methods))
    for b in sorted(set(stress_buckets)):
        idx = [i for i in range(len(stress_labels)) if stress_buckets[i] == b]
        row = f"  {b:<20} {len(idx):>2} "
        for m in methods:
            if m == "regex":
                flags = stress_regex
            else:
                flags = stress_flags[m]
            acc = float((flags[idx] == stress_labels[idx]).mean())
            row += f" {acc:>10.3f}"
        _safe(row)
    _safe("")

    # ---- every misclassified stress example in full, with probability ----
    _safe("STRESS SET misclassified examples in full (DEVELOPMENT set):")
    for m in methods:
        if m == "regex":
            flags = stress_regex
            probs = None
        else:
            flags = stress_flags[m]
            probs = predict_proba(models[m], stress_feats)
        wrong = [i for i in range(len(stress_labels)) if flags[i] != stress_labels[i]]
        if not wrong:
            _safe(f"  {m:<12}: none — classified all 18 correctly")
            continue
        _safe(f"  {m:<12}: {len(wrong)} misclassified")
        for i in wrong:
            kind = ("MISSED ATTACK" if stress_labels[i] == 1
                    else "FALSE ALARM (blocked legitimate)")
            prob_str = f"prob={probs[i]:.3f}" if probs is not None else "regex flag"
            _safe(f"      [{kind}] {stress_records[i]['id']} "
                  f"({stress_buckets[i]}) true={stress_labels[i]} pred={flags[i]} "
                  f"{prob_str}")
            _safe(f"        \"{stress_texts[i]}\"")
    _safe("")

    # =====================================================================
    # 4. FINAL SUMMARY — did v2 change false positives / missed attacks?
    # =====================================================================
    _safe("=" * 78)
    _safe("FINAL SUMMARY — v2 vs v1, per set (FP = false positives, FN = missed attacks)")
    for set_name, res in (("held-out test", test_results),
                          ("fresh set", fresh_results),
                          ("stress set (DEV)", stress_results)):
        _safe(f"  {set_name}:")
        for arch in ("mlp", "linear"):
            v1 = res[f"v1-{arch}"]
            v2 = res[f"v2-{arch}"]
            fp_delta = v2["fp"] - v1["fp"]
            fn_delta = v2["fn"] - v1["fn"]
            fp_txt = ("unchanged" if fp_delta == 0 else
                      f"{'+' if fp_delta > 0 else ''}{fp_delta} vs v1")
            fn_txt = ("unchanged" if fn_delta == 0 else
                      f"{'+' if fn_delta > 0 else ''}{fn_delta} vs v1")
            _safe(f"    {arch:<7} FP {v1['fp']} -> {v2['fp']} ({fp_txt})   "
                  f"FN {v1['fn']} -> {v2['fn']} ({fn_txt})")
    _safe("=" * 78)


if __name__ == "__main__":
    main()
