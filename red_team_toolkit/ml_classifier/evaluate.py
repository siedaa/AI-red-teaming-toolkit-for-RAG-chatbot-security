"""
evaluate.py — Step 5 of the ML classifier sub-project (PART A: single-shot test).

WHAT this script does:
    Loads the two BEST checkpoints saved by train.py (mlp_best.pt and
    linear_best.pt), scores them ONCE on the held-out TEST set, and compares
    them against two simple baselines (majority-class and a character-length
    rule) so we can tell whether the neural models are actually learning
    something beyond a trivial trick.

*** EVALUATION DISCIPLINE — READ THIS BEFORE CHANGING ANYTHING ***
    The test set (embeddings_test.npy / labels_test.npy) is evaluated EXACTLY
    ONCE, here. The score you see below is the final answer. No hyperparameter
    change, no retraining, no architecture swap may be made BECAUSE OF this
    score. If the results are disappointing, we report them honestly and move
    on — tuning against the test set would inflate it and destroy its meaning
    (that failure is called DATA LEAKAGE / "teaching to the exam").

    All day-to-day decisions use the VALIDATION split carved from the training
    data inside train.py. The test set is touched here and nowhere else.

HOW the models are used (no training happens in this file):
    We build FRESH model objects from each checkpoint's saved config, load the
    saved weights with load_state_dict, switch to model.eval() (which turns
    Dropout off), and run everything under torch.no_grad() (no gradients are
    needed for scoring, so this saves memory and speeds things up).

HOW to run it (from the ml_classifier/ folder):
    python evaluate.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from model import build_model

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_NPY = SCRIPT_DIR / "embeddings_train.npy"    # (57, 3072) — for the length baseline
LABELS_TRAIN_NPY = SCRIPT_DIR / "labels_train.npy"  # (57,)
TEST_NPY = SCRIPT_DIR / "embeddings_test.npy"       # (14, 3072) — evaluated ONCE here
LABELS_TEST_NPY = SCRIPT_DIR / "labels_test.npy"    # (14,)
TRAIN_JSON = SCRIPT_DIR / "dataset_train.json"      # texts, for the length baseline
TEST_JSON = SCRIPT_DIR / "dataset_test.json"        # texts, for the per-example table
CKPT_DIR = SCRIPT_DIR / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# METRICS BY HAND (no sklearn)
# ---------------------------------------------------------------------------
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Count the four confusion-matrix cells. Positive class = attack (1).

      TP (true positive)  : we predicted ATTACK and it really was an attack.
      FP (false positive) : we predicted ATTACK but it was legitimate.
                            -> a false alarm; a blocked legitimate user.
      TN (true negative)  : we predicted LEGITIMATE and it really was.
      FN (false negative) : we predicted LEGITIMATE but it was an attack.
                            -> a MISS; the attack slipped through.

    We compute them by simple boolean masks: compare predictions to truth.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())   # flagged, and truly attack
    fp = int(((y_pred == 1) & (y_true == 0)).sum())   # flagged, but legitimate
    tn = int(((y_pred == 0) & (y_true == 0)).sum())   # not flagged, legitimate
    fn = int(((y_pred == 0) & (y_true == 1)).sum())   # not flagged, but attack
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def safe_div(num: float, den: float) -> float:
    """
    Divide with a guard for division by zero.

    WHY the guard? If a model never predicts "attack", then (TP + FP) == 0 and
    precision would be 0/0. In plain math that's undefined; in code it raises
    ZeroDivisionError. We define it as 0.0 so one degenerate model can't crash
    the whole evaluation — and so a "perfect-looking" NaN never sneaks in.
    """
    return num / den if den else 0.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Build the full metrics dict from raw counts. Each formula in plain English:

      ACCURACY  = (TP + TN) / ALL
          Of every example, what fraction did we get right? Simple, but it can
          look high when classes are imbalanced, so we never report it alone.

      PRECISION = TP / (TP + FP)
          Of everything we FLAGGED as an attack, how many really were? High
          precision means few false alarms.

      RECALL    = TP / (TP + FN)
          Of the real attacks, how many did we CATCH? High recall means few
          misses. (Recall is the safety-critical number here: a missed attack
          is worse than an annoyed legitimate user.)

      F1        = 2 * precision * recall / (precision + recall)
          The harmonic mean of precision and recall — a single number that is
          high only when BOTH are high. (A plain average could be dragged up
          by one strong metric while the other is terrible.)
    """
    c = confusion_counts(y_true, y_pred)
    total = c["tp"] + c["fp"] + c["tn"] + c["fn"]
    accuracy = safe_div(c["tp"] + c["tn"], total)
    precision = safe_div(c["tp"], c["tp"] + c["fp"])
    recall = safe_div(c["tp"], c["tp"] + c["fn"])
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, **c}


def print_metrics(name: str, m: dict) -> None:
    """Pretty-print one model's confusion matrix + the four metrics."""
    print(f"  {name}")
    print(f"    confusion : TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print(f"    accuracy  : {m['accuracy']:.3f}")
    print(f"    precision : {m['precision']:.3f}")
    print(f"    recall    : {m['recall']:.3f}")
    print(f"    f1        : {m['f1']:.3f}")


# ---------------------------------------------------------------------------
# LOAD A CHECKPOINT (fresh model + saved weights)
# ---------------------------------------------------------------------------
def load_checkpoint(name: str):
    """
    Rebuild a fresh model from a checkpoint's saved config and load its weights.

    WHY rebuild from config instead of saving the whole model object?
    Because a whole-model pickle is brittle (it breaks if the class definition
    moves or changes). Saving just the weights (state_dict) + a small config
    dict is portable and explicit: we reconstruct the architecture ourselves,
    then pour the numbers back in.
    """
    path = CKPT_DIR / f"{name}_best.pt"
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model_type"], input_dim=cfg["input_dim"],
                        hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()   # Dropout off; we're scoring, not training.
    return model, cfg


def predict_proba(model, features: np.ndarray) -> np.ndarray:
    """
    Run the model on features and return the ATTACK PROBABILITY for each row.

    The model outputs a raw LOGIT (an unbounded number). To turn it into a
    probability between 0 and 1 we apply the sigmoid function:
        sigmoid(logit) = 1 / (1 + exp(-logit))
    which squashes any real number into (0, 1). torch.no_grad() wraps this so
    PyTorch doesn't waste memory building a gradient graph for a pure scoring
    pass.
    """
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(DEVICE)
        logits = model(x)                    # (N,) raw logits
        probs = torch.sigmoid(logits)        # (N,) probabilities in (0, 1)
    return probs.cpu().numpy()


def load_json_texts(path: Path) -> list[str]:
    """Return just the text field of every record in a dataset_*.json."""
    with open(path, encoding="utf-8") as f:
        return [r["text"] for r in json.load(f)]


# ---------------------------------------------------------------------------
# BASELINES
# ---------------------------------------------------------------------------
def majority_class_baseline(train_labels: np.ndarray, test_labels: np.ndarray) -> np.ndarray:
    """
    (a) MAJORITY-CLASS baseline: always predict whichever class is more common
        in the TRAINING set, for every test example.

    WHY compute it from TRAIN, not TEST? Because using the test set's own
        labels to pick the majority would be cheating (it peeked at the
        answers). We learn the rule on train and apply it unchanged.

    Our train set is 29 legitimate vs 28 attack, so this baseline predicts
    "legitimate" for everything — its test accuracy is just the test set's
        legitimate fraction. Any real model must beat this to be worth anything.
    """
    majority = 1 if (train_labels == 1).sum() > (train_labels == 0).sum() else 0
    return np.full_like(test_labels, majority, dtype=int)


def fit_length_baseline(texts: list[str], labels: list[int]) -> tuple[int, str, float]:
    """
    (b) CHARACTER-LENGTH baseline using ONLY the training texts.

    The idea (a deliberately simple "trick" baseline): maybe attacks are just
    LONGER (or shorter) than legitimate prompts, and a one-number rule can
    separate them without any neural network at all.

    We search every candidate length threshold, in both directions, and keep
    whichever gives the best accuracy ON THE TRAINING TEXTS:
        direction "long" : predict attack when len(text) >  threshold
        direction "short": predict attack when len(text) <  threshold

    Returns (best_train_accuracy, threshold, direction). This threshold is
    then applied UNCHANGED to the test set — never re-fit on test, or the
    comparison would be unfair (see the majority baseline's comment).
    """
    lengths = np.array([len(t) for t in texts])
    labels = np.asarray(labels)
    candidates = sorted(set(lengths.tolist()))

    best_acc, best_thr, best_dir = -1.0, candidates[0], "long"
    for thr in candidates:
        for direction in ("long", "short"):
            if direction == "long":
                preds = (lengths > thr).astype(int)
            else:
                preds = (lengths < thr).astype(int)
            acc = float((preds == labels).mean())
            if acc > best_acc:          # strictly better -> keep; ties keep the first
                best_acc, best_thr, best_dir = acc, thr, direction
    return best_acc, best_thr, best_dir


def apply_length_baseline(texts: list[str], thr: int, direction: str) -> np.ndarray:
    """Apply a fitted length rule to new texts (same comparison as above)."""
    lengths = np.array([len(t) for t in texts])
    if direction == "long":
        return (lengths > thr).astype(int)
    return (lengths < thr).astype(int)


def print_length_stats(train_texts: list[str], train_labels: list[int]) -> None:
    """
    Print the mean and median CHARACTER length of attacks vs legitimate in the
    TRAIN set. This shows whether the length rule has any real signal (e.g.
    attacks averaging much longer than legitimate prompts).
    """
    lengths = np.array([len(t) for t in train_texts])
    labels = np.asarray(train_labels)
    att = lengths[labels == 1]
    leg = lengths[labels == 0]
    print(f"  attack     : mean {att.mean():.1f}  median {np.median(att):.1f} chars  (n={len(att)})")
    print(f"  legitimate : mean {leg.mean():.1f}  median {np.median(leg):.1f} chars  (n={len(leg)})")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 64)
    print("PART A — SINGLE-SHOT TEST EVALUATION (test set touched exactly once)")
    print("=" * 64)
    print("No hyperparameter/model change will be made because of these scores.\n")

    device_note = "cuda" if DEVICE.type == "cuda" else "cpu"
    print(f"device: {device_note}\n")

    # ---- load the TEST embeddings/labels (the one and only time) ----
    test_feats = np.load(TEST_NPY)      # (14, 3072)
    test_labels = np.load(LABELS_TEST_NPY)  # (14,)
    train_labels = np.load(LABELS_TRAIN_NPY)  # (57,) — only labels needed for majority baseline
    test_texts = load_json_texts(TEST_JSON)
    train_texts = load_json_texts(TRAIN_JSON)

    # ---- length stats on the TRAIN set (context for the length baseline) ----
    print("CHARACTER-LENGTH STATS (train set only):")
    print_length_stats(train_texts, train_labels)
    print()

    # ---- fit the length baseline on TRAIN texts, then freeze it ----
    len_train_acc, len_thr, len_dir = fit_length_baseline(train_texts, train_labels.tolist())
    print(f"LENGTH BASELINE fit on train: best train accuracy {len_train_acc:.3f} "
          f"with rule 'attack if len(text) {">" if len_dir == "long" else "<"} {len_thr}'\n")

    # =====================================================================
    # 1. SCORE BOTH NEURAL MODELS ON THE TEST SET
    # =====================================================================
    results = {}   # name -> metrics dict

    for name in ("mlp", "linear"):
        model, cfg = load_checkpoint(name)
        probs = predict_proba(model, test_feats)
        preds = (probs > 0.5).astype(int)   # threshold 0.5 == logit > 0
        m = compute_metrics(test_labels, preds)
        results[name] = m

        print("-" * 64)
        print(f"MODEL: {name}  (hidden={cfg['hidden_dim']}, dropout={cfg['dropout']}, "
              f"embedding={cfg['embedding_model']})")
        print_metrics(name, m)

        # ---- per-example table ----
        print("    per-example (prob = P(attack) = sigmoid(logit)):")
        print("      #  true  prob   ok | text")
        for i, (lab, prob, pred) in enumerate(zip(test_labels, probs, preds)):
            text = test_texts[i]
            prefix = text[:44] + ("..." if len(text) > 44 else "")
            ok = "v" if pred == lab else "X"
            print(f"      {i:>2}  {lab:>4}  {prob:.3f}  {ok:>2} | {prefix}")
        print()

    # =====================================================================
    # 2. MISCLASSIFIED TEST EXAMPLES (full text, for the neural models)
    # =====================================================================
    print("-" * 64)
    print("MISCLASSIFIED TEST EXAMPLES (full text):")
    for name in ("mlp", "linear"):
        model, _ = load_checkpoint(name)
        probs = predict_proba(model, test_feats)
        preds = (probs > 0.5).astype(int)
        wrong = [i for i in range(len(test_labels)) if preds[i] != test_labels[i]]
        print(f"  {name}: {len(wrong)} misclassified")
        for i in wrong:
            kind = "MISSED ATTACK" if test_labels[i] == 1 else "FALSE ALARM (blocked legitimate)"
            print(f"    [{kind}] true={test_labels[i]} pred={preds[i]} prob={probs[i]:.3f}")
            print(f"      \"{test_texts[i]}\"")
    print()

    # =====================================================================
    # 3. BASELINES ON THE TEST SET
    # =====================================================================
    print("-" * 64)
    print("BASELINES:")

    # (a) majority class
    maj_preds = majority_class_baseline(train_labels, test_labels)
    results["majority"] = compute_metrics(test_labels, maj_preds)
    print_metrics("majority-class (always predict the more common train class)", results["majority"])

    # (b) character-length rule (threshold learned on train, applied unchanged)
    len_preds = apply_length_baseline(test_texts, len_thr, len_dir)
    results["length"] = compute_metrics(test_labels, len_preds)
    print_metrics("character-length baseline", results["length"])
    print()

    # =====================================================================
    # 4. SUMMARY TABLE
    # =====================================================================
    print("=" * 64)
    print(f"{'method':<16} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}   TP/FP/TN/FN")
    print("-" * 64)
    for name in ("majority", "length", "linear", "mlp"):
        m = results[name]
        print(f"{name:<16} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
              f"{m['recall']:>6.3f} {m['f1']:>6.3f}   "
              f"{m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}")
    print("=" * 64)
    print("\nRemember: this is ONE evaluation on 14 test examples. cv.py (Part B)")
    print("re-estimates stability with repeated cross-validation on all 71 examples.")


if __name__ == "__main__":
    main()
