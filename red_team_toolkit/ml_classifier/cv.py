"""
cv.py — Step 5 of the ML classifier sub-project (PART B: cross-validation).

WHAT this script does:
    Trains the MLP and the linear probe from scratch, MANY times, using
    repeated stratified k-fold cross-validation, and reports the MEAN and
    STANDARD DEVIATION of accuracy and F1. It also runs a character-length
    baseline through the exact same folds so we can compare like-for-like.

WHY cross-validation, when evaluate.py already gave us a test score?
    evaluate.py scored the models ONCE on a 14-example test set. With only 14
    examples, a single lucky (or unlucky) split can swing the number a lot —
    one flipped example moves accuracy by 7 percentage points. Cross-validation
    trains and evaluates on MANY different held-out slices and reports the
    spread, telling us how STABLE the result is.

*** IMPORTANT — THIS DOES NOT REPLACE PART A'S SINGLE-SHOT RESULT ***
    This script concatenates the train and test embeddings (57 + 14 = 71) to
    get more data for the stability estimate. That is acceptable HERE, and
    only here, for one reason: Part A already produced its single-shot test
    score BEFORE this runs, and we are NOT using these CV numbers to pick a
    hyperparameter, architecture, or threshold. No model change will be made
    because of anything printed below. CV here is a DIAGNOSTIC (how noisy is
    our estimate?), not a tuning knob. Using CV scores to choose settings and
    then reporting them as "the" score would be its own form of leakage.

THE PROTOCOL (deliberately simple and honest):
    - 5 folds, stratified (each fold keeps the attack/legit proportion).
    - Repeated 3 times with seeds 0, 1, 2 (different shuffles each repeat).
    - Every fold trains a FRESH model for a FIXED 100 epochs.
    - NO early stopping, NO checkpoint selection using the held-out fold.
      The held-out fold is used ONLY to score the FINAL-epoch model.
      (In train.py we picked the best epoch by validation loss; here the
      held-out fold is a TEST fold, so touching it to choose an epoch would
      leak. We just train the fixed budget and score the end result.)

HOW to run it (from the ml_classifier/ folder):
    python cv.py
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import PromptEmbeddingDataset
from model import build_model
from evaluate import compute_metrics   # reuse the by-hand metric formulas

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_NPY = SCRIPT_DIR / "embeddings_train.npy"
LABELS_TRAIN_NPY = SCRIPT_DIR / "labels_train.npy"
TEST_NPY = SCRIPT_DIR / "embeddings_test.npy"
LABELS_TEST_NPY = SCRIPT_DIR / "labels_test.npy"
TRAIN_JSON = SCRIPT_DIR / "dataset_train.json"
TEST_JSON = SCRIPT_DIR / "dataset_test.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed training protocol (same as train.py's defaults, minus the scheduler
# and checkpoint selection — see the module docstring for WHY they're gone).
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 8
HIDDEN = 64
DROPOUT = 0.3
K_FOLDS = 5
REPEAT_SEEDS = (0, 1, 2)


# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed python/numpy/torch so a fold's init + shuffles are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# STRATIFIED K-FOLD SPLIT (manual, no sklearn)
# ---------------------------------------------------------------------------
def stratified_kfold_indices(labels: np.ndarray, k: int, seed: int) -> list[list[int]]:
    """
    Split indices into k stratified folds. Returns a list of k lists; each
    inner list holds the HELD-OUT indices for that fold.

    HOW it works (per class): shuffle that class's indices with the seeded RNG,
    then deal them round-robin into the k folds (index i goes to fold i % k).
    Because each class is dealt separately, every fold receives ~1/k of the
    attacks and ~1/k of the legitimate prompts — the stratification that keeps
    a tiny, imbalanced dataset from producing a fold that is all one class.
    """
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in (0, 1):                       # 0 = legitimate, 1 = attack
        cls_idx = np.flatnonzero(labels == cls).tolist()
        rng.shuffle(cls_idx)
        for i, j in enumerate(cls_idx):
            folds[i % k].append(j)           # round-robin deal into k folds
    for f in folds:
        rng.shuffle(f)                       # don't leave folds class-ordered
    return folds


# ---------------------------------------------------------------------------
# ONE FOLD'S TRAINING + SCORING (the compact fit() helper)
# ---------------------------------------------------------------------------
def fit(model_type: str, train_x, train_y, test_x, test_y, seed: int) -> dict:
    """
    Train a FRESH model on the training fold, then score the FINAL-epoch model
    on the held-out fold. Returns the metrics dict from evaluate.compute_metrics.

    Deliberately NO scheduler, NO validation split, NO checkpointing: the only
    thing that happens is EPOCHS passes of plain AdamW training. The held-out
    fold is touched once, at the end, purely to score.
    """
    set_seed(seed)

    model = build_model(model_type, input_dim=train_x.shape[1],
                        hidden_dim=HIDDEN, dropout=DROPOUT).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # shuffle=True: a different batch order every epoch (standard training).
    loader = DataLoader(PromptEmbeddingDataset(train_x, train_y),
                        batch_size=BATCH_SIZE, shuffle=True)

    # ---- fixed-epoch training loop (same shape as train.py's inner loop) ----
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)   # model returns raw logits
            loss.backward()
            optimizer.step()

    # ---- score the FINAL model on the held-out fold (once, at the end) ----
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(test_x.astype(np.float32)).to(DEVICE))
    preds = (logits > 0).cpu().numpy().astype(int)   # logit > 0 == prob > 0.5
    return compute_metrics(test_y, preds)


# ---------------------------------------------------------------------------
# LENGTH BASELINE (threshold chosen on the TRAINING folds only)
# ---------------------------------------------------------------------------
def length_baseline_fold(train_texts, train_labels, test_texts) -> np.ndarray:
    """
    Fit the character-length rule on the training fold's texts, then apply it
    UNCHANGED to the held-out fold's texts. Same search as evaluate.py's
    fit_length_baseline, but here it is re-fit inside every fold so the
    held-out fold never influences the chosen threshold.
    """
    lengths = np.array([len(t) for t in train_texts])
    labels = np.asarray(train_labels)
    best_acc, best_thr, best_dir = -1.0, 0, "long"
    for thr in sorted(set(lengths.tolist())):
        for direction in ("long", "short"):
            preds = (lengths > thr).astype(int) if direction == "long" \
                else (lengths < thr).astype(int)
            acc = float((preds == labels).mean())
            if acc > best_acc:
                best_acc, best_thr, best_dir = acc, thr, direction

    test_lengths = np.array([len(t) for t in test_texts])
    if best_dir == "long":
        return (test_lengths > best_thr).astype(int)
    return (test_lengths < best_thr).astype(int)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 64)
    print("PART B — REPEATED STRATIFIED K-FOLD CROSS-VALIDATION (stability check)")
    print("=" * 64)
    print("This does NOT replace Part A's single-shot test score, and no model")
    print("setting will be chosen from these numbers.\n")
    print(f"device: {DEVICE.type}   folds: {K_FOLDS}   repeats: {len(REPEAT_SEEDS)} "
          f"(seeds {list(REPEAT_SEEDS)})   epochs/fold: {EPOCHS}\n")

    # ---- concatenate train + test embeddings (71 examples) for CV only ----
    emb = np.concatenate([np.load(TRAIN_NPY), np.load(TEST_NPY)])          # (71, 3072)
    labels = np.concatenate([np.load(LABELS_TRAIN_NPY), np.load(LABELS_TEST_NPY)])  # (71,)
    train_records = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    test_records = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    # Texts in the SAME order as the embeddings: embed_dataset.py embedded the
    # train JSON then the test JSON, in file order, so concatenating both JSONs
    # in that order keeps text i aligned with embedding row i.
    all_texts = [r["text"] for r in train_records] + [r["text"] for r in test_records]

    assert emb.shape[0] == labels.shape[0] == len(all_texts) == 71, "alignment broke"
    print(f"combined dataset: {emb.shape[0]} examples  "
          f"({int((labels==0).sum())} legit / {int((labels==1).sum())} attack)\n")

    # Storage for every fold's scores, per method.
    scores = {"mlp": {"acc": [], "f1": []},
              "linear": {"acc": [], "f1": []},
              "length": {"acc": [], "f1": []}}

    t0 = time.time()
    fold_counter = 0
    for repeat in REPEAT_SEEDS:
        folds = stratified_kfold_indices(labels, k=K_FOLDS, seed=repeat)
        print(f"repeat (seed {repeat}):")
        for fold_id, held_out in enumerate(folds):
            fold_counter += 1
            # Everything NOT in held_out becomes the training fold.
            held_out_set = set(held_out)
            train_idx = [i for i in range(len(labels)) if i not in held_out_set]

            tr_x, tr_y = emb[train_idx], labels[train_idx]
            te_x, te_y = emb[held_out], labels[held_out]
            tr_texts = [all_texts[i] for i in train_idx]
            te_texts = [all_texts[i] for i in held_out]

            # A distinct, reproducible seed per (repeat, fold) so each fold's
            # weight init and batch shuffling are independent but repeatable.
            seed = repeat * 100 + fold_id

            m_mlp = fit("mlp", tr_x, tr_y, te_x, te_y, seed)
            m_lin = fit("linear", tr_x, tr_y, te_x, te_y, seed)

            # Length baseline: threshold fit on THIS fold's training texts only.
            len_preds = length_baseline_fold(tr_texts, tr_y.tolist(), te_texts)
            m_len = compute_metrics(te_y, len_preds)

            for name, m in (("mlp", m_mlp), ("linear", m_lin), ("length", m_len)):
                scores[name]["acc"].append(m["accuracy"])
                scores[name]["f1"].append(m["f1"])

            print(f"  fold {fold_id}  (held out {len(te_y)}: "
                  f"{int((te_y==0).sum())} legit / {int((te_y==1).sum())} attack)  "
                  f"mlp acc {m_mlp['accuracy']:.3f} f1 {m_mlp['f1']:.3f} | "
                  f"linear acc {m_lin['accuracy']:.3f} f1 {m_lin['f1']:.3f} | "
                  f"length acc {m_len['accuracy']:.3f} f1 {m_len['f1']:.3f}")
        print()

    elapsed = time.time() - t0
    print(f"trained + scored {fold_counter} folds "
          f"({fold_counter} MLP + {fold_counter} linear models) in {elapsed:.1f}s\n")

    # =====================================================================
    # AGGREGATE: mean +/- standard deviation across all folds
    # =====================================================================
    # WHY report a standard deviation at all? With a dataset this small, the
    # per-fold scores bounce around a lot. A single mean like "0.97" hides
    # that. The std tells us the typical size of that bounce: a small std
    # means the result is stable across different splits; a large std means
    # the mean is fragile and a different split could tell a different story.
    print("=" * 64)
    print(f"{'method':<10} {'accuracy (mean +/- std)':<26} {'f1 (mean +/- std)':<22} folds")
    print("-" * 64)
    for name in ("mlp", "linear", "length"):
        acc = np.array(scores[name]["acc"])
        f1 = np.array(scores[name]["f1"])
        print(f"{name:<10} {acc.mean():.3f} +/- {acc.std():.3f}          "
              f"{f1.mean():.3f} +/- {f1.std():.3f}         {len(acc)}")
    print("=" * 64)
    print("\n(std = population standard deviation across all fold scores.)")
    print("Compare with Part A's single-shot test result (mlp/linear = 1.000 on")
    print("14 examples). If CV mean is lower or its std is large, the perfect")
    print("test score was partly luck of a small, easy split — report honestly.")


if __name__ == "__main__":
    main()
