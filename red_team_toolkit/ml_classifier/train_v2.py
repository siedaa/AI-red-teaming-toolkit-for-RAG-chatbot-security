"""
train_v2.py — v2 of the prompt-classifier training loop (augmented training pool).

*** BACKGROUND — WHY v2 EXISTS ***
    The v1 classifier flagged all off-topic and small-talk prompts as attacks
    (see evaluate_stress.py / the stress-set run). The cause: v1's legitimate
    class contained ONLY on-topic paper questions, so instead of learning
    "attack vs legitimate" the model learned a spurious shortcut: "on-topic
    (paper) vs not-on-topic". Any legitimate prompt that wasn't about the
    paper — a biryani recipe, "hi", "thanks, that was helpful!" — landed on
    the "not on-topic" side and was flagged as an attack. This failure is a
    classic case of DISTRIBUTION SHIFT: the training data's input distribution
    did not match the distribution of real user traffic the model would face.

    v2 fixes this by BROADENING the legitimate class (off-topic, small talk,
    multilingual, short paper questions) and adding more attack styles (short,
    multilingual, format-disguised, long social-engineered) via aug_set.json.
    The 62 new examples are concatenated with the original 57 into one
    training pool of 119.

*** EVALUATION DISCIPLINE — READ THIS BEFORE CHANGING ANYTHING ***
    We do NOT tune any hyperparameter, threshold, or architecture setting
    based on results from the held-out test set, the fresh set, or the stress
    set. v2 uses EXACTLY train.py's defaults:
        AdamW lr 1e-3, weight_decay 1e-2, StepLR(step_size=30, gamma=0.5),
        100 epochs, batch size 8, hidden 64, dropout 0.3, seed 42.
    We run training ONCE per model. If the results are disappointing, we
    report them honestly rather than re-tuning.

    The stress set is now a DEVELOPMENT set: its failures (the off-topic /
    small-talk false positives) are what motivated this change, so it is no
    longer an independent estimate of generalization. All output labels it as
    such. v2's honest generalization numbers come from the held-out test set
    and the fresh set — neither of which is loaded or touched by this script.

*** THIS SCRIPT NEVER LOADS THE TEST, FRESH, OR STRESS EMBEDDINGS ***
    train_v2.py loads ONLY embeddings_train.npy + embeddings_aug.npy (and
    their labels). The test/fresh/stress files are intentionally not opened
    here — using them to make training decisions would leak evaluation data
    into training (data leakage). See the DATA RULE block below.

HOW to run it (from the ml_classifier/ folder):
    python train_v2.py --model mlp
    python train_v2.py --model linear

WHAT it writes into checkpoints_v2/:
    <model>_best.pt       best-by-validation-loss weights + config (incl. pos_weight)
    <model>_history.json  per-epoch metrics
    <model>_curves.png    loss + accuracy plots
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import PromptEmbeddingDataset
from model import build_model, count_parameters

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_NPY = SCRIPT_DIR / "embeddings_train.npy"    # (57, 3072) — original training set
LABELS_TRAIN_NPY = SCRIPT_DIR / "labels_train.npy"  # (57,)
AUG_NPY = SCRIPT_DIR / "embeddings_aug.npy"         # (62, 3072) — new augmentation set
LABELS_AUG_NPY = SCRIPT_DIR / "labels_aug.npy"      # (62,)
META_JSON = SCRIPT_DIR / "embedding_meta.json"      # records the embedding model
CKPT_DIR = SCRIPT_DIR / "checkpoints_v2"

# DATA RULE (restated here so it is impossible to miss):
#   The TEST, FRESH, and STRESS files below are intentionally NOT referenced
#   anywhere in this script. Loading them during training would leak
#   evaluation data into the training process. See the module docstring.
# embeddings_test.npy / labels_test.npy     ->  used ONLY by evaluate.py
# embeddings_fresh.npy / labels_fresh.npy   ->  used ONLY by evaluate_fresh.py
# embeddings_stress.npy / labels_stress.npy ->  used ONLY by evaluate_stress.py
#                                              and evaluate_v2.py (as a DEV set)


# ---------------------------------------------------------------------------
# REPRODUCIBILITY (same as train.py)
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed every random source so a run is REPRODUCIBLE."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# STRATIFIED TRAIN/VALIDATION SPLIT (manual, no sklearn — same as train.py)
# ---------------------------------------------------------------------------
def stratified_split_indices(labels: np.ndarray, val_frac: float, seed: int):
    """
    Split indices into (train_idx, val_idx) while keeping the SAME
    attack-vs-legitimate proportion in both halves — a STRATIFIED split.

    WHY stratified: with a small, imbalanced dataset a plain random split can
    accidentally dump most of one class into validation, leaving the train set
    starved of that class. Splitting each class separately guarantees both
    halves look like the whole pool.
    """
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for cls in (0, 1):                       # 0 = legitimate, 1 = attack
        cls_idx = np.flatnonzero(labels == cls).tolist()
        rng.shuffle(cls_idx)
        n_val = round(len(cls_idx) * val_frac)   # ~20% of this class
        val_idx.extend(cls_idx[:n_val])          # first slice -> validation
        train_idx.extend(cls_idx[n_val:])        # the rest    -> train
    rng.shuffle(train_idx)                    # interleave classes so the
    rng.shuffle(val_idx)                      # batches aren't class-ordered
    return train_idx, val_idx


def class_counts(labels: np.ndarray) -> str:
    """Return e.g. '64 legit / 31 attack' for a label array."""
    n_legit = int((labels == 0).sum())
    n_attack = int((labels == 1).sum())
    return f"{n_legit} legit / {n_attack} attack"


# ---------------------------------------------------------------------------
# ONE EPOCH OF TRAINING / EVALUATION (same as train.py)
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device):
    """
    Run one pass over `loader`. If optimizer is given we're TRAINING; if
    optimizer is None we're EVALUATING. Returns (average_loss, accuracy).

    NOTE: the SAME weighted `criterion` (BCEWithLogitsLoss with pos_weight) is
    used for both training and validation, so the validation loss is measured
    on the same scale as the training loss — apples to apples.
    """
    is_training = optimizer is not None
    model.train(is_training)     # toggles Dropout on/off (and grad tracking)

    total_loss, correct, n = 0.0, 0, 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        # ---- forward + loss ----
        logits = model(features)                 # (batch,) raw logits
        loss = criterion(logits, labels)         # weighted BCEWithLogitsLoss

        if is_training:
            # ---- backward + update (training only) ----
            optimizer.zero_grad()   # reset gradients from the previous step
            loss.backward()         # compute d(loss)/d(weight) for every weight
            optimizer.step()        # nudge each weight to reduce the loss

        # ---- bookkeeping (no gradients needed -> no_grad saves memory) ----
        total_loss += loss.item() * features.size(0)      # sum over the batch
        preds = (logits > 0).to(labels.dtype)             # logit>0 => class 1
        correct += (preds == labels).sum().item()
        n += features.size(0)

    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# PLOTTING (same as train.py)
# ---------------------------------------------------------------------------
def plot_curves(history, out_path: Path, title: str) -> bool:
    """Plot train vs validation loss and accuracy, saved to out_path."""
    try:
        import matplotlib
        matplotlib.use("Agg")        # render to a file, not a window (headless)
        import matplotlib.pyplot as plt
    except Exception as err:         # noqa: BLE001
        _safe(f"  [plot skipped] matplotlib unavailable: {err}")
        return False

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="train loss")
    ax1.plot(epochs, history["val_loss"], label="validation loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("weighted BCE-with-logits loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="train accuracy")
    ax2.plot(epochs, history["val_acc"], label="validation accuracy")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _safe(msg: str) -> None:
    """UTF-8-safe print for Windows cp1252 consoles (no non-ASCII expected here,
    but kept for consistency with the embedding scripts)."""
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train the v2 attack/legitimate prompt classifier.")
    p.add_argument("--model", choices=["mlp", "linear"], default="mlp",
                   help="which architecture to train (default: mlp)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p.add_argument("--hidden", type=int, default=64, help="hidden width (mlp only)")
    p.add_argument("--dropout", type=float, default=0.3, help="dropout rate (mlp only)")
    p.add_argument("--weight-decay", type=float, default=1e-2, help="L2 penalty for AdamW")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # ---- reproducibility + device selection ----
    set_seed(args.seed)
    # Even on a CPU-only machine we still SELECT a device and .to(device) the
    # model and batches. On CPU these calls are free no-ops, but writing the
    # code this way means it runs unchanged the moment CUDA becomes available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---- load the TRAINING pool ONLY (original + aug; never test/fresh/stress) ----
    emb_train = np.load(TRAIN_NPY)       # (57, 3072)
    lab_train = np.load(LABELS_TRAIN_NPY)  # (57,)
    emb_aug = np.load(AUG_NPY)           # (62, 3072)
    lab_aug = np.load(LABELS_AUG_NPY)    # (62,)

    # Concatenate into one pool. np.concatenate stacks them along axis 0 (rows).
    features = np.concatenate([emb_train, emb_aug], axis=0)   # (119, 3072)
    labels = np.concatenate([lab_train, lab_aug], axis=0)     # (119,)
    print(f"training pool: {features.shape[0]} examples "
          f"(original {emb_train.shape[0]} + aug {emb_aug.shape[0]})  "
          f"({class_counts(labels)})\n")

    # ---- build the train/validation split ----
    train_idx, val_idx = stratified_split_indices(labels, val_frac=0.2, seed=args.seed)
    train_feats, train_labs = features[train_idx], labels[train_idx]
    val_feats, val_labs = features[val_idx], labels[val_idx]

    print(f"train split      : {len(train_idx):>3} examples  ({class_counts(train_labs)})")
    print(f"validation split : {len(val_idx):>3} examples  ({class_counts(val_labs)})")
    print(f"test/fresh/stress: not loaded (held out entirely for evaluation)\n")

    train_ds = PromptEmbeddingDataset(train_feats, train_labs)
    val_ds = PromptEmbeddingDataset(val_feats, val_labs)

    # ---- DataLoaders ----
    # train: shuffle=True so each epoch sees examples in a different order.
    # validation: shuffle=False because order doesn't affect an averaged metric.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ---- CLASS IMBALANCE: pos_weight ----
    # Our pool has more legitimate (label 0) than attack (label 1) examples.
    # Without correction, BCEWithLogitsLoss would treat the majority class as
    # "more important" and bias the model toward predicting legitimate —
    # exactly the wrong bias for a security tool that must CATCH attacks.
    #
    # pos_weight multiplies the loss contributed by POSITIVE (attack) examples.
    #   pos_weight = (# label-0 training examples) / (# label-1 training examples)
    # If there are twice as many legitimate as attack examples, pos_weight = 2,
    # so each attack example counts DOUBLE, restoring balance. We compute it
    # from the TRAINING part ONLY (not validation) to avoid peeking.
    #
    # IMPORTANT: we must use the TRAINING split's counts, not the whole pool's,
    # because validation is held out and must not influence training decisions.
    n_neg_train = int((train_labs == 0).sum())   # legitimate examples in train
    n_pos_train = int((train_labs == 1).sum())   # attack examples in train
    pos_weight_value = n_neg_train / n_pos_train
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32)
    print(f"pos_weight (from TRAIN split only): {pos_weight_value:.4f} "
          f"= {n_neg_train} legit / {n_pos_train} attack\n")

    # ---- model / loss / optimizer / scheduler ----
    # input_dim is read from the actual data so a different embedding size
    # would just work, instead of silently mismatching the Linear layer.
    input_dim = train_feats.shape[1]
    model = build_model(args.model, input_dim=input_dim,
                        hidden_dim=args.hidden, dropout=args.dropout).to(device)
    print(f"model: {args.model}  |  trainable parameters: {count_parameters(model):,}\n")

    # BCEWithLogitsLoss with pos_weight: the SAME weighted loss is used for
    # training AND validation so the two are measured on the same scale.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # StepLR halves the learning rate every 30 epochs (same as train.py).
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # ------------------------------------------------------------------
    # TRAINING (fixed budget; no early stopping on the val set for model
    # selection beyond best-loss checkpointing, same as train.py)
    # ------------------------------------------------------------------
    CKPT_DIR.mkdir(exist_ok=True)
    best_path = CKPT_DIR / f"{args.model}_best.pt"
    history_path = CKPT_DIR / f"{args.model}_history.json"
    curves_path = CKPT_DIR / f"{args.model}_curves.png"

    # Embedding model name, recorded so a checkpoint is fully self-describing.
    embedding_model = "unknown"
    if META_JSON.is_file():
        embedding_model = json.loads(META_JSON.read_text(encoding="utf-8")).get("model", "unknown")

    config = {
        "model_type": args.model,
        "input_dim": input_dim,
        "hidden_dim": args.hidden,
        "dropout": args.dropout,
        "embedding_model": embedding_model,
        "pos_weight": pos_weight_value,   # recorded so a reload reproduces the loss
    }

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    best_epoch = -1
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        # torch.no_grad(): during evaluation we never compute gradients.
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()   # decay the LR on schedule

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Track + save the BEST model (by validation loss).
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "config": config,
                        "epoch": epoch, "val_loss": val_loss}, best_path)

        # Print epoch 1, every 10th, and the last.
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:>3} | lr {optimizer.param_groups[0]['lr']:.2e} | "
                  f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.3f}")

    elapsed = time.time() - t0
    print(f"\ntraining done in {elapsed:.1f}s | best val loss {best_val_loss:.4f} at epoch {best_epoch}")

    # ---- CHECKPOINT ROUND TRIP ----
    # Prove that a checkpoint on disk can rebuild the model from scratch:
    # build a FRESH model using the saved config (including pos_weight), load
    # the saved weights, re-score the validation set with the SAME weighted
    # loss, and confirm the loss matches the recorded best.
    print("\ncheckpoint round trip:")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    fresh = build_model(ckpt["config"]["model_type"],
                        input_dim=ckpt["config"]["input_dim"],
                        hidden_dim=ckpt["config"]["hidden_dim"],
                        dropout=ckpt["config"]["dropout"]).to(device)
    fresh.load_state_dict(ckpt["state_dict"])
    fresh.eval()
    # Rebuild the SAME weighted criterion from the checkpoint's recorded pos_weight.
    fresh_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(ckpt["config"]["pos_weight"], dtype=torch.float32).to(device)
    )
    with torch.no_grad():
        re_val_loss, re_val_acc = run_epoch(fresh, val_loader, fresh_criterion, None, device)
    match = abs(re_val_loss - best_val_loss) < 1e-6
    print(f"  reloaded val loss {re_val_loss:.6f} vs recorded best {best_val_loss:.6f} "
          f"-> {'MATCH [OK]' if match else 'MISMATCH [FAIL]'}")
    print(f"  reloaded val acc  {re_val_acc:.3f}")

    # ---- save history + plot curves ----
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nsaved history -> {history_path.name}")
    if plot_curves(history, curves_path, f"v2 {args.model} (hidden={args.hidden}, "
                                          f"dropout={args.dropout}, lr={args.lr}, "
                                          f"pos_weight={pos_weight_value:.3f})"):
        print(f"saved curves  -> {curves_path.name}")
    else:
        print("curves NOT saved (matplotlib unavailable).")

    print(f"best checkpoint -> {best_path.name}")


if __name__ == "__main__":
    main()
