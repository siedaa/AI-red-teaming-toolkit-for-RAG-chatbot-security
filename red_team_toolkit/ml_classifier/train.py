"""
train.py — Step 4 of the ML classifier sub-project (training loop + CLI).

WHAT this script does:
    Trains either model from model.py on the TRAINING embeddings, evaluates it
    on a held-out VALIDATION split, saves the best checkpoint, and plots
    learning curves. It also has a --sanity mode that proves the pipeline can
    overfit a tiny batch (a fast correctness check before any real training).

*** THE MOST IMPORTANT RULE IN THIS FILE — NO TEST DATA, EVER. ***
    This script loads ONLY embeddings_train.npy / labels_train.npy.
    embeddings_test.npy and labels_test.npy are NEVER opened here.

    WHY? Because the test set is our final, untouched exam. If we used it to
    pick hyperparameters, choose the best epoch, or tune the architecture, it
    would stop measuring how well we generalize and start measuring how well
    we memorized it — the metric would be inflated and meaningless. This
    failure is called DATA LEAKAGE, and reserving the test set until the very
    last step is the single most important discipline in machine learning.
    The validation split below (carved from the TRAINING data) is what we use
    for all day-to-day decisions; the test set is only touched once, in the
    next step's final evaluation.

HOW to run it (from the ml_classifier/ folder):
    python train.py --sanity                 # fast overfit check, no checkpoints
    python train.py --model mlp              # train the MLP
    python train.py --model linear           # train the baseline

WHAT it writes into checkpoints/:
    <model>_best.pt     best-by-validation-loss weights + config
    <model>_final.pt    last-epoch weights + config
    <model>_history.json  per-epoch metrics
    <model>_curves.png  loss + accuracy plots
"""

import argparse
import json
import os
import random
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
TRAIN_NPY = SCRIPT_DIR / "embeddings_train.npy"   # (57, 3072) float32
LABELS_NPY = SCRIPT_DIR / "labels_train.npy"      # (57,)      int64
META_JSON = SCRIPT_DIR / "embedding_meta.json"    # records the embedding model
CKPT_DIR = SCRIPT_DIR / "checkpoints"

# DATA RULE (restated here so it is impossible to miss):
#   The TEST files below are intentionally NOT referenced anywhere in this
#   script. Loading them during training would leak the final exam into the
#   study process. See the module docstring.
# embeddings_test.npy / labels_test.npy  ->  used ONLY by the next step's
#                                            final evaluation script.


# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """
    Seed every random source so a run is REPRODUCIBLE: same shuffles, same
    dropout masks, same weight init -> same numbers every time.

    We must seed all three because they are independent RNGs:
      - random     : Python's own RNG (used for the train/val index shuffle)
      - numpy      : NumPy's RNG
      - torch      : PyTorch's RNG (weight init + dropout masks)
    torch.use_deterministic_algorithms() is left off: some ops have no
    deterministic CPU/GPU implementation, and our tiny model doesn't need it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # seeds CPU (+ CUDA if present)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# STRATIFIED TRAIN/VALIDATION SPLIT (manual, no sklearn)
# ---------------------------------------------------------------------------
def stratified_split_indices(labels: np.ndarray, val_frac: float, seed: int):
    """
    Split indices into (train_idx, val_idx) while keeping the SAME
    attack-vs-legitimate proportion in both halves — a STRATIFIED split.

    WHY stratified, not just "shuffle and slice the first 20%"?
        With only 57 examples, a plain random split can accidentally dump most
        of one class into validation, leaving the train set starved of that
        class. We would then under-train on it AND get a misleading validation
        score. Splitting each class separately (each contributing ~20% to
        validation) guarantees both halves look like the whole dataset.

    HOW it works: for each class, shuffle its indices with the seeded RNG,
    then move round(count * val_frac) of them into validation.
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
    """Return e.g. '23 legit / 22 attack' for a label array."""
    n_legit = int((labels == 0).sum())
    n_attack = int((labels == 1).sum())
    return f"{n_legit} legit / {n_attack} attack"


# ---------------------------------------------------------------------------
# ONE EPOCH OF TRAINING / EVALUATION
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device):
    """
    Run one pass over `loader`. If optimizer is given we're TRAINING (gradients
    computed, weights updated); if optimizer is None we're EVALUATING.

    Returns (average_loss, accuracy). Accuracy is the fraction of examples
    where the model's prediction matches the true label.

    HOW a prediction is made from a LOGIT:  prediction = (logit > 0).
    Because the model outputs raw logits, and sigmoid(logit) > 0.5 exactly
    when logit > 0, comparing the logit against 0 gives the >50%-probability
    class WITHOUT computing a sigmoid. (See model.py's header.)
    """
    is_training = optimizer is not None
    model.train(is_training)     # toggles Dropout on/off (and grad tracking)

    total_loss, correct, n = 0.0, 0, 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        # ---- forward + loss ----
        logits = model(features)                 # (batch,) raw logits
        loss = criterion(logits, labels)         # BCEWithLogitsLoss

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
# PLOTTING
# ---------------------------------------------------------------------------
def plot_curves(history, out_path: Path, title: str) -> bool:
    """
    Plot train vs validation loss and accuracy, saved to out_path.

    Returns True on success, False if matplotlib is unavailable. We import
    matplotlib lazily (inside the function) so training works even if plotting
    can't be set up.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")        # render to a file, not a window (headless)
        import matplotlib.pyplot as plt
    except Exception as err:         # noqa: BLE001
        print(f"  [plot skipped] matplotlib unavailable: {err}")
        return False

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="train loss")
    ax1.plot(epochs, history["val_loss"], label="validation loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE-with-logits loss")
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


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train the attack/legitimate prompt classifier.")
    p.add_argument("--model", choices=["mlp", "linear"], default="mlp",
                   help="which architecture to train (default: mlp)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p.add_argument("--hidden", type=int, default=64, help="hidden width (mlp only)")
    p.add_argument("--dropout", type=float, default=0.3, help="dropout rate (mlp only)")
    p.add_argument("--weight-decay", type=float, default=1e-2, help="L2 penalty for AdamW")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sanity", action="store_true",
                   help="overfit one fixed batch to near-zero loss, then exit")
    return p.parse_args()


def load_train_data():
    """
    Load the TRAINING embeddings + labels ONLY.

    A loud reminder of what we are deliberately NOT loading:
        embeddings_test.npy, labels_test.npy   (reserved for final evaluation)
    """
    features = np.load(TRAIN_NPY)   # (57, 3072) float32
    labels = np.load(LABELS_NPY)    # (57,)      int64  -> cast to float in Dataset
    return features, labels


def main():
    args = parse_args()

    # ---- reproducibility + device selection ----
    set_seed(args.seed)
    # Even on a CPU-only machine we still SELECT a device and .to(device) the
    # model and batches. On CPU these calls are free no-ops, but writing the
    # code this way means it runs unchanged the moment CUDA becomes available
    # — no scattered "if cuda" checks throughout the loop.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---- load TRAIN data only (test set is off-limits here) ----
    features, labels = load_train_data()

    # ---- build the train/validation split ----
    train_idx, val_idx = stratified_split_indices(labels, val_frac=0.2, seed=args.seed)
    train_feats, train_labs = features[train_idx], labels[train_idx]
    val_feats, val_labs = features[val_idx], labels[val_idx]

    print(f"train split      : {len(train_idx):>3} examples  ({class_counts(train_labs)})")
    print(f"validation split : {len(val_idx):>3} examples  ({class_counts(val_labs)})")
    print(f"test split       :   0 examples - held out entirely for final evaluation\n")

    train_ds = PromptEmbeddingDataset(train_feats, train_labs)
    val_ds = PromptEmbeddingDataset(val_feats, val_labs)

    # ---- DataLoaders ----
    # train: shuffle=True so each epoch sees examples in a different order
    #        (prevents the model from memorising batch order).
    # validation: shuffle=False because order doesn't affect an averaged metric
    #        and a fixed order makes runs comparable.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ---- model / loss / optimizer / scheduler ----
    # input_dim is read from the actual data so a different embedding size
    # would just work, instead of silently mismatching the Linear layer.
    input_dim = train_feats.shape[1]
    model = build_model(args.model, input_dim=input_dim,
                        hidden_dim=args.hidden, dropout=args.dropout).to(device)
    print(f"model: {args.model}  |  trainable parameters: {count_parameters(model):,}\n")

    criterion = nn.BCEWithLogitsLoss()          # fused sigmoid + cross-entropy
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # StepLR halves the learning rate every 30 epochs. Decaying LR later in
    # training lets the model make big moves early, then fine-tune precisely.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # ------------------------------------------------------------------
    # SANITY MODE: overfit ONE batch to prove the pipeline is sound.
    # ------------------------------------------------------------------
    if args.sanity:
        print("=" * 60)
        print("SANITY MODE: overfit a single fixed batch of 8 examples")
        print("=" * 60)
        # Fresh model with dropout DISABLED and weight decay DISABLED so the
        # only thing happening is genuine gradient descent on a tiny problem.
        # A correct pipeline must drive this loss toward ~0; if it doesn't,
        # something is broken (bad labels, wrong loss, dead gradients, ...).
        san_model = build_model("mlp", input_dim=input_dim,
                                hidden_dim=args.hidden, dropout=0.0).to(device)
        san_opt = torch.optim.AdamW(san_model.parameters(), lr=args.lr, weight_decay=0.0)

        # One FIXED batch: take the first 8 training examples (already shuffled
        # by the seeded split above, so this is an arbitrary-but-reproducible 8).
        xb = train_feats[:args.batch_size]
        yb = train_labs[:args.batch_size]
        xb = torch.from_numpy(xb).float().to(device)
        yb = torch.from_numpy(yb).float().to(device)
        print(f"batch: {xb.shape[0]} examples | labels: {yb.tolist()}\n")

        san_model.train()
        report_steps = {1, 25, 50, 100, 150, 200}
        for step in range(1, 201):
            san_opt.zero_grad()
            logits = san_model(xb)                 # (8,) raw logits
            loss = criterion(logits, yb)
            loss.backward()
            san_opt.step()
            if step in report_steps:
                print(f"  step {step:>3}  loss = {loss.item():.6f}")

        print("\nA correct pipeline drives this toward ~0 (the model has enough")
        print("capacity to memorise 8 examples). If it plateaus high, there is")
        print("a bug somewhere in the data/loss/model wiring.")
        return   # sanity mode exits WITHOUT saving any checkpoints

    # ------------------------------------------------------------------
    # REAL TRAINING
    # ------------------------------------------------------------------
    CKPT_DIR.mkdir(exist_ok=True)
    best_path = CKPT_DIR / f"{args.model}_best.pt"
    final_path = CKPT_DIR / f"{args.model}_final.pt"
    history_path = CKPT_DIR / f"{args.model}_history.json"
    curves_path = CKPT_DIR / f"{args.model}_curves.png"

    # Embedding model name, recorded so a checkpoint is fully self-describing:
    # whoever loads it knows exactly which embedding space the weights expect.
    embedding_model = "unknown"
    if META_JSON.is_file():
        embedding_model = json.loads(META_JSON.read_text(encoding="utf-8")).get("model", "unknown")

    config = {
        "model_type": args.model,
        "input_dim": input_dim,
        "hidden_dim": args.hidden,
        "dropout": args.dropout,
        "embedding_model": embedding_model,
    }

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    best_epoch = -1
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        # torch.no_grad(): during evaluation we never compute gradients, which
        # saves memory and speeds things up (gradients are only for learning).
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

    # ---- save FINAL-epoch weights too (for comparison / research) ----
    torch.save({"state_dict": model.state_dict(), "config": config,
                "epoch": args.epochs,
                "val_loss": history["val_loss"][-1]}, final_path)

    # ---- CHECKPOINT ROUND TRIP ----
    # Prove that a checkpoint on disk can rebuild the model from scratch:
    # build a FRESH model using the saved config, load the saved weights,
    # re-score the validation set, and confirm the loss matches the recorded
    # best. This catches silent corruption / config mismatches.
    print("\ncheckpoint round trip:")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    fresh = build_model(ckpt["config"]["model_type"],
                        input_dim=ckpt["config"]["input_dim"],
                        hidden_dim=ckpt["config"]["hidden_dim"],
                        dropout=ckpt["config"]["dropout"]).to(device)
    fresh.load_state_dict(ckpt["state_dict"])
    fresh.eval()
    with torch.no_grad():
        re_val_loss, re_val_acc = run_epoch(fresh, val_loader, criterion, None, device)
    match = abs(re_val_loss - best_val_loss) < 1e-6
    print(f"  reloaded val loss {re_val_loss:.6f} vs recorded best {best_val_loss:.6f} "
          f"-> {'MATCH [OK]' if match else 'MISMATCH [FAIL]'}")
    print(f"  reloaded val acc  {re_val_acc:.3f}")

    # ---- save history + plot curves ----
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nsaved history -> {history_path.name}")
    if plot_curves(history, curves_path, f"{args.model} (hidden={args.hidden}, "
                                          f"dropout={args.dropout}, lr={args.lr})"):
        print(f"saved curves  -> {curves_path.name}")
    else:
        print("curves NOT saved (matplotlib unavailable).")

    print(f"best checkpoint -> {best_path.name}   final weights -> {final_path.name}")


if __name__ == "__main__":
    main()
