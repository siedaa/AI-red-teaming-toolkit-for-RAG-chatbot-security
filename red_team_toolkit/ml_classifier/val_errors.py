"""
val_errors.py — Which validation examples does the v2 classifier get wrong?

WHAT this script does:
    Recreates the EXACT train/validation split used to train v2 (the same
    concatenate order and the same seeded stratified splitter), loads the two
    FROZEN v2 checkpoints, and reports every misclassified VALIDATION example
    in full text, for both the MLP and the linear model. It also prints the
    5 least-confident MLP predictions (closest to the 0.5 threshold, i.e. the
    examples the model was most unsure about) and confirms how many
    TRAINING-split examples are misclassified (a sanity check on overfitting).

WHY RECOMPUTE THE SPLIT INSTEAD OF SAVING IT?
    train_v2.py seeds everything (seed 42) and uses a DETERMINISTIC splitter
    (stratified_split_indices), so re-running the same function on the same
    concatenated pool reproduces the identical split bit-for-bit. We import
    the function from train_v2.py rather than copying it, so if the splitter
    ever changed, this script would see the same change — no silent drift.

*** THIS SCRIPT ONLY READS SAVED FILES. IT DOES NOT TRAIN, AND IT DOES NOT
    WRITE, MODIFY, OR OVERWRITE ANY CHECKPOINT, DATASET, OR EMBEDDING. ***

HOW to run it (from the ml_classifier/ folder):
    python val_errors.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Reuse the SAME model builder the training script used, so the architecture
# matches the checkpoint exactly.
from model import build_model

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import the splitter from train_v2.py. Importing has NO side effects because
# train_v2.py wraps its training loop in `if __name__ == "__main__":`.
from train_v2 import stratified_split_indices, class_counts

V2_CKPT_DIR = SCRIPT_DIR / "checkpoints_v2"
THRESHOLD = 0.5   # same fixed threshold used everywhere else
LEAST_CONFIDENT_N = 5   # how many borderline MLP predictions to print
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe(msg: str) -> None:
    """UTF-8-safe print for Windows cp1252 consoles (aug texts include
    non-ASCII multilingual prompts)."""
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def load_texts(path: Path) -> list[str]:
    """Load texts in FILE order. fresh_eval_set.json is a bare LIST;
    aug_set.json / stress_set.json are OBJECTS with a "records" key."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"] if isinstance(data, dict) else data
    return [r["text"] for r in records]


def load_v2_model(name: str):
    """Rebuild a FRESH model from the checkpoint's saved config and load its
    weights. model.eval() turns off dropout; torch.no_grad() (used at the call
    site) disables gradient tracking — both are required for frozen inference.
    `weights_only=False` is needed because the checkpoint contains a config
    dict alongside the tensors (PyTorch 2.6+ defaults weights_only=True)."""
    ckpt = torch.load(V2_CKPT_DIR / f"{name}_best.pt", map_location=DEVICE,
                      weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model_type"], input_dim=cfg["input_dim"],
                        hidden_dim=cfg["hidden_dim"],
                        dropout=cfg["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict(model, feats: np.ndarray) -> np.ndarray:
    """Return attack probabilities for every row of `feats`. No gradients,
    model in eval mode — a pure read-only forward pass."""
    with torch.no_grad():
        x = torch.from_numpy(feats).float().to(DEVICE)
        return torch.sigmoid(model(x)).cpu().numpy()


def main() -> None:
    _safe("=" * 78)
    _safe("v2 VALIDATION-SPLIT ERRORS (read-only; nothing retrained or changed)")
    _safe("=" * 78)

    # ---- 1. rebuild the exact v2 train/validation split ----
    emb_train = np.load(SCRIPT_DIR / "embeddings_train.npy")
    emb_aug = np.load(SCRIPT_DIR / "embeddings_aug.npy")
    lab_train = np.load(SCRIPT_DIR / "labels_train.npy")
    lab_aug = np.load(SCRIPT_DIR / "labels_aug.npy")

    # SAME concatenate order as train_v2.py: train first, then aug.
    features = np.concatenate([emb_train, emb_aug], axis=0)   # (119, 3072)
    labels = np.concatenate([lab_train, lab_aug], axis=0)     # (119,)

    # Matching text list in the same order: dataset_train.json then aug_set.json.
    train_texts = load_texts(SCRIPT_DIR / "dataset_train.json")
    aug_texts = load_texts(SCRIPT_DIR / "aug_set.json")
    texts = train_texts + aug_texts

    assert len(texts) == features.shape[0], \
        f"text count ({len(texts)}) != embedding rows ({features.shape[0]})"

    # Same seeded splitter as train_v2.py (val_frac=0.2, seed=42).
    train_idx, val_idx = stratified_split_indices(labels, val_frac=0.2, seed=42)
    val_labs = labels[val_idx]

    # Hard assertion: the validation split must be 14 legit / 10 attack.
    n_legit = int((val_labs == 0).sum())
    n_attack = int((val_labs == 1).sum())
    assert (n_legit, n_attack) == (14, 10), \
        f"unexpected validation counts: {n_legit} legit / {n_attack} attack"

    _safe(f"training pool : {features.shape[0]} "
          f"({class_counts(labels)})")
    _safe(f"train split   : {len(train_idx)} ({class_counts(labels[train_idx])})")
    _safe(f"validation    : {len(val_idx)} "
          f"({n_legit} legit / {n_attack} attack)  [asserted OK]\n")

    # ---- 2. load the two frozen v2 models ----
    models = {"v2-mlp": load_v2_model("mlp"), "v2-linear": load_v2_model("linear")}
    _safe("loaded v2-mlp and v2-linear (frozen, eval mode, threshold 0.5)\n")

    # ---- score the validation split ----
    val_feats = features[val_idx]
    probs = {name: predict(m, val_feats) for name, m in models.items()}
    preds = {name: (p > THRESHOLD).astype(int) for name, p in probs.items()}

    # ---- 3. print every misclassified validation example in full ----
    for name in ("v2-mlp", "v2-linear"):
        _safe("-" * 78)
        _safe(f"{name} validation misclassifications")
        _safe("-" * 78)
        wrong = [j for j in range(len(val_idx)) if preds[name][j] != val_labs[j]]
        if not wrong:
            _safe("  none — classified all validation examples correctly")
        for j in wrong:
            gi = val_idx[j]   # global index into the 119-item pool
            kind = ("MISSED ATTACK" if val_labs[j] == 1
                    else "FALSE ALARM (blocked legitimate)")
            _safe(f"  [{kind}] pool_index={gi}")
            _safe(f"    true={int(val_labs[j])}  "
                  f"pred_prob={probs[name][j]:.4f}  pred={preds[name][j]}")
            _safe(f"    text: \"{texts[gi]}\"")
        _safe("")

    # ---- 5 least-confident MLP validation predictions ----
    _safe("-" * 78)
    _safe("5 LEAST-CONFIDENT v2-mlp validation predictions "
          "(closest to the 0.5 threshold)")
    _safe("-" * 78)
    # "least confident" = probability closest to 0.5, i.e. smallest |p - 0.5|.
    confidence_gap = np.abs(probs["v2-mlp"] - THRESHOLD)
    order = np.argsort(confidence_gap)[:LEAST_CONFIDENT_N]
    for rank, j in enumerate(order, start=1):
        gi = val_idx[j]
        _safe(f"{rank}. p={probs['v2-mlp'][j]:.4f}  true={int(val_labs[j])}  "
              f"pred={preds['v2-mlp'][j]}  pool_index={gi}")
        _safe(f"   text: \"{texts[gi]}\"")
    _safe("")

    # ---- confirm training-split misclassification count ----
    train_feats = features[train_idx]
    for name in ("v2-mlp", "v2-linear"):
        p = predict(models[name], train_feats)
        pred = (p > THRESHOLD).astype(int)
        n_wrong = int((pred != labels[train_idx]).sum())
        _safe(f"{name} training-split misclassifications: "
              f"{n_wrong}/{len(train_idx)}")

    _safe("\n(nothing was retrained or modified — this script only READS "
          "saved embeddings and checkpoints)")
    _safe("=" * 78)


if __name__ == "__main__":
    main()
