"""
margin_report.py — Descriptive margin report over the v2 checkpoints.

WHAT this script does (READ-ONLY — no API calls, no training, no file writes):
    Scores the FROZEN v2 checkpoints (linear + mlp, eval mode, no_grad) on
    three saved sets:
        1. the held-out TEST set (embeddings_test.npy)
        2. the FRESH set (embeddings_fresh.npy)
        3. the v2 VALIDATION split (reproduced with train_v2.py's
           stratified_split_indices, seed 42 — same code path as val_errors.py)
    For each (model, set) it prints:
        - the LOWEST attack probability   (worst-case attack the model scores)
        - the HIGHEST legitimate probability (worst-case false-alarm risk)
        - the GAP between them (how separated the two classes are)
    Then it POOLS all three sets and prints the same three numbers plus the 5
    highest-scoring LEGITIMATE prompts (text + score) for each model.

WHAT the gap means:
    gap = (highest legit score) - (lowest attack score).
    A POSITIVE gap means a legit prompt scores HIGHER than the lowest attack —
    the two classes OVERLAP in score space, and any threshold sits inside that
    overlap. A NEGATIVE gap means all legit scores sit below all attack scores
    (perfect separation) on that set.

*** DESCRIPTIVE ONLY — MUST NOT BE USED TO CHANGE A THRESHOLD. ***
    This report exists to BUILD INTUITION about score distributions, nothing
    more. The 0.5 threshold in ml_guard.py was fixed during evaluation and the
    project's evaluation discipline forbids tuning it (or any hyperparameter)
    against held-out/test/fresh results. Do NOT read a number here and "adjust"
    the threshold because of it. If a future data collection justifies a new
    threshold, that is a NEW evaluation cycle with its own fresh data — not an
    edit made because of this descriptive report.

HOW to run (from the ml_classifier/ folder):
    python margin_report.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Reuse the model builder + the exact split used by training.
from model import build_model

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import the splitter from train_v2.py. Importing has NO side effects because
# train_v2.py guards its training loop with `if __name__ == "__main__":`.
from train_v2 import stratified_split_indices

V2_CKPT_DIR = SCRIPT_DIR / "checkpoints_v2"
THRESHOLD = 0.5   # the FIXED ml_guard threshold — reported against, never tuned
TOP_LEGIT_N = 5   # how many highest-scoring legit prompts to print
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe(msg: str) -> None:
    """UTF-8-safe print for Windows cp1252 consoles."""
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def load_v2_model(name: str):
    """Rebuild a fresh model from the checkpoint's saved config, load the
    trained weights, and set eval() (dropout off). torch.no_grad() is applied
    at the predict() call site. weights_only=False because the checkpoint
    stores a config dict alongside the tensors (PyTorch 2.6+ default)."""
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
    """Attack probabilities for every row — pure read-only forward pass."""
    with torch.no_grad():
        x = torch.from_numpy(feats).float().to(DEVICE)
        return torch.sigmoid(model(x)).cpu().numpy()


def class_stats(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Return (lowest_attack, highest_legit, gap).

    lowest_attack  = min score among label==1 (attacks) — if no attacks, NaN.
    highest_legit  = max score among label==0 (legitimate) — if none, NaN.
    gap            = highest_legit - lowest_attack (positive = overlap).
    """
    attacks = scores[labels == 1]
    legits = scores[labels == 0]
    low_att = float(attacks.min()) if len(attacks) else float("nan")
    high_leg = float(legits.max()) if len(legits) else float("nan")
    return low_att, high_leg, high_leg - low_att


def print_set_stats(model_name: str, set_name: str, scores: np.ndarray,
                    labels: np.ndarray) -> None:
    low_att, high_leg, gap = class_stats(scores, labels)
    _safe(f"  {model_name:<7} {set_name:<12} lowest_attack={low_att:.4f}  "
          f"highest_legit={high_leg:.4f}  gap={gap:+.4f}")


def main() -> None:
    _safe("=" * 78)
    _safe("V2 MARGIN REPORT (descriptive only — MUST NOT be used to change a threshold)")
    _safe("=" * 78)
    _safe("gap = highest legit score - lowest attack score; positive = overlap.")
    _safe(f"Fixed threshold (reported against, never tuned): {THRESHOLD}\n")

    # ---- load the three evaluation sets (saved embeddings + labels) ----
    emb_test = np.load(SCRIPT_DIR / "embeddings_test.npy")
    lab_test = np.load(SCRIPT_DIR / "labels_test.npy")
    emb_fresh = np.load(SCRIPT_DIR / "embeddings_fresh.npy")
    lab_fresh = np.load(SCRIPT_DIR / "labels_fresh.npy")

    # Reproduce the v2 validation split EXACTLY as train_v2.py / val_errors.py:
    # concatenate train then aug (same order), split with seed 42, val_frac 0.2.
    emb_train = np.load(SCRIPT_DIR / "embeddings_train.npy")
    emb_aug = np.load(SCRIPT_DIR / "embeddings_aug.npy")
    lab_train = np.load(SCRIPT_DIR / "labels_train.npy")
    lab_aug = np.load(SCRIPT_DIR / "labels_aug.npy")
    features = np.concatenate([emb_train, emb_aug], axis=0)
    labels = np.concatenate([lab_train, lab_aug], axis=0)

    # Matching text list (dataset_train.json then aug_set.json, in file order).
    def _load_texts(path: Path) -> list[str]:
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data["records"] if isinstance(data, dict) else data
        return [r["text"] for r in records]

    texts = (_load_texts(SCRIPT_DIR / "dataset_train.json")
             + _load_texts(SCRIPT_DIR / "aug_set.json"))
    assert len(texts) == features.shape[0], "text/embedding count mismatch"

    train_idx, val_idx = stratified_split_indices(labels, val_frac=0.2, seed=42)
    emb_val, lab_val = features[val_idx], labels[val_idx]

    # Sanity: validation must be 14 legit / 10 attack (as asserted in val_errors.py).
    n_legit = int((lab_val == 0).sum())
    n_attack = int((lab_val == 1).sum())
    assert (n_legit, n_attack) == (14, 10), \
        f"unexpected validation counts: {n_legit}/{n_attack}"

    sets = {
        "test": (emb_test, lab_test),
        "fresh": (emb_fresh, lab_fresh),
        "val": (emb_val, lab_val),
    }
    _safe(f"sets: test n={len(lab_test)}, fresh n={len(lab_fresh)}, "
          f"val n={len(lab_val)} ({n_legit} legit / {n_attack} attack)\n")

    # ---- score everything with both frozen v2 models ----
    models = {"linear": load_v2_model("linear"), "mlp": load_v2_model("mlp")}
    all_scores: dict[str, dict[str, np.ndarray]] = {"linear": {}, "mlp": {}}
    for m_name, model in models.items():
        for s_name, (emb, lab) in sets.items():
            all_scores[m_name][s_name] = predict(model, emb)

    # ---- per (model, set) stats ----
    _safe("-" * 78)
    _safe("PER-SET STATS")
    _safe("-" * 78)
    for m_name in ("linear", "mlp"):
        for s_name in ("test", "fresh", "val"):
            emb, lab = sets[s_name]
            print_set_stats(m_name, s_name, all_scores[m_name][s_name], lab)
        _safe("")

    # ---- pooled stats over all three sets ----
    _safe("-" * 78)
    _safe("POOLED (test + fresh + val)")
    _safe("-" * 78)
    pooled_labels = np.concatenate([lab_test, lab_fresh, lab_val])
    pooled_val_texts = [texts[i] for i in val_idx]
    pooled_texts = (list(_load_texts(SCRIPT_DIR / "dataset_test.json"))
                    + list(_load_texts(SCRIPT_DIR / "fresh_eval_set.json"))
                    + pooled_val_texts)
    for m_name in ("linear", "mlp"):
        pooled_scores = np.concatenate([
            all_scores[m_name]["test"],
            all_scores[m_name]["fresh"],
            all_scores[m_name]["val"],
        ])
        print_set_stats(m_name, "pooled", pooled_scores, pooled_labels)

        # 5 highest-scoring LEGITIMATE prompts (text + score) for this model.
        legit_mask = pooled_labels == 0
        legit_scores = pooled_scores[legit_mask]
        legit_texts = [t for t, m in zip(pooled_texts, legit_mask) if m]
        top = np.argsort(-legit_scores)[:TOP_LEGIT_N]
        _safe(f"  {m_name}: {TOP_LEGIT_N} highest-scoring LEGITIMATE prompts:")
        for rank, j in enumerate(top, start=1):
            _safe(f"    {rank}. score={legit_scores[j]:.4f}  \"{legit_texts[j]}\"")
        _safe("")

    _safe("=" * 78)
    _safe("DESCRIPTIVE ONLY. Do NOT use any number above to change a threshold.")
    _safe("=" * 78)


if __name__ == "__main__":
    main()
