"""
build_dataset.py — Step 1 of the ML classifier sub-project (DATA ONLY, no PyTorch yet).

WHAT this script does:
    Builds a small labeled dataset that we will later use to train a PyTorch
    classifier that predicts whether a user query is an "attack" (1) or a
    "legitimate" question (0) — a second guardrail layer alongside input_guard().

WHY we start with data (not the model):
    Every supervised machine learning project follows the same shape:
        labeled examples  ->  train a model  ->  evaluate  ->  deploy.
    Garbage in = garbage out: a model can only learn the distinctions we give it
    examples of. So the very first step is assembling a clean, balanced,
    well-split dataset. That is exactly what this script does.

HOW to run it:
    python red_team_toolkit/ml_classifier/build_dataset.py

WHAT it produces:
    red_team_toolkit/ml_classifier/dataset_train.json   (80% of the examples)
    red_team_toolkit/ml_classifier/dataset_test.json    (20% of the examples)

    Every record in those files looks like:
        {
          "text":     "the actual query string",
          "label":     0 or 1,          # 0 = legitimate, 1 = attack
          "category": "..."             # attack type, or "legitimate"
        }

NOTES:
    - This file deliberately imports NO machine learning libraries. Data prep is
      ordinary Python (lists, JSON, a bit of shuffling) so the logic is easy to
      follow before we introduce tensors and models in the next step.
"""

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------
# Path(__file__) is "where this script file lives". Resolving it makes the
# paths absolute, so the script works no matter which folder you run it from.
SCRIPT_DIR = Path(__file__).resolve().parent

# The attack prompts live one folder up (red_team_toolkit/attack_prompts.json).
ATTACKS_PATH = SCRIPT_DIR.parent / "attack_prompts.json"

# The two files we will write, next to this script (ml_classifier/).
TRAIN_OUT = SCRIPT_DIR / "dataset_train.json"
TEST_OUT = SCRIPT_DIR / "dataset_test.json"

# ---------------------------------------------------------------------------
# 2. SPLIT SETTINGS
# ---------------------------------------------------------------------------
# TEST_FRACTION = 0.2 means "hold out 20% of the data for testing".
# Why hold anything out? We evaluate the model on data it has NEVER seen —
# otherwise we would just be memorizing, not learning.
TEST_FRACTION = 0.2

# A fixed random seed makes the split reproducible: running this script twice
# always produces the exact same train/test files (same shuffle order every
# time). Without a seed, every run would shuffle differently and you could
# never reproduce an experiment.
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# 3. THE LEGITIMATE (label = 0) EXAMPLES
# ---------------------------------------------------------------------------
# WHY do we need negative (legitimate) examples at all?
#   A classifier trained ONLY on attacks would learn "everything is an attack"
#   — 100% recall on attacks, 0% usefulness. To learn a decision boundary, the
#   model needs clear examples of BOTH sides: "this is what an attack looks
#   like" AND "this is what a normal question looks like".
#
# WHY this variety of phrasing?
#   Direct questions ("What is X?"), explanations ("Explain how Y works"),
#   why-questions ("Why did Z...?"), comparisons, and descriptions all sound
#   different on the surface. If every legitimate question had the exact same
#   shape (say, all starting with "What is..."), the model could cheat by
#   learning "starts with 'What is' = legitimate" instead of learning what an
#   attack actually is. Varying the style forces it to learn meaning, not
#   surface patterns.
#
# The first 5 entries are the questions from our earlier regression test of the
# chatbot (they must be answered correctly with guardrails active), and 3 more
# come from the Streamlit app's built-in example buttons.
LEGITIMATE_QUESTIONS = [
    # --- 5 regression-test questions (label = 0) ---
    "What is multi-head attention?",
    "What BLEU score did the Transformer achieve on English-to-French translation?",
    "Describe the encoder-decoder architecture.",
    "Why did the authors choose scaled dot-product attention over additive attention?",
    "What is positional encoding and why is it needed?",

    # --- 3 example queries from app.py's "Try an example" buttons ---
    "What BLEU score did the Transformer achieve on English-to-French translation, "
    "and how does it compare to the training cost of previous models?",
    "What is multi-head attention and why is it used instead of single-head attention?",
    "What regularization techniques does the Transformer use during training?",

    # --- architecture / structure ---
    "How many encoder and decoder layers does the base Transformer model use?",
    "Explain the role of the position-wise feed-forward network in each Transformer layer.",
    "Why are residual connections added around each sub-layer?",
    "What is layer normalization and where is it applied?",

    # --- encoder-decoder mechanics ---
    "How does encoder-decoder attention decide which source tokens to focus on?",
    "What masks are applied to attention during training, and why are they needed?",
    "In the decoder's self-attention, why are future positions masked out?",

    # --- attention mechanisms ---
    "Explain scaled dot-product attention, one step at a time.",
    "Why is the attention score divided by the square root of d_k before the softmax?",
    "How does splitting attention across multiple heads help the model capture "
    "different relationships?",
    "When is self-attention more efficient than recurrent processing?",

    # --- positional encoding / inputs ---
    "How are positional encodings combined with the input embeddings?",
    "Why did the authors use fixed sinusoidal functions instead of learned "
    "positional embeddings?",
    "What happens to the input embeddings before they enter the encoder?",

    # --- training details ---
    "Which optimizer and learning-rate schedule were used to train the Transformer?",
    "Explain the learning-rate warmup schedule — why increase the rate before "
    "decreasing it?",
    "How does label smoothing affect training and the final BLEU score?",
    "How long did it take to train the base model, and what hardware was used?",
    "What dropout rate does the big model use during training?",

    # --- results / tables ---
    "What BLEU score did the big model achieve on WMT 2014 English-to-German translation?",
    "Compare the Transformer's English-to-French results with earlier "
    "state-of-the-art systems.",
    "What does Table 3 show about the training cost of self-attention compared "
    "to recurrence?",

    # --- comparisons to other models ---
    "Why is the Transformer more parallelizable than a recurrent network?",
    "How does the computational complexity of convolutional layers compare to "
    "self-attention per layer?",
    "Which previous sequence-to-sequence models are used as baselines in the paper?",
    "In what situations might a recurrent network still be preferable to a Transformer?",

    # --- config / data ---
    "What are the main configuration differences between the base and big models?",
    "What dataset and tokenization approach were used for the English-to-German "
    "experiments?",
]


# ---------------------------------------------------------------------------
# 4. LOAD THE ATTACK EXAMPLES (label = 1)
# ---------------------------------------------------------------------------
def load_attack_examples() -> list[dict]:
    """
    Read attack_prompts.json and convert each attack into our record format.

    The attack file already contains 35 curated attacks across 5 categories
    (prompt_injection, jailbreak, role_confusion, system_prompt_leakage,
    data_extraction) — the same ones our red-team harness fires at the chatbot.
    Reusing them means the classifier learns from the exact attack surface we
    have already documented, instead of a different, invented set.
    """
    with open(ATTACKS_PATH, encoding="utf-8") as f:
        attack_prompts = json.load(f)

    # Sanity check: if the source file ever changes, fail loudly right here
    # instead of silently building a wrong dataset later.
    if len(attack_prompts) != 35:
        raise ValueError(
            f"Expected 35 attack prompts in {ATTACKS_PATH}, found {len(attack_prompts)}."
        )

    records = []
    for attack in attack_prompts:
        records.append(
            {
                "text": attack["prompt"],        # the raw attack string
                "label": 1,                      # 1 = attack
                "category": attack["category"],  # e.g. "jailbreak"
            }
        )
    return records


def build_legitimate_examples() -> list[dict]:
    """Wrap each legitimate question in the same record shape as the attacks."""
    records = []
    for question in LEGITIMATE_QUESTIONS:
        records.append(
            {
                "text": question,
                "label": 0,             # 0 = legitimate
                "category": "legitimate",  # so every record has a category field
            }
        )
    return records


# ---------------------------------------------------------------------------
# 5. STRATIFIED 80/20 SPLIT
# ---------------------------------------------------------------------------
def stratified_split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split records into (train, test) while keeping the same attack-vs-legitimate
    proportion in BOTH halves — this is called a *stratified* split.

    WHY NOT just shuffle everything and take the first 20%?
        On a big dataset a plain random split is fine. But with only ~70
        records, pure luck can put, say, 11 of the 15 attacks into the test set
        and leave the train set with almost no attacks to learn from. Then:
          - training sees few attacks  -> the model barely learns them, and
          - the test set is skewed     -> the reported accuracy is misleading.
        Stratifying (splitting each class separately, in the same 80/20 ratio)
        guarantees both halves look like the full dataset — roughly 50% attacks
        and 50% legitimate in each.

    HOW it works, step by step:
      1. Take all records of ONE class (e.g. all legitimate).
      2. Shuffle just that class.
      3. Carve off TEST_FRACTION (20%) of it for the test set.
      4. Repeat for the OTHER class (attacks).
      5. Shuffle the resulting train and test lists so they aren't ordered by
         label (an ordered file could accidentally teach someone — or a
         future training loop — a fake pattern based on position).
    """
    rng = random.Random(RANDOM_SEED)
    train: list[dict] = []
    test: list[dict] = []

    # Loop over both classes: label 0 (legitimate) and label 1 (attack).
    for label in (0, 1):
        group = [r for r in records if r["label"] == label]
        rng.shuffle(group)  # shuffle WITHIN the class, reproducibly

        # round(35 * 0.2) = 7 test examples per class.
        n_test = round(len(group) * TEST_FRACTION)
        test.extend(group[:n_test])      # first slice -> test
        train.extend(group[n_test:])     # the rest    -> train

    # Final shuffle so labels are interleaved in the saved files.
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


# ---------------------------------------------------------------------------
# 6. SAVE + REPORT
# ---------------------------------------------------------------------------
def class_breakdown(records: list[dict]) -> str:
    """Return e.g. '28 attack (50.0%) / 28 legitimate (50.0%)' for the summary."""
    n_attack = sum(1 for r in records if r["label"] == 1)
    n_legit = len(records) - n_attack
    total = len(records)
    return (
        f"{n_attack} attack ({n_attack / total:.1%}) / "
        f"{n_legit} legitimate ({n_legit / total:.1%})"
    )


def save_json(path: Path, records: list[dict]) -> None:
    """Write records as pretty-printed UTF-8 JSON (indent=2 keeps it readable)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> None:
    # --- step A: build both classes ---
    attack_records = load_attack_examples()       # 35 x label=1
    legit_records = build_legitimate_examples()   # 35 x label=0
    all_records = attack_records + legit_records  # 70 total

    # --- step B: split (stratified, reproducible) ---
    train_records, test_records = stratified_split(all_records)

    # --- step C: write the two dataset files ---
    save_json(TRAIN_OUT, train_records)
    save_json(TEST_OUT, test_records)

    # --- step D: human-readable summary ---
    print("=" * 60)
    print("ML CLASSIFIER DATASET — BUILD SUMMARY")
    print("=" * 60)
    print(f"Total examples        : {len(all_records)} "
          f"({len(attack_records)} attack + {len(legit_records)} legitimate)")
    print(f"Train set (80%)       : {len(train_records):>3}   "
          f"{class_breakdown(train_records)}")
    print(f"Test set  (20%)       : {len(test_records):>3}   "
          f"{class_breakdown(test_records)}")
    print("-" * 60)
    print(f"Saved: {TRAIN_OUT.name} ({len(train_records)} records)")
    print(f"Saved: {TEST_OUT.name} ({len(test_records)} records)")
    print("Labels: 0 = legitimate, 1 = attack")
    print("=" * 60)


if __name__ == "__main__":
    main()
