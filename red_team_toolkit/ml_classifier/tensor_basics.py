"""
tensor_basics.py — Step 3 of the ML classifier sub-project (PART C: tensor primer).

WHAT this script does:
    A hands-on introduction to the torch tensors our classifier will be built
    on. It loads the .npy files produced by embed_dataset.py, converts them to
    torch tensors, demonstrates a handful of essential tensor operations, and
    answers one important question about our data:

        "Do attack embeddings actually CLUSTER AWAY FROM legitimate embeddings?"

WHY check clustering before training anything?
    Our whole idea is that gemini-embedding-001 places similar meanings near
    each other. If attacks genuinely live in one region of the 3072-dimensional
    space and legitimate questions in another, a simple classifier has a real
    signal to learn. If the two groups are hopelessly interleaved, no model
    will do well — better to know NOW than after an hour of training.

HOW to run it:
    python red_team_toolkit/ml_classifier/tensor_basics.py

REQUIRES: embeddings/labels .npy files from embed_dataset.py + torch (CPU).
"""

# numpy reads .npy files; torch does the math from here on.
import numpy as np
import torch
# torch.nn.functional holds stateless operations on tensors (normalize, relu...).
import torch.nn.functional as F
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. LOAD .npy FILES AND CONVERT TO TORCH TENSORS
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. LOADING FILES AS TENSORS")
print("=" * 60)

# np.load() reads the .npy format back into a numpy array.
# torch.from_numpy() then wraps that array as a torch tensor.
#   - It is ZERO-COPY: the tensor shares the same memory as the numpy array.
#     Great for loading speed; just remember editing one edits the other.
emb_train_np = np.load(ML_DIR / "embeddings_train.npy")
emb_test_np = np.load(ML_DIR / "embeddings_test.npy")
lab_train_np = np.load(ML_DIR / "labels_train.npy")
lab_test_np = np.load(ML_DIR / "labels_test.npy")

emb_train = torch.from_numpy(emb_train_np)  # (57, 3072) float32
emb_test = torch.from_numpy(emb_test_np)    # (14, 3072) float32
lab_train = torch.from_numpy(lab_train_np)  # (57,)      int64
lab_test = torch.from_numpy(lab_test_np)    # (14,)      int64

for name, t in [("emb_train", emb_train), ("emb_test", emb_test),
                ("lab_train", lab_train), ("lab_test", lab_test)]:
    print(f"  {name:9s}  shape={tuple(t.shape)}  dtype={t.dtype}")

# WHY float32 for features (the embeddings)?
#   - It is the standard float width in deep learning: 4 bytes per number.
#     float64 would double memory/compute for no accuracy gain, and many
#     accelerated kernels are optimized for (and some only support) float32.
#
# WHY will labels need to become FLOAT later?
#   - The loss function we plan to use, BCEWithLogitsLoss (binary cross
#     entropy with logits), compares a predicted probability (float) against
#     a target probability (float 0.0 or 1.0). It expects float labels, not
#     integer class ids. So at training time we will do: lab_train.float().
#     (A different loss — CrossEntropyLoss — would instead want int64 labels;
#     each loss has its own dtype contract. This is a very common beginner
#     error: "expected scalar type Long but found Float" means these dtypes
#     were mixed up.)
print("\n  NOTE: labels are int64 here; the training step will call .float()")
print("        on them because BCEWithLogitsLoss expects float targets.")


# ---------------------------------------------------------------------------
# 2. BASIC TENSOR OPERATIONS (the everyday toolkit)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("2. BASIC TENSOR OPERATIONS")
print("=" * 60)

# --- indexing: pull out ONE training example (one query's 3072-dim vector) ---
first_attack_vector = emb_train[0]           # row 0 of the 2-D tensor
print(f"  emb_train[0]        -> shape={tuple(first_attack_vector.shape)} "
      f"(one example, all 3072 dims)")
print(f"  emb_train[0].shape  -> {first_attack_vector.shape}  "
      f"(torch.Size is just a tuple-like)")

# --- slicing: look at only the FIRST 8 dimensions of that example ---
first_8_dims = emb_train[0, :8]              # [row, columns 0..7]
print(f"  emb_train[0, :8]    -> shape={tuple(first_8_dims.shape)}  "
      f"values[:3]={first_8_dims[:3].tolist()}")

# --- .mean(): average value across ALL numbers in a tensor ---
overall_mean = emb_train.mean()
print(f"  emb_train.mean()    -> {overall_mean.item():.6f}  "
      f"(scalar: the average of 57 x 3072 numbers)")

# --- F.normalize: rescale a vector to UNIT LENGTH (L2 norm = 1) ---
v = emb_train[0]
v_norm = F.normalize(v, dim=0)   # dim=0 because v is 1-D; dim=-1 also works
norm_before = torch.linalg.vector_norm(v)
norm_after = torch.linalg.vector_norm(v_norm)
print(f"  |v|        = {norm_before.item():.6f}   (original vector length)")
print(f"  |v_normed| = {norm_after.item():.6f}   (after F.normalize — always ~1.0)")
print("  WHY normalize? Cosine similarity is easier (and numerically safer)")
print("  when vectors are unit length: the dot product of two unit vectors")
print("  IS their cosine similarity.")


# ---------------------------------------------------------------------------
# 3. DO THE TWO CLASSES ACTUALLY SEPARATE? (cosine similarity analysis)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("3. AVERAGE COSINE SIMILARITIES (within-class vs cross-class)")
print("=" * 60)

# WHAT is cosine similarity?
#   Imagine each embedding as an arrow from the origin into 3072-D space.
#   Cosine similarity compares the ANGLE between two arrows:
#       +1.0  -> same direction  (very similar meaning)
#        0.0  -> perpendicular   (unrelated)
#       -1.0  -> opposite direction.
#   It ignores arrow LENGTH and looks only at direction — which is exactly
#   what we want when comparing the "meaning" of two sentences.

# Split the TRAINING set into the two classes using a boolean mask.
#   y_train == 1 is a boolean tensor; indexing with it keeps only the rows
#   where the condition is True. (~ negates the mask.)
attack_mask = lab_train == 1           # True for attack rows
legit_mask = lab_train == 0            # True for legitimate rows
A = emb_train[attack_mask]             # (28, 3072) all training attacks
L = emb_train[legit_mask]              # (29, 3072) all training legitimate
print(f"  training attacks: {A.shape[0]}   training legitimate: {L.shape[0]}")

# Normalize every row to unit length, then the dot product of two rows =
# their cosine similarity (see comment above F.normalize).
A_n = F.normalize(A, dim=1)            # (28, 3072), every row |v| = 1
L_n = F.normalize(L, dim=1)            # (29, 3072)

# --- attack vs attack: mean over all DISTINCT pairs ---
# A_n @ A_n.T gives a (28, 28) matrix where entry [i, j] is the cosine
# similarity of attack i with attack j. The diagonal is always 1.0 (an
# example compared with itself) and [i, j] duplicates [j, i], so we take
# only the UPPER TRIANGLE to get each pair exactly once.
sim_aa = A_n @ A_n.T                                   # (28, 28)
iu = torch.triu_indices(A.shape[0], A.shape[0], offset=1)  # upper-triangle coords
avg_attack_attack = sim_aa[iu[0], iu[1]].mean()

# --- legitimate vs legitimate: same trick ---
sim_ll = L_n @ L_n.T                                   # (29, 29)
iu2 = torch.triu_indices(L.shape[0], L.shape[0], offset=1)
avg_legit_legit = sim_ll[iu2[0], iu2[1]].mean()

# --- attack vs legitimate: every cross pair, no diagonal to exclude ---
sim_al = A_n @ L_n.T                                   # (28, 29)
avg_attack_legit = sim_al.mean()

print()
print(f"  attack  vs attack   (within) : {avg_attack_attack.item():.4f}")
print(f"  legit   vs legit    (within) : {avg_legit_legit.item():.4f}")
print(f"  attack  vs legit    (cross)  : {avg_attack_legit.item():.4f}")
print()
print("  INTERPRETATION:")
print("  If the two WITHIN-group numbers are clearly HIGHER than the")
print("  cross-group number, each class forms its own 'neighborhood' in")
print("  embedding space — the embeddings carry a usable attack-vs-legit")
print("  signal and a classifier should be able to learn a decision")
print("  boundary between the neighborhoods. If all three numbers are")
print("  about the same, the classes are interleaved and even a strong")
print("  model would struggle — we would need better features or more data.")
print("=" * 60)
