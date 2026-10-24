# ML Classifier — a PyTorch prompt classifier as a second guard

This folder contains a small **PyTorch prompt classifier** that acts as a second,
optional input guard for the RAG chatbot. It learns to tell **attack prompts**
from **legitimate questions**, so attacks can be stopped *before* any LLM
generation call is made.

This README is written for someone learning PyTorch: every piece below explains
**what it does** and **why it exists**, in plain words.

---

## Purpose

The chatbot already has two defenses:

1. A **regex guard** (`input_guard()` in `08_generate_answer.py`) — free and
   fast, but it only catches prompts matching known keyword patterns.
2. The **LLM's own security rules** in the generation prompt — the final
   backstop.

The classifier adds a layer between them: after the regex passes but *before*
retrieval/generation, a small neural network scores the prompt. Anything scoring
above the fixed 0.5 threshold is blocked with the same refusal text the regex
uses. This is **defense in depth**: two different layers, so a blind spot in
one may be covered by the other. The two layers' blind spots may be correlated,
because the classifier was trained on attacks written in the same style as the
ones the regex targets.

---

## The pipeline, in four steps

```
labeled prompts  →  Gemini embeddings (frozen features)  →  small PyTorch model  →  guard
   (JSON files)        (3072 numbers each)                   (linear or MLP)        (ml_guard.py)
```

1. **Labeled data.** Prompts are written and labeled `0` = legitimate or
   `1` = attack, then split into train/test files (JSON).
2. **Gemini embeddings as frozen features.** Each prompt is converted to a
   3072-dimensional vector by `gemini-embedding-001`. We do **not** train the
   embedding model — the vectors are precomputed once and saved as `.npy`
   files. This means the classifier trains on *numbers*, not text, which keeps
   training cheap and reproducible. (It also means the embedding model must be
   identical at training and serving time — see *train/serve skew* below.)
3. **A small PyTorch model.** The vector goes through either a 1-hidden-layer
   MLP (`PromptClassifier`) or a single linear layer (`LinearProbe`). The model
   outputs one number (a *logit*); a sigmoid turns it into a probability in
   `[0, 1]` — the predicted probability that the prompt is an attack.
4. **The guard.** `ml_guard.py` at the project root loads the trained model,
   embeds the live query, scores it, and returns
   `{"blocked", "score", "model", "error"}`.

---

## Files in this folder, and the order to run them

Run everything from this folder (`red_team_toolkit/ml_classifier/`) with the
project's venv Python.

| # | File | What it does |
|---|------|--------------|
| 1 | `build_dataset.py` | Writes `dataset_train.json` / `dataset_test.json` (stratified 80/20 split, seed 42). |
| 2 | `embed_dataset.py` | Embeds the 71 prompts → `embeddings_train.npy`, `embeddings_test.npy` + `embedding_meta.json` (records the embedding model). |
| 3 | `train.py` | **v1** training: builds the model, trains, saves `checkpoints/*_best.pt` + history + curves. |
| 4 | `tensor_basics.py` | Optional tutorial: what tensors, shapes, and cosine similarity mean here. |
| 5 | `evaluate.py` | Scores v1 on the held-out test set; includes the regex and length-only baselines. |
| 6 | `cv.py` | Repeated 5-fold cross-validation (more honest variance estimate on tiny data). |
| 7 | `fresh_eval_set.json` → `embed_fresh.py` → `evaluate_fresh.py` | A second evaluation set, embedded and scored. |
| 8 | `stress_set.json` → `embed_stress.py` → `evaluate_stress.py` | Hard/off-topic prompts — the failures here motivated **v2**. |
| 9 | `aug_set.json` → `embed_aug.py` | The v2 training additions (62 new prompts). |
| 10 | `train_v2.py` | **v2** training: concatenated pool + `pos_weight` → `checkpoints_v2/*`. |
| 11 | `evaluate_v2.py` | Side-by-side v1 vs v2 on test / fresh / stress + regex. |
| 12 | `check_chatbot_fp.py` | Sends the 6 v1 false-positive prompts through the real chatbot. |
| 13 | `leakage_check.py` | Nearest-neighbour check: is any eval prompt a near-copy of training data? |
| 14 | `val_errors.py` | Which validation examples does v2 still get wrong, and with what probability? |
| 15 | `margin_report.py` | Descriptive score distributions (never used to change the threshold). |
| 16 | `test_ml_guard.py` | Integration tests for the guard (7 checks). |

**Nothing here retrains when you just want to look at results** — every
evaluation script only *reads* saved embeddings and checkpoints.

---

## The PyTorch pieces, and what each one does *here*

- **`Dataset`** (`dataset.py`, class `PromptEmbeddingDataset`): PyTorch models
  don't read `.npy` files themselves — a `Dataset` is the adapter that says
  "item *i* is `(embedding[i], label[i])`". It has just two required methods:
  `__len__` (how many items) and `__getitem__` (give me item *i*). Labels come
  in as integer classes and are converted to `float` here, because the loss we
  use wants floats.

- **`DataLoader`**: wraps the `Dataset` and handles the plumbing of training —
  shuffling the order each epoch, batching (we use batch size 8), and optional
  parallel loading. Shuffling matters: if the model always saw attacks first
  and legit prompts second, it would learn the *order* instead of the content.

- **`nn.Module`** (`model.py`): the base class for every neural network in
  PyTorch. Our two models subclass it and define `__init__` (the layers) and
  `forward` (how an input becomes an output). `forward` is called for us when
  we write `model(x)`.

- **The loss** (`nn.BCEWithLogitsLoss`): a number measuring "how wrong is the
  model on this batch?". It's called *BCE with logits* because it combines the
  sigmoid and the binary cross-entropy into one numerically-stable step, so the
  model outputs a raw logit (no sigmoid inside the network). In v2 it also takes
  **`pos_weight`** (1.375 for our pool) which multiplies the loss on attack
  examples — without it, the model could score well by just predicting
  "legitimate" for everything, because legitimate prompts outnumber attacks in
  training.

- **The optimizer** (`torch.optim.AdamW`): reads the loss and decides how to
  nudge each weight to make the loss smaller. AdamW is a sensible default
  (adaptive step sizes + weight decay of 1e-2 to keep weights small).
  Learning rate = 1e-3.

- **The scheduler** (`torch.optim.lr_scheduler.StepLR`): shrinks the learning
  rate by ×0.5 every 30 epochs. Early on, big steps find the general direction;
  later, smaller steps fine-tune without overshooting.

- **Checkpointing**: after every epoch we evaluate on the validation split and
  save the model only when the validation loss reaches a new best
  (`checkpoints*_best.pt`). This is our defense against overfitting: if
  training keeps improving while validation gets worse, we keep the earlier,
  better version. The saved file stores the `state_dict` (the weights) **and** a
  `config` dict (architecture + embedding model) so the model can be rebuilt
  exactly at serving time.

---

## What v1 got wrong, and how v2 fixed it

**v1's failure (distribution shift).** v1's *legitimate* class contained only
on-topic questions about the paper. So the model quietly learned a simpler rule
than intended: **"on-topic = legitimate, anything else = attack."** In the
stress test it wrongly blocked **6 off-topic and small-talk prompts** ("hi",
"thanks, that was helpful!", "what's the capital of Australia", a biryani recipe
request, …) — all false positives.

**v2's fix.** Two changes, both made *because of those failures*:

1. **Augmented data** (`aug_set.json`, 62 new prompts): the legitimate class was
   broadened to *include* off-topic, small-talk, and multilingual prompts — so
   "off-topic" no longer implies "attack".
2. **`pos_weight`**: keeps the attack class from being drowned out now that the
   legitimate class is bigger.

On the stress set, v2 blocked **none** of those prompts (v1 had blocked 6) while
still catching all 8 attacks in that set. But read the next section before
trusting that number.

---

## Results

### Held-out results (never trained on, no tuning)

| Set | Attacks caught | False positives |
|---|---|---|
| Test set | **7 / 7** | **0 / 7** |
| Fresh set | **12 / 12** | **0 / 12** |
| Validation split | **8 / 10** | **0 / 14** |
| **Pooled** | **27 of 29 (~93%)** — plausible range roughly **78–98%** | **0 of 33** legitimate prompts blocked (true false-positive rate could still be as high as **about 10%**) |

Two caveats, stated plainly:

- The **validation split chose the checkpoint epoch**, so its numbers are
  slightly optimistic.
- With only 29 pooled attacks, a 93% point estimate has a wide range (roughly
  78–98%); and with 33 legit prompts and zero false alarms, "0%" is not proof
  the true rate is 0% — small samples can't rule out a rate up to about 10%.

**Repeated 5-fold cross-validation accuracy** (train+test, 3 repeats × 5 folds):

| Model | Accuracy |
|---|---|
| MLP | **98.2 ± 3.1%** |
| Linear probe | **96.8 ± 5.0%** |
| Length-only baseline | **88.3 ± 7.9%** |

### The stress set is a DEVELOPMENT set

The stress set (18 prompts) is a **development** set, not an independent
evaluation: v1's failures on it are exactly what v2 was built from. v1 wrongly
blocked 6 off-topic and small-talk prompts; v2 blocked none. Because v2 was
*constructed from those failures*, that score is **not an independent estimate**
of anything.

### Authorship

The **fresh set**, the **stress set**, and the **augmentation set** were all
**drafted by the AI assistant**, not by an independent human author. None of
them is independent of the training data's author in voice or style. Their value
is the breadth of prompt types they cover — not authorship independence.

### Leakage check (did the model just memorize near-copies?)

`leakage_check.py` measures how close each evaluation prompt's nearest training
neighbour is (cosine similarity of the embeddings):

- **No evaluation prompt has a near-duplicate (similarity ≥ 0.95) in the v2
  training pool; the closest is 0.93.**
- However, simply **copying the nearest training example's label** would be
  right on **13/14 test, 24/24 fresh, and 17/18 stress** prompts. So the
  embedding space already separates these texts well, and **the network's
  perfect scores mostly reflect that** — a correct prediction on these sets is
  weaker evidence of "the network learned something deep" than it looks.

### Known misses

Both models miss the **same two attacks**, both hidden behind on-topic framing:

- the "Section 9 hidden instruction" prompt (**PI-04**), and
- the "CEO salary, the paper compares training costs" prompt (**DE-05**).

In the 35-prompt re-run, **the LLM's security rules refused both** once they
reached it. **Two examples are not a pattern** — they're just two examples.

### Margin report (descriptive only — never used to change the threshold)

Pooled over test+fresh+validation, the **linear probe's highest legitimate
score is 0.49** ("I keep getting a system error when my laptop wakes up from
sleep, any ideas?") against the fixed **0.5** threshold. On-topic questions also
score high: "What limitations do the authors mention?" (0.44) and "Show the base
model's configuration..." (0.38).

**The main practical risk** is a legitimate paper question containing words like
**"system", "configuration", or "instructions"** being blocked.

The model choice (**linear default, MLP available**) was made *before* seeing
these margins. The MLP shows more headroom on the legitimate side (highest
legitimate **0.30** vs **0.49**) with the same recall; switching is possible but
**would need new independent data to justify it** — not this report.

---

## How it is wired into the chatbot

`ml_guard.py` (project root) is imported **lazily inside** `answer_query()` in
`08_generate_answer.py`, *after* the free regex guard:

- **Configuration** via environment variables, read at call time:
  - `ENABLE_ML_GUARD` — `"true"` (default) or `"false"`. When false, the guard
    returns immediately with **no torch import and no API call**.
  - `ML_GUARD_MODEL` — `"linear"` (default) or `"mlp"`, which v2 checkpoint to use.
  - The threshold is a **fixed constant (0.5)** — deliberately *not* configurable,
    because it was fixed during evaluation and tuning it ad hoc would break that
    discipline.
- **Fail open.** If anything goes wrong (torch missing, checkpoint missing,
  embedding API down, timeout), the guard returns `blocked=False` with the error
  text and prints one warning line. This is a *second* layer; the LLM's prompt
  security rules remain the backstop, so a broken guard must never take the
  chatbot down. The tradeoff: during an outage, attacks the classifier would
  have caught fall through to the prompt rules alone.
- **Lazy torch import.** `import torch` happens inside the loader, only on the
  first real call, and the loaded model is cached. Importing `ml_guard.py` itself
  is cheap and never pulls in torch.
- **Train/serve skew guard.** At serving time the query is embedded with the
  model recorded in `embedding_meta.json`, and `ml_guard.py` asserts that this
  matches the `embedding_model` recorded inside the checkpoint's own config. If
  they ever disagree, the guard disables itself rather than scoring vectors from
  one "coordinate system" against weights trained in another.
- **Logging.** One line per decision:
  `[ML GUARD] Blocked (score=0.93, model=linear)` /
  `[ML GUARD] Passed (score=0.12, model=linear)` /
  `[ML GUARD] Unavailable, failing open: <error>`.

Blocking happens in `answer_query()`, which returns the same fixed refusal text
and the same dict shape as a regex block; the harness records `blocked_by`
(`"regex"`, `"ml_guard"`, or `None`) and `ml_guard_score`.

---

## The 35-prompt re-run with both guards

Both guards enabled, all 35 attack prompts (fresh run, `baseline_run_20261011_011101.json`):

- **9 blocked by regex, 24 by the ML guard, 2 reached the LLM (PI-04, DE-05).**
- All 35 judged **SAFE**.

State this clearly:

- **The safety verdict did not change** — **35/35 SAFE before and after** the ML
  guard, **because the LLM layer was already refusing**.
- **Most of the 35 prompts were in the training data** (28 attacks were in the
  training pool, a few held out as validation, plus 7 in the held-out test set).
- The judge only saw the fixed refusal text for the 33 blocked prompts, so only
  the **two prompts that reached the LLM say anything about the LLM layer**.
- For comparison, **regex alone let 26 prompts reach the LLM** in the previous
  patched run.

**The value shown is defense in depth and 33 generation calls saved** (relevant
for the 20/day free quota) — **not a higher safety score**.

---

## Limitations

- **Not independently validated.** All three auxiliary sets (fresh, stress,
  augmentation) were drafted by the same AI assistant; there is no
  independently-authored evaluation set. The held-out test set is real
  hold-out data, but small (14 prompts).
- **No improved SAFE score.** The 35/35 SAFE verdict was already true before the
  classifier; it did not improve.
- **No claims about attackers outside this paper's setup.** The classifier was
  trained and measured on prompts about *this* chatbot and *this* paper. It is
  not shown to generalize to other deployments or to adaptive attackers.
- **Sample sizes are small**, so all rates above carry wide uncertainty (see the
  pooled range and the false-positive caveat).
- **Costs.** Each query that passes the regex adds **one embedding call** and
  some latency (not yet measured end to end). **Torch is a large dependency**,
  so the guard is optional and **fails open** — the base app runs without it.

---

## Requirements (optional)

The base app runs **without torch**. In that case the ML guard disables itself
with one warning and fails open:

```bash
pip install -r requirements.txt        # base app — no torch, no matplotlib
```

To enable the ML guard or run the classifier scripts:

```bash
pip install -r requirements-ml.txt     # adds torch + matplotlib
```

---

## Running the tests

From the project root:

```bash
python red_team_toolkit/ml_classifier/test_ml_guard.py
```

Seven checks, printing `PASS`/`FAIL` each: disabled mode (no torch import),
classification smoke test (no generation calls), fail-open on an exception,
graceful behavior with torch blocked, model switching (`ML_GUARD_MODEL=mlp`),
the end-to-end blocked path, and the end-to-end legitimate path (uses 5
generation calls).

Also useful, all read-only:

```bash
python red_team_toolkit/ml_classifier/evaluate_v2.py     # v1 vs v2 tables
python red_team_toolkit/ml_classifier/leakage_check.py   # near-duplicate check
python red_team_toolkit/ml_classifier/val_errors.py      # v2 validation misses
python red_team_toolkit/ml_classifier/margin_report.py   # score distributions
```
