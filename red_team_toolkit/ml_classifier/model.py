"""
model.py — Step 4 of the ML classifier sub-project.

WHAT this file does:
    Defines the two neural-network architectures we will train to separate
    "attack" prompts from "legitimate" prompts, given a 3072-dim embedding.

THE TWO MODELS (they differ only in capacity):
    1. PromptClassifier (the "mlp"):  Linear -> ReLU -> Dropout -> Linear
       A tiny 2-layer network. Can learn NON-LINEAR boundaries (e.g. "attacks
       are over here AND over there, legitimate is in between").
    2. LinearProbe (the "linear"):    a single Linear layer.
       Can only draw ONE straight hyperplane through the embedding space.
       It is the honest BASELINE: if the MLP only barely beats it, the
       non-linearity wasn't buying us much.

THE MOST IMPORTANT DESIGN RULE IN THIS FILE:
    forward() returns RAW LOGITS (unbounded real numbers), NOT probabilities.
    There is deliberately NO sigmoid() at the end of the model.

WHY NO SIGMOID INSIDE THE MODEL?
    We train with nn.BCEWithLogitsLoss(), which = sigmoid() THEN binary
    cross-entropy, but computed in a single fused, numerically-stable step.
    If we put a sigmoid in forward() AND use BCEWithLogitsLoss, we would
    apply sigmoid TWICE — wrong math, and it silently wrecks training.
    The rule of thumb:
        - Model outputs logits.
        - Loss (BCEWithLogitsLoss) does the sigmoid internally.
        - When we later need an actual probability (0..1) for a prediction,
          we apply torch.sigmoid() ourselves, only at that point.
    (train.py turns logits into class predictions with `logit > 0`, which is
    equivalent to `sigmoid(logit) > 0.5` — no sigmoid needed for accuracy.)
"""

import torch
from torch import nn


class PromptClassifier(nn.Module):
    """
    A small feed-forward network (an MLP): Linear -> ReLU -> Dropout -> Linear.

    nn.Module is the base class for ALL neural networks in PyTorch. Subclassing
    it gives us, for free: parameter tracking, .to(device), .train()/.eval()
    mode switching (which toggles Dropout), and .state_dict() for checkpointing.
    """

    def __init__(self, input_dim: int = 3072, hidden_dim: int = 64, dropout: float = 0.3):
        """
        Parameters
        ----------
        input_dim  : size of each input vector (3072 = our embedding dimension).
        hidden_dim : width of the single hidden layer. Bigger = more capacity
                     but also more prone to overfitting on our tiny dataset.
        dropout    : probability of randomly zeroing a hidden unit during
                     TRAINING only. A regularization technique: it stops any
                     one unit from becoming a crutch, forcing redundancy.
                     (In eval() mode Dropout is automatically turned off.)
        """
        super().__init__()  # required: initialize the nn.Module machinery

        # nn.Sequential chains layers in order: input -> layer1 -> layer2 -> ...
        self.net = nn.Sequential(
            # Layer 1: project 3072-dim input down to hidden_dim features.
            # Each output neuron learns to detect some combination of input
            # patterns. bias=True adds a learnable offset per neuron.
            nn.Linear(input_dim, hidden_dim),

            # ReLU = max(0, x). WHY is a non-linearity REQUIRED?
            # Two stacked Linear layers WITHOUT an activation between them
            # collapse back into ONE linear layer (mathematically, composing
            # two linear maps is just another linear map). ReLU breaks that,
            # letting the network carve out curved, non-linear decision
            # boundaries. Without it this "MLP" could not learn anything a
            # single Linear couldn't.
            nn.ReLU(),

            # Dropout: randomly zeroes hidden units with probability `dropout`
            # during training. Regularization against overfitting. Off in eval.
            nn.Dropout(p=dropout),

            # Layer 2: project hidden_dim back down to a SINGLE number — the
            # logit for "is this an attack?". One output because it's binary.
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the model's output for a batch of inputs.

        x    : (batch_size, 3072)
        returns : (batch_size,)  — one RAW LOGIT per example.

        self.net(x) produces shape (batch_size, 1); squeeze(-1) removes that
        trailing size-1 dimension so we get (batch_size,). Keeping the output
        1-D makes it line up exactly with our (batch_size,) float labels,
        which is what BCEWithLogitsLoss expects.
        """
        return self.net(x).squeeze(-1)   # (batch, 1) -> (batch,)  RAW LOGITS


class LinearProbe(nn.Module):
    """
    The baseline: a SINGLE Linear layer from 3072 -> 1. No hidden layer, no
    ReLU, no dropout.

    This is the "is the data linearly separable?" test. It draws exactly one
    straight decision boundary. If this baseline scores nearly as well as the
    MLP, it means the classes are already close to linearly separable in
    embedding space and the extra hidden layer wasn't necessary.
    """

    def __init__(self, input_dim: int = 3072):
        super().__init__()
        # One affine map: 3072 inputs -> 1 logit. Same "raw logit" convention
        # as the MLP (no sigmoid), so the training code is identical for both.
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 3072) -> returns (batch,) raw logits."""
        return self.linear(x).squeeze(-1)   # (batch, 1) -> (batch,)


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of TRAINABLE parameters in a model.

    Each nn.Linear holds a weight matrix and a bias vector; each is a
    torch.Tensor with a .numel() (number of elements). We only count tensors
    with requires_grad=True, because frozen tensors aren't learned.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_type: str, input_dim: int = 3072,
                hidden_dim: int = 64, dropout: float = 0.3) -> nn.Module:
    """
    Factory helper: build a model by name.

    Having ONE place to map a name -> architecture is what makes the
    checkpoint "round trip" possible: we save the config dict alongside the
    weights, and later rebuild a FRESH model from that config before loading
    the weights back in (see train.py).
    """
    if model_type == "mlp":
        return PromptClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if model_type == "linear":
        return LinearProbe(input_dim=input_dim)
    raise ValueError(f"Unknown model_type {model_type!r} — expected 'mlp' or 'linear'.")
