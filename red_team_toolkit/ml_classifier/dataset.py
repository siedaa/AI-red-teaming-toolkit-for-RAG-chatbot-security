"""
dataset.py — Step 4 of the ML classifier sub-project.

WHAT this file does:
    Defines a `PromptEmbeddingDataset`, a small Python class that wraps our
    pre-computed embedding vectors + labels so that PyTorch's DataLoader can
    hand them to the model in mini-batches during training.

WHY does PyTorch need a Dataset class at all?
    PyTorch separates "WHERE the data lives" from "HOW it is fed to the model":
      - Dataset   = "how do I fetch one example (features, label) by index?"
      - DataLoader = "how do I bunch examples into batches, shuffle them, and
                     use multiple workers?" (see train.py)
    By implementing the tiny Dataset "interface" (just __len__ and __getitem__),
    our data becomes compatible with the entire PyTorch training ecosystem —
    shuffling, batching, pinned memory, etc. — without rewriting any of it.

THE INTERFACE (the only two methods we must provide):
    __len__(self)        -> return HOW MANY examples there are
    __getitem__(self, i) -> return the i-th example as a (features, label) pair
    That's it. DataLoader calls these for us.

WHY float32 for BOTH features and labels here?
    - Features: the embedding vectors. float32 is the standard float width for
      neural-network inputs (half the memory of float64, and what the model's
      Linear layers expect by default).
    - Labels: we are using BCEWithLogitsLoss (binary cross-entropy). That loss
      compares the model's predicted probability against a target that MUST be
      a float of the SAME shape as the prediction. Our labels arrive from the
      .npy files as int64 class ids (0 / 1), so this class converts them to
      float32 (0.0 / 1.0). Getting this wrong raises the classic error:
          "Expected floating point type for target with dtype = Long"
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class PromptEmbeddingDataset(Dataset):
    """
    Wraps an (N, 3072) feature array and an (N,) label array into something
    PyTorch's DataLoader understands.

    We store the data as torch tensors up front (not numpy), so __getitem__
    just does a cheap integer index and returns a tensor slice — no per-item
    conversion cost inside the training loop.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Parameters
        ----------
        features : np.ndarray of shape (N, 3072), float32
            One embedding vector per prompt.
        labels : np.ndarray of shape (N,), any int/float 0/1
            One binary label per prompt (0 = legitimate, 1 = attack).
        """
        # torch.from_numpy() shares the array's memory (zero-copy), which is
        # fast. .float() guarantees float32 even if the source was int64.
        # (Embeddings are already float32, so that call is a no-op safety net.)
        self.features = torch.from_numpy(np.asarray(features)).float()

        # Labels MUST become float32 targets for BCEWithLogitsLoss (see header).
        # .float() turns int64 0/1 into float32 0.0/1.0.
        self.labels = torch.from_numpy(np.asarray(labels)).float()

        # Cheap sanity check: the two arrays must describe the same number of
        # examples, otherwise we'd silently misalign features with labels.
        assert self.features.shape[0] == self.labels.shape[0], (
            f"features has {self.features.shape[0]} rows but "
            f"labels has {self.labels.shape[0]} — they must match."
        )

    def __len__(self) -> int:
        """How many examples? DataLoader uses this to know the epoch size."""
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the (features, label) pair for one example.

        `features[index]` -> a (3072,) float32 tensor
        `labels[index]`   -> a scalar () float32 tensor (0.0 or 1.0)

        DataLoader will stack several of these rows into a batch:
            stacked features -> (batch_size, 3072)
            stacked labels   -> (batch_size,)
        which is exactly the shape our model's forward() expects.
        """
        return self.features[index], self.labels[index]
