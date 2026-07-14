from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.datasets import MNIST


def load_mnist_dataset(
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root = Path(config.dataset.data_root)
    train_set = MNIST(root=str(root), train=True, download=bool(config.dataset.download))
    test_set = MNIST(root=str(root), train=False, download=bool(config.dataset.download))

    return (
        train_set.data.to(dtype=torch.float32),
        train_set.targets.to(dtype=torch.long),
        test_set.data.to(dtype=torch.float32),
        test_set.targets.to(dtype=torch.long),
    )
