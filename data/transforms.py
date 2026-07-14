from __future__ import annotations

import torch
from mmhdc.utils import HDTransform


def transform_features(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    *,
    model_dim: int,
    transform_seed: int,
    normalize: bool,
    batch_size: int | None,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    transform = HDTransform(
        in_channels=X_train_flat.shape[1],
        out_channels=model_dim,
        seed=transform_seed,
        batch_size=batch_size,
        normalize=normalize,
        device=device,
        dtype=torch.float32,
    )
    return transform(X_train_flat), transform(X_test_flat)

