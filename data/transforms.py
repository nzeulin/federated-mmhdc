from __future__ import annotations

import torch
from mmhdc.utils import HDTransform


def transform_features(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    *,
    model_dim: int,
    transform_seed: int,
    normalize_input: bool,
    normalize_hypervectors: bool,
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
        normalize=normalize_input,
        device=device,
        dtype=torch.float32,
    )
    X_train_hd = transform(X_train_flat)
    X_test_hd = transform(X_test_flat)

    if normalize_hypervectors:
        eps = torch.finfo(X_train_hd.dtype).eps
        X_train_hd = X_train_hd / torch.linalg.vector_norm(
            X_train_hd,
            dim=1,
            keepdim=True,
        ).clamp_min(eps)
        X_test_hd = X_test_hd / torch.linalg.vector_norm(
            X_test_hd,
            dim=1,
            keepdim=True,
        ).clamp_min(eps)

    return X_train_hd, X_test_hd
