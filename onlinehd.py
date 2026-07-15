from __future__ import annotations

import math

import torch


class OnlineHD(torch.nn.Module):
    """OnlineHD classifier implementation.
    """

    def __init__(
        self,
        num_classes: int,
        out_channels: int,
        lr: float = 0.035,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes < 1
        ):
            raise ValueError("num_classes must be a positive integer.")
        if (
            isinstance(out_channels, bool)
            or not isinstance(out_channels, int)
            or out_channels < 1
        ):
            raise ValueError("out_channels must be a positive integer.")
        if isinstance(lr, bool):
            raise ValueError("lr must be a positive finite number.")
        try:
            learning_rate = float(lr)
        except (TypeError, ValueError) as error:
            raise ValueError("lr must be a positive finite number.") from error
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("lr must be a positive finite number.")
        if (
            not isinstance(dtype, torch.dtype)
            or not torch.empty((), dtype=dtype).is_floating_point()
        ):
            raise ValueError("dtype must be a real floating-point torch dtype.")

        self.num_classes = num_classes
        self.out_channels = out_channels
        self.lr = learning_rate
        self.dtype = dtype
        self.prototypes = torch.nn.Parameter(
            torch.zeros(num_classes, out_channels, device=device, dtype=dtype),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict one class index for each input hypervector."""
        hypervectors = self._prepare_hypervectors(x, allow_empty=True)
        return self._cosine_scores(hypervectors).argmax(dim=1)

    @torch.no_grad()
    def initialize(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Initialize prototypes as ``lr`` times each class hypervector sum."""
        hypervectors, labels = self._prepare_training_batch(x, y)
        prototype_sums = torch.zeros_like(self.prototypes)
        prototype_sums.index_add_(0, labels, hypervectors)
        self.prototypes.copy_(prototype_sums)

    @torch.no_grad()
    def step(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Apply one OnlineHD update pass for a complete batch."""
        hypervectors, labels = self._prepare_training_batch(x, y)
        scores = self._cosine_scores(hypervectors)
        predictions = scores.argmax(dim=1)
        mistakes = predictions != labels
        if not mistakes.any():
            return

        mistake_rows = mistakes.nonzero(as_tuple=False).flatten()
        mistake_vectors = hypervectors.index_select(0, mistake_rows)
        true_labels = labels.index_select(0, mistake_rows)
        predicted_labels = predictions.index_select(0, mistake_rows)
        mistake_scores = scores.index_select(0, mistake_rows)

        true_weights = mistake_scores.gather(1, true_labels.unsqueeze(1)).squeeze(1)
        predicted_weights = mistake_scores.gather(
            1, predicted_labels.unsqueeze(1)
        ).squeeze(1)

        prototype_update = torch.zeros_like(self.prototypes)
        prototype_update.index_add_(
            0,
            true_labels,
            true_weights.unsqueeze(1) * mistake_vectors,
        )
        prototype_update.index_add_(
            0,
            predicted_labels,
            -predicted_weights.unsqueeze(1) * mistake_vectors,
        )
        self.prototypes.add_(prototype_update, alpha=self.lr)

    def _cosine_scores(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(self.prototypes.dtype).eps
        x_norms = torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(eps)
        prototype_norms = torch.linalg.vector_norm(
            self.prototypes, dim=1, keepdim=True
        ).clamp_min(eps)
        return (x / x_norms) @ (self.prototypes / prototype_norms).T

    def _prepare_training_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hypervectors = self._prepare_hypervectors(x, allow_empty=False)
        if not torch.is_tensor(y):
            raise TypeError("y must be a torch.Tensor.")
        if y.ndim not in {1, 2} or (y.ndim == 2 and y.shape[1] != 1):
            raise ValueError("y must have shape [num_samples] or [num_samples, 1].")
        if y.numel() != hypervectors.shape[0]:
            raise ValueError(
                "x and y must contain the same number of samples: "
                f"got {hypervectors.shape[0]} and {y.numel()}."
            )
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if y.dtype not in integer_dtypes:
            raise ValueError("y must contain integer class indices.")

        labels = y.reshape(-1).to(device=self.prototypes.device, dtype=torch.long)
        min_label = int(labels.min().item())
        max_label = int(labels.max().item())
        if min_label < 0 or max_label >= self.num_classes:
            raise ValueError(
                f"y values must be in [0, {self.num_classes - 1}], "
                f"got range [{min_label}, {max_label}]."
            )
        return hypervectors, labels

    def _prepare_hypervectors(
        self,
        x: torch.Tensor,
        *,
        allow_empty: bool,
    ) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor.")
        if x.ndim != 2:
            raise ValueError("x must have shape [num_samples, out_channels].")
        if x.shape[1] != self.out_channels:
            raise ValueError(
                f"Expected x to have {self.out_channels} features, got {x.shape[1]}."
            )
        if not allow_empty and x.shape[0] == 0:
            raise ValueError("x must contain at least one sample.")
        if not x.is_floating_point():
            raise ValueError("x must have a real floating-point dtype.")
        if not torch.isfinite(x).all().item():
            raise ValueError("x must contain only finite values.")
        hypervectors = x.to(
            device=self.prototypes.device,
            dtype=self.prototypes.dtype,
        )
        if not torch.isfinite(hypervectors).all().item():
            raise ValueError("x must remain finite when converted to the model dtype.")
        return hypervectors


__all__ = ["OnlineHD"]
