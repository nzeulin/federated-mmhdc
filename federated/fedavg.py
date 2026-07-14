from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from tqdm import tqdm

from . import split_iid, split_noniid


@dataclass
class FedAvgResult:
    global_prototypes: torch.Tensor
    eval_rounds: torch.Tensor
    accuracies: torch.Tensor
    global_epoch_durations: torch.Tensor


class FedAvg:
    def __init__(
        self,
        *,
        num_classes: int,
        model_dim: int,
        lr: float,
        C: float,
        margin_width: float,
        no_margin: bool = False,
        backend: str = "python",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_classes = num_classes
        self.model_dim = model_dim
        self.lr = lr
        self.C = C
        self.margin_width = margin_width
        self.no_margin = no_margin
        self.backend = backend
        self.device = torch.device(device)
        self.dtype = dtype

    @staticmethod
    @torch.no_grad()
    def local_update(
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        epochs: int,
        batch_size: int,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
    ):
        x_local = x
        y_local = y.reshape(-1).to(dtype=torch.long)

        for _ in range(epochs):
            if shuffle:
                perm = torch.randperm(x_local.shape[0], generator=generator, device="cpu")
                x_epoch = x_local.index_select(0, perm.to(x_local.device))
                y_epoch = y_local.index_select(0, perm.to(y_local.device))
            else:
                x_epoch = x_local
                y_epoch = y_local

            for start in range(0, x_epoch.shape[0], batch_size):
                end = min(start + batch_size, x_epoch.shape[0])
                batch_x = x_epoch[start:end].to(device=model.prototypes.device)
                batch_y = y_epoch[start:end].to(device=model.prototypes.device)
                model.step(batch_x, batch_y)

        return model

    @staticmethod
    @torch.no_grad()
    def global_update(local_prototypes: Sequence[torch.Tensor]) -> torch.Tensor:
        if not local_prototypes:
            raise ValueError("local_prototypes must contain at least one tensor.")
        return torch.stack([prototype.detach().clone() for prototype in local_prototypes]).mean(dim=0)

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        *,
        X_test: torch.Tensor | None = None,
        y_test: torch.Tensor | None = None,
        num_clients: int,
        global_epochs: int,
        local_epochs: int,
        batch_size: int,
        chunks: int = 1,
        noniid: bool = False,
        classes_per_client: int = 2,
        shuffle: bool = False,
        seed: int = 0,
        eval_global_epochs: int = 1,
        show_progress: bool = False,
        method_name: str = "fedavg",
        experiment_number: int = 1,
        num_experiments: int = 1,
    ) -> FedAvgResult:
        if chunks < 1:
            raise ValueError("chunks must be at least 1.")
        if global_epochs < 1:
            raise ValueError("global_epochs must be at least 1.")
        if local_epochs < 1:
            raise ValueError("local_epochs must be at least 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if eval_global_epochs < 1:
            raise ValueError("eval_global_epochs must be at least 1.")

        X_train = X_train.to(dtype=self.dtype)
        y_train = y_train.reshape(-1).to(dtype=torch.long)
        if X_test is not None:
            X_test = X_test.to(dtype=self.dtype)
        if y_test is not None:
            y_test = y_test.reshape(-1).to(dtype=torch.long)
        if X_train.shape[1] != self.model_dim:
            raise ValueError(
                f"Expected X_train to have {self.model_dim} features, "
                f"got {X_train.shape[1]}."
            )

        if noniid:
            client_indices = split_noniid(y_train, num_clients, classes_per_client, seed)
        else:
            client_indices = split_iid(y_train, num_clients, seed)

        if chunks > self.model_dim:
            raise ValueError("chunks must not exceed model_dim.")

        # Initializing prototypes for every client
        initial_prototypes = []
        for indices in client_indices:
            model = self._make_model(out_channels=self.model_dim)
            X_client = X_train.index_select(0, indices).contiguous()
            y_client = y_train.index_select(0, indices).contiguous()
            model.initialize(X_client.to(self.device), y_client.to(self.device))
            initial_prototypes.append(model.prototypes.detach().clone())

        global_prototypes = self.global_update(initial_prototypes).to(
            device=self.device,
            dtype=self.dtype,
        )

        # For reproducibility
        schedule_generator = torch.Generator(device="cpu")
        schedule_generator.manual_seed(seed)

        full_positions = torch.arange(self.model_dim, dtype=torch.long)
        chunk_schedule: list[torch.Tensor] = []
        eval_rounds: list[int] = []
        accuracies: list[float] = []
        global_epoch_durations: list[float] = []

        description = (
            f"{method_name} | D={self.model_dim} | C={chunks} | "
            f"exp={experiment_number}/{num_experiments}"
        )
        global_epoch_bar = tqdm(
            total=global_epochs,
            desc=description,
            disable=not show_progress,
            leave=False,
        )
        for global_epoch in range(global_epochs):
            self._synchronize_device()
            epoch_start = time.perf_counter()

            if chunks == 1:
                current_positions = full_positions
            else:
                # Randomly selected indices of HDC sub-models
                if global_epoch % chunks == 0:
                    shuffled = full_positions[torch.randperm(self.model_dim, generator=schedule_generator)]
                    chunk_schedule = list(torch.tensor_split(shuffled, chunks))
                current_positions = chunk_schedule[global_epoch % chunks]

            current_positions = current_positions.to(dtype=torch.long)
            local_prototypes = []
            for client_id, indices in enumerate(client_indices):
                model = self._make_model(out_channels=current_positions.numel())
                # Materialize contiguous local tensors. This avoids depending on
                # advanced-indexing view behavior and keeps the optional C++
                # backend on simple dense tensors.
                X_client = X_train.index_select(0, indices).index_select(1, current_positions).contiguous()
                y_client = y_train.index_select(0, indices).contiguous()

                # Every update round starts from the latest global slice and
                # writes back the averaged local changes for the same positions.
                prototype_slice = global_prototypes.index_select(1, current_positions.to(self.device))
                with torch.no_grad():
                    model.prototypes.copy_(prototype_slice.contiguous())

                local_generator = torch.Generator(device="cpu")
                local_generator.manual_seed(seed + global_epoch * max(num_clients, 1) + client_id)
                self.local_update(
                    model,
                    X_client,
                    y_client,
                    epochs=local_epochs,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    generator=local_generator,
                )
                local_prototypes.append(model.prototypes.detach().clone())

            global_slice = self.global_update(local_prototypes).to(device=self.device, dtype=self.dtype)
            # Write only the active coordinates back into the global C x D model.
            global_prototypes[:, current_positions.to(self.device)] = global_slice

            self._synchronize_device()
            global_epoch_durations.append(time.perf_counter() - epoch_start)

            eval_positions = full_positions

            round_number = global_epoch + 1
            if X_test is not None and y_test is not None and round_number % eval_global_epochs == 0:
                accuracy = self.evaluate(X_test, y_test, global_prototypes, eval_positions)
                eval_rounds.append(round_number)
                accuracies.append(accuracy)
                if show_progress:
                    global_epoch_bar.set_postfix_str(f"acc={accuracy:.4f}")

            global_epoch_bar.update(1)

        global_epoch_bar.close()

        return FedAvgResult(
            global_prototypes=global_prototypes.detach().cpu(),
            eval_rounds=torch.as_tensor(eval_rounds, dtype=torch.long),
            accuracies=torch.as_tensor(accuracies, dtype=torch.float32),
            global_epoch_durations=torch.as_tensor(global_epoch_durations, dtype=torch.float64),
        )

    @torch.no_grad()
    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        prototypes: torch.Tensor,
        positions: torch.Tensor | Iterable[int],
    ) -> float:
        pos = torch.as_tensor(list(positions) if not torch.is_tensor(positions) else positions, dtype=torch.long)
        X_eval = X.index_select(1, pos.to(X.device)).to(device=self.device, dtype=self.dtype)
        P_eval = prototypes.index_select(1, pos.to(prototypes.device)).to(device=self.device, dtype=self.dtype)
        y_eval = y.to(device=self.device, dtype=torch.long)
        pred = torch.argmax(X_eval @ P_eval.T, dim=1)
        return (pred == y_eval).to(dtype=torch.float32).mean().item()

    def _make_model(self, out_channels: int):
        from mmhdc import MultiMMHDC

        return MultiMMHDC(
            num_classes=self.num_classes,
            out_channels=out_channels,
            lr=self.lr,
            C=self.C,
            margin_width=self.margin_width,
            no_margin=self.no_margin,
            device=str(self.device),
            backend=self.backend,
            dtype=self.dtype,
        )

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
