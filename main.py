from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import random
from pathlib import Path
from typing import Any

import torch
from torchvision import datasets
from tqdm import tqdm

from federated import encode_labels
from federated.fedavg import FedAvg


def load_config(config_ref: str):
    if config_ref.endswith(".py") or os.path.exists(config_ref):
        path = Path(config_ref).resolve()
        spec = importlib.util.spec_from_file_location("runtime_config", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load config file: {config_ref}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(config_ref)

    if not hasattr(module, "get_config"):
        raise AttributeError(f"Config '{config_ref}' must define get_config().")
    return module.get_config()


def load_dataset(config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset_map = {
        "mnist": datasets.MNIST,
        "fashion-mnist": datasets.FashionMNIST,
    }
    dataset_name = config.dataset.name.lower()
    if dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset '{config.dataset.name}'. Expected one of {sorted(dataset_map)}.")

    dataset_cls = dataset_map[dataset_name]
    root = Path(config.dataset.data_root)
    train_set = dataset_cls(root=str(root), train=True, download=bool(config.dataset.download))
    test_set = dataset_cls(root=str(root), train=False, download=bool(config.dataset.download))

    return (
        train_set.data.to(dtype=torch.float32),
        train_set.targets.to(dtype=torch.long),
        test_set.data.to(dtype=torch.float32),
        test_set.targets.to(dtype=torch.long),
    )


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
    from mmhdc.utils import HDTransform

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


def plot_accuracy(eval_rounds: torch.Tensor, accuracies: torch.Tensor, output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rounds = eval_rounds.detach().cpu()
    curves = accuracies.detach().cpu()
    mean = curves.mean(dim=0)
    lower = torch.quantile(curves, 0.05, dim=0)
    upper = torch.quantile(curves, 0.95, dim=0)

    plt.figure(figsize=(8, 5))
    plt.plot(rounds.numpy(), mean.numpy(), label="Mean accuracy")
    plt.fill_between(rounds.numpy(), lower.numpy(), upper.numpy(), alpha=0.25, label="5-95 percentile")
    plt.xlabel("Global update round")
    plt.ylabel("Test accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return dict(config)


def run(config) -> dict[str, Any]:
    # Fixing random seeds for reproducibility
    base_seed = int(config.reproducibility.base_seed)
    random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    X_train, y_train_raw, X_test, y_test_raw = load_dataset(config)
    y_train, y_test, label_encoder = encode_labels(y_train_raw, y_test_raw)

    device = config.device

    all_accuracies = []
    eval_rounds = None
    final_prototypes = []

    if config.fl.method != "fedavg":
        raise ValueError(f"Unsupported FL method '{config.fl.method}'. Only 'fedavg' is implemented.")

    for experiment in tqdm(range(int(config.training.num_experiments)), desc="Experiments"):
        experiment_seed = base_seed + experiment
        transform_seed = int(config.transform.seed) + experiment_seed

        X_train_hd, X_test_hd = transform_features(
            X_train,
            X_test,
            model_dim=int(config.dataset.model_dim),
            transform_seed=transform_seed,
            normalize=bool(config.transform.normalize),
            batch_size=config.transform.batch_size,
            device=device,
        )

        trainer = FedAvg(
            num_classes=int(config.dataset.num_classes),
            model_dim=int(config.dataset.model_dim),
            lr=float(config.model.learning_rate),
            C=float(config.model.C),
            margin_width=float(config.model.margin_width),
            no_margin=bool(config.model.no_margin),
            backend=str(config.model.backend),
            device=device,
            dtype=torch.float32,
        )
        result = trainer.fit(
            X_train_hd,
            y_train,
            X_test=X_test_hd,
            y_test=y_test,
            num_clients=int(config.fl.num_clients),
            global_epochs=int(config.training.global_epochs),
            local_epochs=int(config.training.local_epochs),
            batch_size=int(config.fl.batch_size),
            chunks=int(config.fl.chunks),
            noniid=bool(config.fl.noniid),
            classes_per_client=int(config.fl.classes_per_client),
            shuffle=bool(config.fl.shuffle),
            seed=experiment_seed,
            eval_global_epochs=int(config.training.eval_global_epochs),
            print_progress=True,
            experiment_index=experiment,
            num_experiments=int(config.training.num_experiments),
        )

        if eval_rounds is None:
            eval_rounds = result.eval_rounds
        elif not torch.equal(eval_rounds, result.eval_rounds):
            raise RuntimeError("Evaluation rounds differ between experiments.")

        all_accuracies.append(result.accuracies)
        final_prototypes.append(result.global_prototypes)

    accuracy_tensor = torch.stack(all_accuracies)
    assert eval_rounds is not None

    output_dir = Path(config.output.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / config.output.results_filename
    plot_path = output_dir / config.output.plot_filename

    results = {
        "config": _config_to_dict(config),
        "eval_rounds": eval_rounds,
        "accuracies": accuracy_tensor,
        "final_prototypes": final_prototypes,
        "label_classes": list(label_encoder.classes_),
    }
    torch.save(results, results_path)
    plot_accuracy(eval_rounds, accuracy_tensor, plot_path)
    print(f"Saved results to {results_path}")
    print(f"Saved accuracy plot to {plot_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated MM-HDC")
    parser.add_argument(
        "--config",
        default="configs.fedavg.config_mnist",
        help="Python config module or path to a .py config file.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
