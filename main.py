from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torchvision import datasets

from federated import encode_labels
from federated.fedavg import FedAvg


@dataclass(frozen=True)
class RunSpec:
    id: str
    label: str
    method: str
    variant: str
    divisor: int
    model_dim: int
    chunks: int


def build_run_specs(methods: Sequence[str], chunks: Sequence[int], model_dim: int) -> list[RunSpec]:
    if isinstance(methods, (str, bytes)) or not methods:
        raise ValueError("config.fl.method must be a non-empty list.")
    if isinstance(chunks, (str, bytes)) or not chunks:
        raise ValueError("config.fl.chunks must be a non-empty list.")
    if model_dim < 1:
        raise ValueError("config.dataset.model_dim must be at least 1.")

    method_names = [str(method).lower() for method in methods]
    if len(set(method_names)) != len(method_names):
        raise ValueError("config.fl.method must not contain duplicates.")
    unsupported = [method for method in method_names if method != "fedavg"]
    if unsupported:
        raise ValueError(f"Unsupported FL method(s): {unsupported}. Only 'fedavg' is implemented.")

    chunk_counts = []
    for chunk in chunks:
        count = int(chunk)
        if isinstance(chunk, bool) or count != chunk:
            raise ValueError("config.fl.chunks values must be integers.")
        chunk_counts.append(count)
    if len(set(chunk_counts)) != len(chunk_counts):
        raise ValueError("config.fl.chunks must not contain duplicates.")
    if any(chunk < 1 for chunk in chunk_counts):
        raise ValueError("config.fl.chunks values must be positive.")
    if any(chunk > model_dim for chunk in chunk_counts):
        raise ValueError("config.fl.chunks values must not exceed model_dim.")

    reduced_dims = [model_dim // chunk for chunk in chunk_counts]
    if len(set(reduced_dims)) != len(reduced_dims):
        raise ValueError("config.fl.chunks produces duplicate reduced model dimensions.")

    specs = []
    for method in method_names:
        for divisor, reduced_dim in zip(chunk_counts, reduced_dims):
            if divisor == 1:
                specs.append(
                    RunSpec(
                        id=f"{method}_full_d{model_dim}_c1",
                        label=f"FedAvg D={model_dim}, chunks=1",
                        method=method,
                        variant="full_dim",
                        divisor=divisor,
                        model_dim=model_dim,
                        chunks=1,
                    )
                )
                continue

            specs.append(
                RunSpec(
                    id=f"{method}_reduced_d{reduced_dim}_for_c{divisor}",
                    label=f"FedAvg D={reduced_dim}, chunks=1 (D/{divisor} baseline)",
                    method=method,
                    variant="reduced_dim",
                    divisor=divisor,
                    model_dim=reduced_dim,
                    chunks=1,
                )
            )
            specs.append(
                RunSpec(
                    id=f"{method}_full_d{model_dim}_c{divisor}",
                    label=f"FedAvg D={model_dim}, chunks={divisor}",
                    method=method,
                    variant="full_dim",
                    divisor=divisor,
                    model_dim=model_dim,
                    chunks=divisor,
                )
            )
    return specs


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


def _run_styles(runs: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_dims = list(dict.fromkeys(int(run["model_dim"]) for run in runs))
    color_map = plt.get_cmap("tab10")
    colors = {model_dim: color_map(index % color_map.N) for index, model_dim in enumerate(model_dims)}
    return {
        str(run["id"]): {
            "color": colors[int(run["model_dim"])],
            "linestyle": "-" if int(run["chunks"]) == 1 else "--",
        }
        for run in runs
    }


def plot_accuracy(eval_rounds: torch.Tensor, runs: Sequence[dict[str, Any]], output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rounds = eval_rounds.detach().cpu()
    styles = _run_styles(runs)

    plt.figure(figsize=(8, 5))
    for run in runs:
        curves = run["accuracies"].detach().cpu()
        mean = curves.mean(dim=0)
        lower = torch.quantile(curves, 0.05, dim=0)
        upper = torch.quantile(curves, 0.95, dim=0)
        style = styles[str(run["id"])]
        plt.plot(
            rounds.numpy(),
            mean.numpy(),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            label=str(run["label"]),
        )
        plt.fill_between(
            rounds.numpy(),
            lower.numpy(),
            upper.numpy(),
            color=style["color"],
            alpha=0.1,
        )
    plt.xlabel("Evaluation epoch")
    plt.ylabel("Test accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_eval_wall_times(eval_rounds: torch.Tensor, global_epoch_durations: torch.Tensor) -> torch.Tensor:
    rounds = eval_rounds.detach().cpu().to(dtype=torch.long)
    durations = global_epoch_durations.detach().cpu().to(dtype=torch.float64)
    cumulative = durations.cumsum(dim=1)
    return cumulative.index_select(1, rounds - 1)


def compute_mean_eval_wall_times(
    eval_rounds: torch.Tensor,
    global_epoch_durations: torch.Tensor,
) -> torch.Tensor:
    mean_durations = global_epoch_durations.mean(dim=0, keepdim=True)
    return compute_eval_wall_times(eval_rounds, mean_durations).squeeze(0)


def plot_accuracy_by_time(
    eval_rounds: torch.Tensor,
    runs: Sequence[dict[str, Any]],
    output_path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _run_styles(runs)

    plt.figure(figsize=(8, 5))
    for run in runs:
        durations = run["global_epoch_durations"].detach().cpu().to(dtype=torch.float64)
        mean_times = compute_mean_eval_wall_times(eval_rounds, durations)
        curves = run["accuracies"].detach().cpu()
        mean = curves.mean(dim=0)
        lower = torch.quantile(curves, 0.05, dim=0)
        upper = torch.quantile(curves, 0.95, dim=0)
        style = styles[str(run["id"])]
        plt.plot(
            mean_times.numpy(),
            mean.numpy(),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            label=str(run["label"]),
        )
        plt.fill_between(
            mean_times.numpy(),
            lower.numpy(),
            upper.numpy(),
            color=style["color"],
            alpha=0.1,
        )
    plt.xlabel("Training wall-clock time (s)")
    plt.ylabel("Test accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _wall_time_plot_path(plot_path: str | Path) -> Path:
    path = Path(plot_path)
    return path.with_name(f"{path.stem}_by_time{path.suffix}")


def _config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return dict(config)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_fedavg(
    config: Any,
    spec: RunSpec,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    experiment_seed: int,
):
    trainer = FedAvg(
        num_classes=int(config.dataset.num_classes),
        model_dim=spec.model_dim,
        lr=float(config.model.learning_rate),
        C=float(config.model.C),
        margin_width=float(config.model.margin_width),
        no_margin=bool(config.model.no_margin),
        backend=str(config.model.backend),
        device=config.device,
        dtype=torch.float32,
    )
    return trainer.fit(
        X_train,
        y_train,
        X_test=X_test,
        y_test=y_test,
        num_clients=int(config.fl.num_clients),
        global_epochs=int(config.training.global_epochs),
        local_epochs=int(config.training.local_epochs),
        batch_size=int(config.fl.batch_size),
        chunks=spec.chunks,
        noniid=bool(config.fl.noniid),
        classes_per_client=int(config.fl.classes_per_client),
        shuffle=bool(config.fl.shuffle),
        seed=experiment_seed,
        eval_global_epochs=int(config.training.eval_global_epochs),
        show_progress=True,
        method_name=spec.method,
    )


_METHOD_RUNNERS: dict[str, Callable[..., Any]] = {
    "fedavg": _fit_fedavg,
}


def run(config) -> dict[str, Any]:
    full_model_dim = int(config.dataset.model_dim)
    run_specs = build_run_specs(config.fl.method, config.fl.chunks, full_model_dim)
    num_experiments = int(config.training.num_experiments)
    if num_experiments < 1:
        raise ValueError("config.training.num_experiments must be at least 1.")

    base_seed = int(config.reproducibility.base_seed)
    _seed_everything(base_seed)

    X_train, y_train_raw, X_test, y_test_raw = load_dataset(config)
    y_train, y_test, label_encoder = encode_labels(y_train_raw, y_test_raw)
    assert y_test is not None

    eval_rounds = None
    runs = [
        {
            **asdict(spec),
            "accuracies": [],
            "global_epoch_durations": [],
            "final_prototypes": [],
        }
        for spec in run_specs
    ]
    runs_by_id = {str(run["id"]): run for run in runs}

    for experiment in range(num_experiments):
        experiment_seed = base_seed + experiment
        transform_seed = int(config.transform.seed) + experiment_seed

        X_train_hd, X_test_hd = transform_features(
            X_train,
            X_test,
            model_dim=full_model_dim,
            transform_seed=transform_seed,
            normalize=bool(config.transform.normalize),
            batch_size=config.transform.batch_size,
            device=config.device,
        )

        for spec in run_specs:
            print(
                f"Evaluating {spec.method}: model_dim={spec.model_dim}, "
                f"chunks={spec.chunks}, experiment={experiment + 1}/{num_experiments}"
            )
            _seed_everything(experiment_seed)
            X_train_run = X_train_hd[:, :spec.model_dim].contiguous()
            X_test_run = X_test_hd[:, :spec.model_dim].contiguous()
            runner = _METHOD_RUNNERS[spec.method]
            result = runner(
                config,
                spec,
                X_train_run,
                y_train,
                X_test_run,
                y_test,
                experiment_seed=experiment_seed,
            )

            if eval_rounds is None:
                eval_rounds = result.eval_rounds
            elif not torch.equal(eval_rounds, result.eval_rounds):
                raise RuntimeError(f"Evaluation rounds differ for run '{spec.id}'.")

            run = runs_by_id[spec.id]
            run["accuracies"].append(result.accuracies)
            run["global_epoch_durations"].append(result.global_epoch_durations)
            run["final_prototypes"].append(result.global_prototypes)

    assert eval_rounds is not None
    for run in runs:
        run["accuracies"] = torch.stack(run["accuracies"])
        run["global_epoch_durations"] = torch.stack(run["global_epoch_durations"])
        run["eval_wall_times"] = compute_eval_wall_times(eval_rounds, run["global_epoch_durations"])

    output_dir = Path(config.output.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / config.output.results_filename
    plot_path = output_dir / config.output.plot_filename
    time_plot_path = _wall_time_plot_path(plot_path)

    results = {
        "config": _config_to_dict(config),
        "eval_rounds": eval_rounds,
        "runs": runs,
        "label_classes": list(label_encoder.classes_),
    }
    torch.save(results, results_path)
    plot_accuracy(eval_rounds, runs, plot_path)
    plot_accuracy_by_time(eval_rounds, runs, time_plot_path)
    print(f"Saved results to {results_path}")
    print(f"Saved accuracy plot by evaluation epoch to {plot_path}")
    print(f"Saved accuracy plot by wall-clock time to {time_plot_path}")
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
