from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

import data
import utils
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


def run(config: Any) -> dict[str, Any]:
    full_model_dim = int(config.dataset.model_dim)
    run_specs = build_run_specs(config.fl.method, config.fl.chunks, full_model_dim)
    num_experiments = int(config.training.num_experiments)
    if num_experiments < 1:
        raise ValueError("config.training.num_experiments must be at least 1.")

    base_seed = int(config.reproducibility.base_seed)
    utils.seed_everything(base_seed)

    X_train, y_train_raw, X_test, y_test_raw = data.load_dataset(config)
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

        X_train_hd, X_test_hd = data.transform_features(
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
            utils.seed_everything(experiment_seed)
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

            run_record = runs_by_id[spec.id]
            run_record["accuracies"].append(result.accuracies)
            run_record["global_epoch_durations"].append(result.global_epoch_durations)
            run_record["final_prototypes"].append(result.global_prototypes)

    assert eval_rounds is not None
    for run_record in runs:
        run_record["accuracies"] = torch.stack(run_record["accuracies"])
        run_record["global_epoch_durations"] = torch.stack(
            run_record["global_epoch_durations"]
        )
        run_record["eval_wall_times"] = utils.compute_eval_wall_times(
            eval_rounds,
            run_record["global_epoch_durations"],
        )

    output_dir = Path(config.output.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / config.output.results_filename
    plot_path = output_dir / config.output.plot_filename
    time_plot_path = utils.wall_time_plot_path(plot_path)

    results = {
        "config": utils.config_to_dict(config),
        "eval_rounds": eval_rounds,
        "runs": runs,
        "label_classes": list(label_encoder.classes_),
    }
    torch.save(results, results_path)
    utils.plot_accuracy(eval_rounds, runs, plot_path)
    utils.plot_accuracy_by_time(eval_rounds, runs, time_plot_path)
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
    config = utils.load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
