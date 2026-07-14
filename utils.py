from __future__ import annotations

import importlib
import importlib.util
import random
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config(config_ref: str) -> Any:
    if config_ref.endswith(".py") or Path(config_ref).exists():
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


def _run_styles(runs: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
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


def plot_accuracy(
    eval_rounds: torch.Tensor,
    runs: Sequence[dict[str, Any]],
    output_path: str | Path,
) -> None:
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


def compute_eval_wall_times(
    eval_rounds: torch.Tensor,
    global_epoch_durations: torch.Tensor,
) -> torch.Tensor:
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


def wall_time_plot_path(plot_path: str | Path) -> Path:
    path = Path(plot_path)
    return path.with_name(f"{path.stem}_by_time{path.suffix}")


def config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return dict(config)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
