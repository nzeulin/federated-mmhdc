from __future__ import annotations

from typing import Any

from .cwru import load_cwru_dataset
from .har import load_har_dataset
from .mnist import load_mnist_dataset


def load_dataset(config: Any) -> tuple[Any, ...]:
    dataset_name = config.dataset.name.lower()
    if dataset_name in {"har", "uci-har"}:
        return load_har_dataset(
            config.dataset.data_root,
            download=bool(config.dataset.download),
        )
    if dataset_name in {"cwru", "cwru-bearing"}:
        return load_cwru_dataset(
            config.dataset.data_root,
            manifest_path=config.dataset.manifest_path,
            cache_dir=config.dataset.cache_dir,
            download=bool(config.dataset.download),
            sensor_channel=config.dataset.sensor_channel,
            loads=config.dataset.loads,
            fault_diameters=config.dataset.fault_diameters,
            outer_race_position=config.dataset.outer_race_position,
            window_size=int(config.dataset.window_size),
            train_candidate_stride=int(config.dataset.train_candidate_stride),
            test_stride=int(config.dataset.test_stride),
            train_windows_per_group=int(config.dataset.train_windows_per_group),
            test_windows_per_group=int(config.dataset.test_windows_per_group),
            return_metadata=bool(config.dataset.return_metadata),
            return_class_mapping=bool(config.dataset.return_class_mapping),
            seed=int(config.dataset.seed),
        )
    if dataset_name == "mnist":
        return load_mnist_dataset(config)

    supported = ["cwru", "cwru-bearing", "har", "mnist", "uci-har"]
    raise ValueError(f"Unsupported dataset '{config.dataset.name}'. Expected one of {supported}.")
