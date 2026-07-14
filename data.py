from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmhdc.utils import HDTransform
from torchvision.datasets import MNIST


HAR_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)


def load_dataset(config: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset_name = config.dataset.name.lower()
    if dataset_name in {"har", "uci-har"}:
        return load_har_dataset(
            config.dataset.data_root,
            download=bool(config.dataset.download),
        )
    if dataset_name != "mnist":
        supported = ["har", "mnist", "uci-har"]
        raise ValueError(f"Unsupported dataset '{config.dataset.name}'. Expected one of {supported}.")

    root = Path(config.dataset.data_root)
    train_set = MNIST(root=str(root), train=True, download=bool(config.dataset.download))
    test_set = MNIST(root=str(root), train=False, download=bool(config.dataset.download))

    return (
        train_set.data.to(dtype=torch.float32),
        train_set.targets.to(dtype=torch.long),
        test_set.data.to(dtype=torch.float32),
        test_set.targets.to(dtype=torch.long),
    )


def _har_dataset_paths(data_root: str | Path) -> dict[str, Path]:
    root = Path(data_root)
    return {
        "X_train": root / "train" / "X_train.txt",
        "y_train": root / "train" / "y_train.txt",
        "X_test": root / "test" / "X_test.txt",
        "y_test": root / "test" / "y_test.txt",
    }


def download_har_dataset(data_root: str | Path) -> None:
    root = Path(data_root)
    paths = _har_dataset_paths(root)
    if all(path.is_file() for path in paths.values()):
        return

    root.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading UCI HAR dataset from {HAR_DATASET_URL}")
    with tempfile.TemporaryDirectory(prefix="uci-har-", dir=root.parent) as temp_dir:
        temp_root = Path(temp_dir)
        outer_archive = temp_root / "uci_har.zip"
        inner_archive = temp_root / "UCI HAR Dataset.zip"
        staged_root = temp_root / "staged"

        with urllib.request.urlopen(HAR_DATASET_URL, timeout=120) as response:
            with outer_archive.open("wb") as output:
                shutil.copyfileobj(response, output)

        with zipfile.ZipFile(outer_archive) as outer_zip:
            try:
                with outer_zip.open("UCI HAR Dataset.zip") as source:
                    with inner_archive.open("wb") as output:
                        shutil.copyfileobj(source, output)
            except KeyError as error:
                raise RuntimeError(
                    "UCI HAR outer archive does not contain the expected nested archive."
                ) from error

        with zipfile.ZipFile(inner_archive) as inner_zip:
            for destination in paths.values():
                relative_path = destination.relative_to(root)
                member = f"UCI HAR Dataset/{relative_path}"
                staged_path = staged_root / relative_path
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with inner_zip.open(member) as source:
                        with staged_path.open("wb") as output:
                            shutil.copyfileobj(source, output)
                except KeyError as error:
                    raise RuntimeError(f"UCI HAR inner archive is missing '{member}'.") from error

        for destination in paths.values():
            relative_path = destination.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_root / relative_path, destination)


def load_har_dataset(
    data_root: str | Path,
    *,
    download: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    paths = _har_dataset_paths(data_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing and download:
        download_har_dataset(data_root)
        missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"UCI HAR dataset files are missing: {missing}")

    X_train = torch.from_numpy(np.loadtxt(paths["X_train"], dtype=np.float32, ndmin=2))
    y_train = torch.from_numpy(np.loadtxt(paths["y_train"], dtype=np.int64, ndmin=1))
    X_test = torch.from_numpy(np.loadtxt(paths["X_test"], dtype=np.float32, ndmin=2))
    y_test = torch.from_numpy(np.loadtxt(paths["y_test"], dtype=np.int64, ndmin=1))

    if X_train.shape[0] != y_train.numel() or X_test.shape[0] != y_test.numel():
        raise ValueError("UCI HAR feature and label counts do not match.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("UCI HAR train and test feature dimensions do not match.")

    return X_train, y_train, X_test, y_test


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
