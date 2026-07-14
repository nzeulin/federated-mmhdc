from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly


CWRU_CLASS_MAPPING = {
    0: "normal",
    1: "ball_007",
    2: "ball_014",
    3: "ball_021",
    4: "inner_007",
    5: "inner_014",
    6: "inner_021",
    7: "outer_007",
    8: "outer_014",
    9: "outer_021",
}

_CWRU_CLASS_IDS = {
    ("normal", 0): 0,
    ("ball", 7): 1,
    ("ball", 14): 2,
    ("ball", 21): 3,
    ("inner", 7): 4,
    ("inner", 14): 5,
    ("inner", 21): 6,
    ("outer", 7): 7,
    ("outer", 14): 8,
    ("outer", 21): 9,
}

_CWRU_SUPPORTED_LOADS = {0, 1, 2, 3}
_CWRU_SUPPORTED_DIAMETERS = {7, 14, 21}
_CWRU_TARGET_SAMPLING_RATE = 12_000
_CWRU_CACHE_VERSION = 1


@dataclass(frozen=True)
class CWRURecording:
    recording_id: str
    relative_path: str
    fault_type: str
    fault_diameter: int
    motor_load: int
    rpm: float | None
    fault_bearing_location: str | None
    outer_race_position: str | None
    de_sampling_rate: int | None
    fe_sampling_rate: int | None
    has_de: bool
    has_fe: bool
    source_url: str | None


def load_cwru_dataset(
    raw_dir: str | Path,
    *,
    manifest_path: str | Path,
    cache_dir: str | Path,
    download: bool = True,
    sensor_channel: str = "DE",
    loads: tuple[int, ...] | list[int] = (1, 2, 3),
    fault_diameters: tuple[int, ...] | list[int] = (7, 14, 21),
    outer_race_position: str = "6",
    window_size: int = 100,
    train_candidate_stride: int = 1,
    test_stride: int | None = None,
    train_windows_per_group: int = 660,
    test_windows_per_group: int = 25,
    return_metadata: bool = False,
    return_class_mapping: bool = False,
    seed: int = 42,
) -> tuple[Any, ...]:
    """Prepare the manifest-selected CWRU RES-HD reconstruction.

    The source paper does not report its exact stride or split procedure. This
    adapter reserves a deterministic tail test region and samples training
    windows only from the preceding signal region.
    """
    if test_stride is None:
        test_stride = window_size

    resolved_config = _resolve_cwru_config(
        sensor_channel=sensor_channel,
        loads=loads,
        fault_diameters=fault_diameters,
        outer_race_position=outer_race_position,
        window_size=window_size,
        train_candidate_stride=train_candidate_stride,
        test_stride=test_stride,
        train_windows_per_group=train_windows_per_group,
        test_windows_per_group=test_windows_per_group,
        seed=seed,
    )

    manifest_path = Path(manifest_path)
    manifest_bytes = _read_required_file(manifest_path, "CWRU manifest")
    manifest_checksum = hashlib.sha256(manifest_bytes).hexdigest()
    recordings = _parse_cwru_manifest(manifest_bytes, manifest_path)
    selected = _select_cwru_recordings(recordings, resolved_config)

    cache_config = {
        **resolved_config,
        "manifest_checksum": manifest_checksum,
        "return_metadata": bool(return_metadata),
        "return_class_mapping": bool(return_class_mapping),
        "implementation_version": _CWRU_CACHE_VERSION,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_path = Path(cache_dir) / f"cwru_{cache_key}.pt"
    if cache_path.is_file():
        payload = _load_cwru_cache(cache_path, cache_config)
        return _format_cwru_output(payload, return_metadata, return_class_mapping)

    raw_root = Path(raw_dir)
    if download:
        _download_cwru_source_files(raw_root, recordings)
    _validate_cwru_source_files(raw_root, recordings)

    train_windows = []
    train_labels = []
    test_windows = []
    test_labels = []
    train_metadata: list[dict[str, Any]] = []
    test_metadata: list[dict[str, Any]] = []

    for recording in selected:
        source_path = raw_root / recording.relative_path
        signal, rpm = _load_cwru_signal(
            source_path,
            resolved_config["sensor_channel"],
            recording.rpm,
        )
        original_length = int(signal.size)
        original_rate = _cwru_sampling_rate(recording, resolved_config["sensor_channel"])
        signal, downsampled = _resample_cwru_signal(signal, original_rate)

        label_id = _cwru_label_id(recording)
        group_train, group_test, group_train_meta, group_test_meta = _window_cwru_recording(
            signal,
            recording,
            rpm=rpm,
            label_id=label_id,
            original_sampling_rate=original_rate,
            original_signal_length=original_length,
            downsampled=downsampled,
            config=resolved_config,
            include_metadata=return_metadata,
        )
        train_windows.append(group_train)
        test_windows.append(group_test)
        train_labels.append(
            np.full(resolved_config["train_windows_per_group"], label_id, dtype=np.int64)
        )
        test_labels.append(
            np.full(resolved_config["test_windows_per_group"], label_id, dtype=np.int64)
        )
        train_metadata.extend(group_train_meta)
        test_metadata.extend(group_test_meta)

    X_train = torch.from_numpy(np.concatenate(train_windows, axis=0))
    y_train = torch.from_numpy(np.concatenate(train_labels, axis=0))
    X_test = torch.from_numpy(np.concatenate(test_windows, axis=0))
    y_test = torch.from_numpy(np.concatenate(test_labels, axis=0))

    payload: dict[str, Any] = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "resolved_config": cache_config,
    }
    if return_metadata:
        payload["train_metadata"] = train_metadata
        payload["test_metadata"] = test_metadata
    if return_class_mapping:
        payload["class_mapping"] = dict(CWRU_CLASS_MAPPING)

    _save_cwru_cache(cache_path, payload)
    return _format_cwru_output(payload, return_metadata, return_class_mapping)


def download_cwru_dataset(
    raw_dir: str | Path,
    *,
    manifest_path: str | Path,
) -> None:
    """Download manifest-referenced CWRU recordings that are not cached locally."""
    manifest_path = Path(manifest_path)
    manifest_bytes = _read_required_file(manifest_path, "CWRU manifest")
    recordings = _parse_cwru_manifest(manifest_bytes, manifest_path)
    raw_root = Path(raw_dir)
    _download_cwru_source_files(raw_root, recordings)
    _validate_cwru_source_files(raw_root, recordings)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _resolve_cwru_config(
    *,
    sensor_channel: str,
    loads: tuple[int, ...] | list[int],
    fault_diameters: tuple[int, ...] | list[int],
    outer_race_position: str,
    window_size: int,
    train_candidate_stride: int,
    test_stride: int,
    train_windows_per_group: int,
    test_windows_per_group: int,
    seed: int,
) -> dict[str, Any]:
    channel = str(sensor_channel).upper()
    if channel not in {"DE", "FE"}:
        raise ValueError("sensor_channel must be either 'DE' or 'FE'.")

    selected_loads = tuple(int(load) for load in loads)
    if not selected_loads or len(set(selected_loads)) != len(selected_loads):
        raise ValueError("loads must be non-empty and contain unique values.")
    unsupported_loads = sorted(set(selected_loads) - _CWRU_SUPPORTED_LOADS)
    if unsupported_loads:
        raise ValueError(f"Unsupported CWRU motor loads: {unsupported_loads}.")

    selected_diameters = tuple(int(diameter) for diameter in fault_diameters)
    if not selected_diameters or len(set(selected_diameters)) != len(selected_diameters):
        raise ValueError("fault_diameters must be non-empty and contain unique values.")
    unsupported_diameters = sorted(set(selected_diameters) - _CWRU_SUPPORTED_DIAMETERS)
    if unsupported_diameters:
        raise ValueError(f"Unsupported CWRU fault diameters: {unsupported_diameters}.")

    position = _normalize_outer_race_position(outer_race_position)
    return {
        "sensor_channel": channel,
        "loads": sorted(selected_loads),
        "fault_diameters": sorted(selected_diameters),
        "outer_race_position": position,
        "window_size": _positive_int("window_size", window_size),
        "train_candidate_stride": _positive_int(
            "train_candidate_stride", train_candidate_stride
        ),
        "test_stride": _positive_int("test_stride", test_stride),
        "train_windows_per_group": _positive_int(
            "train_windows_per_group", train_windows_per_group
        ),
        "test_windows_per_group": _positive_int(
            "test_windows_per_group", test_windows_per_group
        ),
        "seed": int(seed),
    }


def _normalize_outer_race_position(position: str) -> str:
    normalized = str(position).lower().replace("o'clock", "").replace(":00", "").strip()
    if normalized not in {"3", "6", "12"}:
        raise ValueError("outer_race_position must be one of '3', '6', or '12'.")
    return normalized


def _read_required_file(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise FileNotFoundError(f"{description} is not readable: {path}") from error


def _parse_cwru_manifest(manifest_bytes: bytes, manifest_path: Path) -> list[CWRURecording]:
    try:
        document = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"CWRU manifest is not valid JSON: {manifest_path}") from error
    if not isinstance(document, dict) or not isinstance(document.get("recordings"), list):
        raise ValueError("CWRU manifest must contain a 'recordings' list.")

    recordings = [_parse_cwru_recording(item) for item in document["recordings"]]
    recording_ids = [recording.recording_id for recording in recordings]
    if len(set(recording_ids)) != len(recording_ids):
        raise ValueError("CWRU manifest recording IDs must be unique.")
    return recordings


def _parse_cwru_recording(item: Any) -> CWRURecording:
    required = {
        "recording_id",
        "path",
        "fault_type",
        "fault_diameter",
        "motor_load",
        "rpm",
        "fault_bearing_location",
        "outer_race_position",
        "de_sampling_rate",
        "fe_sampling_rate",
        "has_de",
        "has_fe",
    }
    if not isinstance(item, dict) or not required.issubset(item):
        missing = sorted(required - set(item)) if isinstance(item, dict) else sorted(required)
        raise ValueError(f"CWRU manifest entry is missing required fields: {missing}.")

    relative_path = Path(str(item["path"]))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"CWRU manifest path must be relative: {relative_path}")

    fault_type = str(item["fault_type"]).lower()
    if fault_type not in {"normal", "ball", "inner", "outer"}:
        raise ValueError(f"Unsupported CWRU fault type: {fault_type}")

    diameter = int(item["fault_diameter"])
    if fault_type == "normal" and diameter != 0:
        raise ValueError("CWRU normal recordings must have fault diameter zero.")
    if fault_type != "normal" and diameter not in _CWRU_SUPPORTED_DIAMETERS:
        raise ValueError(f"Unsupported CWRU fault diameter: {diameter}")

    motor_load = int(item["motor_load"])
    if motor_load not in _CWRU_SUPPORTED_LOADS:
        raise ValueError(f"Unsupported CWRU motor load: {motor_load}")
    if not isinstance(item["has_de"], bool) or not isinstance(item["has_fe"], bool):
        raise ValueError("CWRU manifest channel-availability fields must be booleans.")

    fault_bearing_location = item["fault_bearing_location"]
    if fault_type != "normal" and fault_bearing_location is None:
        raise ValueError("CWRU fault recordings must define a fault-bearing location.")

    outer_position = item["outer_race_position"]
    if fault_type == "outer":
        if outer_position is None:
            raise ValueError("CWRU outer-race recordings must define an outer-race position.")
        outer_position = _normalize_outer_race_position(str(outer_position))
    elif outer_position is not None:
        outer_position = str(outer_position)

    return CWRURecording(
        recording_id=str(item["recording_id"]),
        relative_path=relative_path.as_posix(),
        fault_type=fault_type,
        fault_diameter=diameter,
        motor_load=motor_load,
        rpm=None if item["rpm"] is None else float(item["rpm"]),
        fault_bearing_location=(
            None
            if fault_bearing_location is None
            else str(fault_bearing_location)
        ),
        outer_race_position=outer_position,
        de_sampling_rate=(
            None if item["de_sampling_rate"] is None else int(item["de_sampling_rate"])
        ),
        fe_sampling_rate=(
            None if item["fe_sampling_rate"] is None else int(item["fe_sampling_rate"])
        ),
        has_de=item["has_de"],
        has_fe=item["has_fe"],
        source_url=None if item.get("source_url") is None else str(item["source_url"]),
    )


def _select_cwru_recordings(
    recordings: list[CWRURecording],
    config: dict[str, Any],
) -> list[CWRURecording]:
    grouped: dict[tuple[str, int, int], list[CWRURecording]] = {}
    for recording in recordings:
        if recording.motor_load not in config["loads"]:
            continue
        if recording.fault_type == "normal":
            group_key = ("normal", 0, recording.motor_load)
        else:
            if recording.fault_diameter not in config["fault_diameters"]:
                continue
            if (
                recording.fault_type == "outer"
                and recording.outer_race_position != config["outer_race_position"]
            ):
                continue
            group_key = (
                recording.fault_type,
                recording.fault_diameter,
                recording.motor_load,
            )
        grouped.setdefault(group_key, []).append(recording)

    expected_groups = []
    for load in config["loads"]:
        expected_groups.append(("normal", 0, load))
        for fault_type in ("ball", "inner", "outer"):
            for diameter in config["fault_diameters"]:
                expected_groups.append((fault_type, diameter, load))

    selected = []
    for group in expected_groups:
        matches = grouped.get(group, [])
        if len(matches) != 1:
            raise ValueError(
                "CWRU manifest must select exactly one recording for "
                f"class/load group {group}; found {len(matches)}."
            )
        recording = matches[0]
        _validate_cwru_channel(recording, config["sensor_channel"])
        selected.append(recording)

    return sorted(
        selected,
        key=lambda recording: (_cwru_label_id(recording), recording.motor_load),
    )


def _validate_cwru_channel(recording: CWRURecording, sensor_channel: str) -> None:
    has_channel = recording.has_de if sensor_channel == "DE" else recording.has_fe
    if not has_channel:
        raise ValueError(
            f"CWRU recording '{recording.recording_id}' does not provide {sensor_channel} data."
        )
    sampling_rate = _cwru_sampling_rate(recording, sensor_channel)
    if sampling_rate not in {_CWRU_TARGET_SAMPLING_RATE, 48_000}:
        raise ValueError(
            f"CWRU recording '{recording.recording_id}' has unsupported "
            f"{sensor_channel} sampling rate {sampling_rate}."
        )


def _cwru_sampling_rate(recording: CWRURecording, sensor_channel: str) -> int:
    sampling_rate = (
        recording.de_sampling_rate if sensor_channel == "DE" else recording.fe_sampling_rate
    )
    if sampling_rate is None:
        raise ValueError(
            f"CWRU recording '{recording.recording_id}' has no known "
            f"{sensor_channel} sampling rate."
        )
    return sampling_rate


def _cwru_label_id(recording: CWRURecording) -> int:
    try:
        return _CWRU_CLASS_IDS[(recording.fault_type, recording.fault_diameter)]
    except KeyError as error:
        raise ValueError(
            "CWRU recording does not belong to the fixed ten-class task: "
            f"{recording.recording_id}."
        ) from error


def _validate_cwru_source_files(
    raw_root: Path,
    recordings: list[CWRURecording],
) -> None:
    missing = [recording for recording in recordings if not (raw_root / recording.relative_path).is_file()]
    if not missing:
        return
    details = [
        f"{recording.relative_path} ({recording.source_url or 'no source URL in manifest'})"
        for recording in missing
    ]
    raise FileNotFoundError(
        f"CWRU MATLAB files are missing from '{raw_root}': {details}"
    )


def _download_cwru_source_files(
    raw_root: Path,
    recordings: list[CWRURecording],
) -> None:
    missing = [
        recording
        for recording in recordings
        if not (raw_root / recording.relative_path).is_file()
    ]
    missing_urls = [recording.recording_id for recording in missing if not recording.source_url]
    if missing_urls:
        raise ValueError(
            "CWRU manifest entries are missing source URLs for recordings: "
            f"{missing_urls}."
        )

    for recording in missing:
        destination = raw_root / recording.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        print(f"Downloading CWRU recording {recording.recording_id} from {recording.source_url}")
        try:
            with urllib.request.urlopen(recording.source_url, timeout=120) as response:
                with temporary_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            os.replace(temporary_path, destination)
        except Exception as error:
            raise RuntimeError(
                f"Cannot download CWRU recording '{recording.recording_id}' "
                f"from {recording.source_url}."
            ) from error
        finally:
            temporary_path.unlink(missing_ok=True)


def _load_cwru_signal(
    source_path: Path,
    sensor_channel: str,
    manifest_rpm: float | None,
) -> tuple[np.ndarray, float | None]:
    try:
        contents = loadmat(source_path)
    except Exception as error:
        raise RuntimeError(f"Cannot read CWRU MATLAB file: {source_path}") from error

    suffix = f"_{sensor_channel}_time"
    signal_keys = [key for key in contents if key.endswith(suffix)]
    if len(signal_keys) != 1:
        raise ValueError(
            f"CWRU file '{source_path}' must contain exactly one variable ending "
            f"in '{suffix}'; found {signal_keys}."
        )

    signal = np.asarray(contents[signal_keys[0]], dtype=np.float64).reshape(-1)
    if signal.size == 0:
        raise ValueError(f"CWRU signal is empty in '{source_path}'.")
    if not np.isfinite(signal).all():
        raise ValueError(f"CWRU signal contains nonfinite values in '{source_path}'.")

    rpm_keys = [key for key in contents if key == "RPM" or key.endswith("_RPM")]
    if len(rpm_keys) > 1:
        raise ValueError(f"CWRU file '{source_path}' contains multiple RPM variables: {rpm_keys}.")
    if not rpm_keys:
        return signal, manifest_rpm

    rpm_values = np.asarray(contents[rpm_keys[0]], dtype=np.float64).reshape(-1)
    if rpm_values.size != 1 or not np.isfinite(rpm_values[0]):
        raise ValueError(f"CWRU RPM value is invalid in '{source_path}'.")
    return signal, float(rpm_values[0])


def _resample_cwru_signal(
    signal: np.ndarray,
    sampling_rate: int,
) -> tuple[np.ndarray, bool]:
    if sampling_rate == _CWRU_TARGET_SAMPLING_RATE:
        return signal, False
    if sampling_rate != 48_000:
        raise ValueError(f"Unsupported CWRU sampling rate: {sampling_rate}.")

    processed = resample_poly(signal, up=1, down=4)
    if processed.size == 0 or not np.isfinite(processed).all():
        raise ValueError("CWRU downsampling produced an empty or nonfinite signal.")
    return processed, True


def _cwru_recording_seed(
    seed: int,
    recording_id: str,
    sensor_channel: str,
    window_size: int,
) -> int:
    material = json.dumps(
        [int(seed), str(recording_id), str(sensor_channel), int(window_size)],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], byteorder="little")


def _window_cwru_recording(
    signal: np.ndarray,
    recording: CWRURecording,
    *,
    rpm: float | None,
    label_id: int,
    original_sampling_rate: int,
    original_signal_length: int,
    downsampled: bool,
    config: dict[str, Any],
    include_metadata: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    window_size = config["window_size"]
    test_quota = config["test_windows_per_group"]
    test_stride = config["test_stride"]
    test_region_length = window_size + (test_quota - 1) * test_stride
    test_region_start = int(signal.size) - test_region_length
    if test_region_start < window_size:
        raise ValueError(
            f"CWRU recording '{recording.recording_id}' is too short for the requested "
            f"train/test windows after resampling: {signal.size} samples."
        )

    train_starts = np.arange(
        0,
        test_region_start - window_size + 1,
        config["train_candidate_stride"],
        dtype=np.int64,
    )
    train_quota = config["train_windows_per_group"]
    if train_starts.size < train_quota:
        raise ValueError(
            f"CWRU recording '{recording.recording_id}' provides {train_starts.size} "
            f"training candidates, fewer than the requested {train_quota}."
        )

    generator = np.random.default_rng(
        _cwru_recording_seed(
            config["seed"],
            recording.recording_id,
            config["sensor_channel"],
            window_size,
        )
    )
    train_starts = np.sort(generator.choice(train_starts, size=train_quota, replace=False))
    test_starts = test_region_start + np.arange(test_quota, dtype=np.int64) * test_stride

    train_windows = np.stack(
        [signal[start : start + window_size] for start in train_starts]
    ).astype(np.float32, copy=False)
    test_windows = np.stack(
        [signal[start : start + window_size] for start in test_starts]
    ).astype(np.float32, copy=False)
    if not np.isfinite(train_windows).all() or not np.isfinite(test_windows).all():
        raise ValueError(f"CWRU windows contain nonfinite values for '{recording.recording_id}'.")

    if not include_metadata:
        return train_windows, test_windows, [], []

    common_metadata = {
        "recording_id": recording.recording_id,
        "source_file": recording.relative_path,
        "sensor_channel": config["sensor_channel"],
        "original_sampling_rate": original_sampling_rate,
        "processed_sampling_rate": _CWRU_TARGET_SAMPLING_RATE,
        "original_signal_length": original_signal_length,
        "processed_signal_length": int(signal.size),
        "downsampled": downsampled,
        "label_id": label_id,
        "label_name": CWRU_CLASS_MAPPING[label_id],
        "fault_type": recording.fault_type,
        "fault_diameter": recording.fault_diameter,
        "motor_load": recording.motor_load,
        "rpm": rpm,
        "fault_bearing_location": recording.fault_bearing_location,
        "outer_race_position": recording.outer_race_position,
    }
    train_metadata = [
        _cwru_window_metadata(common_metadata, "train", int(start), window_size)
        for start in train_starts
    ]
    test_metadata = [
        _cwru_window_metadata(common_metadata, "test", int(start), window_size)
        for start in test_starts
    ]
    return train_windows, test_windows, train_metadata, test_metadata


def _cwru_window_metadata(
    common_metadata: dict[str, Any],
    split: str,
    start: int,
    window_size: int,
) -> dict[str, Any]:
    return {
        "sample_id": f"{split}:{common_metadata['recording_id']}:{start}",
        "split": split,
        **common_metadata,
        "start_index": start,
        "end_index": start + window_size,
    }


def _load_cwru_cache(cache_path: Path, expected_config: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(f"Cannot read CWRU cache: {cache_path}") from error
    if not isinstance(payload, dict) or payload.get("resolved_config") != expected_config:
        raise RuntimeError(f"CWRU cache configuration is invalid: {cache_path}")
    return payload


def _save_cwru_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_path.name}.",
        dir=cache_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _format_cwru_output(
    payload: dict[str, Any],
    return_metadata: bool,
    return_class_mapping: bool,
) -> tuple[Any, ...]:
    tensors = (
        payload["X_train"],
        payload["y_train"],
        payload["X_test"],
        payload["y_test"],
    )
    if not return_metadata and not return_class_mapping:
        return tensors

    dataset_info: dict[str, Any] = {
        "preprocessing": dict(payload["resolved_config"]),
    }
    if return_metadata:
        dataset_info["train_metadata"] = payload["train_metadata"]
        dataset_info["test_metadata"] = payload["test_metadata"]
    if return_class_mapping:
        dataset_info["class_mapping"] = dict(payload["class_mapping"])
    return (*tensors, dataset_info)

