#!/usr/bin/env python3
"""
heatmap_fix.py

Drop-in replacement for heatmap.py.

What this fixes
---------------
The original heatmap.py assumes CSI is already packed as:

    <data_path>/csi/csi_<exp_name>.npz

with an array named "csi" and shape:

    (timestamp, tx, rx, subcarrier)

Your sensor-agent capture program saves reconstructed CSI arrays as sidecar .npy
files under:

    <data_path>/<exp_name>/arrays/csi.rx.*/<timestamp>.npy

This script supports both formats:

1. Legacy NPZ:
       artifacts/csi/csi_<exp_name>.npz
       artifacts/<exp_name>/csi/csi_<exp_name>.npz
       artifacts/csi_<exp_name>.npz

2. New capture sidecar layout:
       artifacts/<exp_name>/arrays/csi.rx.*/*.npy

It assembles the sidecar files into CSI shape:

    (timestamp, tx, rx, subcarrier)

then runs the same ToF-Doppler / AoA-ToF-Doppler pipeline as your original file.
"""

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import savemat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from utilsforheatmap import (
    heatmap_setup,
    CSI_preprocessing,
    create_steering_matrix_ToF_Doppler,
    create_steering_matrix_F3D,
    smoothed_CSI,
    calculate_correlation_matrix,
    pipeline_3D,
    run_music_algorithm,
)
from plot_utils import generate_heatmaps
from datetime import datetime, timezone

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _natural_key(s: str):
    """Natural sort key: csi.rx.2 before csi.rx.10."""
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(s))]


def _topic_sort_key(path: Path):
    return _natural_key(path.name)


def _as_bool01(flag: bool) -> str:
    return "yes" if flag else "no"


def _expected_num_tx(config: dict) -> int:
    # Your config currently has num_tx at top level.
    return int(config.get("num_tx", config.get("heatmap_setting", {}).get("num_tx", 1)))


def _expected_num_rx(config: dict) -> int:
    # heatmap_setting.num_antenna is what utilsforheatmap validates against.
    return int(config["heatmap_setting"].get("num_antenna", config.get("num_rx", 1)))


def _expected_num_subcarriers(config: dict) -> int:
    return int(config["heatmap_setting"]["num_subcarriers"])


def _describe_array(a: np.ndarray) -> str:
    return f"shape={tuple(a.shape)}, dtype={a.dtype}"


def _timestamp_from_path(path: Path) -> str:
    """
    Convert CSI sidecar filename to a stable timestamp token.

    The capture subscriber saves arrays as e.g.:
        2026-01-17T08-36-42.347Z.npy

    We keep the stem exactly as the filename-safe timestamp token:
        2026-01-17T08-36-42.347Z
    """
    return path.stem


def _safe_timestamp_token(ts: Optional[str], fallback_index: int) -> str:
    """Return a filename-safe timestamp token."""
    if ts is None or str(ts).strip() == "":
        return f"no_timestamp_{fallback_index:06d}"
    token = str(ts).strip()
    # Your capture filenames already use '-' instead of ':'; keep that convention.
    token = token.replace(":", "-").replace("/", "-").replace("\\", "-").replace(" ", "T")
    return token

def _timestamp_token_to_unix_seconds(ts: str) -> float:
    """
    Convert CSI filename timestamp token to Unix epoch seconds.

    Example:
        2026-01-17T08-28-32.888Z
        2026-01-17T08:28:32.888Z
    """
    text = str(ts).strip()
    if text == "":
        raise ValueError("empty CSI timestamp token")

    try:
        return float(text)
    except ValueError:
        pass

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    # CSI filenames use '-' instead of ':' in the time field.
    # Convert only the HH-MM-SS part back to HH:MM:SS.
    if "T" in text:
        date_part, time_part = text.split("T", 1)
        if len(time_part) >= 8 and time_part[2] == "-" and time_part[5] == "-":
            time_part = time_part[:2] + ":" + time_part[3:5] + ":" + time_part[6:]
            text = date_part + "T" + time_part

    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()

def _decode_npz_timestamps(z) -> Optional[List[str]]:
    """Load timestamp arrays from a NPZ object if present."""
    for key in ("csi_timestamps_iso", "csi_timestamps", "timestamps", "timestamp"):
        if key in z:
            raw = z[key]
            out = []
            for x in raw.tolist():
                if isinstance(x, bytes):
                    out.append(x.decode("utf-8", errors="replace"))
                else:
                    out.append(str(x))
            return out
    return None


def _read_sidecar_reference_timestamps(args, exp_name: str) -> Optional[List[str]]:
    """
    Read timestamp tokens from the first CSI sidecar topic without loading arrays.
    Useful when an old cache has CSI but no timestamp metadata.
    """
    try:
        exp_dir = _find_experiment_dir(args.data_path, exp_name)
        arrays_dir = exp_dir / "arrays"
        topic_dirs = sorted(
            [p for p in arrays_dir.glob(args.csi_topic_glob) if p.is_dir()],
            key=_topic_sort_key,
        )
        if not topic_dirs:
            return None
        files = sorted(topic_dirs[0].glob("*.npy"), key=lambda p: _natural_key(p.name))
        if args.max_csi_files is not None and args.max_csi_files > 0:
            files = files[: args.max_csi_files]
        return [_timestamp_from_path(f) for f in files]
    except Exception:
        return None


# -----------------------------------------------------------------------------
# CSI loading and normalization
# -----------------------------------------------------------------------------

def _load_npz_csi(npz_path: Path) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Load legacy/cache csi_<exp>.npz. Prefer key 'csi'. If not present, use first key.
    Also load timestamp metadata when the cache contains it.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        timestamps = _decode_npz_timestamps(z)
        if "csi" in z:
            return z["csi"], timestamps
        keys = [k for k in z.keys() if k not in {"csi_timestamps_iso", "csi_timestamps", "timestamps", "timestamp"}]
        if not keys:
            raise ValueError(f"{npz_path} has no CSI array")
        print(f"[CSI] NPZ does not contain key 'csi'; using first array key: {keys[0]}")
        return z[keys[0]], timestamps


def _maybe_convert_real_imag_pair(arr: np.ndarray, config: dict, mode: str, source_name: str) -> np.ndarray:
    """
    Convert arrays with a trailing real/imag pair to complex.

    mode:
      - none: never convert
      - last: convert last axis if its length is 2
      - auto: convert only if last axis is 2 and some earlier axis looks like subcarrier
    """
    mode = (mode or "auto").lower()
    if mode not in {"auto", "last", "none"}:
        raise ValueError("complex_pair_mode must be one of: auto, last, none")

    if mode == "none" or np.iscomplexobj(arr):
        return arr

    if arr.ndim < 2 or arr.shape[-1] != 2:
        return arr

    expected_sc = _expected_num_subcarriers(config)
    should_convert = mode == "last" or any(dim == expected_sc for dim in arr.shape[:-1])

    if should_convert:
        print(f"[CSI] Converting trailing real/imag pair to complex for {source_name}: {_describe_array(arr)}")
        return arr[..., 0] + 1j * arr[..., 1]

    return arr


def _squeeze_redundant_singletons(arr: np.ndarray) -> np.ndarray:
    """
    Remove singleton dimensions after the timestamp/file axis when they only make
    shape inference harder. Do not remove axis 0 because it is time/file count.
    """
    if arr.ndim <= 4:
        return arr

    squeeze_axes = tuple(ax for ax in range(1, arr.ndim) if arr.shape[ax] == 1)
    if squeeze_axes:
        arr = np.squeeze(arr, axis=squeeze_axes)
    return arr


def _normalize_csi_shape(
    arr: np.ndarray,
    config: dict,
    source_name: str,
    complex_pair_mode: str = "auto",
) -> np.ndarray:
    """
    Normalize loaded CSI to shape:

        (time, tx, rx, subcarrier)

    Handles common source shapes after stacking files:

        (time, tx, rx, subcarrier)
        (time, rx, tx, subcarrier)
        (time, tx, subcarrier)          -> rx dimension added
        (time, rx, subcarrier)          -> tx dimension added
        (time, subcarrier)              -> tx/rx dimensions added
        (file_time, inner_time, tx, rx, subcarrier)

    Also optionally handles trailing real/imag pairs:

        (..., subcarrier, 2) -> complex(..., subcarrier)
    """
    arr = np.asarray(arr)
    arr = _maybe_convert_real_imag_pair(arr, config, complex_pair_mode, source_name)
    arr = _squeeze_redundant_singletons(arr)

    expected_tx = _expected_num_tx(config)
    expected_sc = _expected_num_subcarriers(config)

    # If each saved .npy already contains a short time sequence, flatten file-time
    # and inner-time into a single timestamp axis.
    if arr.ndim == 5:
        arr = arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
        arr = _squeeze_redundant_singletons(arr)

    if arr.ndim == 4:
        # Axes are [time, ?, ?, ?]. Detect subcarrier axis, tx axis, and rx axis.
        axes = [1, 2, 3]

        sc_candidates = [ax for ax in axes if arr.shape[ax] == expected_sc]
        if sc_candidates:
            sc_axis = sc_candidates[0]
        else:
            # Fallback: subcarrier is usually the largest non-time dimension.
            sc_axis = max(axes, key=lambda ax: arr.shape[ax])
            print(
                f"[WARN] Could not find expected subcarrier axis={expected_sc} in {source_name}; "
                f"using largest axis {sc_axis} with length {arr.shape[sc_axis]}."
            )

        tx_candidates = [ax for ax in axes if ax != sc_axis and arr.shape[ax] == expected_tx]
        if tx_candidates:
            tx_axis = tx_candidates[0]
        else:
            # Fallback: choose the smallest remaining axis as tx.
            remaining = [ax for ax in axes if ax != sc_axis]
            tx_axis = min(remaining, key=lambda ax: arr.shape[ax])
            print(
                f"[WARN] Could not find expected tx axis={expected_tx} in {source_name}; "
                f"using axis {tx_axis} with length {arr.shape[tx_axis]}."
            )

        rx_axis = [ax for ax in axes if ax not in (tx_axis, sc_axis)][0]
        arr = np.transpose(arr, (0, tx_axis, rx_axis, sc_axis))

    elif arr.ndim == 3:
        # Shape is likely (time, tx, subcarrier) or (time, rx, subcarrier).
        # Move the detected subcarrier axis to the end first.
        if arr.shape[-1] != expected_sc:
            sc_candidates = [ax for ax in [1, 2] if arr.shape[ax] == expected_sc]
            if sc_candidates:
                arr = np.moveaxis(arr, sc_candidates[0], -1)
            else:
                # Fallback: largest of axes 1/2 is subcarrier.
                sc_axis = max([1, 2], key=lambda ax: arr.shape[ax])
                arr = np.moveaxis(arr, sc_axis, -1)
                print(
                    f"[WARN] Could not find expected subcarrier axis={expected_sc} in {source_name}; "
                    f"using largest axis as subcarrier. New shape={arr.shape}."
                )

        if arr.shape[1] == expected_tx:
            # (time, tx, subcarrier) -> (time, tx, 1, subcarrier)
            arr = arr[:, :, None, :]
        else:
            # (time, rx, subcarrier) -> (time, 1, rx, subcarrier)
            arr = arr[:, None, :, :]

    elif arr.ndim == 2:
        # Shape is likely (time, subcarrier).
        if arr.shape[-1] != expected_sc:
            raise ValueError(
                f"Unsupported 2D CSI shape from {source_name}: {arr.shape}; "
                f"expected last dimension to be num_subcarriers={expected_sc}."
            )
        arr = arr[:, None, None, :]

    elif arr.ndim == 1:
        # Single CSI vector. Rare for legacy; useful for testing one frame.
        if arr.shape[0] != expected_sc:
            raise ValueError(
                f"Unsupported 1D CSI shape from {source_name}: {arr.shape}; "
                f"expected num_subcarriers={expected_sc}."
            )
        arr = arr[None, None, None, :]

    else:
        raise ValueError(
            f"Unsupported CSI shape from {source_name}: {arr.shape}. "
            "Expected 1D, 2D, 3D, 4D, or 5D array."
        )

    if arr.ndim != 4:
        raise ValueError(f"Failed to normalize CSI from {source_name}; got shape {arr.shape}")

    return arr


def _load_topic_stack(
    topic_dir: Path,
    config: dict,
    complex_pair_mode: str,
    max_files: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load one artifacts/<exp>/arrays/csi.rx.X directory and return its CSI stack plus
    timestamp tokens parsed from the .npy filenames.
    """
    files = sorted(topic_dir.glob("*.npy"), key=lambda p: _natural_key(p.name))
    if max_files is not None and max_files > 0:
        files = files[:max_files]

    if not files:
        raise FileNotFoundError(f"No .npy files in {topic_dir}")

    timestamps = [_timestamp_from_path(f) for f in files]

    print(f"[CSI] Loading {len(files)} files from {topic_dir}")

    frames = []
    first_shape = None
    first_dtype = None
    for f in files:
        a = np.load(f, allow_pickle=False)
        if first_shape is None:
            first_shape = a.shape
            first_dtype = a.dtype
        elif a.shape != first_shape:
            raise ValueError(
                f"Inconsistent CSI shapes in {topic_dir}: first={first_shape}, {f.name}={a.shape}"
            )
        frames.append(a)

    stacked = np.stack(frames, axis=0)
    print(f"[CSI] Topic {topic_dir.name}: raw stacked shape={stacked.shape}, dtype={first_dtype}")

    normalized = _normalize_csi_shape(
        stacked,
        config=config,
        source_name=str(topic_dir),
        complex_pair_mode=complex_pair_mode,
    )
    print(f"[CSI] Topic {topic_dir.name}: normalized shape={normalized.shape}, dtype={normalized.dtype}")

    # If each file expands to multiple time samples, duplicate or interpolate timestamp tokens.
    if normalized.shape[0] != len(timestamps):
        if normalized.shape[0] % len(timestamps) == 0:
            repeat = normalized.shape[0] // len(timestamps)
            timestamps = [ts for ts in timestamps for _ in range(repeat)]
        else:
            print(
                f"[WARN] Timestamp count {len(timestamps)} does not match normalized time axis "
                f"{normalized.shape[0]} for {topic_dir.name}; using index fallback for extra frames."
            )
            timestamps = [timestamps[min(i, len(timestamps) - 1)] if timestamps else f"sample_{i:06d}" for i in range(normalized.shape[0])]

    return normalized, timestamps


def _find_experiment_dir(data_path: str, exp_name: str) -> Path:
    """
    Support both:

      --data_path /.../artifacts
      --data_path /.../artifacts/<exp_name>
    """
    data_root = Path(data_path).expanduser().resolve()

    candidates: List[Path] = []
    if data_root.name == exp_name:
        candidates.append(data_root)
    candidates.append(data_root / exp_name)

    # Also support data_path already pointing at an arrays parent accidentally.
    if data_root.name == "arrays":
        candidates.append(data_root.parent)

    for c in candidates:
        if (c / "arrays").exists():
            return c

    raise FileNotFoundError(
        "Cannot find capture artifact arrays directory. Tried:\n"
        + "\n".join(str(c / "arrays") for c in candidates)
    )


def _load_sidecar_csi_from_artifacts(args, exp_name: str, config: dict) -> np.ndarray:
    """
    Load new capture layout:

        data_path/<exp_name>/arrays/csi.rx.*/timestamp.npy

    Multiple csi.rx.* directories are concatenated along RX axis after each
    topic is normalized to (time, tx, rx, subcarrier), unless each topic already
    contains a full RX array.
    """
    exp_dir = _find_experiment_dir(args.data_path, exp_name)
    arrays_dir = exp_dir / "arrays"

    topic_dirs = sorted(
        [p for p in arrays_dir.glob(args.csi_topic_glob) if p.is_dir()],
        key=_topic_sort_key,
    )

    if not topic_dirs:
        raise FileNotFoundError(
            f"No CSI topic directories found under {arrays_dir} with glob '{args.csi_topic_glob}'."
        )

    print(f"[CSI] Experiment directory: {exp_dir}")
    print(f"[CSI] Topic glob: {args.csi_topic_glob}")
    print("[CSI] Topic dirs: " + ", ".join(p.name for p in topic_dirs))

    topic_stacks: List[np.ndarray] = []
    topic_timestamps: List[List[str]] = []
    used_topic_dirs: List[Path] = []
    for topic_dir in topic_dirs:
        try:
            stack, timestamps = _load_topic_stack(
                topic_dir=topic_dir,
                config=config,
                complex_pair_mode=args.complex_pair_mode,
                max_files=args.max_csi_files,
            )
            topic_stacks.append(stack)
            topic_timestamps.append(timestamps)
            used_topic_dirs.append(topic_dir)
        except FileNotFoundError as e:
            print(f"[WARN] {e}; skipping")

    if not topic_stacks:
        raise FileNotFoundError(f"No usable CSI .npy files found in {arrays_dir}")

    # Align by sorted order and trim to the shortest topic length.
    min_t = min(s.shape[0] for s in topic_stacks)
    if any(s.shape[0] != min_t for s in topic_stacks):
        print(f"[WARN] CSI topic lengths differ; trimming all topics to {min_t} samples")
    topic_stacks = [s[:min_t] for s in topic_stacks]
    topic_timestamps = [ts[:min_t] for ts in topic_timestamps]

    # Use the first loaded CSI topic as the reference clock for heatmap filenames.
    # In your capture layout, each .npy filename is produced from the receive timestamp.
    reference_timestamps = topic_timestamps[0]
    print(f"[CSI] Timestamp reference topic: {used_topic_dirs[0].name}, count={len(reference_timestamps)}")

    expected_rx = _expected_num_rx(config)
    rx_counts = [s.shape[2] for s in topic_stacks]

    mode = args.rx_concat_mode.lower()
    if mode not in {"auto", "concat", "first"}:
        raise ValueError("--rx_concat_mode must be one of: auto, concat, first")

    if mode == "first":
        csi = topic_stacks[0]
        print(f"[CSI] rx_concat_mode=first; using topic {used_topic_dirs[0].name}")
    elif mode == "concat":
        csi = np.concatenate(topic_stacks, axis=2)
        print(f"[CSI] rx_concat_mode=concat; concatenated RX counts {rx_counts}")
    else:
        # auto mode:
        # - If every topic already has all expected RX antennas, avoid duplicating.
        # - Otherwise concatenate csi.rx.* along RX axis.
        if len(topic_stacks) > 1 and all(rx == expected_rx for rx in rx_counts):
            print(
                f"[WARN] Each CSI topic already has rx={expected_rx}. "
                f"Using first topic only: {used_topic_dirs[0].name}. "
                "Use --rx_concat_mode concat if this is not what you want."
            )
            csi = topic_stacks[0]
        else:
            csi = np.concatenate(topic_stacks, axis=2)
            print(f"[CSI] rx_concat_mode=auto; concatenated RX counts {rx_counts}")

    print(f"[CSI] Combined CSI shape: {csi.shape}, dtype={csi.dtype}  # (time, tx, rx, subcarrier)")
    return csi, reference_timestamps


def _cast_csi_for_cache(csi: np.ndarray, dtype_name: str) -> np.ndarray:
    dtype_name = (dtype_name or "none").lower()
    if dtype_name == "none":
        return csi
    if dtype_name == "complex64":
        return csi.astype(np.complex64, copy=False)
    if dtype_name == "complex128":
        return csi.astype(np.complex128, copy=False)
    if dtype_name == "float32":
        return csi.astype(np.float32, copy=False)
    if dtype_name == "float64":
        return csi.astype(np.float64, copy=False)
    raise ValueError("--csi_dtype must be one of: none, complex64, complex128, float32, float64")


def load_csi_for_experiment(args, exp_name: str, config: dict) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Main CSI loader.

    1. Try legacy NPZ.
    2. Try capture artifact sidecar NPY layout.
    3. Optionally cache assembled CSI as data_path/csi/csi_<exp>.npz.
    """
    data_root = Path(args.data_path).expanduser().resolve()

    legacy_candidates = [
        data_root / "csi" / f"csi_{exp_name}.npz",
        data_root / exp_name / "csi" / f"csi_{exp_name}.npz",
        data_root / f"csi_{exp_name}.npz",
    ]

    if args.force_rebuild_csi_cache:
        print("[CSI] --force_rebuild_csi_cache set; skipping legacy/cache NPZ lookup.")
    else:
        for p in legacy_candidates:
            if p.exists():
                print(f"[CSI] Loading legacy/cache NPZ: {p}")
                csi, timestamps = _load_npz_csi(p)
                csi = _normalize_csi_shape(
                    csi,
                    config=config,
                    source_name=str(p),
                    complex_pair_mode=args.complex_pair_mode,
                )
                csi = _cast_csi_for_cache(csi, args.csi_dtype)
                if timestamps is None or len(timestamps) != csi.shape[0]:
                    sidecar_ts = _read_sidecar_reference_timestamps(args, exp_name)
                    if sidecar_ts is not None and len(sidecar_ts) >= csi.shape[0]:
                        timestamps = sidecar_ts[: csi.shape[0]]
                        print(f"[CSI] Loaded timestamps from sidecar reference because cache had none/mismatch.")
                    else:
                        print("[WARN] No timestamp metadata available for cached CSI; filenames will use sample indices.")
                        timestamps = None
                print(f"[CSI] Loaded CSI shape: {csi.shape}, dtype={csi.dtype}")
                return csi, timestamps

    print("[CSI] Legacy/cache NPZ not found or skipped. Falling back to captured sidecar .npy artifacts.")
    csi, timestamps = _load_sidecar_csi_from_artifacts(args=args, exp_name=exp_name, config=config)
    csi = _cast_csi_for_cache(csi, args.csi_dtype)

    if not args.no_cache_csi:
        cache_dir = data_root / "csi"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"csi_{exp_name}.npz"
        print(f"[CSI] Caching assembled CSI to: {cache_file}")
        np.savez_compressed(cache_file, csi=csi, csi_timestamps_iso=np.asarray(timestamps, dtype=str))

    return csi, timestamps


def apply_antenna_order(csi: np.ndarray, config: dict, skip: bool = False) -> np.ndarray:
    """Apply config['antenna_order'] with explicit diagnostics."""
    if skip:
        print("[CSI] --skip_antenna_order set; not reordering RX antennas.")
        return csi

    order = config.get("antenna_order", None)
    if order is None:
        print("[CSI] No antenna_order in config; not reordering RX antennas.")
        return csi

    order = list(order)
    if not order:
        print("[CSI] Empty antenna_order in config; not reordering RX antennas.")
        return csi

    max_idx = max(order)
    if csi.shape[2] <= max_idx:
        raise ValueError(
            f"CSI has only {csi.shape[2]} RX antennas after loading, but config antenna_order "
            f"uses index {max_idx}: {order}. Loaded CSI shape is {csi.shape}.\n"
            "Check whether all csi.rx.* topics were captured and whether --rx_concat_mode is correct."
        )

    csi = csi[:, :, order, :]
    print(f"[CSI] Applied antenna_order={order}; shape now {csi.shape}")
    return csi


def load_config(args) -> dict:
    if args.config_path is not None:
        config_path = Path(args.config_path).expanduser().resolve()
    else:
        config_path = Path(__file__).resolve().parent / "config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"[CFG] Loaded config: {config_path}")
    print(
        f"[CFG] expected tx={_expected_num_tx(config)}, "
        f"rx={_expected_num_rx(config)}, "
        f"subcarriers={_expected_num_subcarriers(config)}"
    )
    return config



# -----------------------------------------------------------------------------
# Timestamp-aware, fixed-size heatmap image generation
# -----------------------------------------------------------------------------

def _compute_timestamp_indices(
    num_heatmaps: int,
    num_csi_timestamps: Optional[int],
    alignment: str = "center",
    explicit_offset: Optional[int] = None,
) -> List[int]:
    """
    Map each heatmap frame to a CSI timestamp index.

    When the heatmap pipeline outputs fewer frames than raw CSI files, the safest
    default is center alignment: for N CSI samples and M heatmaps, use offset
    floor((N-M)/2). For your example N=373 and M=351, heatmap 0 maps to CSI 11.
    """
    if num_csi_timestamps is None or num_csi_timestamps <= 0:
        return list(range(num_heatmaps))

    diff = num_csi_timestamps - num_heatmaps
    if explicit_offset is not None:
        offset = int(explicit_offset)
    else:
        alignment = (alignment or "center").lower()
        if alignment == "start":
            offset = 0
        elif alignment == "end":
            offset = max(0, diff)
        elif alignment == "center":
            offset = max(0, diff // 2)
        else:
            raise ValueError("--timestamp_alignment must be one of: center, start, end")

    indices = []
    for i in range(num_heatmaps):
        j = i + offset
        j = max(0, min(j, num_csi_timestamps - 1))
        indices.append(j)
    return indices


def _normalization_bounds(spectrum: np.ndarray, args) -> Tuple[Optional[float], Optional[float]]:
    mode = args.heatmap_normalization.lower()
    if mode == "per_frame":
        return None, None
    if mode == "global":
        return float(np.nanmin(spectrum)), float(np.nanmax(spectrum))
    if mode == "fixed":
        if args.heatmap_vmin is None or args.heatmap_vmax is None:
            raise ValueError("--heatmap_normalization fixed requires --heatmap_vmin and --heatmap_vmax")
        return float(args.heatmap_vmin), float(args.heatmap_vmax)
    raise ValueError("--heatmap_normalization must be one of: global, per_frame, fixed")


def _save_full_canvas_heatmap(
    data: np.ndarray,
    filepath: Path,
    width: int,
    height: int,
    cmap: str,
    interpolation: str,
    vmin: Optional[float],
    vmax: Optional[float],
    dpi: int = 100,
) -> None:
    """Save an exact-size PNG/JPG heatmap with no axes, title, legend, or border."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_axis_off()
    fig.savefig(filepath, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    # Enforce exact pixel size even if a backend changes rounding behavior.
    if filepath.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        with Image.open(filepath) as im:
            if im.size != (width, height):
                im = im.resize((width, height), resample=Image.Resampling.NEAREST)
                im.save(filepath)


def _save_plot_style_heatmap(
    data: np.ndarray,
    filepath: Path,
    heatmap_type: str,
    heatmap_setting,
    sample_idx: int,
    cmap: str,
    interpolation: str,
    vmin: Optional[float],
    vmax: Optional[float],
) -> None:
    """Optional old-style diagnostic plot with axes/colorbar/title."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if heatmap_type == "ToF-Doppler":
        x_axis_data = heatmap_setting.fd_axis
        y_axis_data = heatmap_setting.tau_axis
        x_label = "Doppler Shift (Hz)"
        y_label = "ToF (s)"
        title = f"ToF-Doppler Spectrum - Sample {sample_idx}"
    else:
        x_axis_data = np.arange(data.shape[1])
        y_axis_data = np.arange(data.shape[0])
        x_label = "X"
        y_label = "Y"
        title = f"{heatmap_type} - Sample {sample_idx}"

    plt.figure(figsize=(10, 8))
    im = plt.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[x_axis_data[0], x_axis_data[-1], y_axis_data[0], y_axis_data[-1]],
        cmap=cmap,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.colorbar(im, label="Power")
    plt.savefig(filepath, dpi=150)
    plt.close()


def generate_heatmaps(
    exp_name: str,
    heatmap_type: str,
    spectrum: np.ndarray,
    heatmap_setting,
    output_folder: str,
    args,
    csi_timestamps: Optional[Sequence[str]] = None,
    start_sample: Optional[int] = None,
    end_sample: Optional[int] = None,
    plot_gt: bool = False,
):
    """
    Timestamp-aware figure saver for ToF-Doppler heatmaps.

    Filenames use the mapped CSI timestamp:
        <csi_timestamp>__hm<heatmap_index>__csi<csi_index>.png

    A CSV manifest is always saved for synchronization with 6DoF/video streams.
    """
    if heatmap_type != "ToF-Doppler":
        # Keep original behavior for other heatmap types.
        return generate_heatmaps_plot(
            exp_name,
            heatmap_type,
            spectrum=spectrum,
            heatmap_setting=heatmap_setting,
            output_folder=output_folder,
            plot_gt=plot_gt,
        )

    spectrum = spectrum.reshape(
        spectrum.shape[0],
        heatmap_setting.tau_axis.shape[0],
        heatmap_setting.fd_axis.shape[0],
    )

    total_samples = spectrum.shape[0]
    if start_sample is None:
        start_sample = 0
    if end_sample is None:
        end_sample = total_samples
    else:
        end_sample = min(end_sample, total_samples)
    if start_sample < 0 or start_sample >= total_samples:
        raise ValueError(f"start_sample={start_sample} out of bounds for {total_samples} heatmaps")
    if end_sample <= start_sample:
        raise ValueError(f"end_sample={end_sample} must be greater than start_sample={start_sample}")

    output_path = Path(output_folder) / "heatmap_result" / "figures" / exp_name / heatmap_type
    output_path.mkdir(parents=True, exist_ok=True)

    vmin_global, vmax_global = _normalization_bounds(spectrum, args)
    timestamp_indices = _compute_timestamp_indices(
        num_heatmaps=total_samples,
        num_csi_timestamps=len(csi_timestamps) if csi_timestamps is not None else None,
        alignment=args.timestamp_alignment,
        explicit_offset=args.heatmap_timestamp_offset,
    )

    print(f"[OUT] Heatmap output folder: {output_path}")
    print(f"[OUT] CSI timestamps available: {len(csi_timestamps) if csi_timestamps is not None else 0}")
    print(f"[OUT] Heatmap frames: {total_samples}")
    if csi_timestamps is not None and len(csi_timestamps) != total_samples:
        used_offset = timestamp_indices[0] if timestamp_indices else 0
        print(
            f"[OUT] Timestamp count differs from heatmap count; using {args.timestamp_alignment} alignment "
            f"with offset={used_offset}."
        )
    print(f"[OUT] Render mode: {args.heatmap_render_mode}, fixed size: {args.heatmap_image_width}x{args.heatmap_image_height}")

    manifest_rows = []
    processed_count = 0
    failed_count = 0

    from tqdm import tqdm
    for sample_idx in tqdm(range(start_sample, end_sample), desc="Saving heatmaps", unit="frames"):
        try:
            current_data = spectrum[sample_idx, :, :]
            if args.heatmap_normalization == "per_frame":
                vmin = float(np.nanmin(current_data))
                vmax = float(np.nanmax(current_data))
            else:
                vmin, vmax = vmin_global, vmax_global

            csi_idx = timestamp_indices[sample_idx]
            csi_ts = csi_timestamps[csi_idx] if csi_timestamps is not None and csi_idx < len(csi_timestamps) else None
            ts_token = _safe_timestamp_token(csi_ts, csi_idx)

            filename = f"{ts_token}__hm{sample_idx:06d}__csi{csi_idx:06d}.{args.heatmap_save_format}"
            filepath = output_path / filename

            if args.heatmap_render_mode == "image":
                _save_full_canvas_heatmap(
                    current_data,
                    filepath=filepath,
                    width=args.heatmap_image_width,
                    height=args.heatmap_image_height,
                    cmap=args.heatmap_cmap,
                    interpolation=args.heatmap_interpolation,
                    vmin=vmin,
                    vmax=vmax,
                )
            elif args.heatmap_render_mode == "plot":
                _save_plot_style_heatmap(
                    current_data,
                    filepath=filepath,
                    heatmap_type=heatmap_type,
                    heatmap_setting=heatmap_setting,
                    sample_idx=sample_idx,
                    cmap=args.heatmap_cmap,
                    interpolation=args.heatmap_interpolation,
                    vmin=vmin,
                    vmax=vmax,
                )
            else:
                raise ValueError("--heatmap_render_mode must be image or plot")

            manifest_rows.append({
                "heatmap_index": sample_idx,
                "csi_index": csi_idx,
                "csi_timestamp": csi_ts or "",
                "filename": filename,
                "filepath": str(filepath),
                "normalization": args.heatmap_normalization,
                "vmin": vmin if vmin is not None else "",
                "vmax": vmax if vmax is not None else "",
                "width": args.heatmap_image_width if args.heatmap_render_mode == "image" else "",
                "height": args.heatmap_image_height if args.heatmap_render_mode == "image" else "",
            })
            processed_count += 1
        except Exception as e:
            print(f"[WARN] Heatmap sample {sample_idx} failed: {e}")
            failed_count += 1
            plt.close("all")

    # Always save a manifest; this is the robust way to republish synchronized streams.
    import csv
    manifest_path = output_path / "heatmap_timestamp_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "heatmap_index", "csi_index", "csi_timestamp", "filename", "filepath",
            "normalization", "vmin", "vmax", "width", "height",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata_path = output_path / "heatmap_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "exp_name": exp_name,
            "heatmap_type": heatmap_type,
            "total_heatmaps": int(total_samples),
            "total_csi_timestamps": int(len(csi_timestamps)) if csi_timestamps is not None else 0,
            "timestamp_alignment": args.timestamp_alignment,
            "heatmap_timestamp_offset": args.heatmap_timestamp_offset,
            "render_mode": args.heatmap_render_mode,
            "image_width": args.heatmap_image_width,
            "image_height": args.heatmap_image_height,
            "normalization": args.heatmap_normalization,
            "manifest": str(manifest_path),
            "processed": processed_count,
            "failed": failed_count,
        }, f, indent=2)

    print(f"[OUT] Saved {processed_count} heatmap images; failed={failed_count}")
    print(f"[OUT] Saved timestamp manifest: {manifest_path}")
    print(f"[OUT] Saved metadata: {metadata_path}")

    return {
        "processed": processed_count,
        "failed": failed_count,
        "output_folder": str(output_path),
        "manifest": str(manifest_path),
    }

# -----------------------------------------------------------------------------
# Heatmap generation pipeline
# -----------------------------------------------------------------------------

def heatmap_gen(args):
    config = load_config(args)

    for exp_name in args.exp_names:
        plot_gt = False

        print("----single human pose experiment in 2026----")
        print(f"Experiment: {exp_name}")

        # Load CSI as (timestamp, tx, rx, subcarrier).
        CSI, csi_timestamps = load_csi_for_experiment(args=args, exp_name=exp_name, config=config)
        CSI = apply_antenna_order(CSI, config=config, skip=args.skip_antenna_order)

        if args.max_samples is not None and args.max_samples > 0:
            old_t = CSI.shape[0]
            CSI = CSI[: args.max_samples]
            if csi_timestamps is not None:
                csi_timestamps = list(csi_timestamps)[: args.max_samples]
            print(f"[CSI] --max_samples={args.max_samples}; trimmed time axis {old_t} -> {CSI.shape[0]}")

        print(f"[CSI] Final CSI shape before preprocessing: {CSI.shape}, dtype={CSI.dtype}")

        # Prepare heatmap parameters.
        heatmap_setting = heatmap_setup(config)

        # Data preprocessing.
        CSI_mov = CSI_preprocessing(config, CSI, heatmap_setting)
        print(f"[CSI] CSI after preprocessing: {CSI_mov.shape}, dtype={CSI_mov.dtype}")

        # Free memory.
        del CSI
        gc.collect()

        # Heatmap pipeline.
        for heatmap_type in args.heatmap_type:
            print(f"Starting {heatmap_type} method:")

            if heatmap_type == "ToF-Doppler":
                steering_matrix_ToF_Doppler = create_steering_matrix_ToF_Doppler(heatmap_setting)

                # Output:
                # (timestamp, tx, steps_smooth_AoA, steps_smooth_ToF, window_rx, window_subcarrier)
                CSI_smoothed = smoothed_CSI(heatmap_type, heatmap_setting, CSI_mov)

                print("Calculating correlation matrix")
                R = calculate_correlation_matrix(CSI_smoothed, heatmap_type=heatmap_type)
                del CSI_smoothed
                gc.collect()

                spectrums = run_music_algorithm(R, steering_matrix_ToF_Doppler)
                del R, steering_matrix_ToF_Doppler
                gc.collect()

                if args.save_mat:
                    mat_path = Path(args.save_path) / "heatmap_result" / "mat" / exp_name / heatmap_type
                    mat_path.mkdir(parents=True, exist_ok=True)

                    idx = np.arange(spectrums.shape[0], dtype=np.int64)

                    spectrum_flat = spectrums.astype(np.float64)
                    spectrum_3d = spectrum_flat.reshape(
                        spectrum_flat.shape[0],
                        heatmap_setting.tau_axis.size,
                        heatmap_setting.fd_axis.size,
                    )

                    timestamp_indices = _compute_timestamp_indices(
                        num_heatmaps=spectrum_3d.shape[0],
                        num_csi_timestamps=len(csi_timestamps) if csi_timestamps is not None else None,
                        alignment=args.timestamp_alignment,
                        explicit_offset=args.heatmap_timestamp_offset,
                    )

                    if csi_timestamps is None:
                        raise ValueError(
                            "Cannot save RDITH heatmap timestamps because CSI timestamps are missing. "
                            "Run with --force_rebuild_csi_cache or check CSI sidecar filenames."
                        )

                    timestamps = np.asarray(
                        [
                            _timestamp_token_to_unix_seconds(csi_timestamps[csi_idx])
                            for csi_idx in timestamp_indices
                        ],
                        dtype=np.float64,
                    )

                    csi_timestamp_tokens = np.asarray(
                        [str(csi_timestamps[csi_idx]) for csi_idx in timestamp_indices],
                        dtype=object,
                    )

                    center_frequency = np.asarray([heatmap_setting.f_center], dtype=np.float64)

                    mat_file = mat_path / "smoothed_CSI_avg.mat"
                    savemat(
                        mat_file,
                        {
                            "idx": idx,
                            "csi_idx": np.asarray(timestamp_indices, dtype=np.int64),
                            "spectrum": spectrum_3d,
                            "timestamps": timestamps,
                            "csi_timestamp_tokens": csi_timestamp_tokens,
                            "tau_axis": heatmap_setting.tau_axis,
                            "fd_axis": heatmap_setting.fd_axis,
                            "center_frequency": center_frequency,
                        },
                    )
                    print(f"[OUT] Saved MAT: {mat_file}")

                    npz_file = mat_path / "smoothed_CSI_avg.npz"
                    np.savez_compressed(
                        npz_file,
                        spectrum=spectrum_3d,
                        timestamps=timestamps,
                        csi_idx=np.asarray(timestamp_indices, dtype=np.int64),
                        csi_timestamp_tokens=np.asarray(
                            [str(csi_timestamps[csi_idx]) for csi_idx in timestamp_indices],
                            dtype=str,
                        ),
                        tau_axis=heatmap_setting.tau_axis,
                        fd_axis=heatmap_setting.fd_axis,
                        center_frequency=center_frequency,
                    )
                    print(f"[OUT] Saved NPZ: {npz_file}")

                if args.save_fig:
                    generate_heatmaps(
                        exp_name,
                        heatmap_type,
                        spectrum=spectrums,
                        heatmap_setting=heatmap_setting,
                        output_folder=args.save_path,
                        args=args,
                        csi_timestamps=csi_timestamps,
                        start_sample=args.start_sample,
                        end_sample=args.end_sample,
                        plot_gt=plot_gt,
                    )

                del spectrums
                gc.collect()

            elif heatmap_type == "AoA-ToF-Doppler":
                steering_matrix_3D = create_steering_matrix_F3D(heatmap_setting)

                # Output:
                # (timestamp, tx, steps_smooth_AoA, steps_smooth_ToF,
                #  window_rx, window_subcarriers, window_sample)
                CSI_smoothed = smoothed_CSI(heatmap_type, heatmap_setting, CSI_mov)

                pipeline_3D(
                    exp_name,
                    args,
                    CSI_smoothed,
                    steering_matrix_3D,
                    heatmap_setting,
                    start_sample=args.start_sample,
                    end_sample=args.end_sample,
                    plot_gt=plot_gt,
                )

                del CSI_smoothed, steering_matrix_3D
                gc.collect()

            else:
                raise ValueError(
                    f"Unsupported heatmap_type '{heatmap_type}'. "
                    "Available: ToF-Doppler, AoA-ToF-Doppler"
                )

        del CSI_mov
        gc.collect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate CSI heatmaps from legacy NPZ or sensor-agent sidecar NPY artifacts."
    )

    # Data config.
    parser.add_argument(
        "--data_path",
        default="/home/tonic/guan125/1exp_data/20250512",
        help="Root data path. For your capture program, use /home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/artifacts",
    )
    parser.add_argument(
        "--save_path",
        default="/home/tonic/guan125/1exp_data/20250512",
        help="Output root for heatmap_result/figures and heatmap_result/mat.",
    )
    parser.add_argument(
        "--exp_names",
        "--exp_name",
        dest="exp_names",
        nargs="+",
        default=["csi_20250512-tof-2-2"],
        help="Experiment name(s). Alias --exp_name is supported for your current run script.",
    )
    parser.add_argument(
        "--config_path",
        default=None,
        help="Optional config.json path. Default: config.json next to this script.",
    )

    # Plot type setting.
    parser.add_argument(
        "--heatmap_type",
        nargs="+",
        default=["ToF-Doppler"],
        help="Available: ToF-Doppler AoA-ToF-Doppler",
    )
    parser.add_argument("--save_fig", dest="save_fig", action="store_true")
    parser.set_defaults(save_fig=False)
    parser.add_argument("--save_mat", dest="save_mat", action="store_true")
    parser.set_defaults(save_mat=False)

    # Sidecar capture loader options.
    parser.add_argument(
        "--csi_topic_glob",
        default="csi.rx.*",
        help="Topic directory glob under artifacts/<exp_name>/arrays/, e.g. csi.rx.* or csi.rx.4",
    )
    parser.add_argument(
        "--rx_concat_mode",
        default="auto",
        choices=["auto", "concat", "first"],
        help=(
            "How to combine multiple csi.rx.* topic directories. "
            "auto: concatenate unless each topic already has full RX count; "
            "concat: always concatenate along RX axis; first: use first topic only."
        ),
    )
    parser.add_argument(
        "--complex_pair_mode",
        default="auto",
        choices=["auto", "last", "none"],
        help="Convert trailing real/imag pair (..., 2) to complex. Default auto.",
    )
    parser.add_argument(
        "--csi_dtype",
        default="none",
        choices=["none", "complex64", "complex128", "float32", "float64"],
        help="Optional dtype cast after loading. Use complex64 to reduce cache size if your CSI is complex.",
    )
    parser.add_argument(
        "--max_csi_files",
        type=int,
        default=None,
        help="Debug only: load at most this many .npy files per CSI topic.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Debug only: keep at most this many timestamps after loading/assembling.",
    )
    parser.add_argument(
        "--skip_antenna_order",
        action="store_true",
        help="Do not apply config['antenna_order'].",
    )

    # Cache controls.
    parser.add_argument(
        "--no_cache_csi",
        action="store_true",
        help="Do not cache assembled sidecar CSI as data_path/csi/csi_<exp_name>.npz.",
    )
    parser.add_argument(
        "--force_rebuild_csi_cache",
        action="store_true",
        help="Ignore existing data_path/csi/csi_<exp_name>.npz and rebuild from sidecar .npy files.",
    )

    # Optional sample range for 3D pipeline. ToF-Doppler plotting still uses plot_utils defaults.
    parser.add_argument("--start_sample", type=int, default=None)
    parser.add_argument("--end_sample", type=int, default=None)

    # Timestamp-aware heatmap image saving.
    parser.add_argument(
        "--heatmap_render_mode",
        default="image",
        choices=["image", "plot"],
        help="image: full-canvas fixed-size heatmap; plot: diagnostic plot with axes/colorbar.",
    )
    parser.add_argument("--heatmap_image_width", type=int, default=640)
    parser.add_argument("--heatmap_image_height", type=int, default=480)
    parser.add_argument(
        "--heatmap_save_format",
        default="png",
        choices=["png", "jpg", "jpeg", "pdf"],
        help="Image output format. For synchronization use png.",
    )
    parser.add_argument("--heatmap_cmap", default="jet")
    parser.add_argument("--heatmap_interpolation", default="nearest")
    parser.add_argument(
        "--heatmap_normalization",
        default="global",
        choices=["global", "per_frame", "fixed"],
        help="global keeps color scale consistent across all heatmaps.",
    )
    parser.add_argument("--heatmap_vmin", type=float, default=None)
    parser.add_argument("--heatmap_vmax", type=float, default=None)
    parser.add_argument(
        "--timestamp_alignment",
        default="center",
        choices=["center", "start", "end"],
        help="How to map fewer heatmaps to more CSI timestamps. For 373 CSI and 351 heatmaps, center uses offset 11.",
    )
    parser.add_argument(
        "--heatmap_timestamp_offset",
        type=int,
        default=None,
        help="Manual CSI-index offset for heatmap 0. Overrides --timestamp_alignment.",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    try:
        heatmap_gen(args=args)
    except Exception as e:
        print("\n[ERROR] heatmap_fix.py failed:", file=sys.stderr)
        print(f"  {type(e).__name__}: {e}", file=sys.stderr)
        print("\n[HINT] Common checks:", file=sys.stderr)
        print("  1. Is --data_path pointing to the artifacts root?", file=sys.stderr)
        print("     /home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/artifacts", file=sys.stderr)
        print("  2. Does artifacts/<exp_name>/arrays/csi.rx.* contain .npy files?", file=sys.stderr)
        print("  3. Does config.json num_subcarriers match the captured CSI shape?", file=sys.stderr)
        print("  4. If RX count is wrong, try --rx_concat_mode concat or inspect the printed topic shapes.", file=sys.stderr)
        print("  5. If old cache has no timestamp metadata, run once with --force_rebuild_csi_cache.", file=sys.stderr)
        raise
