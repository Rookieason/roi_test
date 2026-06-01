from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data_types import CandidateROI, RFBlob, as_candidate_roi


def plot_world_frame(
    blobs: list[RFBlob],
    rois: list[CandidateROI],
    pose_kinematics_at_t: dict,
    roi_features: dict[str, dict[str, float]],
    output_path: str,
    config: dict,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    head = np.asarray(pose_kinematics_at_t.get("position_world", [0, 0, 0]), dtype=float)
    forward = _unit(np.asarray(pose_kinematics_at_t.get("head_forward_world", [0, 0, 1]), dtype=float))
    ax.scatter([head[0]], [head[2]], c="black", marker="x", s=80, label="HMD")
    ax.arrow(head[0], head[2], forward[0] * 0.5, forward[2] * 0.5, width=0.01, color="black")

    if blobs:
        energies = np.asarray([max(0.0, b.residual_energy) for b in blobs], dtype=float)
        sizes = 40.0 + 160.0 * energies / max(float(energies.max()), 1e-12)
        ax.scatter(
            [b.centroid_world[0] for b in blobs],
            [b.centroid_world[2] for b in blobs],
            c=energies,
            s=sizes,
            cmap="magma",
            label="RF blobs",
        )
    for roi_value in rois:
        roi = as_candidate_roi(roi_value)
        score = roi_features.get(roi.roi_id, {}).get("rf_residual_energy", 0.0)
        ax.scatter([roi.center_world[0]], [roi.center_world[2]], marker="s", s=60, label=f"ROI {roi.roi_id}")
        ax.text(roi.center_world[0], roi.center_world[2], f"{roi.roi_id}\n{score:.2g}", fontsize=8)
        if roi.bbox_min_world is not None and roi.bbox_max_world is not None:
            x0, z0 = roi.bbox_min_world[0], roi.bbox_min_world[2]
            w = roi.bbox_max_world[0] - roi.bbox_min_world[0]
            h = roi.bbox_max_world[2] - roi.bbox_min_world[2]
            ax.add_patch(plt.Rectangle((x0, z0), w, h, fill=False, linewidth=1.0))
    ax.set_xlabel("world x")
    ax.set_ylabel("world z")
    ax.set_title("RDITH world frame")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    _save(fig, output_path)


def plot_heatmap_comparison(
    standard_heatmap: dict,
    residual_heatmap_or_cells: dict | np.ndarray,
    frame_idx: int,
    output_path: str,
) -> None:
    standard = _project_heatmap_frame(np.asarray(standard_heatmap["spectrum"])[frame_idx])
    residual = _residual_frame_to_grid(residual_heatmap_or_cells, frame_idx, standard.shape)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(standard, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("standard")
    axes[1].imshow(residual, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("residual")
    for ax in axes:
        ax.set_xlabel("Doppler bin")
        ax.set_ylabel("ToF bin")
    fig.tight_layout()
    _save(fig, output_path)


def plot_roi_scores(
    roi_features_at_frame: dict[str, dict[str, float]],
    output_path: str,
    score_key: str = "rf_residual_energy",
    top_k: int = 20,
) -> None:
    items = sorted(
        [(roi_id, feats.get(score_key, 0.0)) for roi_id, feats in roi_features_at_frame.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]
    fig, ax = plt.subplots(figsize=(8, 4))
    if items:
        labels, values = zip(*items)
        ax.bar(labels, values)
        ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel(score_key)
    ax.set_title("ROI RF scores")
    fig.tight_layout()
    _save(fig, output_path)


def plot_feature_timeseries(
    exported_features: dict,
    feature_names: list[str],
    output_path: str,
    aggregate: str = "max_per_frame",
) -> None:
    x = np.asarray(exported_features.get("X", np.empty((0, len(feature_names)))), dtype=float)
    timestamps = np.asarray(exported_features.get("timestamps", np.arange(x.shape[0])), dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    keys = [
        "rf_residual_energy",
        "rf_motion_to_roi_alignment",
        "rf_time_to_contact_score",
        "rf_visibility_conflict",
        "rf_temporal_growth",
    ]
    for key in keys:
        if key not in feature_names or x.size == 0:
            continue
        values = x[:, feature_names.index(key)]
        ax.plot(timestamps, values, label=key)
    ax.set_xlabel("timestamp")
    ax.set_title("RDITH feature time series")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)


def plot_feature_matrix_health(ml_dataset: dict, output_path: str) -> dict:
    x = np.asarray(ml_dataset.get("X", np.empty((0, 0))), dtype=float)
    names = list(ml_dataset.get("feature_names", []))
    if x.size == 0:
        summary = {"nan_ratio": {}, "inf_ratio": {}, "zero_row_ratio": 0.0}
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No feature matrix", ha="center", va="center")
        _save(fig, output_path)
        return summary
    nan_ratio = np.mean(np.isnan(x), axis=0)
    inf_ratio = np.mean(np.isinf(x), axis=0)
    finite = np.where(np.isfinite(x), x, np.nan)
    summary = {
        "nan_ratio": dict(zip(names, nan_ratio.tolist())),
        "inf_ratio": dict(zip(names, inf_ratio.tolist())),
        "mean": dict(zip(names, _safe_nan_stat(finite, "mean").tolist())),
        "std": dict(zip(names, _safe_nan_stat(finite, "std").tolist())),
        "min": dict(zip(names, _safe_nan_stat(finite, "min").tolist())),
        "max": dict(zip(names, _safe_nan_stat(finite, "max").tolist())),
        "zero_row_ratio": float(np.mean(np.all(np.nan_to_num(x) == 0.0, axis=1))),
    }
    fig, ax = plt.subplots(figsize=(max(8, 0.25 * len(names)), 4))
    ax.bar(np.arange(len(names)) - 0.2, nan_ratio, width=0.4, label="NaN")
    ax.bar(np.arange(len(names)) + 0.2, inf_ratio, width=0.4, label="Inf")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Feature matrix health")
    fig.tight_layout()
    _save(fig, output_path)
    return summary


def write_visualization_report(output_dir: str, summary: dict, image_paths: list[str]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    links = "\n".join(f'<li><a href="{Path(p).name}">{Path(p).name}</a></li>' for p in image_paths)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>RDITH Debug Report</title></head>
<body><h1>RDITH Debug Report</h1><pre>{json.dumps(summary, indent=2, default=str)}</pre><ul>{links}</ul></body></html>
"""
    with open(out / "index.html", "w", encoding="utf-8") as f:
        f.write(html)


def _project_heatmap_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return np.nanmax(frame, axis=1)
    if frame.ndim == 4:
        return np.nanmax(frame, axis=(0, 2))
    return frame


def _residual_frame_to_grid(residual: dict | np.ndarray, frame_idx: int, shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=float)
    if isinstance(residual, dict) and residual.get("mode") == "sparse":
        for record in residual.get("frames", [])[frame_idx]:
            tau_idx, _, fd_idx = record["cell_index"][-3:]
            grid[tau_idx, fd_idx] += record.get("residual_energy", 0.0)
        return grid
    arr = np.asarray(residual)
    if arr.ndim == 2 and arr.shape[1] >= 8:
        for row in arr[arr[:, 0] == frame_idx]:
            grid[int(row[1]), int(row[3])] += row[5]
        return grid
    if arr.ndim >= 3:
        return _project_heatmap_frame(arr[frame_idx])
    return grid


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else np.zeros_like(value)


def _safe_nan_stat(values: np.ndarray, stat: str) -> np.ndarray:
    out = np.full(values.shape[1], np.nan, dtype=float)
    for idx in range(values.shape[1]):
        col = values[:, idx]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        if stat == "mean":
            out[idx] = float(np.mean(finite))
        elif stat == "std":
            out[idx] = float(np.std(finite))
        elif stat == "min":
            out[idx] = float(np.min(finite))
        elif stat == "max":
            out[idx] = float(np.max(finite))
    return out


def _save(fig, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
