from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import RDITHCalibration, load_rdith_calibration


def build_progress_summary(
    heatmap: dict,
    poses: list,
    candidate_rois: list | None,
    residual_heatmap: dict,
    blobs: list,
    tracked_blobs: list,
    ml_dataset: dict | None,
    warnings: list[str],
    config: dict,
) -> dict:
    residual_frames = residual_heatmap.get("frames", [])
    calibration = config.get("_rdith_calibration")
    geometry_mode = ""
    if calibration is not None:
        geometry_mode = calibration.geometry_mode if isinstance(calibration, RDITHCalibration) else load_rdith_calibration(calibration).geometry_mode
    return {
        "num_heatmap_frames": int(np.asarray(heatmap.get("spectrum")).shape[0]) if heatmap.get("spectrum") is not None else 0,
        "num_pose_samples": len(poses or []),
        "num_aligned_pose_frames": int(np.asarray(heatmap.get("timestamps", [])).shape[0]),
        "num_roi_frames": 0 if candidate_rois is None else len(candidate_rois),
        "num_total_rois": 0 if candidate_rois is None else sum(len(frame) for frame in candidate_rois),
        "num_active_cells": int(sum(len(frame) for frame in config.get("_active_cells", []))),
        "num_residual_cells": int(sum(len(frame) for frame in residual_frames)),
        "num_blobs": int(sum(len(frame) for frame in blobs)),
        "num_tracked_blobs": int(sum(len(frame) for frame in tracked_blobs)),
        "num_feature_rows": 0 if ml_dataset is None else int(ml_dataset["X"].shape[0]),
        "feature_names": [] if ml_dataset is None else list(ml_dataset.get("feature_names", [])),
        "warnings": list(warnings),
        "stop_reasons": [],
        "geometry_mode": geometry_mode,
        "heatmap_type": str(heatmap.get("heatmap_type", "")),
        "axis_order": list(heatmap.get("axis_order", [])),
    }


def write_progress_summary(summary: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

