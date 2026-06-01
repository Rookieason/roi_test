from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .blob_extraction import extract_rf_blobs, track_rf_blobs
from .calibration import calibration_to_dict, load_rdith_calibration
from .data_types import CandidateROI, Pose6DoF
from .heatmap_adapter import generate_standard_heatmap_from_csi, load_heatmap_result
from .ml_export import export_rdith_features, export_residual_heatmap
from .pose_utils import align_pose_to_heatmap_timestamps, compute_head_kinematics
from .progress import build_progress_summary, write_progress_summary
from .residual_heatmap import (
    build_residual_heatmap,
    compute_residual_motion_cells,
    estimate_rf_velocity,
    extract_active_heatmap_cells,
    map_heatmap_cells_to_world,
)
from .roi_features import compute_global_intent_features, merge_features_for_ml, rf_roi_align
from .validation import validate_rdith_inputs


def run_rdith_pipeline(
    heatmap_path: Optional[str],
    raw_csi_path: Optional[str],
    pose_path: str,
    roi_path: Optional[str],
    calibration_path: Optional[str],
    config_path: str,
    output_path: str,
    heatmap_type: str = "AoA-ToF-Doppler",
    export_progress_summary: bool = True,
    save_intermediate_dir: Optional[str] = None,
) -> dict:
    config = _load_json(config_path)
    warnings: list[str] = []
    poses = _load_poses(pose_path)
    if heatmap_path:
        heatmap = load_heatmap_result(heatmap_path, heatmap_type)
    elif raw_csi_path:
        csi = np.load(raw_csi_path, allow_pickle=False)
        heatmap = generate_standard_heatmap_from_csi(csi, config, heatmap_type)
    else:
        raise ValueError("Need either heatmap_path or raw_csi_path")

    # RDITH no longer requires a separate antenna/Tx/Rx calibration file.
    # The existing heatmap generator owns antenna geometry.  calibration_path is
    # optional and should contain only world-frame alignment overrides such as
    # rf_origin_world / rf_rotation_world_from_rf.
    calibration_obj = load_rdith_calibration(calibration_path, heatmap_result=heatmap, config=config)
    calibration = calibration_to_dict(calibration_obj)
    if calibration_obj.geometry_mode == "tof_only_pseudo":
        warnings.append("Using tof_only_pseudo geometry: RF positions are pseudo-positions, not exact 3D cells.")
    if calibration_path is None:
        warnings.append("No calibration_path provided; using config/heatmap metadata plus default RF world origin/rotation.")

    candidate_rois = _load_rois(roi_path) if roi_path else None
    validate_rdith_inputs(heatmap, poses, candidate_rois, calibration)
    aligned_poses = align_pose_to_heatmap_timestamps(poses, heatmap["timestamps"])
    kinematics = compute_head_kinematics(aligned_poses, smooth_window=int(config.get("rdith_smooth_window", 3)))
    active_cells = extract_active_heatmap_cells(heatmap, **config.get("active_cell_config", {}))
    cell_map = map_heatmap_cells_to_world(heatmap, calibration)
    rf_motion_cells = estimate_rf_velocity(active_cells, cell_map, heatmap, calibration)
    residual_cells = compute_residual_motion_cells(rf_motion_cells, kinematics, calibration)
    residual_heatmap = build_residual_heatmap(heatmap, residual_cells, output_mode="sparse")
    blobs = extract_rf_blobs(residual_heatmap, **config.get("blob_config", {}))
    tracked_blobs = track_rf_blobs(blobs, **config.get("tracking_config", {}))
    global_features = compute_global_intent_features(tracked_blobs, kinematics, config.get("sector_config", {}))

    ml_dataset = None
    if candidate_rois is not None:
        roi_rf_features = rf_roi_align(tracked_blobs, candidate_rois, kinematics, config.get("roi_feature_config", {}))
        ml_dataset = merge_features_for_ml(candidate_rois, roi_rf_features, global_features)
        export_rdith_features(ml_dataset, output_path, format=Path(output_path).suffix.lstrip(".") or "npz")
    else:
        export_residual_heatmap(residual_heatmap, global_features, output_path)

    progress_config = dict(config)
    progress_config["_active_cells"] = active_cells
    progress_config["_rdith_calibration"] = calibration
    summary = build_progress_summary(
        heatmap,
        poses,
        candidate_rois,
        residual_heatmap,
        blobs,
        tracked_blobs,
        ml_dataset,
        warnings,
        progress_config,
    )
    if export_progress_summary:
        summary_path = str(Path(output_path).with_name("rdith_progress_summary.json"))
        write_progress_summary(summary, summary_path)
    if save_intermediate_dir:
        _save_intermediates(save_intermediate_dir, active_cells, residual_cells, blobs, ml_dataset, summary)

    return {
        "heatmap": heatmap,
        "residual_heatmap": residual_heatmap,
        "tracked_blobs": tracked_blobs,
        "global_features": global_features,
        "ml_dataset": ml_dataset,
        "warnings": warnings,
        "progress_summary": summary,
    }


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_poses(path: str) -> list[Pose6DoF | dict]:
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("poses", data) if isinstance(data, dict) else data
        if p.suffix.lower() == ".csv":
            return _load_poses_from_csv(p)
    if p.is_dir():
        csv_files = sorted([*p.glob("*.csv")])
        if csv_files:
            poses: list[Pose6DoF | dict] = []
            for csv_file in csv_files:
                poses.extend(_load_poses_from_csv(csv_file))
            return poses
    raise ValueError(f"Unsupported pose_path: {path}")


def _load_poses_from_csv(path: Path) -> list[Pose6DoF | dict]:
    poses: list[Pose6DoF | dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pose = _parse_pose_csv_row(row)
            if pose is not None:
                poses.append(pose)
    return poses


def _parse_pose_csv_row(row: dict[str, str]) -> dict[str, object] | None:
    payload_json = row.get("payload_json") or row.get("payload")
    payload: dict[str, object] | None = None
    if payload_json:
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            payload = None
    if payload is None:
        payload = {k: v for k, v in row.items() if v is not None and v != ""}

    ts_utc = payload.get("ts_utc") or row.get("ts_utc")
    if ts_utc is None or ts_utc == "":
        recv_unix = row.get("recv_unix")
        if recv_unix:
            try:
                return {
                    "timestamp": float(recv_unix),
                    "position_world": _parse_float_list(payload.get("pos") or row.get("pos")),
                    "rotation_world_from_head": _rotation_matrix_from_yaw_pitch_roll_deg(
                        _parse_float_list(payload.get("rot_deg") or payload.get("rot") or row.get("rot_deg") or row.get("rot"))
                    ),
                }
            except Exception:
                return None

    try:
        timestamp = _parse_iso_timestamp(str(ts_utc))
    except ValueError:
        return None

    position = _parse_float_list(payload.get("pos") or payload.get("position") or row.get("pos"))
    rotation_values = _parse_float_list(payload.get("rot_deg") or payload.get("rot") or row.get("rot_deg") or row.get("rot"))
    if position is None or rotation_values is None:
        return None
    return {
        "timestamp": timestamp,
        "position_world": position,
        "rotation_world_from_head": _rotation_matrix_from_yaw_pitch_roll_deg(rotation_values),
    }


def _parse_float_list(value: object | None) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            return [float(x) for x in json.loads(text)]
        except Exception:
            pass
    if "," in text:
        try:
            return [float(x.strip()) for x in text.split(",") if x.strip()]
        except Exception:
            pass
    try:
        return [float(text)]
    except Exception:
        return None


def _parse_iso_timestamp(value: str) -> float:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError as exc:
        raise ValueError(f"Unsupported timestamp format: {value}") from exc


def _rotation_matrix_from_yaw_pitch_roll_deg(yaw_pitch_roll: list[float]) -> list[list[float]]:
    if len(yaw_pitch_roll) != 3:
        raise ValueError("rot_deg must be a list of three values [yaw,pitch,roll]")
    yaw, pitch, roll = [float(x) for x in yaw_pitch_roll]
    cy = np.cos(np.deg2rad(yaw))
    sy = np.sin(np.deg2rad(yaw))
    cp = np.cos(np.deg2rad(pitch))
    sp = np.sin(np.deg2rad(pitch))
    cr = np.cos(np.deg2rad(roll))
    sr = np.sin(np.deg2rad(roll))
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    return (rz @ ry @ rx).tolist()


def _load_rois(path: str) -> list[list[CandidateROI | dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("frames", data) if isinstance(data, dict) else data


def _serialize_blob(blob):
    return {
        "blob_id": int(blob.blob_id),
        "frame_idx": int(blob.frame_idx),
        "timestamp": float(blob.timestamp),
        "centroid_grid": [float(blob.centroid_grid[0]), float(blob.centroid_grid[1])],
        "bbox_grid": [
            int(blob.bbox_grid[0]),
            int(blob.bbox_grid[1]),
            int(blob.bbox_grid[2]),
            int(blob.bbox_grid[3]),
        ],
        "centroid_world": np.asarray(blob.centroid_world, dtype=float).tolist(),
        "velocity_world": (
            None
            if blob.velocity_world is None
            else np.asarray(blob.velocity_world, dtype=float).tolist()
        ),
        "energy": float(blob.energy),
        "residual_energy": float(blob.residual_energy),
        "doppler_mean_hz": float(blob.doppler_mean_hz),
        "doppler_bandwidth_hz": float(blob.doppler_bandwidth_hz),
        "doppler_entropy": float(blob.doppler_entropy),
        "confidence": float(blob.confidence),
        "num_supporting_cells": int(blob.num_supporting_cells),
        "lifetime": int(blob.lifetime),
        "metadata": blob.metadata,
    }

def _save_intermediates(
    output_dir: str,
    active_cells: list[np.ndarray],
    residual_cells: list[list[dict]],
    blobs: list[list],
    ml_dataset: Optional[dict],
    summary: dict,
) -> None:
    import csv

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "active_cells.npz", frames=np.asarray(active_cells, dtype=object))
    np.savez_compressed(out / "residual_cells.npz", frames=np.asarray(residual_cells, dtype=object))
    with open(out / "blobs.json", "w", encoding="utf-8") as f:
        json.dump(
            [[_serialize_blob(blob) for blob in frame] for frame in blobs],
            f,
            default=str,
            indent=2,
        )
    if ml_dataset is not None:
        with open(out / "roi_features.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "roi_id", *ml_dataset["feature_names"]])
            for timestamp, roi_id, row in zip(ml_dataset["timestamps"], ml_dataset["roi_ids"], ml_dataset["X"]):
                writer.writerow([timestamp, roi_id, *row.tolist()])
    write_progress_summary(summary, str(out / "progress_summary.json"))
