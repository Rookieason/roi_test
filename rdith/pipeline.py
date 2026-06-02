from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .blob_extraction import extract_rf_blobs, track_rf_blobs
from .calibration import calibration_to_dict, load_rdith_calibration
from .data_types import CandidateROI, Pose6DoF, as_pose6dof
from .heatmap_adapter import generate_standard_heatmap_from_csi, load_heatmap_result
from .ml_export import export_rdith_features, export_residual_heatmap
from .pose_utils import align_pose_to_heatmap_timestamps, compute_head_kinematics
from .progress import build_progress_summary, write_progress_summary
from .residual_heatmap import (
    build_residual_heatmap,
    compute_residual_motion_cells,
    estimate_rf_velocity,
    extract_active_heatmap_cells,
    filter_residual_motion_cells,
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
    verbose: bool = False,
) -> dict:
    config = _load_json(config_path)
    warnings: list[str] = []
    if verbose:
        _print_header("RDITH input paths")
        _print_kv("heatmap_path", heatmap_path)
        _print_kv("raw_csi_path", raw_csi_path)
        _print_kv("pose_path", pose_path)
        _print_kv("roi_path", roi_path)
        _print_kv("calibration_path", calibration_path)
        _print_kv("config_path", config_path)
        _print_kv("output_path", output_path)
        _print_kv("heatmap_type", heatmap_type)
    poses = _load_poses(pose_path)
    if heatmap_path:
        heatmap = load_heatmap_result(heatmap_path, heatmap_type)
    elif raw_csi_path:
        csi = np.load(raw_csi_path, allow_pickle=False)
        heatmap = generate_standard_heatmap_from_csi(csi, config, heatmap_type)
    else:
        raise ValueError("Need either heatmap_path or raw_csi_path")
    if verbose:
        _print_heatmap_diagnostics(heatmap)
        _print_pose_diagnostics(poses, np.asarray(heatmap["timestamps"], dtype=float))

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
    if verbose:
        _print_calibration_diagnostics(calibration)

    candidate_rois = _load_rois(roi_path) if roi_path else None
    validate_rdith_inputs(heatmap, poses, candidate_rois, calibration)
    aligned_poses = align_pose_to_heatmap_timestamps(poses, heatmap["timestamps"])
    kinematics = compute_head_kinematics(aligned_poses, smooth_window=int(config.get("rdith_smooth_window", 3)))
    if verbose:
        _print_kinematics_diagnostics(kinematics)
    active_cells = extract_active_heatmap_cells(heatmap, **config.get("active_cell_config", {}))
    if verbose:
        _print_cell_count_diagnostics("active cells", active_cells)
    cell_map = map_heatmap_cells_to_world(heatmap, calibration)
    if verbose:
        _print_cell_map_diagnostics(cell_map)
    rf_motion_cells = estimate_rf_velocity(active_cells, cell_map, heatmap, calibration)
    if verbose:
        _print_rf_motion_diagnostics(rf_motion_cells)
    residual_cells_all = compute_residual_motion_cells(rf_motion_cells, kinematics, calibration)
    if verbose:
        _print_residual_diagnostics("residual before filtering", residual_cells_all)
    residual_cells = filter_residual_motion_cells(
        residual_cells_all,
        **config.get("residual_cell_config", {}),
    )
    _print_active_residual_difference(active_cells, residual_cells, residual_cells_all)
    if verbose:
        _print_filter_diagnostics(residual_cells_all, residual_cells, config.get("residual_cell_config", {}))
        _print_residual_diagnostics("residual after filtering", residual_cells)
    residual_heatmap = build_residual_heatmap(heatmap, residual_cells, output_mode="sparse")
    blobs = extract_rf_blobs(residual_heatmap, **config.get("blob_config", {}))
    tracked_blobs = track_rf_blobs(blobs, **config.get("tracking_config", {}))
    global_features = compute_global_intent_features(tracked_blobs, kinematics, config.get("sector_config", {}))
    if verbose:
        _print_blob_diagnostics(blobs, tracked_blobs)
        _print_global_feature_diagnostics(global_features, kinematics)

    ml_dataset = None
    if candidate_rois is not None:
        if verbose:
            _print_roi_diagnostics(candidate_rois)
        roi_rf_features = rf_roi_align(tracked_blobs, candidate_rois, kinematics, config.get("roi_feature_config", {}))
        ml_dataset = merge_features_for_ml(candidate_rois, roi_rf_features, global_features)
        if verbose:
            _print_ml_diagnostics(ml_dataset)
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
    if verbose:
        _print_summary_diagnostics(summary, warnings)

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


def _print_header(title: str) -> None:
    print(f"\n[RDITH] {title}")


def _print_kv(key: str, value) -> None:
    print(f"  {key}: {value}")


def _finite_stats(values) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite": 0, "min": None, "p50": None, "mean": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "p50": float(np.percentile(finite, 50)),
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _print_stats(label: str, values) -> None:
    stats = _finite_stats(values)
    print(
        f"  {label}: count={stats['count']} finite={stats['finite']} "
        f"min={stats['min']} p50={stats['p50']} mean={stats['mean']} "
        f"p95={stats['p95']} max={stats['max']}"
    )


def _print_heatmap_diagnostics(heatmap: dict) -> None:
    _print_header("heatmap diagnostics")
    spectrum = np.asarray(heatmap["spectrum"])
    _print_kv("spectrum_shape", tuple(spectrum.shape))
    _print_kv("axis_order", heatmap.get("axis_order"))
    _print_kv("heatmap_type", heatmap.get("heatmap_type"))
    _print_kv("source_path", heatmap.get("metadata", {}).get("source_path"))
    _print_kv("metadata", heatmap.get("metadata", {}))
    _print_axis_diagnostics("timestamps", heatmap.get("timestamps"))
    _print_axis_diagnostics("tau_axis", heatmap.get("tau_axis"))
    _print_axis_diagnostics("fd_axis", heatmap.get("fd_axis"))
    _print_axis_diagnostics("theta_deg_axis", heatmap.get("theta_deg_axis"))
    _print_stats("spectrum_energy", spectrum)


def _print_axis_diagnostics(name: str, axis) -> None:
    if axis is None:
        _print_kv(name, None)
        return
    arr = np.asarray(axis, dtype=float).reshape(-1)
    if arr.size == 0:
        _print_kv(name, "empty")
        return
    step = float(np.median(np.diff(arr))) if arr.size > 1 else None
    _print_kv(name, f"len={arr.size} first={arr[0]} last={arr[-1]} median_step={step}")


def _print_pose_diagnostics(poses: list, heatmap_timestamps: np.ndarray) -> None:
    _print_header("pose/timestamp diagnostics")
    pose_list = [as_pose6dof(p) for p in poses]
    pose_ts = np.asarray([p.timestamp for p in pose_list], dtype=float)
    _print_kv("pose_count", len(pose_list))
    if pose_ts.size:
        _print_kv("pose_time_range", f"{pose_ts.min()} -> {pose_ts.max()}")
        if pose_ts.size > 1:
            _print_stats("pose_dt", np.diff(np.sort(pose_ts)))
    _print_kv("heatmap_time_range", f"{heatmap_timestamps.min()} -> {heatmap_timestamps.max()}")
    overlap_start = max(float(pose_ts.min()), float(heatmap_timestamps.min()))
    overlap_end = min(float(pose_ts.max()), float(heatmap_timestamps.max()))
    _print_kv("timestamp_overlap", f"{overlap_start} -> {overlap_end} duration={overlap_end - overlap_start}")
    nearest_deltas = [float(np.min(np.abs(pose_ts - t))) for t in heatmap_timestamps] if pose_ts.size else []
    _print_stats("nearest_pose_delta_for_heatmap_frames", nearest_deltas)


def _print_calibration_diagnostics(calibration: dict) -> None:
    _print_header("calibration diagnostics")
    for key in [
        "geometry_mode",
        "center_frequency",
        "wavelength",
        "rf_origin_world",
        "rf_rotation_world_from_rf",
        "tof_range_scale",
        "range_offset_m",
        "confidence",
    ]:
        _print_kv(key, calibration.get(key))


def _print_kinematics_diagnostics(kinematics: dict) -> None:
    _print_header("6DoF kinematics diagnostics")
    linear = np.asarray(kinematics["linear_velocity_world"], dtype=float)
    angular = np.asarray(kinematics["angular_velocity_world"], dtype=float)
    _print_stats("head_speed_mps", np.linalg.norm(linear, axis=1))
    _print_stats("head_angular_speed_radps", np.linalg.norm(angular, axis=1))
    _print_kv("first_position_world", np.asarray(kinematics["position_world"][0], dtype=float).tolist())
    _print_kv("last_position_world", np.asarray(kinematics["position_world"][-1], dtype=float).tolist())
    _print_kv("first_head_forward_world", np.asarray(kinematics["head_forward_world"][0], dtype=float).tolist())


def _frame_lengths(frames: list) -> np.ndarray:
    return np.asarray([len(frame) for frame in frames], dtype=float)


def _flatten_records(frames: list[list[dict]]) -> list[dict]:
    return [record for frame in frames for record in frame]


def _print_active_residual_difference(
    active_cells: list[np.ndarray],
    residual_cells: list[list[dict]],
    residual_cells_all: list[list[dict]],
) -> None:
    active_by_key: dict[tuple[int, int, int, int], float] = {}
    for fallback_frame_idx, frame_cells in enumerate(active_cells):
        for row in np.asarray(frame_cells):
            frame_idx = int(row[0]) if row.size > 0 else fallback_frame_idx
            tau_idx = int(row[1])
            theta_idx = int(row[2])
            fd_idx = int(row[3])
            active_by_key[(frame_idx, tau_idx, theta_idx, fd_idx)] = float(row[4])

    residual_by_key: dict[tuple[int, int, int, int], float] = {}
    for fallback_frame_idx, records in enumerate(residual_cells):
        for record in records:
            tau_idx, theta_idx, fd_idx = record["cell_index"]
            frame_idx = int(record.get("frame_idx", fallback_frame_idx))
            residual_by_key[(frame_idx, int(tau_idx), int(theta_idx), int(fd_idx))] = float(record.get("residual_energy", 0.0))

    active_values = np.asarray(list(active_by_key.values()), dtype=float)
    residual_values = np.asarray([residual_by_key.get(key, 0.0) for key in active_by_key], dtype=float)
    diff = residual_values - active_values
    scales = []
    for records in residual_cells_all:
        for record in records:
            energy = float(record.get("energy", 0.0))
            scales.append(float(record.get("residual_energy", 0.0)) / max(energy, 1e-12))
    scales_arr = np.asarray(scales, dtype=float)

    print("\n[RDITH] active vs residual heatmap difference")
    print(f"  active_cells: {len(active_by_key)}")
    print(f"  residual_cells_after_filter: {len(residual_by_key)}")
    if active_values.size == 0:
        print("  diff: no active cells to compare")
        return
    close_count = int(np.isclose(active_values, residual_values, rtol=1e-6, atol=1e-9).sum())
    print(f"  active_energy_sum: {float(active_values.sum())}")
    print(f"  residual_energy_sum_on_active_grid: {float(residual_values.sum())}")
    print(f"  diff_residual_minus_active_sum: {float(diff.sum())}")
    print(f"  diff_abs_mean: {float(np.mean(np.abs(diff)))}")
    print(f"  diff_abs_max: {float(np.max(np.abs(diff)))}")
    print(f"  equal_energy_cells: {close_count}/{active_values.size}")
    if scales_arr.size:
        print(
            "  residual_scale_before_filter: "
            f"min={float(np.min(scales_arr))} "
            f"p50={float(np.percentile(scales_arr, 50))} "
            f"mean={float(np.mean(scales_arr))} "
            f"p95={float(np.percentile(scales_arr, 95))} "
            f"max={float(np.max(scales_arr))}"
        )


def _print_cell_count_diagnostics(label: str, cells: list) -> None:
    _print_header(f"{label} diagnostics")
    counts = _frame_lengths(cells)
    _print_kv("total", int(np.sum(counts)))
    _print_stats("count_per_frame", counts)
    if cells and len(cells[0]):
        first = cells[0][0]
        _print_kv("first_frame_first_cell", np.asarray(first).tolist() if not isinstance(first, dict) else first)


def _print_cell_map_diagnostics(cell_map: dict) -> None:
    _print_header("cell map diagnostics")
    positions = np.asarray(cell_map.get("cell_positions_world", []), dtype=float)
    confidence = np.asarray(cell_map.get("cell_geometry_confidence", []), dtype=float)
    _print_kv("axis_order", cell_map.get("axis_order"))
    _print_kv("metadata", cell_map.get("metadata"))
    _print_kv("num_mapped_cells", int(positions.shape[0]) if positions.ndim else 0)
    if positions.size:
        _print_stats("cell_position_x", positions[:, 0])
        _print_stats("cell_position_y", positions[:, 1])
        _print_stats("cell_position_z", positions[:, 2])
    _print_stats("geometry_confidence", confidence)


def _print_rf_motion_diagnostics(rf_motion_cells: list[list[dict]]) -> None:
    _print_header("RF motion diagnostics")
    records = _flatten_records(rf_motion_cells)
    _print_kv("total_rf_motion_cells", len(records))
    _print_stats("measured_doppler_hz", [r.get("measured_doppler_hz", np.nan) for r in records])
    _print_stats("radial_velocity_mps", [r.get("radial_velocity_mps", np.nan) for r in records])
    _print_stats("active_energy", [r.get("energy", np.nan) for r in records])
    if records:
        sample = dict(records[0])
        sample["position_world"] = np.asarray(sample["position_world"], dtype=float).tolist()
        _print_kv("sample_rf_cell", sample)


def _print_residual_diagnostics(label: str, residual_cells: list[list[dict]]) -> None:
    _print_header(f"{label} diagnostics")
    records = _flatten_records(residual_cells)
    _print_kv("total_cells", len(records))
    _print_stats("measured_doppler_hz", [r.get("measured_doppler_hz", np.nan) for r in records])
    _print_stats("expected_doppler_hz", [r.get("expected_doppler_hz", np.nan) for r in records])
    _print_stats("residual_doppler_hz", [r.get("residual_doppler_hz", np.nan) for r in records])
    _print_stats("original_energy", [r.get("energy", np.nan) for r in records])
    _print_stats("residual_energy", [r.get("residual_energy", np.nan) for r in records])
    scales = [r.get("residual_energy", 0.0) / max(float(r.get("energy", 0.0)), 1e-12) for r in records]
    _print_stats("residual_scale", scales)
    if records:
        sample = dict(records[0])
        sample["position_world"] = np.asarray(sample["position_world"], dtype=float).tolist()
        _print_kv("sample_residual_cell", sample)


def _print_filter_diagnostics(before: list[list[dict]], after: list[list[dict]], config: dict) -> None:
    _print_header("residual filtering diagnostics")
    before_counts = _frame_lengths(before)
    after_counts = _frame_lengths(after)
    total_before = float(np.sum(before_counts))
    total_after = float(np.sum(after_counts))
    _print_kv("residual_cell_config", config or {"min_residual_energy": 0.0, "min_residual_scale": 0.2})
    _print_kv("total_before", int(total_before))
    _print_kv("total_after", int(total_after))
    _print_kv("kept_ratio", total_after / max(total_before, 1.0))
    _print_stats("kept_count_per_frame", after_counts)


def _print_blob_diagnostics(blobs: list, tracked_blobs: list) -> None:
    _print_header("blob diagnostics")
    blob_counts = _frame_lengths(blobs)
    tracked_counts = _frame_lengths(tracked_blobs)
    flat = [blob for frame in tracked_blobs for blob in frame]
    _print_kv("total_blobs", int(np.sum(blob_counts)))
    _print_kv("total_tracked_blobs", int(np.sum(tracked_counts)))
    _print_stats("blob_count_per_frame", blob_counts)
    _print_stats("tracked_blob_count_per_frame", tracked_counts)
    _print_stats("blob_residual_energy", [b.residual_energy for b in flat])
    _print_stats("blob_doppler_mean_hz", [b.doppler_mean_hz for b in flat])
    _print_stats("blob_confidence", [b.confidence for b in flat])
    if flat:
        sample = flat[0]
        _print_kv(
            "sample_blob",
            {
                "blob_id": sample.blob_id,
                "frame_idx": sample.frame_idx,
                "centroid_grid": sample.centroid_grid,
                "bbox_grid": sample.bbox_grid,
                "centroid_world": np.asarray(sample.centroid_world, dtype=float).tolist(),
                "energy": sample.energy,
                "residual_energy": sample.residual_energy,
                "doppler_mean_hz": sample.doppler_mean_hz,
                "confidence": sample.confidence,
                "metadata": sample.metadata,
            },
        )


def _print_global_feature_diagnostics(global_features: list[dict], kinematics: dict) -> None:
    _print_header("global feature diagnostics")
    _print_kv("num_feature_frames", len(global_features))
    for key in [
        "rf_surprise",
        "rf_sector_front_left_energy",
        "rf_sector_front_right_energy",
        "rf_sector_rear_left_energy",
        "rf_sector_rear_right_energy",
        "rf_intent_angle_from_head_forward",
    ]:
        _print_stats(key, [frame.get(key, np.nan) for frame in global_features])
    head_speed = np.linalg.norm(np.asarray(kinematics["linear_velocity_world"], dtype=float), axis=1)
    residual_totals = []
    for frame in global_features:
        residual_totals.append(
            sum(float(frame.get(key, 0.0)) for key in frame if key.startswith("rf_sector_") and key.endswith("_energy"))
        )
    ratio = np.asarray(residual_totals, dtype=float) / np.maximum(head_speed, 1e-12)
    _print_stats("residual_sector_energy_to_head_speed", ratio)


def _print_roi_diagnostics(candidate_rois: list) -> None:
    _print_header("ROI diagnostics")
    counts = _frame_lengths(candidate_rois)
    _print_kv("total_rois", int(np.sum(counts)))
    _print_stats("roi_count_per_frame", counts)


def _print_ml_diagnostics(ml_dataset: dict) -> None:
    _print_header("ML dataset diagnostics")
    x = np.asarray(ml_dataset["X"], dtype=float)
    _print_kv("X_shape", tuple(x.shape))
    _print_kv("num_feature_names", len(ml_dataset.get("feature_names", [])))
    _print_kv("feature_names", ml_dataset.get("feature_names", []))
    if x.size:
        _print_kv("nan_count", int(np.isnan(x).sum()))
        _print_kv("inf_count", int(np.isinf(x).sum()))
        _print_stats("feature_matrix_values", x[np.isfinite(x)])


def _print_summary_diagnostics(summary: dict, warnings: list[str]) -> None:
    _print_header("final summary")
    for key, value in summary.items():
        if key != "warnings":
            _print_kv(key, value)
    if warnings:
        _print_header("warnings")
        for warning in warnings:
            print(f"  - {warning}")
