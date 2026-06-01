from __future__ import annotations

from typing import Any

import numpy as np

from .data_types import CandidateROI, RFBlob, as_candidate_roi


ROI_RF_FEATURE_NAMES = [
    "rf_motion_energy",
    "rf_residual_energy",
    "rf_velocity_x",
    "rf_velocity_y",
    "rf_velocity_z",
    "rf_scalar_doppler_mean_hz",
    "rf_scalar_doppler_abs_mean_hz",
    "rf_approach_user",
    "rf_motion_to_roi_alignment",
    "rf_time_to_contact",
    "rf_time_to_contact_score",
    "rf_micro_doppler_bandwidth",
    "rf_doppler_entropy",
    "rf_confidence",
    "rf_visibility_conflict",
    "rf_temporal_growth",
    "rf_blob_count",
    "rf_nearest_blob_distance",
    "rf_mean_residual_doppler_hz",
    "rf_abs_residual_doppler_hz",
    "rf_geometry_confidence",
    "rf_blob_lifetime_mean",
    "rf_blob_lifetime_max",
    "rf_roi_support_distance_weighted",
]

GLOBAL_FEATURE_NAMES = [
    "rf_sector_front_left_energy",
    "rf_sector_front_right_energy",
    "rf_sector_rear_left_energy",
    "rf_sector_rear_right_energy",
    "rf_sector_upper_energy",
    "rf_sector_lower_energy",
    "rf_intent_centroid_x",
    "rf_intent_centroid_y",
    "rf_intent_centroid_z",
    "rf_intent_angle_from_head_forward",
    "rf_surprise",
]


def rf_roi_align(
    tracked_blobs_per_frame: list[list[RFBlob]],
    candidate_rois_per_frame: list[list[CandidateROI]],
    kinematics: dict,
    config: dict,
) -> list[dict[str, dict[str, float]]]:
    return rf_roi_align_v2(tracked_blobs_per_frame, candidate_rois_per_frame, kinematics, config)


def rf_roi_align_v2(
    tracked_blobs_per_frame: list[list[RFBlob]],
    candidate_rois_per_frame: list[list[CandidateROI]],
    kinematics: dict,
    config: dict,
) -> list[dict[str, dict[str, float]]]:
    roi_radius = float(config.get("roi_support_radius_m", config.get("roi_radius_m", 1.0)))
    support_mode = str(config.get("support_mode", "radius"))
    previous_energy: dict[str, float] = {}
    out: list[dict[str, dict[str, float]]] = []
    for frame_idx, rois in enumerate(candidate_rois_per_frame):
        frame_features: dict[str, dict[str, float]] = {}
        kin = _kinematics_at(kinematics, frame_idx)
        blobs = tracked_blobs_per_frame[frame_idx] if frame_idx < len(tracked_blobs_per_frame) else []
        for roi_value in rois:
            roi = as_candidate_roi(roi_value)
            pairs = [(b, roi_support_distance(b, roi, support_mode)) for b in blobs]
            nearby = [(b, d) for b, d in pairs if d <= roi_radius]
            features = compute_roi_rf_features(roi, [b for b, _ in nearby], kin, config, blob_distances=[d for _, d in nearby])
            features["rf_temporal_growth"] = features["rf_residual_energy"] - previous_energy.get(roi.roi_id, 0.0)
            previous_energy[roi.roi_id] = features["rf_residual_energy"]
            frame_features[roi.roi_id] = features
        out.append(frame_features)
    return out


def roi_support_distance(blob: RFBlob, roi: CandidateROI, mode: str) -> float:
    point = np.asarray(blob.centroid_world, dtype=float)
    mode = mode.lower()
    if mode == "bbox" and roi.bbox_min_world is not None and roi.bbox_max_world is not None:
        lower = np.asarray(roi.bbox_min_world, dtype=float)
        upper = np.asarray(roi.bbox_max_world, dtype=float)
        outside = np.maximum(np.maximum(lower - point, point - upper), 0.0)
        return float(np.linalg.norm(outside))
    if mode == "support_points" and roi.support_points_world is not None and len(roi.support_points_world) > 0:
        points = np.asarray(roi.support_points_world, dtype=float)
        return float(np.min(np.linalg.norm(points - point[None, :], axis=1)))
    return float(np.linalg.norm(point - roi.center_world))


def compute_roi_blob_weight(blob: RFBlob, roi: CandidateROI, distance_m: float, config: dict) -> float:
    sigma = float(config.get("roi_distance_sigma_m", config.get("roi_support_radius_m", 1.0)))
    sigma = max(sigma, 1e-6)
    residual = max(float(blob.residual_energy), 0.0)
    confidence = max(float(blob.confidence), 0.0)
    distance_weight = float(np.exp(-(distance_m**2) / (2.0 * sigma**2)))
    return residual * confidence * distance_weight


def compute_roi_rf_features(
    roi: CandidateROI,
    nearby_blobs: list[RFBlob],
    pose_kinematics_at_t: dict,
    config: dict,
    blob_distances: list[float] | None = None,
) -> dict[str, float]:
    if not nearby_blobs:
        return _empty_roi_features(roi)

    if blob_distances is None:
        blob_distances = [float(np.linalg.norm(b.centroid_world - roi.center_world)) for b in nearby_blobs]
    pooling_weights = np.asarray(
        [compute_roi_blob_weight(blob, roi, distance, config) for blob, distance in zip(nearby_blobs, blob_distances)],
        dtype=float,
    )
    energies = np.asarray([max(0.0, b.energy) for b in nearby_blobs], dtype=float)
    residuals = np.asarray([max(0.0, b.residual_energy) for b in nearby_blobs], dtype=float)
    weights = pooling_weights / max(float(pooling_weights.sum()), 1e-12) if pooling_weights.sum() > 0 else energies / max(float(energies.sum()), 1e-12)
    velocities = [b.velocity_world for b in nearby_blobs]
    if all(v is not None for v in velocities):
        v_avg = np.sum(np.asarray(velocities, dtype=float) * weights[:, None], axis=0)
        velocity_features = {
            "rf_velocity_x": float(v_avg[0]),
            "rf_velocity_y": float(v_avg[1]),
            "rf_velocity_z": float(v_avg[2]),
        }
    else:
        v_avg = None
        velocity_features = {"rf_velocity_x": np.nan, "rf_velocity_y": np.nan, "rf_velocity_z": np.nan}

    roi_vec_from_head = _unit(roi.center_world - pose_kinematics_at_t["position_world"])
    approach = np.nan if v_avg is None else float(-np.dot(v_avg, roi_vec_from_head))
    alignments = []
    alignment_weights = []
    ttc_values = []
    for blob in nearby_blobs:
        if blob.velocity_world is None:
            continue
        to_roi = roi.center_world - blob.centroid_world
        direction = _unit(to_roi)
        alignments.append(_cosine(blob.velocity_world, to_roi))
        alignment_weights.append(max(0.0, blob.energy))
        closing_speed = max(0.0, float(np.dot(blob.velocity_world, direction)))
        distance = float(np.linalg.norm(to_roi))
        ttc_values.append(distance / (closing_speed + 1e-6))

    if alignments:
        alignment = float(np.average(alignments, weights=np.asarray(alignment_weights, dtype=float)))
    else:
        alignment = np.nan
    if ttc_values:
        ttc = float(np.min(ttc_values))
        ttc_score = float(np.exp(-ttc / float(config.get("ttc_tau_s", 1.0))))
    else:
        ttc = np.nan
        ttc_score = np.nan

    distances = np.asarray(blob_distances, dtype=float)
    mean_residual_doppler = _weighted_meta(nearby_blobs, "mean_residual_doppler_hz", weights)
    geometry_confidence = _weighted_meta(nearby_blobs, "geometry_confidence", weights, default_key="confidence")
    return {
        "rf_motion_energy": float(np.sum(energies * weights)),
        "rf_residual_energy": float(np.sum(residuals * weights)),
        **velocity_features,
        "rf_scalar_doppler_mean_hz": float(np.average([b.doppler_mean_hz for b in nearby_blobs], weights=weights)),
        "rf_scalar_doppler_abs_mean_hz": float(np.average([abs(b.doppler_mean_hz) for b in nearby_blobs], weights=weights)),
        "rf_approach_user": approach,
        "rf_motion_to_roi_alignment": alignment,
        "rf_time_to_contact": ttc,
        "rf_time_to_contact_score": ttc_score,
        "rf_micro_doppler_bandwidth": float(np.average([b.doppler_bandwidth_hz for b in nearby_blobs], weights=weights)),
        "rf_doppler_entropy": float(np.average([b.doppler_entropy for b in nearby_blobs], weights=weights)),
        "rf_confidence": float(np.average([b.confidence for b in nearby_blobs], weights=weights)),
        "rf_visibility_conflict": float(residuals.sum() * (1.0 - np.clip(roi.visibility, 0.0, 1.0))),
        "rf_temporal_growth": 0.0,
        "rf_blob_count": float(len(nearby_blobs)),
        "rf_nearest_blob_distance": float(distances.min()),
        "rf_mean_residual_doppler_hz": mean_residual_doppler,
        "rf_abs_residual_doppler_hz": _weighted_abs_meta(nearby_blobs, "mean_residual_doppler_hz", weights),
        "rf_geometry_confidence": geometry_confidence,
        "rf_blob_lifetime_mean": float(np.average([b.lifetime for b in nearby_blobs], weights=weights)),
        "rf_blob_lifetime_max": float(max(b.lifetime for b in nearby_blobs)),
        "rf_roi_support_distance_weighted": float(np.average(distances, weights=weights)),
    }


def compute_global_intent_features(
    tracked_blobs_per_frame: list[list[RFBlob]],
    kinematics: dict,
    sector_config: dict,
) -> list[dict[str, float]]:
    previous_total = 0.0
    out: list[dict[str, float]] = []
    for frame_idx, blobs in enumerate(tracked_blobs_per_frame):
        kin = _kinematics_at(kinematics, frame_idx)
        features = {name: 0.0 for name in GLOBAL_FEATURE_NAMES}
        if not blobs:
            out.append(features)
            previous_total = 0.0
            continue
        head_pos = kin["position_world"]
        forward = _unit(kin["head_forward_world"])
        right = _unit(np.cross(forward, np.array([0.0, 1.0, 0.0])))
        if np.linalg.norm(right) < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
        up = _unit(np.cross(right, forward))
        total = 0.0
        centroid_num = np.zeros(3)
        for blob in blobs:
            energy = max(0.0, blob.residual_energy)
            rel = blob.centroid_world - head_pos
            front = float(np.dot(rel, forward))
            lateral = float(np.dot(rel, right))
            vertical = float(np.dot(rel, up))
            if front >= 0 and lateral < 0:
                features["rf_sector_front_left_energy"] += energy
            elif front >= 0 and lateral >= 0:
                features["rf_sector_front_right_energy"] += energy
            elif front < 0 and lateral < 0:
                features["rf_sector_rear_left_energy"] += energy
            else:
                features["rf_sector_rear_right_energy"] += energy
            if vertical >= 0:
                features["rf_sector_upper_energy"] += energy
            else:
                features["rf_sector_lower_energy"] += energy
            total += energy
            centroid_num += energy * blob.centroid_world
        centroid = centroid_num / max(total, 1e-12)
        intent_dir = _unit(centroid - head_pos)
        features["rf_intent_centroid_x"] = float(centroid[0])
        features["rf_intent_centroid_y"] = float(centroid[1])
        features["rf_intent_centroid_z"] = float(centroid[2])
        features["rf_intent_angle_from_head_forward"] = float(np.arccos(np.clip(np.dot(intent_dir, forward), -1.0, 1.0)))
        features["rf_surprise"] = float(total - previous_total)
        previous_total = total
        out.append(features)
    return out


def merge_features_for_ml(
    candidate_rois_per_frame: list[list[CandidateROI]],
    roi_rf_features_per_frame: list[dict[str, dict[str, float]]],
    global_features_per_frame: list[dict[str, float]],
    include_base_features: bool = True,
) -> dict:
    rows: list[list[float]] = []
    roi_ids: list[str] = []
    timestamps: list[float] = []
    base_feature_names = _collect_base_feature_names(candidate_rois_per_frame) if include_base_features else []
    feature_names = base_feature_names + ROI_RF_FEATURE_NAMES + GLOBAL_FEATURE_NAMES
    for frame_idx, rois in enumerate(candidate_rois_per_frame):
        frame_rf = roi_rf_features_per_frame[frame_idx] if frame_idx < len(roi_rf_features_per_frame) else {}
        global_rf = global_features_per_frame[frame_idx] if frame_idx < len(global_features_per_frame) else {}
        for roi_value in rois:
            roi = as_candidate_roi(roi_value)
            rf_features = frame_rf.get(roi.roi_id, _empty_roi_features(roi))
            base = roi.base_features or {}
            row = [float(base.get(name, np.nan)) for name in base_feature_names]
            row += [float(rf_features.get(name, np.nan)) for name in ROI_RF_FEATURE_NAMES]
            row += [float(global_rf.get(name, 0.0)) for name in GLOBAL_FEATURE_NAMES]
            rows.append(row)
            roi_ids.append(roi.roi_id)
            timestamps.append(roi.timestamp)
    return {
        "X": np.asarray(rows, dtype=float),
        "feature_names": feature_names,
        "roi_ids": roi_ids,
        "timestamps": np.asarray(timestamps, dtype=float),
        "metadata": {"include_base_features": include_base_features},
    }


def _empty_roi_features(roi: CandidateROI) -> dict[str, float]:
    return {
        "rf_motion_energy": 0.0,
        "rf_residual_energy": 0.0,
        "rf_velocity_x": np.nan,
        "rf_velocity_y": np.nan,
        "rf_velocity_z": np.nan,
        "rf_scalar_doppler_mean_hz": np.nan,
        "rf_scalar_doppler_abs_mean_hz": np.nan,
        "rf_approach_user": np.nan,
        "rf_motion_to_roi_alignment": np.nan,
        "rf_time_to_contact": np.nan,
        "rf_time_to_contact_score": np.nan,
        "rf_micro_doppler_bandwidth": 0.0,
        "rf_doppler_entropy": 0.0,
        "rf_confidence": 0.0,
        "rf_visibility_conflict": 0.0,
        "rf_temporal_growth": 0.0,
        "rf_blob_count": 0.0,
        "rf_nearest_blob_distance": np.inf,
        "rf_mean_residual_doppler_hz": np.nan,
        "rf_abs_residual_doppler_hz": np.nan,
        "rf_geometry_confidence": 0.0,
        "rf_blob_lifetime_mean": 0.0,
        "rf_blob_lifetime_max": 0.0,
        "rf_roi_support_distance_weighted": np.inf,
    }


def _collect_base_feature_names(candidate_rois_per_frame: list[list[CandidateROI]]) -> list[str]:
    names: set[str] = set()
    for frame in candidate_rois_per_frame:
        for roi_value in frame:
            roi = as_candidate_roi(roi_value)
            if roi.base_features:
                names.update(roi.base_features.keys())
    return sorted(names)


def _kinematics_at(kinematics: dict, frame_idx: int) -> dict:
    return {key: value[frame_idx] for key, value in kinematics.items() if isinstance(value, np.ndarray)}


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else np.zeros_like(value)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def _weighted_meta(blobs: list[RFBlob], key: str, weights: np.ndarray, default_key: str | None = None) -> float:
    vals = []
    for blob in blobs:
        if key in blob.metadata:
            vals.append(blob.metadata[key])
        elif default_key is not None and hasattr(blob, default_key):
            vals.append(getattr(blob, default_key))
        else:
            vals.append(np.nan)
    values = np.asarray(vals, dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        return float("nan")
    w = weights[mask]
    return float(np.sum(values[mask] * w) / max(float(w.sum()), 1e-12))


def _weighted_abs_meta(blobs: list[RFBlob], key: str, weights: np.ndarray) -> float:
    vals = np.asarray([blob.metadata.get(key, np.nan) for blob in blobs], dtype=float)
    mask = np.isfinite(vals)
    if not mask.any():
        return float("nan")
    w = weights[mask]
    return float(np.sum(np.abs(vals[mask]) * w) / max(float(w.sum()), 1e-12))
