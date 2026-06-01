from __future__ import annotations

from typing import Iterable

import numpy as np

from .data_types import Pose6DoF, as_pose6dof


def align_pose_to_heatmap_timestamps(
    poses: list,
    heatmap_timestamps: np.ndarray,
    method: str = "linear",
) -> list[Pose6DoF]:
    if method not in {"linear", "nearest"}:
        raise ValueError("method must be 'linear' or 'nearest'")
    if heatmap_timestamps is None or len(heatmap_timestamps) == 0:
        raise ValueError("Heatmap timestamps are required for pose alignment")

    normalized = sorted((as_pose6dof(p) for p in poses), key=lambda p: p.timestamp)
    if not normalized:
        raise ValueError("At least one 6DoF pose is required")

    pose_ts = np.asarray([p.timestamp for p in normalized], dtype=float)
    target_ts = np.asarray(heatmap_timestamps, dtype=float)
    if np.any(~np.isfinite(pose_ts)) or np.any(~np.isfinite(target_ts)):
        raise ValueError("Pose and heatmap timestamps must be finite numeric values")
    if target_ts.max() < pose_ts.min() or target_ts.min() > pose_ts.max():
        raise ValueError("Pose and heatmap timestamp ranges do not overlap")

    positions = np.asarray([p.position_world for p in normalized], dtype=float)
    rotations = np.asarray([p.rotation_world_from_head for p in normalized], dtype=float)

    aligned: list[Pose6DoF] = []
    for t in target_ts:
        if method == "nearest" or len(normalized) == 1:
            idx = int(np.argmin(np.abs(pose_ts - t)))
            pos = positions[idx]
            rot = rotations[idx]
        else:
            right = int(np.searchsorted(pose_ts, t, side="left"))
            if right <= 0:
                pos, rot = positions[0], rotations[0]
            elif right >= len(pose_ts):
                pos, rot = positions[-1], rotations[-1]
            else:
                left = right - 1
                alpha = float((t - pose_ts[left]) / max(pose_ts[right] - pose_ts[left], 1e-12))
                pos = (1.0 - alpha) * positions[left] + alpha * positions[right]
                rot = _project_to_rotation((1.0 - alpha) * rotations[left] + alpha * rotations[right])
        aligned.append(Pose6DoF(float(t), np.asarray(pos, dtype=float), np.asarray(rot, dtype=float)))
    return aligned


def compute_head_kinematics(
    aligned_poses: list,
    smooth_window: int = 3,
) -> dict:
    poses = [as_pose6dof(p) for p in aligned_poses]
    if len(poses) < 2:
        raise ValueError("At least two aligned poses are required to compute kinematics")

    timestamps = np.asarray([p.timestamp for p in poses], dtype=float)
    positions = np.asarray([p.position_world for p in poses], dtype=float)
    rotations = np.asarray([p.rotation_world_from_head for p in poses], dtype=float)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Aligned poses must have strictly increasing timestamps")

    linear_velocity = np.gradient(positions, timestamps, axis=0, edge_order=1)
    angular_velocity = np.zeros_like(linear_velocity)
    for i in range(len(poses)):
        if i == 0:
            angular_velocity[i] = _angular_velocity_between(rotations[0], rotations[1], timestamps[1] - timestamps[0])
        elif i == len(poses) - 1:
            angular_velocity[i] = _angular_velocity_between(rotations[-2], rotations[-1], timestamps[-1] - timestamps[-2])
        else:
            angular_velocity[i] = _angular_velocity_between(rotations[i - 1], rotations[i + 1], timestamps[i + 1] - timestamps[i - 1])

    linear_velocity = _moving_average(linear_velocity, smooth_window)
    angular_velocity = _moving_average(angular_velocity, smooth_window)
    head_forward = rotations @ np.array([0.0, 0.0, 1.0])

    return {
        "timestamps": timestamps,
        "position_world": positions,
        "rotation_world_from_head": rotations,
        "linear_velocity_world": linear_velocity,
        "angular_velocity_world": angular_velocity,
        "head_forward_world": _normalize_rows(head_forward),
    }


def kinematics_frame(kinematics: dict, frame_idx: int) -> dict:
    return {key: value[frame_idx] for key, value in kinematics.items() if isinstance(value, np.ndarray)}


def _project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def _angular_velocity_between(r0: np.ndarray, r1: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0:
        return np.zeros(3)
    delta = r1 @ r0.T
    trace_value = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(trace_value))
    if abs(angle) < 1e-9:
        return np.zeros(3)
    axis = np.array(
        [
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ],
        dtype=float,
    )
    axis /= max(2.0 * np.sin(angle), 1e-12)
    return axis * angle / dt


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.shape[0] < 3:
        return values
    window = int(window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.vstack([np.convolve(padded[:, col], kernel, mode="valid") for col in range(values.shape[1])]).T


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)

