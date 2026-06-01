from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Pose6DoF:
    timestamp: float
    position_world: np.ndarray
    rotation_world_from_head: np.ndarray


@dataclass
class Pose6DoFQuat:
    timestamp: float
    position_world: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass
class CandidateROI:
    roi_id: str
    timestamp: float
    center_world: np.ndarray
    support_points_world: Optional[np.ndarray] = None
    bbox_min_world: Optional[np.ndarray] = None
    bbox_max_world: Optional[np.ndarray] = None
    visibility: float = 1.0
    in_frustum: bool = True
    occlusion_score: float = 0.0
    base_features: Optional[dict[str, Any]] = None

@dataclass
class RFBlob:
    blob_id: int
    frame_idx: int = 0
    timestamp: float = 0.0

    centroid_grid: tuple[float, float] = (0.0, 0.0)   # (tau_idx, fd_idx), for visualization
    bbox_grid: tuple[int, int, int, int] = (0, 0, 0, 0) # (tau_min, fd_min, tau_max, fd_max)

    centroid_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    velocity_world: Optional[np.ndarray] = None
    energy: float = 0.0
    residual_energy: float = 0.0
    doppler_mean_hz: float = 0.0
    doppler_bandwidth_hz: float = 0.0
    doppler_entropy: float = 0.0
    confidence: float = 0.0
    num_supporting_cells: int = 0
    lifetime: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


def as_pose6dof(value: Pose6DoF | Pose6DoFQuat | dict[str, Any]) -> Pose6DoF:
    if isinstance(value, Pose6DoF):
        return Pose6DoF(
            timestamp=float(value.timestamp),
            position_world=np.asarray(value.position_world, dtype=float),
            rotation_world_from_head=np.asarray(value.rotation_world_from_head, dtype=float),
        )
    if isinstance(value, Pose6DoFQuat):
        return Pose6DoF(
            timestamp=float(value.timestamp),
            position_world=np.asarray(value.position_world, dtype=float),
            rotation_world_from_head=quat_xyzw_to_matrix(value.quaternion_xyzw),
        )
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported pose type: {type(value)!r}")

    timestamp = value.get("timestamp")
    position = value.get("position_world", value.get("position"))
    rotation = value.get("rotation_world_from_head", value.get("rotation_matrix"))
    quat = value.get("quaternion_xyzw", value.get("quaternion"))
    if timestamp is None or position is None:
        raise ValueError("Pose requires timestamp and position_world/position")
    if rotation is None:
        if quat is None:
            raise ValueError("Pose requires rotation matrix or quaternion_xyzw")
        rotation = quat_xyzw_to_matrix(quat)
    return Pose6DoF(
        timestamp=float(timestamp),
        position_world=np.asarray(position, dtype=float),
        rotation_world_from_head=np.asarray(rotation, dtype=float),
    )


def as_candidate_roi(value: CandidateROI | dict[str, Any]) -> CandidateROI:
    if isinstance(value, CandidateROI):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported ROI type: {type(value)!r}")
    center = value.get("center_world", value.get("center"))
    if center is None:
        raise ValueError("ROI requires center_world/center")
    return CandidateROI(
        roi_id=str(value.get("roi_id", value.get("id"))),
        timestamp=float(value.get("timestamp", 0.0)),
        center_world=np.asarray(center, dtype=float),
        support_points_world=_optional_array(value.get("support_points_world")),
        bbox_min_world=_optional_array(value.get("bbox_min_world")),
        bbox_max_world=_optional_array(value.get("bbox_max_world")),
        visibility=float(value.get("visibility", 1.0)),
        in_frustum=bool(value.get("in_frustum", True)),
        occlusion_score=float(value.get("occlusion_score", 0.0)),
        base_features=value.get("base_features"),
    )


def _optional_array(value: Any) -> Optional[np.ndarray]:
    return None if value is None else np.asarray(value, dtype=float)


def quat_xyzw_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=float)
    norm = np.linalg.norm([x, y, z, w])
    if norm == 0:
        raise ValueError("Quaternion norm is zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
