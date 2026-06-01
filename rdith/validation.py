from __future__ import annotations

from typing import Optional

import numpy as np

from .data_types import as_pose6dof


def validate_rdith_inputs(
    heatmap_result: dict,
    poses: list,
    candidate_rois_per_frame: Optional[list],
    calibration: dict,
) -> None:
    spectrum = heatmap_result.get("spectrum")
    if spectrum is None:
        raise ValueError("heatmap spectrum is required")
    spectrum = np.asarray(spectrum)
    if spectrum.ndim not in {3, 4}:
        raise ValueError("heatmap spectrum must be (T,tau,fd) or (T,tau,theta,fd)")
    if heatmap_result.get("fd_axis") is None:
        raise ValueError("fd_axis is required")
    if heatmap_result.get("tau_axis") is None:
        raise ValueError("tau_axis is required")
    if heatmap_result.get("timestamps") is None:
        raise ValueError("heatmap timestamps are required for 6DoF synchronization")
    if heatmap_result.get("axis_order") is None:
        raise ValueError("axis_order is required")

    pose_list = [as_pose6dof(p) for p in poses]
    if not pose_list:
        raise ValueError("timestamped 6DoF poses are required")
    pose_ts = np.asarray([p.timestamp for p in pose_list], dtype=float)
    hm_ts = np.asarray(heatmap_result["timestamps"], dtype=float)
    if hm_ts.max() < pose_ts.min() or hm_ts.min() > pose_ts.max():
        raise ValueError("6DoF and heatmap timestamps do not overlap")

    if "wavelength" not in calibration and "center_frequency" not in calibration and "center_frequency" not in heatmap_result.get("metadata", {}):
        raise ValueError("RDITH requires wavelength or center_frequency from config, heatmap metadata, or optional rdith calibration")
    # rf_origin_world / rf_rotation_world_from_rf are static world-frame alignment.
    # They default to [0,0,0] and identity when omitted, so they are not required
    # runtime inputs and should not be treated as additional data.

    if candidate_rois_per_frame is not None and len(candidate_rois_per_frame) == 0:
        raise ValueError("candidate_rois_per_frame is empty")
    if candidate_rois_per_frame is not None and len(candidate_rois_per_frame) != spectrum.shape[0]:
        raise ValueError("candidate_rois_per_frame length must match heatmap frame count")

