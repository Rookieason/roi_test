from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .calibration import (
    RDITHCalibration,
    RFLinkCalibration,
    heatmap_cell_to_world_candidates,
    load_rdith_calibration,
)
from .doppler_residual import compute_scalar_residual_doppler


LIGHT_SPEED = 299792458.0


def extract_active_heatmap_cells(
    heatmap_result: dict,
    threshold_mode: str = "percentile",
    threshold_value: float = 95.0,
    max_cells_per_frame: int = 2000,
) -> list[np.ndarray]:
    spectrum = np.asarray(heatmap_result["spectrum"])
    if spectrum.ndim not in {3, 4}:
        raise ValueError("Expected heatmap spectrum shape (T,tau,fd) or (T,tau,theta,fd)")
    if threshold_mode not in {"percentile", "absolute", "cfar"}:
        raise ValueError("threshold_mode must be percentile, absolute, or cfar")

    tau_axis = heatmap_result.get("tau_axis")
    fd_axis = heatmap_result.get("fd_axis")
    theta_axis = heatmap_result.get("theta_deg_axis")
    if tau_axis is None or fd_axis is None:
        raise ValueError("tau_axis and fd_axis are required to extract active cells")

    active: list[np.ndarray] = []
    for frame_idx, frame in enumerate(spectrum):
        finite = frame[np.isfinite(frame)]
        if finite.size == 0:
            active.append(np.empty((0, 8), dtype=float))
            continue
        if threshold_mode in {"percentile", "cfar"}:
            threshold = np.percentile(finite, threshold_value)
            if threshold_mode == "cfar":
                threshold = max(threshold, float(np.mean(finite) + 3.0 * np.std(finite)))
        else:
            threshold = threshold_value
        indices = np.argwhere(frame >= threshold)
        if indices.size == 0:
            active.append(np.empty((0, 8), dtype=float))
            continue
        energies = frame[tuple(indices.T)]
        order = np.argsort(energies)[::-1][:max_cells_per_frame]
        indices = indices[order]
        energies = energies[order]
        if spectrum.ndim == 3:
            tau_idx = indices[:, 0]
            theta_idx = np.full_like(tau_idx, -1)
            fd_idx = indices[:, 1]
            theta_value = np.full(indices.shape[0], np.nan)
        else:
            tau_idx = indices[:, 0]
            theta_idx = indices[:, 1]
            fd_idx = indices[:, 2]
            theta_value = np.asarray(theta_axis, dtype=float)[theta_idx]
        rows = np.column_stack(
            [
                np.full(indices.shape[0], frame_idx),
                tau_idx,
                theta_idx,
                fd_idx,
                energies,
                np.asarray(tau_axis, dtype=float)[tau_idx],
                np.asarray(fd_axis, dtype=float)[fd_idx],
                theta_value,
            ]
        )
        active.append(rows.astype(float))
    return active


def map_heatmap_cells_to_world(heatmap_result: dict, calibration: dict | RDITHCalibration) -> dict:
    calibration_obj = load_rdith_calibration(calibration)
    tau_axis = heatmap_result.get("tau_axis")
    fd_axis = heatmap_result.get("fd_axis")
    theta_axis = heatmap_result.get("theta_deg_axis")
    phi_axis = heatmap_result.get("phi_deg_axis")
    if tau_axis is None or fd_axis is None:
        raise ValueError("tau_axis and fd_axis are required")

    tau_axis = np.asarray(tau_axis, dtype=float)
    theta_values = [None] if theta_axis is None else list(np.asarray(theta_axis, dtype=float))
    phi_values = [None] if phi_axis is None else list(np.asarray(phi_axis, dtype=float))
    link_ids = heatmap_result.get("link_ids") or [calibration_obj.links[0].link_id]
    link_by_id = {link.link_id: link for link in calibration_obj.links}

    positions = []
    indices = []
    confidence = []
    links = []
    for link_idx, link_id in enumerate(link_ids):
        link = link_by_id.get(str(link_id), calibration_obj.links[min(link_idx, len(calibration_obj.links) - 1)])
        for tau_idx, tau_s in enumerate(tau_axis):
            for theta_idx, theta in enumerate(theta_values):
                for phi_idx, phi in enumerate(phi_values):
                    candidates = heatmap_cell_to_world_candidates(tau_s, theta, phi, link, calibration_obj)
                    best = candidates[0]
                    positions.append(best["position_world"])
                    confidence.append(float(best["confidence"]))
                    indices.append([link_idx, tau_idx, -1 if theta is None else theta_idx, -1 if phi is None else phi_idx])
                    links.append(link.link_id)
    flat_positions = np.asarray(positions, dtype=float)
    flat_confidence = np.asarray(confidence, dtype=float)
    indices = np.asarray(indices, dtype=int)
    axis_order = ("link", "tau", "theta", "phi")

    return {
        "cell_positions_world": flat_positions,
        "cell_indices": indices,
        "cell_geometry_confidence": flat_confidence,
        "cell_link_ids": links,
        "axis_order": axis_order,
        "metadata": {
            "geometry_mode": calibration_obj.geometry_mode,
            "tof_only": calibration_obj.geometry_mode == "tof_only_pseudo",
            "position_limitation": "ToF-only cells are pseudo-positions, not exact 3D locations"
            if calibration_obj.geometry_mode == "tof_only_pseudo"
            else "",
        },
    }


def estimate_rf_velocity(
    active_cells: list[np.ndarray],
    cell_map: dict,
    heatmap_result: dict,
    calibration: dict,
) -> list[list[dict[str, Any]]]:
    wavelength = _wavelength(calibration, heatmap_result)
    doppler_scale = _doppler_velocity_scale(calibration)
    axis_order = tuple(cell_map.get("axis_order", ()))
    out: list[list[dict[str, Any]]] = []
    for frame_cells in active_cells:
        records: list[dict[str, Any]] = []
        for row in frame_cells:
            frame_idx, tau_idx, theta_idx, fd_idx, energy, tau_value, fd_value, theta_value = row
            map_idx = _lookup_cell_map_index(cell_map, int(tau_idx), int(theta_idx), axis_order)
            position = cell_map["cell_positions_world"][map_idx]
            geometry_conf = float(cell_map["cell_geometry_confidence"][map_idx])
            link_id = cell_map.get("cell_link_ids", ["link_0"])[map_idx]
            radial_velocity = float(fd_value) * wavelength / doppler_scale
            records.append(
                {
                    "frame_idx": int(frame_idx),
                    "link_id": str(link_id),
                    "position_world": position,
                    "velocity_world": None,
                    "measured_doppler_hz": float(fd_value),
                    "scalar_doppler_hz": float(fd_value),
                    "radial_velocity_mps": radial_velocity,
                    "energy": float(energy),
                    "confidence": geometry_conf,
                    "geometry_confidence": geometry_conf,
                    "cell_index": (int(tau_idx), int(theta_idx), int(fd_idx)),
                    "tau_value": float(tau_value),
                    "theta_value": None if np.isnan(theta_value) else float(theta_value),
                }
            )
        out.append(records)
    return out


def compute_expected_6dof_velocity_at_points(
    points_world: np.ndarray,
    pose_kinematics_at_t: dict,
) -> np.ndarray:
    points = np.asarray(points_world, dtype=float)
    head_pos = np.asarray(pose_kinematics_at_t["position_world"], dtype=float)
    linear = np.asarray(pose_kinematics_at_t["linear_velocity_world"], dtype=float)
    angular = np.asarray(pose_kinematics_at_t["angular_velocity_world"], dtype=float)
    return linear[None, :] + np.cross(angular[None, :], points - head_pos[None, :])


def compute_residual_motion_cells(
    rf_motion_cells: list[list[dict[str, Any]]],
    kinematics: dict,
    calibration: dict | RDITHCalibration,
) -> list[list[dict[str, Any]]]:
    calibration_obj = load_rdith_calibration(calibration)
    link_by_id = {link.link_id: link for link in calibration_obj.links}
    doppler_mode = "bistatic" if calibration_obj.geometry_mode in {"bistatic_txrx", "multi_link_bistatic"} else "monostatic"
    residual_frames: list[list[dict[str, Any]]] = []
    for frame_idx, records in enumerate(rf_motion_cells):
        if not records:
            residual_frames.append([])
            continue
        kin = _kinematics_at(kinematics, frame_idx)
        points = np.asarray([r["position_world"] for r in records], dtype=float)
        expected = compute_expected_6dof_velocity_at_points(points, kin)
        frame_out = []
        for idx, record in enumerate(records):
            out = dict(record)
            link = link_by_id.get(record.get("link_id"), calibration_obj.links[0])
            scalar = compute_scalar_residual_doppler(
                measured_fd_hz=float(record.get("measured_doppler_hz", record.get("scalar_doppler_hz"))),
                point_world=record["position_world"],
                expected_6dof_velocity_world=expected[idx],
                link=link,
                wavelength_m=link.wavelength_m,
                mode=doppler_mode,
            )
            out["expected_doppler_hz"] = scalar["expected_fd_hz"]
            out["residual_doppler_hz"] = scalar["residual_fd_hz"]
            out["residual_radial_velocity_mps"] = scalar["residual_radial_velocity_mps"]
            out["residual_energy"] = float(record["energy"]) * scalar["residual_energy_scale"]
            out["residual_velocity_world"] = None
            out["residual_mode"] = "scalar_doppler"
            out["geometry_confidence"] = float(record.get("geometry_confidence", record.get("confidence", link.confidence)))
            frame_out.append(out)
        residual_frames.append(frame_out)
    return residual_frames


def filter_residual_motion_cells(
    residual_motion_cells: list[list[dict[str, Any]]],
    min_residual_energy: float = 0.0,
    min_residual_scale: float = 0.2,
) -> list[list[dict[str, Any]]]:
    """Keep only cells that remain meaningfully unexplained after 6DoF filtering."""

    filtered: list[list[dict[str, Any]]] = []
    for records in residual_motion_cells:
        frame_records = []
        for record in records:
            energy = float(record.get("energy", 0.0))
            residual_energy = float(record.get("residual_energy", 0.0))
            scale = residual_energy / max(energy, 1e-12)
            if residual_energy > min_residual_energy and scale >= min_residual_scale:
                frame_records.append(record)
        filtered.append(frame_records)
    return filtered


def build_residual_heatmap(
    heatmap_result: dict,
    residual_motion_cells: list[list[dict[str, Any]]],
    output_mode: str = "sparse",
) -> dict:
    if output_mode not in {"sparse", "dense"}:
        raise ValueError("output_mode must be sparse or dense")
    if output_mode == "sparse":
        return {"mode": "sparse", "frames": residual_motion_cells, "heatmap_type": heatmap_result["heatmap_type"]}

    dense = np.zeros_like(np.asarray(heatmap_result["spectrum"], dtype=float))
    for frame_idx, records in enumerate(residual_motion_cells):
        for record in records:
            cell_index = record["cell_index"]
            tau_idx = int(cell_index[0])
            theta_idx = int(cell_index[1]) if len(cell_index) > 2 else -1
            fd_idx = int(cell_index[-1])
            if dense.ndim == 3:
                dense[frame_idx, tau_idx, fd_idx] = record["residual_energy"]
            else:
                dense[frame_idx, tau_idx, theta_idx, fd_idx] = record["residual_energy"]
    return {"mode": "dense", "spectrum": dense, "heatmap_type": heatmap_result["heatmap_type"]}


def _tau_to_range(tau_seconds: np.ndarray, calibration: dict | RDITHCalibration) -> np.ndarray:
    calibration_obj = load_rdith_calibration(calibration)
    scale = float(calibration_obj.tof_range_scale)
    offset = float(calibration_obj.range_offset_m)
    return tau_seconds * scale + offset


def _wavelength(calibration: dict | RDITHCalibration, heatmap_result: dict) -> float:
    if isinstance(calibration, RDITHCalibration):
        return calibration.links[0].wavelength_m
    if "wavelength" in calibration:
        return float(calibration["wavelength"])
    center = calibration.get("center_frequency", heatmap_result.get("metadata", {}).get("center_frequency"))
    if center is None:
        raise ValueError("center_frequency or wavelength is required for Doppler velocity")
    return LIGHT_SPEED / float(center)


def _doppler_velocity_scale(calibration: dict | RDITHCalibration) -> float:
    mode = calibration.geometry_mode if isinstance(calibration, RDITHCalibration) else calibration.get("geometry_mode")
    return 1.0 if mode in {"bistatic_txrx", "multi_link_bistatic"} else 2.0


def _direction_from_theta(theta_deg: float, calibration: dict) -> np.ndarray:
    rf_to_world = np.asarray(calibration.get("rf_rotation_world_from_rf", np.eye(3)), dtype=float)
    direction_rf = np.array([np.sin(np.deg2rad(theta_deg)), 0.0, np.cos(np.deg2rad(theta_deg))], dtype=float)
    return _unit(rf_to_world @ direction_rf)


def _lookup_cell_map_index(cell_map: dict, tau_idx: int, theta_idx: int, axis_order: tuple[str, ...]) -> int:
    indices = np.asarray(cell_map["cell_indices"], dtype=int)
    if indices.shape[1] >= 4:
        if theta_idx >= 0:
            matches = np.where((indices[:, 1] == tau_idx) & (indices[:, 2] == theta_idx))[0]
        else:
            matches = np.where(indices[:, 1] == tau_idx)[0]
    elif axis_order == ("tau", "theta"):
        matches = np.where((indices[:, 0] == tau_idx) & (indices[:, 1] == theta_idx))[0]
    else:
        matches = np.where(indices[:, 0] == tau_idx)[0]
    if matches.size == 0:
        raise IndexError(f"No cell map entry for tau={tau_idx}, theta={theta_idx}")
    return int(matches[0])


def _kinematics_at(kinematics: dict, frame_idx: int) -> dict:
    return {key: value[frame_idx] for key, value in kinematics.items() if isinstance(value, np.ndarray)}


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        return np.zeros_like(value, dtype=float)
    return np.asarray(value, dtype=float) / norm
