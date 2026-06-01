from __future__ import annotations

import numpy as np

from .calibration import RDITHCalibration, RFLinkCalibration, bistatic_doppler_projection_vector


def expected_doppler_from_velocity(
    point_world: np.ndarray,
    velocity_world: np.ndarray,
    link: RFLinkCalibration,
    wavelength_m: float,
    mode: str,
) -> float:
    point = np.asarray(point_world, dtype=float)
    velocity = np.asarray(velocity_world, dtype=float)
    if mode == "monostatic":
        radial_direction = _unit(point - link.rx_position_world)
        return float(2.0 * np.dot(velocity, radial_direction) / wavelength_m)
    if mode == "bistatic":
        b = bistatic_doppler_projection_vector(point, link)
        return float(np.dot(velocity, b) / wavelength_m)
    raise ValueError("mode must be monostatic or bistatic")


def compute_scalar_residual_doppler(
    measured_fd_hz: float,
    point_world: np.ndarray,
    expected_6dof_velocity_world: np.ndarray,
    link: RFLinkCalibration,
    wavelength_m: float,
    mode: str,
) -> dict:
    expected_fd = expected_doppler_from_velocity(point_world, expected_6dof_velocity_world, link, wavelength_m, mode)
    residual_fd = float(measured_fd_hz) - expected_fd
    residual_radial_velocity = residual_fd * wavelength_m / (2.0 if mode == "monostatic" else 1.0)
    measured_abs = abs(float(measured_fd_hz))
    # This is a dimensionless keep-ratio for suppressive residual heatmaps.
    # 0 means the measured Doppler is fully explained by 6DoF; 1 means the
    # observed Doppler is largely unexplained.  It must not carry Hz units.
    residual_energy_scale = float(np.clip(abs(residual_fd) / max(measured_abs, 1e-12), 0.0, 1.0))
    return {
        "measured_fd_hz": float(measured_fd_hz),
        "expected_fd_hz": expected_fd,
        "residual_fd_hz": residual_fd,
        "residual_radial_velocity_mps": float(residual_radial_velocity),
        "residual_energy_scale": residual_energy_scale,
    }


def estimate_full_velocity_from_multilink_doppler(
    point_world: np.ndarray,
    measurements: list[dict],
    calibration: RDITHCalibration,
    regularization: float = 1e-3,
) -> dict:
    if len(measurements) < 3:
        return {"velocity_world": None, "condition_number": np.inf, "confidence": 0.0}
    link_by_id = {link.link_id: link for link in calibration.links}
    rows = []
    rhs = []
    for measurement in measurements:
        link = link_by_id.get(str(measurement.get("link_id", calibration.links[0].link_id)))
        if link is None:
            continue
        mode = "bistatic" if calibration.geometry_mode in {"bistatic_txrx", "multi_link_bistatic"} else "monostatic"
        if mode == "bistatic":
            row = bistatic_doppler_projection_vector(point_world, link) / link.wavelength_m
        else:
            row = 2.0 * _unit(np.asarray(point_world, dtype=float) - link.rx_position_world) / link.wavelength_m
        rows.append(row)
        rhs.append(float(measurement["measured_fd_hz"]))
    if len(rows) < 3:
        return {"velocity_world": None, "condition_number": np.inf, "confidence": 0.0}
    a = np.asarray(rows, dtype=float)
    b = np.asarray(rhs, dtype=float)
    condition = float(np.linalg.cond(a))
    if not np.isfinite(condition) or condition > 1e4:
        return {"velocity_world": None, "condition_number": condition, "confidence": 0.0}
    ata = a.T @ a + regularization * np.eye(3)
    atb = a.T @ b
    velocity = np.linalg.solve(ata, atb)
    return {"velocity_world": velocity, "condition_number": condition, "confidence": float(1.0 / (1.0 + condition))}


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return np.asarray(value, dtype=float) / norm if norm > 1e-12 else np.zeros_like(value, dtype=float)
