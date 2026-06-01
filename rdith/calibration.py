from __future__ import annotations

"""Minimal RDITH static system configuration.

Important design choice:
RDITH does NOT duplicate antenna-array geometry that is already owned by the
CSI heatmap generator.  The heatmap generator uses num_rx, antenna spacing,
subcarrier axes, steering vectors, etc. to produce ToF-Doppler or
AoA-ToF-Doppler spectra.

RDITH only needs enough static metadata to place those completed heatmap cells
in the VR/world frame and to convert Doppler Hz to velocity/residual Doppler:

- RF/world origin of the already-generated heatmap coordinate frame.
- RF-to-world rotation of the already-generated heatmap coordinate frame.
- center frequency or wavelength.
- ToF-to-range scale and optional range offset.

These are static system configuration values, not additional runtime sensing
inputs. Runtime observations remain: raw CSI or completed heatmap, and user 6DoF.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


LIGHT_SPEED = 299792458.0
GEOMETRY_MODES = {"tof_only_pseudo", "monostatic_aoa"}


@dataclass
class RFLinkCalibration:
    """Backward-compatible single virtual link.

    Older RDITH code represented the RF setup as explicit Tx/Rx links.  That is
    unnecessary for this project because the heatmap generator has already used
    antenna geometry.  We keep a virtual monostatic link at the heatmap origin so
    existing residual Doppler helper functions can operate without requiring
    extra antenna-position files.
    """

    link_id: str
    tx_position_world: np.ndarray
    rx_position_world: np.ndarray
    tx_rotation_world_from_rf: Optional[np.ndarray]
    rx_rotation_world_from_rf: Optional[np.ndarray]
    center_frequency_hz: float
    wavelength_m: float
    confidence: float = 1.0


@dataclass
class RDITHCalibration:
    geometry_mode: str
    rf_origin_world: np.ndarray
    rf_rotation_world_from_rf: np.ndarray
    center_frequency_hz: float
    wavelength_m: float
    tof_range_scale: float
    range_offset_m: float = 0.0
    coordinate_convention: str = "world: x-right, y-up, z-forward"
    confidence: float = 1.0

    @property
    def links(self) -> list[RFLinkCalibration]:
        # Virtual single link.  This is not new antenna data; it is a convenient
        # monostatic representation of the completed heatmap coordinate frame.
        return [
            RFLinkCalibration(
                link_id="heatmap_origin",
                tx_position_world=self.rf_origin_world,
                rx_position_world=self.rf_origin_world,
                tx_rotation_world_from_rf=self.rf_rotation_world_from_rf,
                rx_rotation_world_from_rf=self.rf_rotation_world_from_rf,
                center_frequency_hz=self.center_frequency_hz,
                wavelength_m=self.wavelength_m,
                confidence=self.confidence,
            )
        ]


def load_rdith_calibration(
    path_or_dict: str | dict | RDITHCalibration | None = None,
    *,
    heatmap_result: Optional[dict] = None,
    config: Optional[dict] = None,
) -> RDITHCalibration:
    """Load or infer minimal RDITH static configuration.

    Accepts an optional JSON path/dict for world-frame alignment.  If omitted,
    values are inferred from the heatmap metadata and/or heatmap generator
    config.  This makes ``--calibration_path`` optional and prevents RDITH from
    asking for duplicate antenna geometry.
    """

    if isinstance(path_or_dict, RDITHCalibration):
        return path_or_dict
    raw: dict[str, Any] = {}
    if isinstance(path_or_dict, (str, Path)):
        with open(path_or_dict, "r", encoding="utf-8") as f:
            raw = json.load(f)
    elif isinstance(path_or_dict, dict):
        raw = dict(path_or_dict)
    elif path_or_dict is not None:
        raise TypeError(f"Unsupported calibration input: {type(path_or_dict)!r}")

    # Allow users to put the minimal RDITH calibration block inside config.json.
    if config:
        cfg_cal = config.get("rdith_calibration") or config.get("world_frame") or {}
        if isinstance(cfg_cal, dict):
            merged = dict(cfg_cal)
            merged.update(raw)
            raw = merged

    hm_meta = (heatmap_result or {}).get("metadata", {}) if heatmap_result else {}
    heatmap_setting = (config or {}).get("heatmap_setting", {}) if config else {}

    center = _first_present(
        raw,
        ["center_frequency_hz", "center_frequency"],
        fallback=_first_present(hm_meta, ["center_frequency_hz", "center_frequency"], fallback=heatmap_setting.get("center_frequency")),
    )
    wavelength = _first_present(raw, ["wavelength_m", "wavelength"], fallback=hm_meta.get("wavelength"))
    if wavelength is None:
        if center is None:
            raise ValueError(
                "RDITH needs center_frequency or wavelength. Put center_frequency in config['heatmap_setting'], "
                "heatmap metadata, or optional rdith_calibration."
            )
        wavelength = LIGHT_SPEED / float(center)
    if center is None:
        center = LIGHT_SPEED / float(wavelength)

    theta_axis_present = heatmap_result is not None and heatmap_result.get("theta_deg_axis") is not None
    mode = raw.get("geometry_mode") or ("monostatic_aoa" if theta_axis_present else "tof_only_pseudo")
    if mode not in GEOMETRY_MODES:
        # Do not support explicit bistatic/multi-link here because that would
        # reintroduce geometry RDITH should not own in this project.
        raise ValueError(f"geometry_mode must be one of {sorted(GEOMETRY_MODES)}")

    rf_origin = np.asarray(raw.get("rf_origin_world", [0.0, 0.0, 0.0]), dtype=float)
    rf_rot = np.asarray(raw.get("rf_rotation_world_from_rf", np.eye(3)), dtype=float)
    if rf_origin.shape != (3,):
        raise ValueError("rf_origin_world must be a length-3 vector")
    if rf_rot.shape != (3, 3):
        raise ValueError("rf_rotation_world_from_rf must be a 3x3 matrix")

    return RDITHCalibration(
        geometry_mode=str(mode),
        rf_origin_world=rf_origin,
        rf_rotation_world_from_rf=rf_rot,
        center_frequency_hz=float(center),
        wavelength_m=float(wavelength),
        tof_range_scale=float(raw.get("tof_range_scale", LIGHT_SPEED / 2.0)),
        range_offset_m=float(raw.get("range_offset_m", 0.0)),
        coordinate_convention=str(raw.get("coordinate_convention", "world: x-right, y-up, z-forward")),
        confidence=float(raw.get("confidence", 1.0)),
    )


def calibration_to_dict(calibration: RDITHCalibration | dict) -> dict:
    if isinstance(calibration, dict):
        return calibration
    return {
        "geometry_mode": calibration.geometry_mode,
        "center_frequency": calibration.center_frequency_hz,
        "wavelength": calibration.wavelength_m,
        "rf_origin_world": calibration.rf_origin_world.tolist(),
        "rf_rotation_world_from_rf": calibration.rf_rotation_world_from_rf.tolist(),
        "tof_range_scale": calibration.tof_range_scale,
        "range_offset_m": calibration.range_offset_m,
        "coordinate_convention": calibration.coordinate_convention,
        "confidence": calibration.confidence,
    }


def heatmap_cell_to_world_candidates(
    tau_s: float,
    theta_deg: Optional[float],
    phi_deg: Optional[float],
    link: RFLinkCalibration,
    calibration: RDITHCalibration,
) -> list[dict]:
    range_m = float(tau_s) * calibration.tof_range_scale + calibration.range_offset_m
    if calibration.geometry_mode == "tof_only_pseudo" or theta_deg is None or not np.isfinite(theta_deg):
        pos = calibration.rf_origin_world + calibration.rf_rotation_world_from_rf @ np.array([0.0, 0.0, range_m])
        return [{"position_world": pos, "confidence": 0.25 * calibration.confidence, "geometry_mode": "tof_only_pseudo"}]

    elev = 0.0 if phi_deg is None or not np.isfinite(phi_deg) else float(phi_deg)
    direction_rf = _direction_from_angles(float(theta_deg), elev)
    direction_world = _unit(calibration.rf_rotation_world_from_rf @ direction_rf)
    pos = calibration.rf_origin_world + direction_world * range_m
    return [{"position_world": pos, "confidence": calibration.confidence, "geometry_mode": "monostatic_aoa"}]


def bistatic_delay_s(point_world: np.ndarray, link: RFLinkCalibration) -> float:
    # Backward-compatible helper.  With the virtual monostatic link this is the
    # two-way path length divided by c.
    point = np.asarray(point_world, dtype=float)
    return float((np.linalg.norm(point - link.tx_position_world) + np.linalg.norm(point - link.rx_position_world)) / LIGHT_SPEED)


def bistatic_doppler_projection_vector(point_world: np.ndarray, link: RFLinkCalibration) -> np.ndarray:
    point = np.asarray(point_world, dtype=float)
    return _unit(point - link.tx_position_world) + _unit(point - link.rx_position_world)


def _first_present(raw: dict, names: list[str], fallback: Any = None) -> Any:
    for name in names:
        if name in raw and raw[name] is not None:
            return raw[name]
    return fallback


def _direction_from_angles(theta_deg: float, phi_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    return _unit(np.array([np.cos(phi) * np.sin(theta), np.sin(phi), np.cos(phi) * np.cos(theta)]))


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return np.asarray(value, dtype=float) / norm if norm > 1e-12 else np.zeros_like(value, dtype=float)
