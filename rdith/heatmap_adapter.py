from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def load_heatmap_result(path: str, heatmap_type: str) -> dict:
    heatmap_type = _canonical_heatmap_type(heatmap_type)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.is_dir():
        files = sorted([*p.glob("*.npz"), *p.glob("*.mat"), *p.glob("*.npy")])
        if not files:
            raise FileNotFoundError(f"No .npz, .mat, or .npy heatmap files in {path}")
        p = files[0]

    metadata = {"source_path": str(p)}
    arrays = _load_array_file(p)

    metadata = {"source_path": str(p)}

    if "center_frequency" in arrays:
        metadata["center_frequency"] = float(np.asarray(arrays["center_frequency"]).squeeze())
        
    spectrum = arrays.get("spectrum", arrays.get("spectrums", arrays.get("arr_0")))
    if spectrum is None:
        raise ValueError(f"{p} does not contain spectrum/spectrums/arr_0")

    result = {
        "spectrum": np.asarray(spectrum),
        "timestamps": _optional_array(arrays, "timestamps", "timestamp"),
        "tau_axis": _optional_array(arrays, "tau_axis"),
        "fd_axis": _optional_array(arrays, "fd_axis"),
        "theta_deg_axis": _optional_array(arrays, "theta_deg_axis", "theta_axis"),
        "phi_deg_axis": _optional_array(arrays, "phi_deg_axis", "phi_axis"),
        "link_ids": _optional_list(arrays, "link_ids"),
        "rx_ids": _optional_list(arrays, "rx_ids"),
        "tx_ids": _optional_list(arrays, "tx_ids"),
        "heatmap_type": heatmap_type,
        "metadata": metadata,
    }
    result["axis_order"] = _infer_axis_order(result)
    _reshape_if_axes_known(result)
    return normalize_heatmap_axes(result)


def normalize_heatmap_axes(heatmap_result: dict) -> dict:
    result = dict(heatmap_result)
    spectrum = np.asarray(result["spectrum"])
    axis_order = tuple(result.get("axis_order") or _infer_axis_order(result))
    supported = {
        ("time", "tau", "fd"),
        ("time", "tau", "theta", "fd"),
        ("time", "link", "tau", "fd"),
        ("time", "link", "tau", "theta", "fd"),
        ("time", "link", "tau", "theta", "phi", "fd"),
    }
    if axis_order not in supported:
        raise ValueError(f"Unsupported or ambiguous heatmap axis_order: {axis_order}")
    if "link" not in axis_order:
        result["link_ids"] = result.get("link_ids") or None
    elif result.get("link_ids") is None:
        result["link_ids"] = [f"link_{i}" for i in range(spectrum.shape[axis_order.index("link")])]
    result["spectrum"] = spectrum
    result["axis_order"] = axis_order
    return result


def iter_heatmap_cells(heatmap_result: dict):
    hm = normalize_heatmap_axes(heatmap_result)
    spectrum = hm["spectrum"]
    axis_order = tuple(hm["axis_order"])
    tau_axis = hm.get("tau_axis")
    fd_axis = hm.get("fd_axis")
    theta_axis = hm.get("theta_deg_axis")
    phi_axis = hm.get("phi_deg_axis")
    link_ids = hm.get("link_ids")
    if tau_axis is None or fd_axis is None:
        raise ValueError("tau_axis and fd_axis are required")
    for index in np.ndindex(spectrum.shape):
        coord = dict(zip(axis_order, index))
        tau_idx = coord.get("tau", -1)
        theta_idx = coord.get("theta", -1)
        phi_idx = coord.get("phi", -1)
        fd_idx = coord.get("fd", -1)
        link_idx = coord.get("link", 0)
        yield {
            "frame_idx": coord["time"],
            "link_id": None if link_ids is None else str(link_ids[link_idx]),
            "tau_idx": tau_idx,
            "theta_idx": theta_idx,
            "phi_idx": phi_idx,
            "fd_idx": fd_idx,
            "tau_s": float(np.asarray(tau_axis)[tau_idx]),
            "theta_deg": None if theta_idx < 0 or theta_axis is None else float(np.asarray(theta_axis)[theta_idx]),
            "phi_deg": None if phi_idx < 0 or phi_axis is None else float(np.asarray(phi_axis)[phi_idx]),
            "fd_hz": float(np.asarray(fd_axis)[fd_idx]),
            "energy": float(spectrum[index]),
        }


def generate_standard_heatmap_from_csi(csi: np.ndarray, config: dict, heatmap_type: str) -> dict:
    from utilsforheatmap import (
        CSI_preprocessing,
        calculate_correlation_matrix,
        create_steering_matrix_F3D,
        create_steering_matrix_ToF_Doppler,
        heatmap_setup,
        run_music_algorithm,
        smoothed_CSI,
    )

    heatmap_type = _canonical_heatmap_type(heatmap_type)
    heatmap_setting = heatmap_setup(config)
    csi_mov = CSI_preprocessing(config, csi, heatmap_setting)
    if heatmap_type == "ToF-Doppler":
        steering = create_steering_matrix_ToF_Doppler(heatmap_setting)
        csi_smoothed = smoothed_CSI(heatmap_type, heatmap_setting, csi_mov)
        r = calculate_correlation_matrix(csi_smoothed, heatmap_type=heatmap_type)
        spectrum = run_music_algorithm(r, steering, heatmap_type=heatmap_type)
        spectrum = spectrum.reshape(len(spectrum), heatmap_setting.tau_axis.size, heatmap_setting.fd_axis.size)
    else:
        # Keep this path side-effect free by using the same inner operations as pipeline_3D.
        steering = create_steering_matrix_F3D(heatmap_setting)
        csi_smoothed = smoothed_CSI(heatmap_type, heatmap_setting, csi_mov)
        frames = []
        for idx in range(csi_smoothed.shape[0]):
            from utilsforheatmap import calculate_correlation_matrix_F3D

            r = calculate_correlation_matrix_F3D(csi_smoothed[idx])
            frame = run_music_algorithm(r, steering, heatmap_type=heatmap_type)
            frame = frame.reshape(
                heatmap_setting.theta_deg_axis.size,
                heatmap_setting.tau_axis.size,
                heatmap_setting.fd_axis.size,
            ).transpose(1, 0, 2)
            frames.append(frame)
        spectrum = np.asarray(frames)

    timestamps = np.arange(spectrum.shape[0], dtype=float) / float(heatmap_setting.fs)
    return {
        "spectrum": spectrum,
        "timestamps": timestamps,
        "tau_axis": heatmap_setting.tau_axis,
        "fd_axis": heatmap_setting.fd_axis,
        "theta_deg_axis": heatmap_setting.theta_deg_axis if heatmap_type == "AoA-ToF-Doppler" else None,
        "phi_deg_axis": None,
        "link_ids": None,
        "rx_ids": None,
        "tx_ids": None,
        "heatmap_type": heatmap_type,
        "axis_order": ("time", "tau", "theta", "fd") if heatmap_type == "AoA-ToF-Doppler" else ("time", "tau", "fd"),
        "metadata": {
            "generated_from_csi": True,
            "fs": heatmap_setting.fs,
            "center_frequency": float(getattr(heatmap_setting, "center_frequency", config.get("heatmap_setting", {}).get("center_frequency", 0.0))),
            "antenna_geometry_owned_by": "heatmap_generator",
        },
    }


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_array_file(path: Path) -> dict:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {key: z[key] for key in z.files}
    if path.suffix.lower() == ".npy":
        return {"arr_0": np.load(path, allow_pickle=False)}
    if path.suffix.lower() == ".mat":
        from scipy.io import loadmat

        return {key: value for key, value in loadmat(path).items() if not key.startswith("__")}
    raise ValueError(f"Unsupported heatmap file extension: {path.suffix}")


def _optional_array(arrays: dict, *names: str) -> Optional[np.ndarray]:
    for name in names:
        if name in arrays:
            value = np.asarray(arrays[name]).squeeze()
            return value.astype(float) if np.issubdtype(value.dtype, np.number) else value
    return None


def _optional_list(arrays: dict, *names: str) -> Optional[list[str]]:
    for name in names:
        if name in arrays:
            value = np.asarray(arrays[name]).squeeze()
            if value.ndim == 0:
                return [str(value.item())]
            out = []
            for item in value.tolist():
                if isinstance(item, bytes):
                    out.append(item.decode("utf-8", errors="replace"))
                else:
                    out.append(str(item))
            return out
    return None


def _canonical_heatmap_type(heatmap_type: str) -> str:
    if heatmap_type not in {"ToF-Doppler", "AoA-ToF-Doppler"}:
        raise ValueError("heatmap_type must be 'ToF-Doppler' or 'AoA-ToF-Doppler'")
    return heatmap_type


def _infer_axis_order(result: dict) -> tuple[str, ...]:
    spectrum = result["spectrum"]
    if result["heatmap_type"] == "ToF-Doppler":
        if spectrum.ndim == 3:
            return ("time", "tau", "fd")
        if spectrum.ndim == 4:
            return ("time", "link", "tau", "fd")
        return ("time", "flat")
    if spectrum.ndim == 4:
        return ("time", "tau", "theta", "fd")
    if spectrum.ndim == 5:
        return ("time", "link", "tau", "theta", "fd")
    if spectrum.ndim == 6:
        return ("time", "link", "tau", "theta", "phi", "fd")
    return ("time", "flat")


def _reshape_if_axes_known(result: dict) -> None:
    spectrum = result["spectrum"]
    tau_axis = result.get("tau_axis")
    fd_axis = result.get("fd_axis")
    theta_axis = result.get("theta_deg_axis")
    if result["heatmap_type"] == "ToF-Doppler" and spectrum.ndim == 2 and tau_axis is not None and fd_axis is not None:
        result["spectrum"] = spectrum.reshape(spectrum.shape[0], tau_axis.size, fd_axis.size)
        result["axis_order"] = ("time", "tau", "fd")
    if (
        result["heatmap_type"] == "AoA-ToF-Doppler"
        and spectrum.ndim == 2
        and tau_axis is not None
        and theta_axis is not None
        and fd_axis is not None
    ):
        result["spectrum"] = spectrum.reshape(spectrum.shape[0], tau_axis.size, theta_axis.size, fd_axis.size)
        result["axis_order"] = ("time", "tau", "theta", "fd")
