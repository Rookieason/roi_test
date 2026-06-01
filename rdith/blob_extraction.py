from __future__ import annotations

import numpy as np

from .data_types import RFBlob


def extract_rf_blobs(
    residual_heatmap: dict,
    clustering_method: str = "dbscan",
    spatial_eps: float = 0.5,
    doppler_eps: float = 4.0,
    min_samples: int = 1,
    weight_by_residual: bool = True,
) -> list[list[RFBlob]]:
    if residual_heatmap.get("mode") != "sparse":
        raise ValueError("extract_rf_blobs currently expects sparse residual heatmap output")
    if clustering_method not in {"dbscan", "connected_components"}:
        raise ValueError("clustering_method must be dbscan or connected_components")

    blobs_per_frame: list[list[RFBlob]] = []
    next_id = 0
    for frame_idx, records in enumerate(residual_heatmap["frames"]):
        if not records:
            blobs_per_frame.append([])
            continue
        labels = _cluster_records(records, spatial_eps, doppler_eps)
        frame_blobs: list[RFBlob] = []
        for label in sorted(set(labels)):
            members = [records[i] for i, v in enumerate(labels) if v == label]
            if len(members) < min_samples:
                continue
            frame_blobs.append(_records_to_blob(next_id, members, weight_by_residual=weight_by_residual))
            next_id += 1
        blobs_per_frame.append(frame_blobs)
    return blobs_per_frame


def track_rf_blobs(
    blobs_per_frame: list[list[RFBlob]],
    max_match_distance: float = 0.5,
    max_missing_frames: int = 5,
) -> list[list[RFBlob]]:
    active_tracks: dict[int, tuple[np.ndarray, int, int]] = {}
    next_track_id = 0
    tracked: list[list[RFBlob]] = []
    for frame_idx, blobs in enumerate(blobs_per_frame):
        assigned_tracks: set[int] = set()
        frame_out: list[RFBlob] = []
        for blob in blobs:
            best_id = None
            best_dist = max_match_distance
            for track_id, (centroid, last_frame, lifetime) in active_tracks.items():
                if track_id in assigned_tracks or frame_idx - last_frame > max_missing_frames:
                    continue
                dist = float(np.linalg.norm(blob.centroid_world - centroid))
                if dist <= best_dist:
                    best_id, best_dist = track_id, dist
            if best_id is None:
                best_id = next_track_id
                next_track_id += 1
                lifetime = 1
            else:
                lifetime = active_tracks[best_id][2] + 1
            blob.blob_id = best_id
            blob.lifetime = lifetime
            active_tracks[best_id] = (blob.centroid_world, frame_idx, lifetime)
            assigned_tracks.add(best_id)
            frame_out.append(blob)
        active_tracks = {
            track_id: value for track_id, value in active_tracks.items() if frame_idx - value[1] <= max_missing_frames
        }
        tracked.append(frame_out)
    return tracked


def compute_micro_doppler_bandwidth(doppler_values_hz: np.ndarray, energies: np.ndarray) -> float:
    values = np.asarray(doppler_values_hz, dtype=float)
    weights = np.maximum(np.asarray(energies, dtype=float), 0.0)
    if values.size == 0 or weights.sum() <= 0:
        return 0.0
    p = weights / weights.sum()
    mean = float(np.sum(p * values))
    return float(np.sqrt(np.sum(p * (values - mean) ** 2)))


def compute_doppler_entropy(doppler_values_hz: np.ndarray, energies: np.ndarray, num_bins: int = 16) -> float:
    values = np.asarray(doppler_values_hz, dtype=float)
    weights = np.maximum(np.asarray(energies, dtype=float), 0.0)
    if values.size == 0 or weights.sum() <= 0:
        return 0.0
    hist, _ = np.histogram(values, bins=num_bins, weights=weights)
    p = hist.astype(float) / max(float(hist.sum()), 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def _cluster_records(records: list[dict], spatial_eps: float, doppler_eps: float) -> list[int]:
    labels = [-1] * len(records)
    current = 0
    for i in range(len(records)):
        if labels[i] != -1:
            continue
        labels[i] = current
        queue = [i]
        while queue:
            j = queue.pop()
            for k in range(len(records)):
                if labels[k] != -1:
                    continue
                spatial_dist = np.linalg.norm(records[j]["position_world"] - records[k]["position_world"])
                doppler_dist = abs(records[j]["scalar_doppler_hz"] - records[k]["scalar_doppler_hz"])
                if spatial_dist <= spatial_eps and doppler_dist <= doppler_eps:
                    labels[k] = current
                    queue.append(k)
        current += 1
    return labels


def _records_to_blob(blob_id: int, records: list[dict], weight_by_residual: bool = True) -> RFBlob:
    energies = np.asarray([max(0.0, r["energy"]) for r in records], dtype=float)
    residuals = np.asarray([max(0.0, r.get("residual_energy", 0.0)) for r in records], dtype=float)
    weight_values = residuals if weight_by_residual and residuals.sum() > 0 else energies
    weights = weight_values / max(float(weight_values.sum()), 1e-12)
    positions = np.asarray([r["position_world"] for r in records], dtype=float)
    centroid = np.sum(positions * weights[:, None], axis=0)
    velocities = [
        r.get("residual_velocity_world") if r.get("residual_velocity_world") is not None else r.get("velocity_world")
        for r in records
    ]

    cell_indices = [r["cell_index"] for r in records]
    tau_indices = np.asarray([idx[0] for idx in cell_indices], dtype=float)
    fd_indices = np.asarray([idx[-1] for idx in cell_indices], dtype=float)

    centroid_grid = (
        float(np.sum(tau_indices * weights)),
        float(np.sum(fd_indices * weights)),
    )

    bbox_grid = (
        int(np.min(tau_indices)),
        int(np.min(fd_indices)),
        int(np.max(tau_indices)),
        int(np.max(fd_indices)),
    )
    if all(v is not None for v in velocities):
        velocity = np.sum(np.asarray(velocities, dtype=float) * weights[:, None], axis=0)
    else:
        velocity = None
    dopplers = np.asarray([r["scalar_doppler_hz"] for r in records], dtype=float)
    residual_energy = float(np.sum(residuals))
    geom_conf = np.asarray([r.get("geometry_confidence", r.get("confidence", 0.0)) for r in records], dtype=float)
    confidence = float(np.sum(geom_conf * weights))
    modes = [r.get("residual_mode", "unknown") for r in records]
    residual_mode = modes[0] if len(set(modes)) == 1 else "mixed"
    residual_dopplers = np.asarray([r.get("residual_doppler_hz", np.nan) for r in records], dtype=float)
    expected_dopplers = np.asarray([r.get("expected_doppler_hz", np.nan) for r in records], dtype=float)
    finite_res = np.isfinite(residual_dopplers)
    sign_balance = 0.0
    if finite_res.any():
        signs = np.sign(residual_dopplers[finite_res])
        sign_balance = float(abs(np.sum(signs)) / max(signs.size, 1))
    frame_idx = int(records[0].get("frame_idx", 0))

    return RFBlob(
        blob_id=blob_id,
        frame_idx=frame_idx,
        timestamp=float(frame_idx),
        centroid_grid=centroid_grid,
        bbox_grid=bbox_grid,
        centroid_world=centroid,
        velocity_world=velocity,
        energy=float(energies.sum()),
        residual_energy=residual_energy,
        doppler_mean_hz=float(np.sum(dopplers * weights)),
        doppler_bandwidth_hz=compute_micro_doppler_bandwidth(dopplers, energies),
        doppler_entropy=compute_doppler_entropy(dopplers, energies),
        confidence=confidence,
        num_supporting_cells=len(records),
        metadata={
            "records": records,
            "residual_mode": residual_mode,
            "mean_residual_doppler_hz": _weighted_nanmean(residual_dopplers, weights),
            "mean_expected_doppler_hz": _weighted_nanmean(expected_dopplers, weights),
            "geometry_confidence": confidence,
            "doppler_sign_balance": sign_balance,
        },
    )


def _weighted_nanmean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values)
    if not mask.any():
        return float("nan")
    w = weights[mask]
    return float(np.sum(values[mask] * w) / max(float(w.sum()), 1e-12))
