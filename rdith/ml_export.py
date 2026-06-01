from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def export_rdith_features(ml_dataset: dict, output_path: str, format: str = "npz") -> None:
    fmt = format.lower()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npz":
        np.savez_compressed(
            path,
            X=ml_dataset["X"],
            feature_names=np.asarray(ml_dataset["feature_names"]),
            roi_ids=np.asarray(ml_dataset["roi_ids"]),
            timestamps=ml_dataset["timestamps"],
        )
        return
    if fmt == "csv":
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "roi_id", *ml_dataset["feature_names"]])
            for timestamp, roi_id, row in zip(ml_dataset["timestamps"], ml_dataset["roi_ids"], ml_dataset["X"]):
                writer.writerow([timestamp, roi_id, *row.tolist()])
        return
    if fmt == "parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("parquet export requires pandas and a parquet engine") from exc
        df = pd.DataFrame(ml_dataset["X"], columns=ml_dataset["feature_names"])
        df.insert(0, "roi_id", ml_dataset["roi_ids"])
        df.insert(0, "timestamp", ml_dataset["timestamps"])
        df.to_parquet(path, index=False)
        return
    raise ValueError("format must be npz, csv, or parquet")


def export_residual_heatmap(residual_heatmap: dict, global_features: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if residual_heatmap.get("mode") == "dense":
        np.savez_compressed(
            path,
            residual_spectrum=residual_heatmap["spectrum"],
            global_features=np.asarray(global_features, dtype=object),
            heatmap_type=residual_heatmap.get("heatmap_type", ""),
        )
        return

    frames = residual_heatmap.get("frames", [])
    flat_rows = []
    for frame_idx, records in enumerate(frames):
        for record in records:
            position = np.asarray(record["position_world"], dtype=float)
            tau_idx, theta_idx, fd_idx = record["cell_index"]
            flat_rows.append(
                [
                    frame_idx,
                    tau_idx,
                    theta_idx,
                    fd_idx,
                    record.get("energy", 0.0),
                    record.get("residual_energy", 0.0),
                    record.get("scalar_doppler_hz", np.nan),
                    record.get("residual_doppler_hz", np.nan),
                    position[0],
                    position[1],
                    position[2],
                    record.get("confidence", 0.0),
                ]
            )
    np.savez_compressed(
        path,
        residual_cells=np.asarray(flat_rows, dtype=float),
        residual_columns=np.asarray(
            [
                "frame_idx",
                "tau_idx",
                "theta_idx",
                "fd_idx",
                "energy",
                "residual_energy",
                "scalar_doppler_hz",
                "residual_doppler_hz",
                "position_x",
                "position_y",
                "position_z",
                "confidence",
            ]
        ),
        global_features=np.asarray(global_features, dtype=object),
        heatmap_type=residual_heatmap.get("heatmap_type", ""),
    )
