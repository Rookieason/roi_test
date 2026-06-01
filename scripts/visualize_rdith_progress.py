#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rdith.data_types import CandidateROI, Pose6DoF
from rdith.heatmap_adapter import load_heatmap_result
from rdith.pose_utils import align_pose_to_heatmap_timestamps, compute_head_kinematics
from rdith.visualization import (
    plot_feature_matrix_health,
    plot_feature_timeseries,
    plot_heatmap_comparison,
    plot_roi_scores,
    plot_world_frame,
    write_visualization_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an RDITH progress/debug visualization report.")
    parser.add_argument("--rdith_output", required=True)
    parser.add_argument("--heatmap_path", default=None)
    parser.add_argument("--pose_path", default=None)
    parser.add_argument("--roi_path", default=None)
    parser.add_argument("--calibration_path", default=None)
    parser.add_argument("--output_dir", default="debug_viz")
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument("--heatmap_type", default="ToF-Doppler", choices=["ToF-Doppler", "AoA-ToF-Doppler"])
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rdith = _load_rdith_npz(args.rdith_output)
    image_paths: list[str] = []
    summary = {"rdith_output": args.rdith_output, "warnings": []}

    if "X" in rdith:
        health = plot_feature_matrix_health(rdith, str(out / "feature_matrix_health.png"))
        image_paths.append(str(out / "feature_matrix_health.png"))
        plot_feature_timeseries(rdith, list(rdith.get("feature_names", [])), str(out / "feature_timeseries_residual_energy.png"))
        image_paths.append(str(out / "feature_timeseries_residual_energy.png"))
        summary["feature_health"] = health
        if rdith.get("roi_ids") is not None and rdith["X"].size:
            names = list(rdith["feature_names"])
            frame_scores = {}
            if "rf_residual_energy" in names:
                idx = names.index("rf_residual_energy")
                for roi_id, value in zip(rdith["roi_ids"], rdith["X"][:, idx]):
                    frame_scores[str(roi_id)] = {"rf_residual_energy": float(value)}
            plot_roi_scores(frame_scores, str(out / "frame_000000_roi_scores.png"))
            image_paths.append(str(out / "frame_000000_roi_scores.png"))
    else:
        plot_feature_matrix_health({"X": np.empty((0, 0)), "feature_names": []}, str(out / "feature_matrix_health.png"))
        image_paths.append(str(out / "feature_matrix_health.png"))

    heatmap = None
    if args.heatmap_path:
        heatmap = load_heatmap_result(args.heatmap_path, args.heatmap_type)
        residual = rdith.get("residual_cells", rdith.get("residual_spectrum", np.empty((0, 12))))
        plot_heatmap_comparison(heatmap, residual, 0, str(out / "frame_000000_heatmap_compare.png"))
        image_paths.append(str(out / "frame_000000_heatmap_compare.png"))

    poses = _load_json_frames(args.pose_path, "poses") if args.pose_path else []
    rois = _load_json_frames(args.roi_path, "frames") if args.roi_path else [[]]
    if poses:
        timestamps = heatmap["timestamps"] if heatmap is not None and heatmap.get("timestamps") is not None else np.asarray([p["timestamp"] for p in poses], dtype=float)
        kin = compute_head_kinematics(align_pose_to_heatmap_timestamps(poses, timestamps))
        plot_world_frame([], rois[0] if rois else [], {k: v[0] for k, v in kin.items()}, {}, str(out / "frame_000000_world.png"), {})
        image_paths.append(str(out / "frame_000000_world.png"))
    else:
        summary["warnings"].append("pose_path missing; skipped world-frame visualization")

    write_visualization_report(str(out), summary, image_paths)


def _load_rdith_npz(path: str) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {key: z[key] for key in z.files}


def _load_json_frames(path: str, key: str):
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(key, data) if isinstance(data, dict) else data


if __name__ == "__main__":
    main()

