#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rdith.pipeline import run_rdith_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RDITH residual RF ROI feature extraction.")
    parser.add_argument("--heatmap_path", default=None)
    parser.add_argument("--raw_csi_path", default=None)
    parser.add_argument("--pose_path", required=True)
    parser.add_argument("--roi_path", default=None)
    parser.add_argument("--calibration_path", default=None, help="Optional world-frame alignment override. Antenna geometry stays in the heatmap generator config.")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--heatmap_type", default="AoA-ToF-Doppler", choices=["ToF-Doppler", "AoA-ToF-Doppler"])
    parser.add_argument("--no_export_progress_summary", action="store_true")
    parser.add_argument("--save_intermediate_dir", default=None)
    args = parser.parse_args()
    params = vars(args)
    params["export_progress_summary"] = not params.pop("no_export_progress_summary")
    run_rdith_pipeline(**params)


if __name__ == "__main__":
    main()
