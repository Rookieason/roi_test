#!/usr/bin/env bash
set -euo pipefail

# Minimal RDITH call.
# Runtime observations: completed heatmap + user 6DoF.
# config.json is the existing heatmap-generator/system config.
# calibration_path is optional and only needed if you want to override world-frame
# alignment (rf_origin_world / rf_rotation_world_from_rf).

python scripts/run_rdith_features.py \
  --heatmap_path example_heatmap.npz \
  --pose_path examples/pose_schema.json \
  --roi_path examples/roi_schema.json \
  --config_path config.json \
  --output_path output/rdith_features.npz \
  --heatmap_type ToF-Doppler \
  --save_intermediate_dir output/debug_intermediate

# Optional world-frame alignment override:
#   --calibration_path examples/calibration_schema.json
