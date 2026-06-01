#!/usr/bin/env bash
set -euo pipefail

# RDITH Phase 1 demo runner
# Generates heatmap from capture artifacts and runs RDITH residual pipeline.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="/home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/artifacts"
EXP_NAME="20260117-082831_8"
SAVE_PATH="/home/tonic/Projects/NSTC/Test"
HEATMAP_TYPE="ToF-Doppler"
CONFIG_PATH="$ROOT_DIR/config.json"
POSE_PATH="/home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/db/$EXP_NAME/agent6.oculus-quest2-60b119dcc35cd177/20260117_0828.csv"
OUTPUT_DIR="$SAVE_PATH/rdith_output_${EXP_NAME}"

mkdir -p "$SAVE_PATH"
mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

echo "[RDITH] Generating heatmap for ${EXP_NAME} (this may take a while)..."
python3 "$ROOT_DIR/heatmap.py" \
  --data_path "$DATA_PATH" \
  --save_path "$SAVE_PATH" \
  --exp_name "$EXP_NAME" \
  --heatmap_type "$HEATMAP_TYPE" \
  --heatmap_render_mode image \
  --save_fig \
  --save_mat \
  --force_rebuild_csi_cache

# Locate produced heatmap file (MAT/NPZ/NPY)
POSSIBLE_MAT="$SAVE_PATH/heatmap_result/mat/${EXP_NAME}/${HEATMAP_TYPE}/smoothed_CSI_avg.mat"
if [ -f "$POSSIBLE_MAT" ]; then
  HEATMAP_FILE="$POSSIBLE_MAT"
else
  # find any supported heatmap file under the mat folder
  HEATMAP_FILE=$(find "$SAVE_PATH/heatmap_result/mat/${EXP_NAME}" -type f \( -name "*.mat" -o -name "*.npz" -o -name "*.npy" \) 2>/dev/null | head -n1 || true)
fi

if [ -z "$HEATMAP_FILE" ] || [ ! -f "$HEATMAP_FILE" ]; then
  echo "ERROR: No heatmap file found under $SAVE_PATH/heatmap_result/mat/${EXP_NAME}" >&2
  echo "Please check that the earlier heatmap generation succeeded." >&2
  exit 2
fi

echo "[RDITH] Using heatmap file: $HEATMAP_FILE"

# Run RDITH Phase 1 pipeline (no ROI required)
echo "[RDITH] Running RDITH pipeline (phase1)"
python3 "$ROOT_DIR/scripts/run_rdith_features.py" \
  --heatmap_path "$HEATMAP_FILE" \
  --pose_path "$POSE_PATH" \
  --config_path "$CONFIG_PATH" \
  --output_path "$OUTPUT_DIR/rdith_residual.npz" \
  --heatmap_type "$HEATMAP_TYPE" \
  --save_intermediate_dir "$OUTPUT_DIR/intermediate"

echo "[RDITH] Running RDITH visualization"
python3 "$ROOT_DIR/scripts/visualize_rdith_phase1.py" \
  --rdith_output_dir "$OUTPUT_DIR" \
  --heatmap_path "$HEATMAP_FILE" \
  --heatmap_type "$HEATMAP_TYPE" \
  --output_dir "$OUTPUT_DIR/vis" \
  --max_frames 8

echo "[RDITH] Done. Outputs under: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR" || true

echo "Key files you may inspect:"
echo " - residual heatmap / features: $OUTPUT_DIR/rdith_residual.npz"
echo " - intermediate: $OUTPUT_DIR/intermediate"
echo " - progress summary (next to the output file): $(dirname "$OUTPUT_DIR/rdith_residual.npz")/rdith_progress_summary.json"
echo " - visualization dashboard: $OUTPUT_DIR/vis/07_rdith_dashboard.html"

exit 0
