echo "----single human pose experiment in 2026----"
# Use roi_selection/heatmap.py instead of AoA_ToF_Doppler/heatmap.py
# because roi_selection version supports BOTH:
#   1. Legacy NPZ format: artifacts/csi/csi_<exp_name>.npz
#   2. New sidecar format: artifacts/<exp_name>/arrays/csi.rx.*/*.npy (current data format)
python /home/tonic/Projects/NSTC/roi_selection/heatmap.py \
    --data_path /home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/artifacts \
    --save_path /home/tonic/Projects/NSTC/Test \
    --exp_name 20260117-082831_8 \
    --heatmap_type ToF-Doppler \
    --heatmap_render_mode image \
    --save_fig \
    --force_rebuild_csi_cache \
    --save_mat 
    
    # Previously generated:
    # --heatmap_render_mode plot \
    # 20260117-082831_8
    # 20260303-070204_long-movement 
    # 20260330-081943_record-5-minutes \

    # --heatmap_render_mode image \
    # --heatmap_image_width 640 \
    # --heatmap_image_height 480 \
    # --heatmap_normalization global \
    # --timestamp_alignment center

    # --not_create_new_steering_matrix
    # --heatmap_type ToF-Doppler \
    # --heatmap_type AoA-ToF-Doppler \

# 20260330-081943_record-5-minutes
# available AoA-ToF-Doppler ToF-Doppler


# python /home/tonic/Projects/NSTC/AoA_ToF_Doppler_and_ToF_Doppler/heatmap_fix.py \
#     --data_path /home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/artifacts \
#     --save_path /home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/heatmaps \
#     --exp_name 20260224-093756_long-movement \
#     --heatmap_type ToF-Doppler \
#     --heatmap_render_mode plot \
#     --save_fig \
#     --force_rebuild_csi_cache \
#     --save_mat 