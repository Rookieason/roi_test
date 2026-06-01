ls /home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/artifacts|grep 20260117| while read event_id
do
  echo "Processing $event_id"
  python /home/tonic/Projects/NSTC/AoA_ToF_Doppler_and_ToF_Doppler/heatmap.py \
    --data_path /home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/artifacts \
    --save_path /home/tonic/Projects/NSTC/sensor-agent-publish-subscribe/heatmaps \
    --exp_name $event_id \
    --heatmap_type ToF-Doppler \
    --heatmap_render_mode image \
    --save_fig \
    --force_rebuild_csi_cache \
    --save_mat 
done    
    # \
    # --heatmap_render_mode image \
    # --heatmap_image_width 640 \
    # --heatmap_image_height 480 \
    # --heatmap_normalization global \
    # --timestamp_alignment center

    # --not_create_new_steering_matrix
    # --heatmap_type ToF-Doppler \
    # --heatmap_type AoA-ToF-Doppler \


# available AoA-ToF-Doppler ToF-Doppler
