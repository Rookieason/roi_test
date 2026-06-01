"""RDITH feature extraction package.

RDITH adds a residual RF motion layer on top of the existing CSI heatmap
generator. It keeps dense heatmaps at the boundary and works mostly with sparse
active cells, RF blobs, and per-ROI feature dictionaries.
"""

from .data_types import CandidateROI, Pose6DoF, Pose6DoFQuat, RFBlob
from .heatmap_adapter import generate_standard_heatmap_from_csi, load_heatmap_result
from .pose_utils import align_pose_to_heatmap_timestamps, compute_head_kinematics
from .residual_heatmap import (
    build_residual_heatmap,
    compute_expected_6dof_velocity_at_points,
    compute_residual_motion_cells,
    estimate_rf_velocity,
    extract_active_heatmap_cells,
    map_heatmap_cells_to_world,
)
from .blob_extraction import (
    compute_doppler_entropy,
    compute_micro_doppler_bandwidth,
    extract_rf_blobs,
    track_rf_blobs,
)
from .calibration import (
    RDITHCalibration,
    RFLinkCalibration,
    bistatic_delay_s,
    bistatic_doppler_projection_vector,
    heatmap_cell_to_world_candidates,
    load_rdith_calibration,
)
from .doppler_residual import (
    compute_scalar_residual_doppler,
    estimate_full_velocity_from_multilink_doppler,
    expected_doppler_from_velocity,
)
from .roi_features import (
    compute_global_intent_features,
    compute_roi_rf_features,
    compute_roi_blob_weight,
    merge_features_for_ml,
    rf_roi_align_v2,
    roi_support_distance,
    rf_roi_align,
)
from .ml_export import export_rdith_features, export_residual_heatmap
from .progress import build_progress_summary
from .validation import validate_rdith_inputs
