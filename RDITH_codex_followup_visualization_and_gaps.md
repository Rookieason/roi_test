# RDITH Follow-up Implementation Spec for Codex

## 0. Objective

This document evaluates the current `roi_selection.zip` implementation against the requested RDITH plan and defines the missing work Codex should implement next.

The current implementation is a good first scaffold. It does match the high-level request in structure:

```text
standard CSI heatmap
    -> active heatmap cells
    -> approximate RF motion cells
    -> 6DoF residual motion
    -> RF blobs
    -> ROI-level features
    -> ML export
```

However, it is not yet a complete or reliable RDITH implementation. It currently contains many placeholder/simplified assumptions, especially in geometry, Doppler residual computation, ROI pooling, and visualization.

Codex should keep the existing structure, but implement the missing pieces below.

---

## 1. Current match assessment

### 1.1 What already matches the plan

The current code includes these modules, which are aligned with the plan:

```text
rdith/data_types.py
rdith/heatmap_adapter.py
rdith/pose_utils.py
rdith/residual_heatmap.py
rdith/blob_extraction.py
rdith/roi_features.py
rdith/ml_export.py
rdith/pipeline.py
rdith/validation.py
scripts/run_rdith_features.py
scripts/visualize_residual_heatmap.py
tests/test_rdith.py
```

The following requested concepts are already present at a scaffold level:

1. Load or generate standard heatmap from CSI.
2. Accept timestamped 6DoF pose.
3. Align pose timestamps to heatmap timestamps.
4. Compute head linear and angular velocity.
5. Extract active heatmap cells.
6. Estimate RF velocity or Doppler motion.
7. Compute residual against 6DoF-predicted motion.
8. Build sparse residual heatmap.
9. Extract RF blobs.
10. Compute ROI-level RF features.
11. Export ML feature matrix.
12. Provide a simple residual heatmap visualization script.

So this code should **not** be discarded. It should be extended and corrected.

---

## 2. Major gaps that must be fixed

## 2.1 Geometry and calibration are too simplified

Current issue:

`map_heatmap_cells_to_world()` maps heatmap cells using a simplified single-origin model:

- ToF-only is mapped along a pseudo RF forward axis.
- AoA-ToF-Doppler uses only `theta` and assumes direction `[sin(theta), 0, cos(theta)]`.
- `phi` is ignored.
- Multiple Tx/Rx pairs are not represented.
- Bistatic geometry is not represented.
- Antenna array pose is oversimplified.

This is acceptable for a smoke test, but not enough for a research implementation.

### Required implementation

Add a calibration schema that supports at least these modes:

```text
geometry_mode = "tof_only_pseudo"
geometry_mode = "monostatic_aoa"
geometry_mode = "bistatic_txrx"
geometry_mode = "multi_link_bistatic"
```

Create a new module:

```text
rdith/calibration.py
```

### Required data structures

```python
@dataclass
class RFLinkCalibration:
    link_id: str
    tx_position_world: np.ndarray
    rx_position_world: np.ndarray
    tx_rotation_world_from_rf: np.ndarray | None
    rx_rotation_world_from_rf: np.ndarray | None
    center_frequency_hz: float
    wavelength_m: float
    confidence: float = 1.0
```

```python
@dataclass
class RDITHCalibration:
    geometry_mode: str
    links: list[RFLinkCalibration]
    rf_origin_world: np.ndarray | None
    rf_rotation_world_from_rf: np.ndarray | None
    tof_range_scale: float
    range_offset_m: float
    coordinate_convention: str
```

### Required functions

```python
def load_rdith_calibration(path_or_dict: str | dict) -> RDITHCalibration:
    """
    Input:
        path_or_dict: JSON path or dictionary containing RF/VR calibration.
    Output:
        RDITHCalibration.
    Definition:
        Parse and validate RF-to-world calibration.
        Must support legacy calibration fields currently used by the code.
    """
```

```python
def heatmap_cell_to_world_candidates(
    tau_s: float,
    theta_deg: float | None,
    phi_deg: float | None,
    link: RFLinkCalibration,
    calibration: RDITHCalibration,
) -> list[dict]:
    """
    Input:
        tau_s: ToF/delay value.
        theta_deg: azimuth or AoA theta, if available.
        phi_deg: elevation, if available.
        link: RF link calibration.
        calibration: global calibration.
    Output:
        List of candidate world-space cells.
    Definition:
        Convert one heatmap coordinate into possible world locations.
        For ToF-only, return a pseudo-position with low confidence.
        For AoA+ToF, return a ray/range position.
        For bistatic Tx/Rx, return ellipsoid-compatible candidate cells or a projected point if extra angle data exists.
    """
```

```python
def bistatic_delay_s(point_world: np.ndarray, link: RFLinkCalibration) -> float:
    """
    Return expected bistatic delay:
        (||x - Tx|| + ||x - Rx||) / c
    """
```

```python
def bistatic_doppler_projection_vector(point_world: np.ndarray, link: RFLinkCalibration) -> np.ndarray:
    """
    Return geometry vector b(x) for bistatic Doppler:
        b(x) = unit(x - Tx) + unit(x - Rx)
    This vector is used to project 3D velocity into measured Doppler.
    """
```

---

## 2.2 Residual Doppler computation must be dimensionally correct

Current issue:

For AoA cells, the code estimates:

```python
radial_velocity = fd * wavelength / 2
velocity_world = direction * radial_velocity
residual_v = velocity_world - expected_6dof_velocity
```

This subtracts a full 3D expected velocity from a velocity vector reconstructed from a single radial Doppler observation. That is only valid if the measurement really provides a full 3D velocity estimate, which it usually does not.

For a single RF link, Doppler is usually a scalar projection. Therefore, residual should usually be computed in Doppler/radial space:

```text
measured_doppler - expected_doppler_from_6DoF
```

Only when multiple independent links are available should the code solve for a full 3D velocity vector.

### Required implementation

Create a new module:

```text
rdith/doppler_residual.py
```

### Required functions

```python
def expected_doppler_from_velocity(
    point_world: np.ndarray,
    velocity_world: np.ndarray,
    link: RFLinkCalibration,
    wavelength_m: float,
    mode: str,
) -> float:
    """
    Input:
        point_world: 3D world point.
        velocity_world: 3D velocity at that point.
        link: RF link geometry.
        wavelength_m: RF wavelength.
        mode: "monostatic" or "bistatic".
    Output:
        Expected Doppler frequency in Hz.
    Definition:
        For monostatic-like radar:
            fd = 2 * dot(v, radial_direction) / wavelength
        For bistatic:
            fd = dot(v, b(x)) / wavelength
        Sign convention must be documented and tested.
    """
```

```python
def compute_scalar_residual_doppler(
    measured_fd_hz: float,
    point_world: np.ndarray,
    expected_6dof_velocity_world: np.ndarray,
    link: RFLinkCalibration,
    wavelength_m: float,
    mode: str,
) -> dict:
    """
    Output keys:
        measured_fd_hz
        expected_fd_hz
        residual_fd_hz
        residual_radial_velocity_mps
        residual_energy_scale
    Definition:
        Compute residual in the same measurement dimension as the RF heatmap.
        This should be the default path for single-link or pseudo-geometry heatmaps.
    """
```

```python
def estimate_full_velocity_from_multilink_doppler(
    point_world: np.ndarray,
    measurements: list[dict],
    calibration: RDITHCalibration,
    regularization: float = 1e-3,
) -> dict:
    """
    Input:
        measurements: list of per-link Doppler observations for the same or nearby point.
    Output:
        velocity_world
        covariance_or_condition_number
        confidence
    Definition:
        Solve least squares:
            fd_l = dot(v, b_l(x)) / lambda_l
        Only return full 3D velocity when at least 3 sufficiently independent projections exist.
        Otherwise return None and force scalar residual path.
    """
```

### Required behavior

Update `compute_residual_motion_cells()` so that:

1. If only scalar Doppler evidence exists, compute scalar residual Doppler.
2. If full 3D velocity is confidently estimated from multiple links, compute vector residual.
3. Store both scalar and vector residual fields when possible.
4. Never subtract a full 3D 6DoF velocity from a one-dimensional RF Doppler measurement as if they were equivalent.

---

## 2.3 Heatmap adapter must preserve axis and link metadata

Current issue:

`load_heatmap_result()` expects a generic `spectrum`, `tau_axis`, `fd_axis`, `theta_deg_axis`, and timestamps. This is useful, but it does not preserve enough information for multi-link or array-aware RF geometry.

### Required implementation

Extend heatmap result dictionaries to include:

```python
{
    "spectrum": np.ndarray,
    "timestamps": np.ndarray,
    "tau_axis": np.ndarray,
    "fd_axis": np.ndarray,
    "theta_deg_axis": np.ndarray | None,
    "phi_deg_axis": np.ndarray | None,
    "link_ids": list[str] | None,
    "rx_ids": list[str] | None,
    "tx_ids": list[str] | None,
    "axis_order": tuple[str, ...],
    "heatmap_type": str,
    "metadata": dict,
}
```

Supported axis orders should include:

```text
("time", "tau", "fd")
("time", "tau", "theta", "fd")
("time", "link", "tau", "fd")
("time", "link", "tau", "theta", "fd")
("time", "link", "tau", "theta", "phi", "fd")
```

### Required functions

```python
def normalize_heatmap_axes(heatmap_result: dict) -> dict:
    """
    Reorder supported heatmap tensors into a canonical internal representation.
    Must not silently guess when metadata is insufficient.
    """
```

```python
def iter_heatmap_cells(heatmap_result: dict):
    """
    Yield active or all cells with explicit named coordinates:
        frame_idx, link_id, tau_idx, theta_idx, phi_idx, fd_idx,
        tau_s, theta_deg, phi_deg, fd_hz, energy
    """
```

---

## 2.4 ROI pooling is too coarse

Current issue:

`rf_roi_align()` currently uses a simple distance radius between RF blob centroid and ROI center:

```python
nearby = [blob if distance(blob.centroid, roi.center) <= roi_radius]
```

This ignores:

- ROI bounding box.
- ROI support points.
- object size.
- frustum/occlusion geometry beyond scalar visibility.
- RF blob uncertainty.
- distance-weighted pooling.

### Required implementation

Update ROI pooling to support three ROI support modes:

```text
support_mode = "radius"
support_mode = "bbox"
support_mode = "support_points"
```

### Required functions

```python
def roi_support_distance(blob: RFBlob, roi: CandidateROI, mode: str) -> float:
    """
    Return distance from RF blob to ROI support.
    If ROI has bbox, use point-to-AABB distance.
    If ROI has support points, use nearest support point or KD-tree.
    Otherwise fall back to center distance.
    """
```

```python
def compute_roi_blob_weight(
    blob: RFBlob,
    roi: CandidateROI,
    distance_m: float,
    config: dict,
) -> float:
    """
    Weight blob contribution by energy, residual energy, confidence, and spatial distance.
    Example:
        weight = residual_energy * confidence * exp(-distance^2 / (2*sigma^2))
    """
```

```python
def rf_roi_align_v2(
    tracked_blobs_per_frame: list[list[RFBlob]],
    candidate_rois_per_frame: list[list[CandidateROI]],
    kinematics: dict,
    config: dict,
) -> list[dict[str, dict[str, float]]]:
    """
    Replacement for current rf_roi_align.
    Must support radius, bbox, and support point pooling.
    Must expose all previous feature names for backward compatibility.
    """
```

---

## 2.5 Visualization is not enough yet

Current status:

`scripts/visualize_residual_heatmap.py` can draw residual heatmap frames as 2D ToF-Doppler scatter/image. This is useful, but it does **not** fully visualize project progress.

It does not show:

1. Standard heatmap vs residual heatmap comparison.
2. Head pose trajectory and head-forward vector.
3. RF blobs in world/user coordinates.
4. Candidate ROIs and their RF scores.
5. ROI feature time series.
6. ML feature matrix health.
7. Per-frame pipeline summary.
8. Whether heatmap, pose, calibration, ROI, and outputs are synchronized.

### Required implementation

Create a new visualization module and CLI:

```text
rdith/visualization.py
scripts/visualize_rdith_progress.py
```

### Required CLI

```bash
python scripts/visualize_rdith_progress.py \
  --rdith_output output/features_or_residual.npz \
  --heatmap_path optional_standard_heatmap.npz \
  --pose_path poses.json \
  --roi_path rois.json \
  --calibration_path calibration.json \
  --output_dir debug_viz \
  --max_frames 50
```

### Required visual outputs

The script must generate these files:

```text
debug_viz/
  index.html
  summary.json
  frame_000000_world.png
  frame_000000_heatmap_compare.png
  frame_000000_roi_scores.png
  feature_timeseries_residual_energy.png
  feature_timeseries_roi_scores.png
  feature_matrix_health.png
```

### Required visualizations

#### A. World-space frame visualization

For each selected frame, plot:

- HMD/user position.
- head-forward vector.
- RF residual blobs.
- candidate ROI centers or bounding boxes.
- line from blob to matched ROI when alignment is high.
- color or marker size based on residual energy.

Function:

```python
def plot_world_frame(
    blobs: list[RFBlob],
    rois: list[CandidateROI],
    pose_kinematics_at_t: dict,
    roi_features: dict[str, dict[str, float]],
    output_path: str,
    config: dict,
) -> None:
    """
    Generate a 2D top-down or 3D scatter view of current RDITH state.
    Must work without requiring a GUI backend.
    """
```

#### B. Standard-vs-residual heatmap comparison

Function:

```python
def plot_heatmap_comparison(
    standard_heatmap: dict,
    residual_heatmap_or_cells: dict | np.ndarray,
    frame_idx: int,
    output_path: str,
) -> None:
    """
    Show original Doppler energy and residual Doppler energy side by side.
    If AoA exists, max-project over AoA for 2D visualization.
    """
```

#### C. ROI score bar chart

Function:

```python
def plot_roi_scores(
    roi_features_at_frame: dict[str, dict[str, float]],
    output_path: str,
    score_key: str = "rf_residual_energy",
    top_k: int = 20,
) -> None:
    """
    Show top-k ROIs by residual energy or another RF score.
    """
```

#### D. Feature time series

Function:

```python
def plot_feature_timeseries(
    exported_features: dict,
    feature_names: list[str],
    output_path: str,
    aggregate: str = "max_per_frame",
) -> None:
    """
    Show how key RF features evolve over time.
    Required features:
        rf_residual_energy
        rf_motion_to_roi_alignment
        rf_time_to_contact_score
        rf_visibility_conflict
        rf_temporal_growth
    """
```

#### E. Feature matrix health check

Function:

```python
def plot_feature_matrix_health(ml_dataset: dict, output_path: str) -> dict:
    """
    Plot and return:
        NaN ratio per feature.
        Inf ratio per feature.
        mean/std/min/max per feature.
        percentage of zero-valued rows.
    """
```

#### F. HTML report

Function:

```python
def write_visualization_report(
    output_dir: str,
    summary: dict,
    image_paths: list[str],
) -> None:
    """
    Write index.html linking all generated plots.
    The report should be simple static HTML.
    """
```

---

## 2.6 Progress summary must be machine-readable

Add a pipeline progress report so Codex or a human can quickly determine what worked.

### Required file

Every run should optionally export:

```text
rdith_progress_summary.json
```

### Required content

```json
{
  "num_heatmap_frames": 0,
  "num_pose_samples": 0,
  "num_aligned_pose_frames": 0,
  "num_roi_frames": 0,
  "num_total_rois": 0,
  "num_active_cells": 0,
  "num_residual_cells": 0,
  "num_blobs": 0,
  "num_tracked_blobs": 0,
  "num_feature_rows": 0,
  "feature_names": [],
  "warnings": [],
  "stop_reasons": [],
  "geometry_mode": "",
  "heatmap_type": "",
  "axis_order": []
}
```

### Required function

```python
def build_progress_summary(
    heatmap: dict,
    poses: list,
    candidate_rois: list | None,
    residual_heatmap: dict,
    blobs: list[list[RFBlob]],
    tracked_blobs: list[list[RFBlob]],
    ml_dataset: dict | None,
    warnings: list[str],
    config: dict,
) -> dict:
    """
    Return a machine-readable summary of the pipeline result.
    """
```

---

## 2.7 Add example input schemas

The current repo needs explicit example files so another AI/developer knows what to feed into the pipeline.

Create:

```text
examples/pose_schema.json
examples/roi_schema.json
examples/calibration_schema.json
examples/run_example.sh
```

### pose_schema.json

```json
{
  "poses": [
    {
      "timestamp": 0.0,
      "position_world": [0.0, 1.6, 0.0],
      "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
    }
  ]
}
```

### roi_schema.json

```json
{
  "frames": [
    [
      {
        "roi_id": "roi_001",
        "timestamp": 0.0,
        "center_world": [1.0, 1.2, 2.0],
        "bbox_min_world": [0.8, 1.0, 1.8],
        "bbox_max_world": [1.2, 1.4, 2.2],
        "visibility": 0.7,
        "in_frustum": true,
        "occlusion_score": 0.2,
        "base_features": {
          "distance_from_gaze_center": 0.3,
          "previous_roi_score": 0.5
        }
      }
    ]
  ]
}
```

### calibration_schema.json

```json
{
  "geometry_mode": "monostatic_aoa",
  "coordinate_convention": "world: x-right, y-up, z-forward",
  "center_frequency": 5570000000.0,
  "wavelength": 0.0538,
  "rf_origin_world": [0.0, 1.2, 0.0],
  "rf_rotation_world_from_rf": [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ],
  "tof_range_scale": 149896229.0,
  "range_offset_m": 0.0,
  "links": [
    {
      "link_id": "link_0",
      "tx_position_world": [0.0, 1.2, 0.0],
      "rx_position_world": [0.0, 1.2, 0.0],
      "center_frequency_hz": 5570000000.0,
      "confidence": 1.0
    }
  ]
}
```

---

## 3. Required code changes by file

## 3.1 Modify `rdith/residual_heatmap.py`

Required changes:

1. Do not assume `fd * wavelength / 2` is always correct.
2. Use calibration geometry mode.
3. Store residual fields explicitly:

```python
record["measured_doppler_hz"]
record["expected_doppler_hz"]
record["residual_doppler_hz"]
record["residual_radial_velocity_mps"]
record["residual_energy"]
record["residual_velocity_world"]  # only if valid full-vector estimate exists
record["residual_mode"]            # "scalar_doppler" or "vector_velocity"
record["geometry_confidence"]
```

4. Preserve backward compatibility with current tests.

---

## 3.2 Modify `rdith/blob_extraction.py`

Required changes:

1. Blob velocity should distinguish between:

```text
scalar Doppler-only blob
vector velocity blob
mixed/unknown blob
```

2. Add fields to `RFBlob.metadata`:

```python
{
  "residual_mode": "scalar_doppler" | "vector_velocity" | "mixed",
  "mean_residual_doppler_hz": float,
  "mean_expected_doppler_hz": float,
  "geometry_confidence": float,
  "doppler_sign_balance": float
}
```

3. Use weighted centroid and confidence as currently done, but include residual energy in weights if configured.

---

## 3.3 Modify `rdith/roi_features.py`

Required changes:

1. Add `support_mode` config.
2. Support bbox/support points.
3. Implement distance-weighted ROI pooling.
4. Preserve existing feature names.
5. Add optional new features:

```text
rf_mean_residual_doppler_hz
rf_abs_residual_doppler_hz
rf_geometry_confidence
rf_blob_lifetime_mean
rf_blob_lifetime_max
rf_roi_support_distance_weighted
```

---

## 3.4 Modify `rdith/pipeline.py`

Required changes:

1. Use `load_rdith_calibration()` instead of raw JSON dictionary when possible.
2. Allow `--export_progress_summary` or always write summary beside output.
3. Return warnings, not only hard errors.
4. Save intermediate outputs when requested:

```text
--save_intermediate_dir debug_intermediate/
```

Intermediate outputs:

```text
active_cells.npz
residual_cells.npz
blobs.json
roi_features.csv
progress_summary.json
```

---

## 3.5 Add `rdith/visualization.py`

Implement all functions listed in Section 2.5.

Use only matplotlib/numpy/standard library unless already available. Do not require a GUI backend. Use `matplotlib.use("Agg")` in CLI scripts.

---

## 3.6 Add `scripts/visualize_rdith_progress.py`

Required behavior:

1. Load exported RDITH `.npz` feature or residual output.
2. Load optional heatmap, pose, ROI, and calibration files.
3. Generate the debug visualization folder.
4. Generate `index.html` and `summary.json`.
5. Never crash because ROI path is missing; if ROI is missing, generate only heatmap/world/global plots that are possible.

---

## 4. Stop conditions Codex should enforce

Immediately stop and report a clear error if any of these are true:

1. No raw CSI and no completed heatmap.
2. Completed heatmap exists but lacks `tau_axis` or `fd_axis`.
3. Heatmap has no timestamps and pose has no way to align.
4. 6DoF poses do not overlap heatmap timestamps.
5. Calibration cannot map RF coordinates to world/user coordinates.
6. ROI feature extraction is requested but ROI frames do not match heatmap frames.
7. Full vector velocity residual is requested but there are not enough independent RF Doppler projections.

Do not silently continue with invalid geometry. If a fallback is used, put it in warnings and progress summary.

---

## 5. Minimum acceptance tests

Add tests beyond the current tests.

### Test 1: scalar Doppler residual correctness

Create a point, a head velocity, and a monostatic link. Verify:

```text
residual_fd = measured_fd - expected_fd
```

### Test 2: vector residual is not used without enough links

With one link only, verify that the result mode is `scalar_doppler`, not `vector_velocity`.

### Test 3: bbox ROI pooling

Create one ROI bbox and two blobs. Verify the closer blob has larger weight.

### Test 4: support-point ROI pooling

Create support points and verify nearest support distance is used.

### Test 5: visualization outputs

Run visualization on synthetic outputs and verify these files exist:

```text
index.html
summary.json
feature_matrix_health.png
```

### Test 6: progress summary

Run the pipeline on synthetic data and verify `progress_summary.json` contains non-empty counts and warnings.

---

## 6. Final judgment

Current code status:

```text
High-level structure: good
Basic function coverage: partial/good
Scientific geometry correctness: incomplete
Residual Doppler correctness: needs correction
ROI feature extraction: partial
Visualization of current progress: insufficient
Testing: minimal smoke tests only
```

Therefore, the current code **partially matches** the request but should not be considered complete.

The next priority is:

1. Fix Doppler residual computation so scalar/vector dimensions are correct.
2. Improve calibration and heatmap geometry handling.
3. Improve ROI pooling with bbox/support-point modes.
4. Add real progress visualization and HTML/debug report.
5. Add machine-readable progress summary.
6. Add example schemas and stronger tests.

