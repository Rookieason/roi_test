# RDITH Phase 1 Visualization and Progress Reporting Tasks for Codex

## 0. Purpose of this document

This document tells Codex exactly what to implement next in the `roi_selection` project.

The current RDITH Phase 1 pipeline already produces numeric intermediate outputs, but it does not yet produce enough visual evidence to show what the project has achieved. The original heatmap generator produces many PNG heatmaps, while RDITH currently produces only a small number of `.npz` and `.json` files. That is not wrong, but it is not enough for debugging, demonstration, or research communication.

The goal of this task is to add a clear visualization and progress-reporting layer for RDITH Phase 1.

Phase 1 scope is strictly:

```text
Raw CSI or pre-generated Doppler heatmap
+
User 6DoF pose
↓
RDITH residual heatmap
↓
active cells / residual cells / motion blobs / global features
↓
visualization outputs
```

Do not require ROI definitions in Phase 1.

ROI pooling and ROI feature extraction are Phase 2 and must remain optional.

---

## 1. Current expected RDITH output structure

After running:

```bash
bash run_rdith_phase1_demo.sh
```

or equivalent, the current output looks like this:

```text
rdith_output_<timestamp>_<id>/
├── intermediate/
│   ├── active_cells.npz
│   ├── residual_cells.npz
│   ├── blobs.json
│   └── progress_summary.json
│
├── rdith_progress_summary.json
└── rdith_residual.npz
```

This file count is acceptable for the numeric RDITH Phase 1 pipeline, but it is not enough for human inspection.

The original heatmap generator may produce many PNG files under something like:

```text
heatmap_result/
├── figures/
│   └── <exp_name>/<heatmap_type>/...
└── mat/
    └── <exp_name>/<heatmap_type>/smoothed_CSI_avg.npz
```

This difference is expected because the original heatmap generator is primarily visualization-oriented, while RDITH is currently data-output-oriented.

Codex must add visualization outputs for RDITH so users can see the transformation from original Doppler heatmap to RDITH residual result.

---

## 2. Meaning of the current RDITH files

Codex should preserve these files and build visualizations from them.

### 2.1 `rdith_residual.npz`

This is the main Phase 1 RDITH output.

It should contain the final residual heatmap or equivalent residual representation.

Expected possible arrays:

```text
residual_map
residual_energy
residual_cells
active_cells
timestamps
tau_axis
fd_axis
velocity_axis
metadata
```

The exact key names may vary in the current code. Codex must inspect the actual `.npz` keys and make the visualization script robust to available keys.

Minimum expectation:

- There must be enough data to reconstruct or plot residual energy over time.
- If only sparse residual cells are available, the visualization script should still produce scatter-style visualizations.

### 2.2 `intermediate/active_cells.npz`

This contains the cells selected from the original heatmap before 6DoF residual filtering.

Conceptual meaning:

```text
Original Doppler heatmap cells whose energy passes threshold.
```

These are not yet RDITH residual cells. They are simply RF/Doppler-active cells.

Visualization meaning:

```text
Where the original Doppler heatmap thinks motion exists.
```

### 2.3 `intermediate/residual_cells.npz`

This contains cells after subtracting or suppressing the motion explainable by the user's 6DoF pose.

Conceptual meaning:

```text
Motion cells that remain important after accounting for HMD 6DoF.
```

This is the core RDITH result.

Visualization meaning:

```text
Where unexplained physical motion remains.
```

### 2.4 `intermediate/blobs.json`

This contains connected components or motion blobs derived from residual cells.

Conceptual meaning:

```text
Groups of residual cells treated as coherent physical motion regions.
```

Each blob should ideally include:

```json
{
  "frame": 12,
  "id": 0,
  "centroid": [tof_bin, doppler_bin],
  "energy": 0.82,
  "size": 127,
  "bbox": [min_row, min_col, max_row, max_col]
}
```

If the current format is different, Codex should add a compatibility layer rather than breaking existing output.

### 2.5 `intermediate/progress_summary.json`

This is a per-step or intermediate summary.

Expected information:

```text
number of original heatmap frames
number of active cells
number of residual cells
number of blobs
thresholds used
```

### 2.6 `rdith_progress_summary.json`

This is the top-level summary of the RDITH run.

It should be human-readable and should include enough metrics to quickly check whether the run was successful.

Codex should extend it if needed.

---

## 3. Main problem to solve

Current RDITH outputs are mostly `.npz` and `.json`. This is technically correct but hard to present.

The user needs visual proof of:

1. What the original heatmap looked like.
2. Which cells were selected as active.
3. Which cells survived as residual after 6DoF filtering.
4. Where the residual motion blobs are.
5. How residual energy changes over time.
6. Whether RDITH is doing something different from the original heatmap.

Codex must implement a visualization layer that creates PNG/HTML files under:

```text
rdith_output_<timestamp>_<id>/vis/
```

---

## 4. Required new visualization outputs

Codex must create the following outputs.

### 4.1 `vis/01_original_heatmap_frame_<k>.png`

Purpose:

Show the original ToF-Doppler or AoA-ToF-Doppler heatmap for selected frames.

Input sources:

- Prefer original heatmap from `heatmap_path`, usually `smoothed_CSI_avg.npz`.
- If unavailable, use data stored in `rdith_residual.npz` if it contains original/active maps.

For ToF-Doppler heatmap:

- x-axis: Doppler frequency or Doppler bin.
- y-axis: ToF/range bin.
- color: normalized heatmap energy.

For AoA-ToF-Doppler heatmap:

- If 4D, choose one of the following:
  - Max projection over AoA.
  - Max projection over Doppler.
  - Configurable projection mode.

Recommended default:

```text
For 3D spectrum: frame x tof x doppler -> plot tof x doppler.
For 4D spectrum: frame x aoa x tof x doppler -> max over aoa, plot tof x doppler.
```

### 4.2 `vis/02_active_cells_overlay_frame_<k>.png`

Purpose:

Show which cells were selected as active before 6DoF residual filtering.

Display:

- Background: original heatmap.
- Overlay: active cells as points or transparent mask.

This answers:

```text
What did the original RF/Doppler heatmap consider to be motion?
```

### 4.3 `vis/03_residual_cells_overlay_frame_<k>.png`

Purpose:

Show which cells remain after 6DoF residual processing.

Display:

- Background: original heatmap or active-cell map.
- Overlay: residual cells as points or mask.

This answers:

```text
What motion remains after removing motion explainable by HMD 6DoF?
```

### 4.4 `vis/04_active_vs_residual_frame_<k>.png`

Purpose:

Direct comparison between active cells and residual cells.

Recommended layout:

```text
Left: original heatmap + active cells
Right: original heatmap + residual cells
```

This is one of the most important visuals for explaining RDITH.

It should make the difference between standard Doppler heatmap and RDITH immediately visible.

### 4.5 `vis/05_blob_tracking_frame_<k>.png`

Purpose:

Show residual blobs detected from residual cells.

Display:

- Background: residual heatmap or original heatmap.
- Overlay:
  - Blob centroid.
  - Blob bounding box if available.
  - Blob id.
  - Blob energy if available.

This answers:

```text
What coherent motion regions did RDITH detect?
```

### 4.6 `vis/06_energy_timeline.png`

Purpose:

Show RDITH progress over time.

Plot one or more of:

```text
original heatmap total energy vs frame
active cell count vs frame
residual cell count vs frame
residual energy vs frame
blob count vs frame
largest blob energy vs frame
```

Minimum required lines:

- active cell count
- residual cell count
- blob count

If energy data exists, also include:

- total active energy
- total residual energy

This visualization is very important because it shows that RDITH is not just generating isolated files; it is tracking motion over time.

### 4.7 `vis/07_rdith_dashboard.html`

Purpose:

Create one human-readable dashboard summarizing the whole run.

This can be static HTML. It does not need a server.

The dashboard should include:

- Run metadata.
- Input paths.
- Number of frames.
- Heatmap type.
- Number of active cells.
- Number of residual cells.
- Number of blobs.
- Average residual energy.
- Peak residual frame.
- Links or embedded thumbnails of key PNGs.
- Explanation of what each output means.

The dashboard should be safe to open locally in a browser.

### 4.8 Optional: `vis/rdith_progress.gif` or `vis/rdith_progress.mp4`

Optional but recommended.

Purpose:

Show frame-by-frame comparison of original heatmap, active cells, residual cells, and blobs.

If implemented, use only dependencies already in the project or lightweight dependencies such as `imageio`.

Do not make this mandatory if it complicates setup.

---

## 5. Required new script

Codex should implement:

```text
scripts/visualize_rdith_phase1.py
```

### 5.1 CLI interface

The script should support:

```bash
python scripts/visualize_rdith_phase1.py \
  --rdith_output_dir <path/to/rdith_output> \
  --heatmap_path <path/to/smoothed_CSI_avg.npz> \
  --output_dir <path/to/rdith_output/vis> \
  --frames 0,10,20,50 \
  --max_frames 8
```

Arguments:

#### `--rdith_output_dir`

Required.

Directory containing:

```text
rdith_residual.npz
rdith_progress_summary.json
intermediate/active_cells.npz
intermediate/residual_cells.npz
intermediate/blobs.json
```

#### `--heatmap_path`

Optional but strongly recommended.

Path to original standard heatmap file.

Usually:

```text
heatmap_result/mat/<exp_name>/<heatmap_type>/smoothed_CSI_avg.npz
```

If not provided, the script must attempt to find the original heatmap from metadata or use whatever original/active map exists in `rdith_residual.npz`.

#### `--output_dir`

Optional.

Default:

```text
<rdith_output_dir>/vis
```

#### `--frames`

Optional.

Comma-separated frame indices to visualize.

Example:

```text
0,10,20,50
```

#### `--max_frames`

Optional.

If `--frames` is not provided, automatically choose up to this many informative frames.

Default: `8`.

Frame selection should prefer frames with high residual energy or high blob count.

#### `--projection`

Optional.

Default: `max_aoa`.

Allowed values:

```text
max_aoa
max_doppler
mean_aoa
slice_center_aoa
```

Used only when original heatmap is 4D or higher.

### 5.2 Robustness requirements

The script must not crash simply because a key is missing.

Instead:

- Print a warning.
- Skip the unavailable visualization.
- Continue generating other outputs.

Example:

```text
[WARN] residual_cells.npz not found. Skipping residual cell overlay.
```

### 5.3 Dependency requirements

Use only common dependencies:

```text
numpy
matplotlib
json
pathlib
argparse
```

Optional:

```text
imageio
```

Do not require heavy visualization frameworks.

---

## 6. Required modifications to `run_rdith_phase1_demo.sh`

After RDITH Phase 1 numeric output is generated, the demo script should automatically run the visualization script.

Add a step like:

```bash
echo "[RDITH] Visualizing RDITH Phase 1 outputs"

python3 "$ROOT_DIR/scripts/visualize_rdith_phase1.py" \
  --rdith_output_dir "$OUTPUT_DIR" \
  --heatmap_path "$HEATMAP_FILE" \
  --output_dir "$OUTPUT_DIR/vis" \
  --max_frames 8
```

At the end of the script, print:

```bash
echo "[RDITH] Done. Main outputs:"
echo "  Numeric output: $OUTPUT_DIR/rdith_residual.npz"
echo "  Summary:        $OUTPUT_DIR/rdith_progress_summary.json"
echo "  Visualization:  $OUTPUT_DIR/vis/07_rdith_dashboard.html"
```

---

## 7. Required progress summary improvements

Codex should ensure that `rdith_progress_summary.json` contains enough information for the visualization script and for humans.

Suggested schema:

```json
{
  "phase": "phase1",
  "input": {
    "heatmap_path": "...",
    "raw_csi_path": "...",
    "pose_path": "...",
    "config_path": "..."
  },
  "heatmap": {
    "heatmap_type": "ToF-Doppler",
    "num_frames": 120,
    "spectrum_shape": [120, 64, 128],
    "has_tau_axis": true,
    "has_fd_axis": true
  },
  "rdith": {
    "active_cell_count_total": 12345,
    "residual_cell_count_total": 2345,
    "active_cell_count_per_frame": [100, 95, 110],
    "residual_cell_count_per_frame": [20, 18, 25],
    "blob_count_total": 80,
    "blob_count_per_frame": [1, 0, 2],
    "residual_energy_per_frame": [0.12, 0.08, 0.2],
    "peak_residual_frame": 32
  },
  "visualization": {
    "created": true,
    "output_dir": ".../vis",
    "dashboard": ".../vis/07_rdith_dashboard.html"
  }
}
```

If some values are unavailable, set them to `null` rather than omitting them.

---

## 8. Required tests

Codex should add tests for the visualization layer.

### 8.1 Test file

Add:

```text
tests/test_phase1_visualization.py
```

### 8.2 Test 1: visualization script can run on synthetic data

Create a temporary output directory with:

```text
rdith_residual.npz
intermediate/active_cells.npz
intermediate/residual_cells.npz
intermediate/blobs.json
rdith_progress_summary.json
```

Also create a synthetic original heatmap:

```text
smoothed_CSI_avg.npz
```

Use a small shape such as:

```text
frames = 5
tof_bins = 16
doppler_bins = 32
```

Run:

```bash
python scripts/visualize_rdith_phase1.py \
  --rdith_output_dir <tmp_rdith> \
  --heatmap_path <tmp_heatmap> \
  --output_dir <tmp_rdith>/vis \
  --max_frames 2
```

Assert that these files exist:

```text
vis/06_energy_timeline.png
vis/07_rdith_dashboard.html
```

At least one frame visualization should also exist.

### 8.3 Test 2: missing optional files should not crash

Create only:

```text
rdith_residual.npz
rdith_progress_summary.json
```

Run visualization.

Expected:

- It should not crash.
- It should still create `07_rdith_dashboard.html`.
- It should include warnings or notes about missing active/residual/blob files.

### 8.4 Test 3: 4D heatmap projection works

Create synthetic heatmap:

```text
shape = [frames, aoa_bins, tof_bins, doppler_bins]
```

Run visualization with:

```bash
--projection max_aoa
```

Assert at least one original heatmap PNG is produced.

### 8.5 Test 4: summary JSON contains visualization metadata

After visualization, check that either:

- `rdith_progress_summary.json` is updated with visualization metadata, or
- a new `vis/visualization_summary.json` is created.

Recommended output:

```text
vis/visualization_summary.json
```

This avoids modifying the original RDITH summary unexpectedly.

---

## 9. Data format compatibility requirements

Codex must support the current project output formats as much as possible.

### 9.1 Heatmap file

The original heatmap file may contain:

```text
spectrum
spectrums
arr_0
```

Codex should load whichever exists first.

Expected spectrum shapes:

```text
[frames, tof, doppler]
[frames, aoa, tof, doppler]
[frames, tof, doppler, something_else]
```

For unknown shapes, print shape and attempt a reasonable projection.

### 9.2 Active/residual cell files

The sparse cell files may contain different key names.

Support common possibilities:

```text
frame
frames
frame_idx
indices
coords
cells
energy
energies
values
```

If the data is dense rather than sparse, support dense mask plotting.

### 9.3 Blob JSON

Support both:

```json
[
  {"frame": 0, "id": 1, "centroid": [10, 20]}
]
```

and:

```json
{
  "blobs": [
    {"frame": 0, "id": 1, "centroid": [10, 20]}
  ]
}
```

If frame information is absent, treat all blobs as global and show them on selected frames with a warning.

---

## 10. Recommended implementation structure

Inside `scripts/visualize_rdith_phase1.py`, implement helper functions:

```python
def load_npz_flexible(path):
    """Load npz/npy and return dict-like arrays."""


def load_original_heatmap(path):
    """Return spectrum, axes, metadata."""


def project_heatmap_frame(spectrum, frame_idx, projection):
    """Convert one heatmap frame to 2D for plotting."""


def load_sparse_cells(path):
    """Load active/residual cells in a robust schema-compatible way."""


def cells_for_frame(cells, frame_idx):
    """Return cell coordinates and values for one frame."""


def load_blobs(path):
    """Load blobs.json and return a list of blob dictionaries."""


def blobs_for_frame(blobs, frame_idx):
    """Return blobs belonging to one frame."""


def select_frames(summary, residual_cells, blobs, max_frames):
    """Choose informative frames with large residual energy/blob count."""


def plot_original_heatmap(...):
    pass


def plot_active_overlay(...):
    pass


def plot_residual_overlay(...):
    pass


def plot_active_vs_residual(...):
    pass


def plot_blob_tracking(...):
    pass


def plot_energy_timeline(...):
    pass


def write_dashboard_html(...):
    pass
```

Avoid putting all logic in `main()`.

---

## 11. Visual explanation text to include in dashboard

The dashboard should include a short explanation section:

```text
RDITH Phase 1 takes the standard Doppler heatmap and user 6DoF pose as input.
The standard heatmap shows all RF/Doppler motion energy.
Active cells are high-energy cells selected from the standard heatmap.
Residual cells are active cells that remain after suppressing motion explainable by the headset 6DoF.
Blobs are connected groups of residual cells and represent coherent unexplained physical motion.
This phase does not require ROI definitions. ROI-specific pooling is reserved for Phase 2.
```

This text is important because many people will incorrectly think RDITH should produce many heatmap PNGs like the original generator. The dashboard must explain that RDITH is a residual-feature pipeline, not merely another heatmap renderer.

---

## 12. What not to do

Do not add ROI requirement to Phase 1.

Do not require `roi.json` for visualization.

Do not remove the existing numeric outputs.

Do not make visualization depend on browser-only or server-based tools.

Do not require real CSI data for tests; use synthetic arrays.

Do not fail the whole script if one visualization cannot be generated.

Do not overwrite the original heatmap generator outputs.

---

## 13. Acceptance criteria

The task is complete when the following command produces both numeric RDITH output and visualization output:

```bash
bash run_rdith_phase1_demo.sh
```

Expected final directory:

```text
rdith_output_<timestamp>_<id>/
├── intermediate/
│   ├── active_cells.npz
│   ├── residual_cells.npz
│   ├── blobs.json
│   └── progress_summary.json
│
├── vis/
│   ├── 01_original_heatmap_frame_*.png
│   ├── 02_active_cells_overlay_frame_*.png
│   ├── 03_residual_cells_overlay_frame_*.png
│   ├── 04_active_vs_residual_frame_*.png
│   ├── 05_blob_tracking_frame_*.png
│   ├── 06_energy_timeline.png
│   ├── 07_rdith_dashboard.html
│   └── visualization_summary.json
│
├── rdith_progress_summary.json
└── rdith_residual.npz
```

Tests should pass:

```bash
pytest tests/test_phase1_visualization.py
```

A human should be able to open:

```text
rdith_output_<timestamp>_<id>/vis/07_rdith_dashboard.html
```

and understand:

1. What original Doppler heatmap was used.
2. What active cells were selected.
3. What residual cells remained after 6DoF processing.
4. What blobs were detected.
5. How residual activity changed over time.
6. Why no ROI definition is required in Phase 1.
