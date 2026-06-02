import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from rdith.blob_extraction import compute_doppler_entropy, compute_micro_doppler_bandwidth, extract_rf_blobs
from rdith.calibration import RFLinkCalibration, load_rdith_calibration
from rdith.data_types import CandidateROI, Pose6DoF
from rdith.doppler_residual import compute_scalar_residual_doppler
from rdith.pipeline import run_rdith_pipeline
import rdith.pipeline as rdith_pipeline
from rdith.pose_utils import align_pose_to_heatmap_timestamps, compute_head_kinematics
from rdith.residual_heatmap import (
    build_residual_heatmap,
    compute_residual_motion_cells,
    estimate_rf_velocity,
    extract_active_heatmap_cells,
    map_heatmap_cells_to_world,
)
from rdith.roi_features import merge_features_for_ml, rf_roi_align
from rdith.roi_features import compute_roi_blob_weight, roi_support_distance
from rdith.validation import validate_rdith_inputs
from rdith.visualization import plot_feature_matrix_health, write_visualization_report


class RDITHTests(unittest.TestCase):
    def _heatmap(self):
        return {
            "spectrum": np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]]),
            "timestamps": np.array([0.0, 1.0]),
            "tau_axis": np.array([1e-9, 2e-9]),
            "fd_axis": np.array([-2.0, 0.0, 2.0]),
            "theta_deg_axis": None,
            "heatmap_type": "ToF-Doppler",
            "axis_order": ("time", "tau", "fd"),
            "metadata": {"center_frequency": 5.57e9},
        }

    def _poses(self):
        return [
            Pose6DoF(0.0, np.array([0.0, 0.0, 0.0]), np.eye(3)),
            Pose6DoF(1.0, np.array([0.0, 0.0, 0.1]), np.eye(3)),
        ]

    def test_pose_alignment_and_kinematics(self):
        aligned = align_pose_to_heatmap_timestamps(self._poses(), np.array([0.0, 0.5, 1.0]))
        self.assertEqual(len(aligned), 3)
        self.assertAlmostEqual(aligned[1].position_world[2], 0.05)
        kin = compute_head_kinematics(aligned)
        self.assertEqual(kin["linear_velocity_world"].shape, (3, 3))

    def test_load_poses_from_agent6_csv(self):
        import csv as csv_module

        rows = [
            {
                "recv_unix": "1768626922.4573631",
                "source": "pose_6dof",
                "endpoint": "tcp://127.0.0.1:51010",
                "type": "json",
                "topic": "agent6.oculus-quest2-c1a18f39db5d1e5e",
                "payload_json": json.dumps(
                    {
                        "ts_utc": "2026-01-17T05:15:22.757Z",
                        "pos": [8.16, 16.803, 147.171],
                        "rot_deg": [77.746, 3.291, -0.346],
                        "rssi_dbm": -36,
                    }
                ),
            },
            {
                "recv_unix": "1768626922.490402",
                "source": "pose_6dof",
                "endpoint": "tcp://127.0.0.1:51010",
                "type": "json",
                "topic": "agent6.oculus-quest2-c1a18f39db5d1e5e",
                "payload_json": json.dumps(
                    {
                        "ts_utc": "2026-01-17T05:15:22.790Z",
                        "pos": [8.268, 16.778, 147.18],
                        "rot_deg": [77.364, 3.336, -0.318],
                        "rssi_dbm": -36,
                    }
                ),
            },
        ]
        fieldnames = [
            "recv_unix",
            "source",
            "endpoint",
            "type",
            "topic",
            "payload_json",
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "poses.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv_module.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            poses = rdith_pipeline._load_poses(str(path))
        self.assertEqual(len(poses), 2)
        self.assertAlmostEqual(poses[0]["timestamp"], 1768626922.757)
        self.assertEqual(poses[0]["position_world"], [8.16, 16.803, 147.171])

    def test_sparse_residual_to_ml_features(self):
        heatmap = self._heatmap()
        calibration = {"center_frequency": 5.57e9, "rf_origin_world": [0, 0, 0], "tof_range_scale": 1.0}
        validate_rdith_inputs(heatmap, self._poses(), [[{}], [{}]], calibration)
        kin = compute_head_kinematics(align_pose_to_heatmap_timestamps(self._poses(), heatmap["timestamps"]))
        active = extract_active_heatmap_cells(heatmap, threshold_value=80, max_cells_per_frame=4)
        cell_map = map_heatmap_cells_to_world(heatmap, calibration)
        rf = estimate_rf_velocity(active, cell_map, heatmap, calibration)
        residual = compute_residual_motion_cells(rf, kin, calibration)
        sparse = build_residual_heatmap(heatmap, residual)
        blobs = extract_rf_blobs(sparse, spatial_eps=1.0, doppler_eps=10.0)
        rois = [
            [CandidateROI("roi1", 0.0, np.array([0.0, 0.0, 0.0]))],
            [CandidateROI("roi1", 1.0, np.array([0.0, 0.0, 0.0]))],
        ]
        roi_features = rf_roi_align(blobs, rois, kin, {"roi_support_radius_m": 10.0})
        ml = merge_features_for_ml(rois, roi_features, [{}, {}])
        self.assertEqual(ml["X"].shape[0], 2)
        self.assertIn("rf_residual_energy", ml["feature_names"])

    def test_dense_residual_uses_tof_doppler_fd_index(self):
        heatmap = self._heatmap()
        residual = [
            [
                {
                    "cell_index": (1, -1, 2),
                    "residual_energy": 9.0,
                }
            ],
            [],
        ]
        dense = build_residual_heatmap(heatmap, residual, output_mode="dense")
        spectrum = dense["spectrum"]
        self.assertEqual(spectrum[0, 1, 2], 9.0)
        self.assertEqual(spectrum[0, 1, 1], 0.0)

    def test_doppler_statistics(self):
        bandwidth = compute_micro_doppler_bandwidth(np.array([-1.0, 1.0]), np.array([1.0, 1.0]))
        entropy = compute_doppler_entropy(np.array([-1.0, 1.0]), np.array([1.0, 1.0]), num_bins=2)
        self.assertAlmostEqual(bandwidth, 1.0)
        self.assertGreater(entropy, 0.0)

    def test_scalar_doppler_residual_correctness(self):
        link = RFLinkCalibration(
            link_id="link_0",
            tx_position_world=np.array([0.0, 0.0, 0.0]),
            rx_position_world=np.array([0.0, 0.0, 0.0]),
            tx_rotation_world_from_rf=None,
            rx_rotation_world_from_rf=None,
            center_frequency_hz=1.0,
            wavelength_m=0.5,
        )
        result = compute_scalar_residual_doppler(
            measured_fd_hz=10.0,
            point_world=np.array([0.0, 0.0, 1.0]),
            expected_6dof_velocity_world=np.array([0.0, 0.0, 1.0]),
            link=link,
            wavelength_m=0.5,
            mode="monostatic",
        )
        self.assertAlmostEqual(result["expected_fd_hz"], 4.0)
        self.assertAlmostEqual(result["residual_fd_hz"], 6.0)

    def test_vector_residual_not_used_with_one_link(self):
        heatmap = self._heatmap()
        calibration = {"geometry_mode": "monostatic_aoa", "center_frequency": 5.57e9, "rf_origin_world": [0, 0, 0]}
        kin = compute_head_kinematics(align_pose_to_heatmap_timestamps(self._poses(), heatmap["timestamps"]))
        active = extract_active_heatmap_cells(heatmap, threshold_value=99, max_cells_per_frame=2)
        cell_map = map_heatmap_cells_to_world(heatmap, calibration)
        rf = estimate_rf_velocity(active, cell_map, heatmap, calibration)
        residual = compute_residual_motion_cells(rf, kin, calibration)
        self.assertTrue(all(r["residual_mode"] == "scalar_doppler" for frame in residual for r in frame))
        self.assertTrue(all(r["residual_velocity_world"] is None for frame in residual for r in frame))

    def test_bbox_and_support_point_roi_pooling(self):
        close = self._blob(np.array([0.1, 0.0, 0.1]), residual=10.0)
        far = self._blob(np.array([3.0, 0.0, 3.0]), residual=10.0)
        roi = CandidateROI(
            "box",
            0.0,
            np.array([0.0, 0.0, 0.0]),
            bbox_min_world=np.array([-0.5, -0.5, -0.5]),
            bbox_max_world=np.array([0.5, 0.5, 0.5]),
        )
        d_close = roi_support_distance(close, roi, "bbox")
        d_far = roi_support_distance(far, roi, "bbox")
        self.assertLess(d_close, d_far)
        self.assertGreater(compute_roi_blob_weight(close, roi, d_close, {}), compute_roi_blob_weight(far, roi, d_far, {}))
        roi.support_points_world = np.array([[2.9, 0.0, 3.1]])
        self.assertLess(roi_support_distance(far, roi, "support_points"), 0.3)

    def test_visualization_outputs(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            ml = {
                "X": np.array([[1.0, np.nan], [0.0, np.inf]]),
                "feature_names": ["rf_residual_energy", "rf_confidence"],
            }
            summary = plot_feature_matrix_health(ml, str(out / "feature_matrix_health.png"))
            write_visualization_report(str(out), summary, [str(out / "feature_matrix_health.png")])
            self.assertTrue((out / "index.html").exists())
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "feature_matrix_health.png").exists())

    def test_pipeline_progress_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            heatmap_path = root / "heatmap.npz"
            np.savez_compressed(
                heatmap_path,
                spectrum=self._heatmap()["spectrum"],
                timestamps=self._heatmap()["timestamps"],
                tau_axis=self._heatmap()["tau_axis"],
                fd_axis=self._heatmap()["fd_axis"],
            )
            pose_path = root / "poses.json"
            roi_path = root / "rois.json"
            cal_path = root / "calibration.json"
            cfg_path = root / "config.json"
            pose_path.write_text(
                '{"poses":[{"timestamp":0.0,"position_world":[0,0,0],"quaternion_xyzw":[0,0,0,1]},'
                '{"timestamp":1.0,"position_world":[0,0,0.1],"quaternion_xyzw":[0,0,0,1]}]}',
                encoding="utf-8",
            )
            roi_path.write_text(
                '{"frames":[[{"roi_id":"r","timestamp":0.0,"center_world":[0,0,0],"visibility":1.0}],'
                '[{"roi_id":"r","timestamp":1.0,"center_world":[0,0,0],"visibility":1.0}]]}',
                encoding="utf-8",
            )
            cal_path.write_text(
                '{"geometry_mode":"tof_only_pseudo","center_frequency":5570000000.0,"rf_origin_world":[0,0,0]}',
                encoding="utf-8",
            )
            cfg_path.write_text('{"active_cell_config":{"threshold_value":80}}', encoding="utf-8")
            result = run_rdith_pipeline(
                heatmap_path=str(heatmap_path),
                raw_csi_path=None,
                pose_path=str(pose_path),
                roi_path=str(roi_path),
                calibration_path=str(cal_path),
                config_path=str(cfg_path),
                output_path=str(root / "features.npz"),
                heatmap_type="ToF-Doppler",
            )
            self.assertTrue((root / "rdith_progress_summary.json").exists())
            self.assertGreater(result["progress_summary"]["num_residual_cells"], 0)
            self.assertTrue(result["progress_summary"]["warnings"])

    def _blob(self, centroid, residual=1.0):
        from rdith.data_types import RFBlob

        return RFBlob(
            blob_id=0,
            timestamp=0.0,
            centroid_world=centroid,
            velocity_world=None,
            energy=1.0,
            residual_energy=residual,
            doppler_mean_hz=1.0,
            doppler_bandwidth_hz=0.0,
            doppler_entropy=0.0,
            confidence=1.0,
            num_supporting_cells=1,
            metadata={"mean_residual_doppler_hz": 1.0, "geometry_confidence": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
