#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize an exported RDITH residual heatmap NPZ.")
    parser.add_argument("input_npz")
    parser.add_argument("--output_dir", default="rdith_residual_debug")
    parser.add_argument("--max_frames", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input_npz, allow_pickle=True) as z:
        if "residual_spectrum" in z:
            _plot_dense(z["residual_spectrum"], output_dir, args.max_frames)
        elif "residual_cells" in z:
            _plot_sparse(z["residual_cells"], output_dir, args.max_frames)
        else:
            raise ValueError("Expected residual_spectrum or residual_cells in NPZ")


def _plot_dense(spectrum: np.ndarray, output_dir: Path, max_frames: int) -> None:
    for frame_idx in range(min(max_frames, spectrum.shape[0])):
        frame = spectrum[frame_idx]
        if frame.ndim == 3:
            frame = frame.max(axis=1)
        plt.figure(figsize=(8, 5))
        plt.imshow(frame, aspect="auto", origin="lower", cmap="magma")
        plt.colorbar(label="Residual energy")
        plt.xlabel("Doppler bin")
        plt.ylabel("ToF bin")
        plt.title(f"RDITH residual frame {frame_idx}")
        plt.tight_layout()
        plt.savefig(output_dir / f"residual_{frame_idx:06d}.png", dpi=150)
        plt.close()


def _plot_sparse(cells: np.ndarray, output_dir: Path, max_frames: int) -> None:
    if cells.size == 0:
        return
    for frame_idx in range(min(max_frames, int(cells[:, 0].max()) + 1)):
        frame = cells[cells[:, 0] == frame_idx]
        if frame.size == 0:
            continue
        plt.figure(figsize=(8, 5))
        plt.scatter(frame[:, 3], frame[:, 1], c=frame[:, 5], cmap="magma", s=18)
        plt.colorbar(label="Residual energy")
        plt.xlabel("Doppler bin")
        plt.ylabel("ToF bin")
        plt.title(f"RDITH residual cells {frame_idx}")
        plt.tight_layout()
        plt.savefig(output_dir / f"residual_{frame_idx:06d}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
