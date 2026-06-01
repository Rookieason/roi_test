#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rdith.heatmap_adapter import load_heatmap_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RDITH Phase 1 outputs.")
    parser.add_argument("--rdith_output_dir", required=True)
    parser.add_argument("--heatmap_path", default=None)
    parser.add_argument("--heatmap_type", default="ToF-Doppler", choices=["ToF-Doppler", "AoA-ToF-Doppler"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--frames", default=None, help="Comma-separated frame indices to visualize.")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument(
        "--projection",
        default="max_aoa",
        choices=["max_aoa", "max_doppler", "mean_aoa", "slice_center_aoa"],
        help="Projection method for 4D heatmaps.",
    )
    args = parser.parse_args()

    rdith_output_dir = Path(args.rdith_output_dir).expanduser().resolve()
    if not rdith_output_dir.exists():
        raise FileNotFoundError(f"RDITH output directory not found: {rdith_output_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else rdith_output_dir / "vis"
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    summary: dict[str, Any] = {
        "rdith_output_dir": str(rdith_output_dir),
        "heatmap_path": args.heatmap_path,
        "projection": args.projection,
        "frames_selected": [],
        "warnings": warnings,
    }

    heatmap = None
    if args.heatmap_path:
        heatmap = load_heatmap(args.heatmap_path, args.heatmap_type, warnings)
    else:
        warnings.append("heatmap_path not provided; original heatmap plots will be skipped.")

    rdith_npz = rdith_output_dir / "rdith_residual.npz"
    if not rdith_npz.exists():
        raise FileNotFoundError(f"Missing rdith_residual.npz in {rdith_output_dir}")
    rdith = load_npz(rdith_npz)

    active_cells = load_cells(rdith_output_dir / "intermediate" / "active_cells.npz", warnings)
    residual_cells = load_cells(rdith_output_dir / "intermediate" / "residual_cells.npz", warnings)
    blobs = load_blobs(rdith_output_dir / "intermediate" / "blobs.json", warnings)
    progress_summary = load_json_if_exists(rdith_output_dir / "rdith_progress_summary.json", warnings)
    visualization_summary = {
        "active_cells_loaded": active_cells is not None,
        "residual_cells_loaded": residual_cells is not None,
        "blobs_loaded": blobs is not None,
        "progress_summary_loaded": progress_summary is not None,
    }

    frame_indices = parse_frames_arg(args.frames, args.max_frames, heatmap, active_cells, residual_cells, blobs, warnings)
    summary["frames_selected"] = frame_indices

    original_shape = infer_heatmap_frame_shape(heatmap)
    residual_counts = frame_counts(residual_cells, frame_indices)
    active_counts = frame_counts(active_cells, frame_indices)
    blob_counts = frame_counts(blobs, frame_indices, blob=True)

    if heatmap is not None:
        for frame_idx in frame_indices:
            plot_original_heatmap(heatmap, frame_idx, output_dir, args.projection)
            if active_cells is not None:
                plot_cells_overlay(
                    heatmap,
                    active_cells,
                    frame_idx,
                    output_dir / f"02_active_cells_overlay_frame_{frame_idx:03d}.png",
                    title=f"Active cells overlay frame {frame_idx}",
                    label="Active cells",
                )
            if residual_cells is not None:
                plot_cells_overlay(
                    heatmap,
                    residual_cells,
                    frame_idx,
                    output_dir / f"03_residual_cells_overlay_frame_{frame_idx:03d}.png",
                    title=f"Residual cells overlay frame {frame_idx}",
                    label="Residual cells",
                    marker="x",
                    color="red",
                )
            if active_cells is not None and residual_cells is not None:
                plot_active_vs_residual(
                    heatmap,
                    active_cells,
                    residual_cells,
                    frame_idx,
                    output_dir / f"04_active_vs_residual_frame_{frame_idx:03d}.png",
                    projection=args.projection,
                )
            if blobs is not None:
                plot_blob_tracking(
                    heatmap,
                    blobs,
                    frame_idx,
                    output_dir / f"05_blob_tracking_frame_{frame_idx:03d}.png",
                    title=f"Residual blobs frame {frame_idx}",
                )
    else:
        warnings.append("Original heatmap unavailable; only cell and blob scatter plots may be generated.")
        for frame_idx in frame_indices:
            if active_cells is not None:
                plot_scatter_cells(
                    active_cells,
                    frame_idx,
                    output_dir / f"02_active_cells_overlay_frame_{frame_idx:03d}.png",
                    title=f"Active cells frame {frame_idx}",
                    label="Active cells",
                )
            if residual_cells is not None:
                plot_scatter_cells(
                    residual_cells,
                    frame_idx,
                    output_dir / f"03_residual_cells_overlay_frame_{frame_idx:03d}.png",
                    title=f"Residual cells frame {frame_idx}",
                    label="Residual cells",
                    marker="x",
                    color="red",
                )
            if active_cells is not None and residual_cells is not None:
                plot_scatter_compare(
                    active_cells,
                    residual_cells,
                    frame_idx,
                    output_dir / f"04_active_vs_residual_frame_{frame_idx:03d}.png",
                )
            if blobs is not None:
                plot_blob_scatter(
                    blobs,
                    frame_idx,
                    output_dir / f"05_blob_tracking_frame_{frame_idx:03d}.png",
                    title=f"Residual blobs frame {frame_idx}",
                )

    plot_energy_timeline(
        output_dir / "06_energy_timeline.png",
        frame_indices,
        active_counts,
        residual_counts,
        blob_counts,
    )

    write_dashboard_html(output_dir / "07_rdith_dashboard.html", progress_summary, visualization_summary, output_dir)
    write_json(output_dir / "visualization_summary.json", {**summary, **visualization_summary})

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(" -", w)

    print(f"Visualization completed: {output_dir}")


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {key: z[key] for key in z.files}


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_if_exists(path: Path, warnings: list[str]) -> Any | None:
    if path.exists():
        try:
            return load_json(path)
        except Exception as exc:
            warnings.append(f"Failed to load JSON {path}: {exc}")
    return None


def load_cells(path: Path, warnings: list[str]) -> np.ndarray | list[Any] | None:
    if not path.exists():
        warnings.append(f"Missing cell file: {path}")
        return None
    try:
        data = np.load(path, allow_pickle=True)
        if "frames" in data:
            return np.asarray(data["frames"], dtype=object)
        if "arr_0" in data and data["arr_0"].dtype == object:
            return np.asarray(data["arr_0"], dtype=object)
        if "arr_0" in data:
            return np.asarray(data["arr_0"], dtype=float)
        return {key: np.asarray(data[key], dtype=object) for key in data.files}
    except Exception as exc:
        warnings.append(f"Failed to load cells from {path}: {exc}")
        return None


def load_blobs(path: Path, warnings: list[str]) -> list[dict[str, Any]] | None:
    if not path.exists():
        warnings.append(f"Missing blob file: {path}")
        return None
    try:
        data = load_json(path)
        if isinstance(data, dict) and "blobs" in data:
            return data["blobs"]
        if isinstance(data, list):
            return data
        warnings.append(f"Unexpected blob JSON structure in {path}")
    except Exception as exc:
        warnings.append(f"Failed to load blobs from {path}: {exc}")
    return None


def load_heatmap(path: str, heatmap_type: str, warnings: list[str]) -> dict[str, Any] | None:
    try:
        return load_heatmap_result(path, heatmap_type)
    except Exception as exc:
        warnings.append(f"Failed to load heatmap from {path}: {exc}")
    return None


def infer_heatmap_frame_shape(heatmap: dict[str, Any] | None) -> tuple[int, int] | None:
    if heatmap is None:
        return None
    spectrum = np.asarray(heatmap["spectrum"])
    if spectrum.ndim == 3:
        return spectrum.shape[1], spectrum.shape[2]
    if spectrum.ndim == 4:
        return spectrum.shape[1], spectrum.shape[3]
    if spectrum.ndim == 2:
        return spectrum.shape[1], spectrum.shape[2] if spectrum.ndim > 1 else (0, 0)
    return (spectrum.shape[1], spectrum.shape[-1])


def frame_counts(cells: Any, frame_indices: list[int], blob: bool = False) -> list[int]:
    if cells is None:
        return [0] * len(frame_indices)
    if blob:
        return [len(blobs_for_frame(cells, idx)) for idx in frame_indices]
    if isinstance(cells, np.ndarray) and cells.dtype == object:
        return [len(cells[idx]) if idx < len(cells) else 0 for idx in frame_indices]
    if isinstance(cells, np.ndarray) and cells.ndim == 2:
        counts = Counter(int(row[0]) for row in cells)
        return [counts[idx] for idx in frame_indices]
    if isinstance(cells, list):
        return [len(cells[idx]) if idx < len(cells) else 0 for idx in frame_indices]
    return [0] * len(frame_indices)


def parse_frames_arg(
    frames_arg: str | None,
    max_frames: int,
    heatmap: dict[str, Any] | None,
    active_cells: Any,
    residual_cells: Any,
    blobs: Any,
    warnings: list[str],
) -> list[int]:
    if frames_arg:
        frames = []
        for token in frames_arg.split(','):
            try:
                frames.append(int(token.strip()))
            except ValueError:
                warnings.append(f"Invalid frame token: {token}")
        return sorted(set(frames))[:max_frames]

    frame_scores: Counter[int] = Counter()
    counts = {}
    if residual_cells is not None:
        if isinstance(residual_cells, np.ndarray) and residual_cells.dtype == object:
            for idx, frame in enumerate(residual_cells):
                counts[idx] = len(frame)
        elif isinstance(residual_cells, np.ndarray) and residual_cells.ndim == 2:
            counts = Counter(int(row[0]) for row in residual_cells)
    if blobs is not None:
        for blob in blobs_for_output(blobs):
            frame = frame_from_blob(blob, warnings)
            frame_scores[frame] += 2
    for frame, cnt in counts.items():
        frame_scores[frame] += cnt

    if frame_scores:
        selected = [frame for frame, _ in frame_scores.most_common(max_frames)]
        return sorted(selected)

    num_frames = 0
    if heatmap is not None:
        spectrum = np.asarray(heatmap["spectrum"])
        num_frames = spectrum.shape[0]
    elif isinstance(active_cells, np.ndarray) and active_cells.dtype == object:
        num_frames = len(active_cells)
    elif isinstance(residual_cells, np.ndarray) and residual_cells.dtype == object:
        num_frames = len(residual_cells)
    elif blobs is not None:
        num_frames = max(frame_from_blob(blob, warnings) for blob in blobs_for_output(blobs)) + 1

    if num_frames <= 0:
        warnings.append("No frame count could be inferred; defaulting to frame 0.")
        return [0]
    return list(range(min(num_frames, max_frames)))


def blobs_for_output(blobs: Any) -> list[dict[str, Any]]:
    if not isinstance(blobs, list):
        raise TypeError("blobs.json must be a list of frames: list[list[blob]].")

    out = []
    for frame_idx, frame_blobs in enumerate(blobs):
        if not isinstance(frame_blobs, list):
            raise TypeError(f"blobs[{frame_idx}] must be a list of blob dicts.")

        for blob in frame_blobs:
            if not isinstance(blob, dict):
                raise TypeError(f"blobs[{frame_idx}] contains non-dict blob: {type(blob)!r}")

            if "frame_idx" not in blob:
                blob["frame_idx"] = frame_idx

            out.append(blob)

    return out


def frame_from_blob(blob: Any, warnings: list[str] | None = None) -> int:
    if not isinstance(blob, dict):
        raise TypeError(f"blob must be dict, got {type(blob)!r}")
    return int(blob["frame_idx"])


def safe_int(value: Any, default: int = 0, warnings: list[str] | None = None, field: str = "frame") -> int:
    try:
        return int(value)
    except Exception:
        if warnings is not None:
            warnings.append(f"Unable to parse blob {field} value {value!r}; using {default}.")
        return default


def cells_for_frame(cells: Any, frame_idx: int) -> list[dict[str, Any]]:
    if cells is None:
        return []
    if isinstance(cells, np.ndarray) and cells.dtype == object:
        if frame_idx < len(cells):
            frame = cells[frame_idx]
            if frame is None:
                return []
            return [normalize_cell_record(r) for r in list(frame)]
        return []
    if isinstance(cells, np.ndarray) and cells.ndim == 2:
        return [normalize_cell_record(row) for row in cells[cells[:, 0] == frame_idx]]
    if isinstance(cells, list):
        if frame_idx < len(cells):
            return [normalize_cell_record(r) for r in list(cells[frame_idx])]
        return []
    return []


def normalize_cell_record(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        out = {}
        if "tau_idx" in record and "fd_idx" in record:
            out["tau_idx"] = int(record["tau_idx"])
            out["fd_idx"] = int(record["fd_idx"])
            out["energy"] = float(record.get("energy", 1.0))
            return out
        if "cell_index" in record:
            idx = list(record["cell_index"])
            out["tau_idx"] = int(idx[0])
            out["fd_idx"] = int(idx[-1])
            out["energy"] = float(record.get("residual_energy", record.get("energy", 1.0)))
            return out
        if "frame_idx" in record and "fd_idx" in record and "tau_idx" in record:
            out["tau_idx"] = int(record["tau_idx"])
            out["fd_idx"] = int(record["fd_idx"])
            out["energy"] = float(record.get("energy", 1.0))
            return out
    if isinstance(record, np.ndarray) or isinstance(record, list):
        values = list(record)
        if len(values) >= 5:
            return {"tau_idx": int(values[1]), "fd_idx": int(values[3]), "energy": float(values[4])}
        if len(values) >= 4:
            return {"tau_idx": int(values[1]), "fd_idx": int(values[3]), "energy": float(values[4]) if len(values) > 4 else 1.0}
    return {"tau_idx": 0, "fd_idx": 0, "energy": 1.0}


def blobs_for_frame(blobs: Any, frame_idx: int) -> list[dict[str, Any]]:
    if blobs is None:
        return []
    blob_list = blobs_for_output(blobs)
    return [blob for blob in blob_list if frame_from_blob(blob) == frame_idx]


def select_blob_centroid(blob: dict[str, Any]) -> tuple[float, float] | None:
    centroid = blob.get("centroid_grid")
    if centroid is None:
        return None
    if isinstance(centroid, list) and len(centroid) >= 2:
        return float(centroid[0]), float(centroid[1])
    return None


def project_heatmap_frame(frame: np.ndarray, projection: str) -> np.ndarray:
    frame = np.asarray(frame, dtype=float)
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3:
        if projection in {"max_aoa", "max_doppler", "mean_aoa"}:
            return np.nanmax(frame, axis=1)
        return np.nanmax(frame, axis=1)
    if frame.ndim >= 4:
        axes = tuple(range(1, frame.ndim - 1))
        return np.nanmax(frame, axis=axes)
    return frame


def plot_original_heatmap(heatmap: dict[str, Any], frame_idx: int, output_dir: Path, projection: str) -> None:
    spectrum = np.asarray(heatmap["spectrum"])
    if frame_idx >= spectrum.shape[0]:
        return
    frame = project_heatmap_frame(spectrum[frame_idx], projection)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(frame, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(f"Original heatmap frame {frame_idx}")
    ax.set_xlabel("Doppler bin")
    ax.set_ylabel("ToF bin")
    fig.colorbar(im, ax=ax, label="Energy")
    save_fig(fig, output_dir / f"01_original_heatmap_frame_{frame_idx:03d}.png")


def plot_cells_overlay(
    heatmap: dict[str, Any],
    cells: Any,
    frame_idx: int,
    output_path: Path,
    title: str,
    label: str,
    marker: str = "o",
    color: str = "yellow",
) -> None:
    spectrum = np.asarray(heatmap["spectrum"])
    if frame_idx >= spectrum.shape[0]:
        return
    frame = project_heatmap_frame(spectrum[frame_idx], "max_aoa")
    records = cells_for_frame(cells, frame_idx)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(frame, aspect="auto", origin="lower", cmap="viridis")
    if records:
        xs = [r["fd_idx"] for r in records]
        ys = [r["tau_idx"] for r in records]
        ax.scatter(xs, ys, c=color, marker=marker, s=30, edgecolors="black", linewidths=0.4, alpha=0.8, label=label)
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Doppler bin")
    ax.set_ylabel("ToF bin")
    fig.colorbar(im, ax=ax, label="Energy")
    save_fig(fig, output_path)


def plot_scatter_cells(
    cells: Any,
    frame_idx: int,
    output_path: Path,
    title: str,
    label: str,
    marker: str = "o",
    color: str = "yellow",
) -> None:
    records = cells_for_frame(cells, frame_idx)
    fig, ax = plt.subplots(figsize=(8, 5))
    if records:
        xs = [r["fd_idx"] for r in records]
        ys = [r["tau_idx"] for r in records]
        ax.scatter(xs, ys, c=color, marker=marker, s=30, edgecolors="black", linewidths=0.4, alpha=0.8, label=label)
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Doppler bin")
    ax.set_ylabel("ToF bin")
    save_fig(fig, output_path)


def plot_scatter_compare(
    active_cells: Any,
    residual_cells: Any,
    frame_idx: int,
    output_path: Path,
) -> None:
    active = cells_for_frame(active_cells, frame_idx)
    residual = cells_for_frame(residual_cells, frame_idx)
    fig, ax = plt.subplots(figsize=(8, 5))
    if active:
        xs = [r["fd_idx"] for r in active]
        ys = [r["tau_idx"] for r in active]
        ax.scatter(xs, ys, c="yellow", marker="o", s=30, edgecolors="black", linewidths=0.4, alpha=0.7, label="active cells")
    if residual:
        xs = [r["fd_idx"] for r in residual]
        ys = [r["tau_idx"] for r in residual]
        ax.scatter(xs, ys, c="red", marker="x", s=35, linewidths=1.0, alpha=0.9, label="residual cells")
    if active or residual:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Active vs residual cells frame {frame_idx}")
    ax.set_xlabel("Doppler bin")
    ax.set_ylabel("ToF bin")
    save_fig(fig, output_path)


def plot_active_vs_residual(
    heatmap: dict[str, Any],
    active_cells: Any,
    residual_cells: Any,
    frame_idx: int,
    output_path: Path,
    projection: str,
) -> None:
    spectrum = np.asarray(heatmap["spectrum"])
    if frame_idx >= spectrum.shape[0]:
        return
    frame = project_heatmap_frame(spectrum[frame_idx], projection)
    active = cells_for_frame(active_cells, frame_idx)
    residual = cells_for_frame(residual_cells, frame_idx)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, records, title, color, marker in [
        (axes[0], active, "Active cells", "yellow", "o"),
        (axes[1], residual, "Residual cells", "red", "x"),
    ]:
        im = ax.imshow(frame, aspect="auto", origin="lower", cmap="viridis")
        if records:
            xs = [r["fd_idx"] for r in records]
            ys = [r["tau_idx"] for r in records]
            ax.scatter(xs, ys, c=color, marker=marker, s=30, edgecolors="black", linewidths=0.4, alpha=0.9, label=title)
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Doppler bin")
        ax.set_ylabel("ToF bin")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Energy")
    save_fig(fig, output_path)


def plot_blob_tracking(
    heatmap: dict[str, Any],
    blobs: Any,
    frame_idx: int,
    output_path: Path,
    title: str,
) -> None:
    spectrum = np.asarray(heatmap["spectrum"])
    if frame_idx >= spectrum.shape[0]:
        return
    frame = project_heatmap_frame(spectrum[frame_idx], "max_aoa")
    frame_blobs = blobs_for_frame(blobs, frame_idx)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(frame, aspect="auto", origin="lower", cmap="viridis")
    for blob in frame_blobs:
        centroid = select_blob_centroid(blob)
        if centroid is None:
            continue
        tau_idx, fd_idx = centroid
        ax.scatter([fd_idx], [tau_idx], c="cyan", edgecolors="black", s=80, marker="*", label=f"blob {blob.get('blob_id', '?')}")
        if blob.get("bbox_grid"):
            bbox = blob["bbox_grid"]
            tau_min, fd_min, tau_max, fd_max = bbox
            rect = plt.Rectangle(
                (fd_min, tau_min),
                fd_max - fd_min + 1,
                tau_max - tau_min + 1,
                fill=False,
                edgecolor="cyan",
                linewidth=1.5,
            )
            ax.add_patch(rect)
    if frame_blobs:
        ax.legend(loc="upper right", fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("Doppler bin")
    ax.set_ylabel("ToF bin")
    fig.colorbar(im, ax=ax, label="Energy")
    save_fig(fig, output_path)


def plot_blob_scatter(blobs: Any, frame_idx: int, output_path: Path, title: str) -> None:
    frame_blobs = blobs_for_frame(blobs, frame_idx)
    fig, ax = plt.subplots(figsize=(8, 5))
    for blob in frame_blobs:
        centroid = select_blob_centroid(blob)
        if centroid is None:
            continue
        tau_idx, fd_idx = centroid
        ax.scatter([fd_idx], [tau_idx], c="cyan", edgecolors="black", s=80, marker="*", label=f"blob {blob.get('blob_id', '?')}")
    if frame_blobs:
        ax.legend(loc="upper right", fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("Doppler bin")
    ax.set_ylabel("ToF bin")
    save_fig(fig, output_path)


def plot_energy_timeline(output_path: Path, frame_indices: list[int], active_counts: list[int], residual_counts: list[int], blob_counts: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame_indices, active_counts, label="active cells")
    ax.plot(frame_indices, residual_counts, label="residual cells")
    ax.plot(frame_indices, blob_counts, label="blobs")
    ax.set_xlabel("frame")
    ax.set_ylabel("count")
    ax.set_title("RDITH frame counts")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    save_fig(fig, output_path)


def write_dashboard_html(output_path: Path, progress_summary: Any, visualization_summary: dict[str, Any], vis_dir: Path) -> None:
    images = sorted([p.name for p in vis_dir.glob("*.png")])
    summary = {
        "progress_summary": progress_summary,
        "visualization_summary": visualization_summary,
    }
    write_json(output_path.with_name("visualization_report.json"), summary)
    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\"><title>RDITH Phase 1 Dashboard</title></head><body>",
        "<h1>RDITH Phase 1 Visualization</h1>",
        "<p>This dashboard shows original heatmap projections, active vs residual cell overlays, blob detections, and frame counts.</p>",
        "<h2>Run summary</h2>",
        f"<pre>{json.dumps(progress_summary, indent=2)}</pre>",
        "<h2>Visualization summary</h2>",
        f"<pre>{json.dumps(visualization_summary, indent=2)}</pre>",
        "<h2>Images</h2>",
        "<ul>",
    ]
    for image in images:
        html_lines.append(f"<li><a href=\"{image}\">{image}</a><br><img src=\"{image}\" style=\"max-width:600px;\"></li>")
    html_lines.extend(["</ul>", "</body></html>"])
    output_path.write_text("\n".join(html_lines), encoding="utf-8")


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


if __name__ == "__main__":
    main()
