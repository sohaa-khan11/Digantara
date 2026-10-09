"""
Digantara Dataset Inspection & Exploration Module.

Provides modular functions to:
1. Inspect FITS headers and metadata.
2. Compute robust image and pixel statistics.
3. Save structured metadata to CSV/JSON.
4. Visualize downsampled overviews and high-resolution crops (linear and percentile/log stretched).
Original raw data is opened in read-only mode and strictly preserved.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def list_fits_files(data_dir: str | Path) -> List[Path]:
    """Return sorted list of all .fits / .fit files in the directory."""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {path}")
    files = sorted(list(path.glob("*.fits")) + list(path.glob("*.fit")))
    logger.info("Found %d FITS files in %s", len(files), path)
    return files


def extract_fits_metadata(filepath: str | Path) -> Dict[str, Any]:
    """
    Extract header metadata and basic array properties from a FITS file.
    Does not modify the original file (opened with memmap=True, mode='readonly').
    """
    filepath = Path(filepath)
    with fits.open(filepath, mode="readonly", memmap=False) as hdul:
        primary_hdu = hdul[0]
        header = primary_hdu.header
        data = primary_hdu.data

        raw_header_dict = {
            k: str(v)
            for k, v in header.items()
            if k and k not in ("COMMENT", "HISTORY")
        }

        meta = {
            "file_name": filepath.name,
            "file_size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
            "bitpix": header.get("BITPIX"),
            "naxis": header.get("NAXIS"),
            "width_naxis1": header.get("NAXIS1"),
            "height_naxis2": header.get("NAXIS2"),
            "instrument": header.get("INSTRUME"),
            "serial": header.get("SERIAL"),
            "exposure_sec": header.get("EXPOSURE"),
            "node_id": header.get("NODEID"),
            "obs_mode": header.get("OBSMODE"),
            "ops_mode": header.get("OPSMODE"),
            "date_obs": header.get("DATE-OBS"),
            "focal_len_mm": header.get("FOCALLEN"),
            "focal_ratio": header.get("FOCRATIO"),
            "obs_lat": header.get("OBS-LAT"),
            "obs_lng": header.get("OBS-LNG"),
            "obs_elv": header.get("OBS-ELV"),
            "bzero": header.get("BZERO"),
            "bscale": header.get("BSCALE"),
            "gain": header.get("GAIN"),
            "frame_no": header.get("FRAMENO"),
            "has_data": data is not None,
            "dtype": str(data.dtype) if data is not None else None,
            "shape": list(data.shape) if data is not None else None,
            "all_header_cards": raw_header_dict,
        }
    return meta


def compute_pixel_statistics(filepath: str | Path) -> Dict[str, Any]:
    """
    Compute comprehensive pixel statistics including min, max, mean, std,
    and percentiles for background / dynamic range evaluation.
    """
    filepath = Path(filepath)
    with fits.open(filepath, mode="readonly", memmap=False) as hdul:
        data = hdul[0].data
        if data is None:
            return {"file_name": filepath.name, "has_data": False}

        # Memory efficient stats: compute min, max, mean, std
        # Sample for fast, accurate percentile estimation on large 60MP arrays
        # (every 4th pixel in both dimensions gives ~3.8 million points)
        sampled = data[::4, ::4].astype(np.float32)

        p1, p5, p50, p95, p99, p99_9 = np.percentile(
            sampled, [1.0, 5.0, 50.0, 95.0, 99.0, 99.9]
        )

        stats = {
            "file_name": filepath.name,
            "min": int(np.min(data)),
            "max": int(np.max(data)),
            "mean": round(float(np.mean(sampled)), 4),
            "std": round(float(np.std(sampled)), 4),
            "median_p50": round(float(p50), 2),
            "p1": round(float(p1), 2),
            "p5": round(float(p5), 2),
            "p95": round(float(p95), 2),
            "p99": round(float(p99), 2),
            "p99_9": round(float(p99_9), 2),
            "dynamic_range": int(np.max(data)) - int(np.min(data)),
        }
    return stats


def export_inspection_summary(
    data_dir: str | Path,
    output_dir: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan all FITS files in data_dir, extract metadata and pixel statistics,
    and save them to CSV and JSON files in output_dir.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = list_fits_files(data_path)
    metadata_records = []
    stats_records = []

    for f in files:
        logger.info("Inspecting %s...", f.name)
        meta = extract_fits_metadata(f)
        stats = compute_pixel_statistics(f)

        # Separate full header dict from tabular metadata
        tabular_meta = {k: v for k, v in meta.items() if k != "all_header_cards"}
        tabular_meta["shape"] = str(tabular_meta["shape"])
        metadata_records.append(tabular_meta)
        stats_records.append(stats)

        # Save individual full header JSON for detailed reference
        json_path = out_path / f"{f.stem}_header.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(meta, jf, indent=2)

    df_meta = pd.DataFrame(metadata_records)
    df_stats = pd.DataFrame(stats_records)
    df_combined = pd.merge(df_meta, df_stats, on="file_name")

    meta_csv = out_path / "metadata_summary.csv"
    stats_csv = out_path / "pixel_statistics.csv"
    combined_csv = out_path / "dataset_summary.csv"

    df_meta.to_csv(meta_csv, index=False)
    df_stats.to_csv(stats_csv, index=False)
    df_combined.to_csv(combined_csv, index=False)

    logger.info("Saved metadata summary to %s", meta_csv)
    logger.info("Saved pixel statistics to %s", stats_csv)
    logger.info("Saved combined summary to %s", combined_csv)

    return df_combined, df_stats


def stretch_image(
    image: np.ndarray,
    lower_pct: float = 1.0,
    upper_pct: float = 99.5,
    mode: str = "percentile",
) -> np.ndarray:
    """
    Apply contrast stretching for optical/astronomical display.
    Modes:
      - 'linear': direct min-max normalization
      - 'percentile': clip to lower and upper percentiles, normalize to [0, 1]
      - 'log': log1p scaling followed by percentile stretch
    """
    img = image.astype(np.float32)
    if mode == "linear":
        vmin, vmax = np.min(img), np.max(img)
        if vmax > vmin:
            return (img - vmin) / (vmax - vmin)
        return np.zeros_like(img)

    if mode == "log":
        img = np.log1p(np.maximum(img, 0))

    vmin = np.percentile(img, lower_pct)
    vmax = np.percentile(img, upper_pct)
    if vmax <= vmin:
        vmax = vmin + 1e-5

    clipped = np.clip(img, vmin, vmax)
    return (clipped - vmin) / (vmax - vmin)


def visualize_fits_sample(
    filepath: str | Path,
    output_dir: str | Path,
    downsample_factor: int = 4,
    crop_size: int = 512,
) -> Path:
    """
    Generate and save a multi-panel visualization of a single FITS file:
      1. Full-frame overview (downsampled) with linear stretch
      2. Full-frame overview with percentile/contrast stretch
      3. High-resolution crop (center region) showing point sources/stars/streaks
      4. Intensity histogram of the image
    """
    filepath = Path(filepath)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(filepath, mode="readonly", memmap=False) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    if data is None:
        raise ValueError(f"No image data found in {filepath}")

    h, w = data.shape
    # Overview downsample
    overview = data[::downsample_factor, ::downsample_factor]

    # Center crop for resolution inspection
    ch, cw = h // 2, w // 2
    half_crop = crop_size // 2
    y1, y2 = max(0, ch - half_crop), min(h, ch + half_crop)
    x1, x2 = max(0, cw - half_crop), min(w, cw + half_crop)
    crop = data[y1:y2, x1:x2]

    # Stretches
    overview_linear = stretch_image(overview, mode="linear")
    overview_stretch = stretch_image(overview, lower_pct=1.0, upper_pct=99.5, mode="percentile")
    crop_stretch = stretch_image(crop, lower_pct=1.0, upper_pct=99.8, mode="percentile")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    fig.suptitle(
        f"FITS Inspection: {filepath.name}\n"
        f"Dimensions: {w}x{h} | Instrument: {header.get('INSTRUME', 'N/A')} | Exp: {header.get('EXPOSURE', 'N/A')}s",
        fontsize=13,
        fontweight="bold",
    )

    # 1. Linear Overview
    im0 = axes[0, 0].imshow(overview_linear, cmap="gray", origin="lower")
    axes[0, 0].set_title(f"Full-Field Overview (Linear Stretch, 1/{downsample_factor} scale)")
    axes[0, 0].set_xlabel("X (pixels / downsampled)")
    axes[0, 0].set_ylabel("Y (pixels / downsampled)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 2. Contrast/Percentile Overview
    im1 = axes[0, 1].imshow(overview_stretch, cmap="magma", origin="lower")
    axes[0, 1].set_title("Full-Field Overview (1%-99.5% Magma Stretch)")
    axes[0, 1].set_xlabel("X (pixels / downsampled)")
    axes[0, 1].set_ylabel("Y (pixels / downsampled)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. High-res center crop
    im2 = axes[1, 0].imshow(crop_stretch, cmap="gray", origin="lower")
    axes[1, 0].set_title(f"High-Res Center Crop ({crop_size}x{crop_size} px at ({cw}, {ch}))")
    axes[1, 0].set_xlabel("X (local crop px)")
    axes[1, 0].set_ylabel("Y (local crop px)")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 4. Intensity Histogram (sampled)
    sample_flat = overview.ravel()
    p99 = np.percentile(sample_flat, 99.5)
    axes[1, 1].hist(
        sample_flat[sample_flat <= p99],
        bins=60,
        color="#2c3e50",
        edgecolor="#34495e",
        density=True,
    )
    axes[1, 1].set_title("Pixel Intensity Distribution (up to 99.5th percentile)")
    axes[1, 1].set_xlabel("Raw ADU Value")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = out_dir / f"{filepath.stem}_inspection.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved visualization figure to %s", save_path)
    return save_path
