"""
Command-line runner for FITS dataset exploration and inspection.

Usage:
    python src/run_exploration.py
    python src/run_exploration.py --data-dir data --output-dir output --num-samples 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local src directory is on import path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fits_inspector import (
    export_inspection_summary,
    list_fits_files,
    logger,
    visualize_fits_sample,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect FITS dataset metadata, pixel stats, and render sample visuals."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing the raw .fits files (default: 'data').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory where metadata summaries and visualizations are saved (default: 'output').",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of representative FITS images to visualize (default: 2).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Downsampling factor for overview plots (default: 4 for speed & memory).",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
        help="Crop dimension in pixels for high-resolution inspection (default: 512).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print(" DIGANTARA AI/ML DATASET EXPLORATION WORKFLOW")
    print("=" * 60)
    print(f"Data directory:   {data_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")

    # 1. Check input files
    fits_files = list_fits_files(data_dir)
    if not fits_files:
        logger.error("No FITS files found in %s", data_dir)
        sys.exit(1)

    # 2. Extract metadata and pixel stats for all files
    print("\n[Step 1/2] Extracting metadata & computing pixel statistics across dataset...")
    df_combined, _ = export_inspection_summary(data_dir, output_dir)

    print("\n--- DATASET SUMMARY SNAPSHOT ---")
    cols_to_print = [
        "file_name",
        "shape",
        "instrument",
        "exposure_sec",
        "min",
        "max",
        "mean",
        "std",
        "p99_9",
    ]
    avail_cols = [c for c in cols_to_print if c in df_combined.columns]
    print(df_combined[avail_cols].to_string(index=False))

    # 3. Generate sample visual inspection figures
    print(f"\n[Step 2/2] Generating inspection visualizations for {min(args.num_samples, len(fits_files))} sample(s)...")
    vis_dir = output_dir / "visualizations"
    for i, file_path in enumerate(fits_files[: args.num_samples]):
        print(f"  Visualizing ({i + 1}/{args.num_samples}): {file_path.name}")
        vis_path = visualize_fits_sample(
            file_path,
            output_dir=vis_dir,
            downsample_factor=args.downsample,
            crop_size=args.crop_size,
        )
        print(f"  -> Generated: {vis_path}")

    print("\n" + "=" * 60)
    print("Exploration completed successfully!")
    print(f"  Metadata & Statistics: {output_dir.resolve()}")
    print(f"  Visualizations:        {vis_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
