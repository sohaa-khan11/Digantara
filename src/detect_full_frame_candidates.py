"""
Full-frame Candidate Detector for 10 FITS Images (Digantara Assessment).

Computes robust local 2D background and noise maps to detect prospective
point sources (blob_star candidates) and linear trails (streak candidates).

IMPORTANT SCIENTIFIC DISCLAIMER:
These detections are algorithmically proposed candidates awaiting human verification.
They are NOT confirmed ground truth and are kept isolated from manual annotations.
"""

import argparse
import gc
from pathlib import Path
from astropy.io import fits
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def estimate_local_background_and_noise(data, block_size=256):
    """
    Estimate local 2D background and robust noise via block quantiles (q84 - q16)/2.
    Interpolates smoothly to full frame dimensions.
    Guards against zero MAD / zero quantile dispersion from integer quantization.
    """
    rows, cols = data.shape
    r_starts = list(range(0, rows, block_size))
    c_starts = list(range(0, cols, block_size))

    grid_med = np.zeros((len(r_starts), len(c_starts)), dtype=np.float32)
    grid_noise = np.zeros((len(r_starts), len(c_starts)), dtype=np.float32)

    for i, r in enumerate(r_starts):
        r_end = min(r + block_size, rows)
        for j, c in enumerate(c_starts):
            c_end = min(c + block_size, cols)
            block = data[r:r_end, c:c_end].astype(np.float32)

            med = float(np.median(block))
            q16, q84 = np.percentile(block, [16.0, 84.0])
            sigma = float((q84 - q16) / 2.0)

            grid_med[i, j] = med
            grid_noise[i, j] = sigma

    zoom_factors = (rows / grid_med.shape[0], cols / grid_med.shape[1])
    bg_local = ndi.zoom(grid_med, zoom_factors, order=1)[:rows, :cols].astype(np.float32)
    noise_local = ndi.zoom(grid_noise, zoom_factors, order=1)[:rows, :cols].astype(np.float32)

    # Fallback for quantized regions where local quantile noise is zero
    sub = data[::8, ::8].astype(np.float32)
    glob_q16, glob_q84 = np.percentile(sub, [16.0, 84.0])
    glob_noise = float((glob_q84 - glob_q16) / 2.0)
    if glob_noise <= 0.0:
        glob_noise = float(np.std(sub))
    if glob_noise <= 0.0:
        glob_noise = 1.0
    del sub

    noise_local[noise_local <= 0.0] = glob_noise
    return bg_local, noise_local


def measure_component(patch_mask, y_slice, x_slice, raw_data, bg_local, noise_local):
    """
    Measure morphology, bounding box, regularized elongation, and SNR for a connected component.
    """
    yy, xx = np.nonzero(patch_mask)
    area = int(len(yy))
    if area == 0:
        return None

    global_y = yy + y_slice.start
    global_x = xx + x_slice.start

    cx = float(np.mean(global_x))
    cy = float(np.mean(global_y))

    bbox_xmin = int(x_slice.start)
    bbox_xmax = int(x_slice.stop)
    bbox_ymin = int(y_slice.start)
    bbox_ymax = int(y_slice.stop)

    bbox_dx = max(1, bbox_xmax - bbox_xmin)
    bbox_dy = max(1, bbox_ymax - bbox_ymin)
    bbox_len = max(bbox_dx, bbox_dy)
    bbox_width = min(bbox_dx, bbox_dy)
    aspect_ratio = float(bbox_len / max(1, bbox_width))

    comp_pixels = raw_data[global_y, global_x].astype(np.float32)
    peak_intensity = float(np.max(comp_pixels))
    mean_bg = float(np.mean(bg_local[global_y, global_x]))
    mean_noise = float(np.mean(noise_local[global_y, global_x]))
    snr = (peak_intensity - mean_bg) / mean_noise if mean_noise > 0 else 0.0

    # Regularized second central moments (variance floor 0.25 prevents 1-px line divergence)
    if area < 3:
        reg_elongation = 1.0
    else:
        y_cent = yy - np.mean(yy)
        x_cent = xx - np.mean(xx)

        mu20 = np.mean(x_cent ** 2)
        mu02 = np.mean(y_cent ** 2)
        mu11 = np.mean(x_cent * y_cent)

        delta = np.sqrt((mu20 - mu02) ** 2 + 4.0 * (mu11 ** 2))
        lambda1 = max(0.0, 0.5 * (mu20 + mu02 + delta))
        lambda2 = max(0.0, 0.5 * (mu20 + mu02 - delta))

        reg_lambda2 = max(lambda2, 0.25)
        reg_elongation = float(np.sqrt(lambda1 / reg_lambda2))

    return {
        "cx": round(cx, 2),
        "cy": round(cy, 2),
        "bbox_xmin": bbox_xmin,
        "bbox_xmax": bbox_xmax,
        "bbox_ymin": bbox_ymin,
        "bbox_ymax": bbox_ymax,
        "bbox_len": bbox_len,
        "bbox_width": bbox_width,
        "aspect_ratio": round(aspect_ratio, 2),
        "area": area,
        "peak_intensity": round(peak_intensity, 2),
        "local_bg": round(mean_bg, 2),
        "local_noise": round(mean_noise, 2),
        "snr": round(float(snr), 2),
        "reg_elongation": round(float(reg_elongation), 2),
    }


def classify_candidate(cand):
    """
    Categorize candidate component into candidate types:
      - candidate_streak: extended linear track
      - candidate_blob_star: compact circular point source
      - candidate_ambiguous_small: tiny or low-SNR feature
    """
    b_len = cand["bbox_len"]
    ar = cand["aspect_ratio"]
    elong = cand["reg_elongation"]
    area = cand["area"]

    # Streak criteria: extended length, high aspect ratio, oriented
    if (b_len >= 8 and ar >= 1.8 and elong >= 1.8 and area >= 8) or (b_len >= 5 and ar >= 2.5):
        return "candidate_streak"
    # Star criteria: compact, aspect ratio near 1, symmetric
    elif area >= 5 and ar <= 1.4 and elong < 1.3:
        return "candidate_blob_star"
    else:
        return "candidate_ambiguous_small"


def detect_candidates_in_fits(fits_path, output_dir, k_sigma=4.0, min_area=3, max_area=5000):
    """
    Run candidate detection on a single FITS file, saving candidate CSV and binary candidate mask.
    """
    fits_path = Path(fits_path)
    output_dir = Path(output_dir)
    masks_dir = output_dir / "candidate_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {fits_path.name}...")
    with fits.open(fits_path, mode="readonly", memmap=False) as hdul:
        data = hdul[0].data

    rows, cols = data.shape
    bg_local, noise_local = estimate_local_background_and_noise(data, block_size=256)
    threshold_map = bg_local + k_sigma * noise_local

    mask = data > threshold_map
    del threshold_map

    labeled_arr, num_features = ndi.label(mask)
    del mask

    slices = ndi.find_objects(labeled_arr)
    candidates = []

    # Prepare candidate mask array (uint8: 0=bg, 1=candidate_blob_star, 2=candidate_streak)
    cand_mask = np.zeros((rows, cols), dtype=np.uint8)

    for idx, sl in enumerate(slices, 1):
        if sl is None:
            continue
        patch = (labeled_arr[sl] == idx)
        patch_area = int(np.sum(patch))

        if patch_area < min_area or patch_area > max_area:
            continue

        props = measure_component(patch, sl[0], sl[1], data, bg_local, noise_local)
        if props is not None:
            props["source_file"] = fits_path.name
            props["candidate_id"] = f"{fits_path.stem}_cand_{len(candidates)+1:05d}"
            props["provenance"] = "automated_candidate_unverified"
            cat = classify_candidate(props)
            props["candidate_category"] = cat

            # Burn candidate mask
            val = 2 if cat == "candidate_streak" else 1
            cand_mask[sl][patch] = val

            candidates.append(props)

    del labeled_arr
    del slices
    del bg_local
    del noise_local
    del data

    # Save compressed binary candidate mask (.npz to conserve disk space)
    mask_file = masks_dir / f"{fits_path.stem}_candidate_mask.npz"
    np.savez_compressed(mask_file, mask=cand_mask)
    del cand_mask
    gc.collect()

    df_cands = pd.DataFrame(candidates)
    print(f"  -> Found {len(df_cands)} candidates (area >= {min_area}). Mask saved to: {mask_file.name}")
    return df_cands


def main():
    parser = argparse.ArgumentParser(description="Full-frame candidate detection across all 10 FITS images.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing FITS images.")
    parser.add_argument("--output-dir", type=str, default="output/automated_candidates", help="Output directory.")
    parser.add_argument("--k-sigma", type=float, default=4.0, help="Noise threshold multiplier.")
    parser.add_argument("--min-area", type=int, default=3, help="Minimum connected component area.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fits_files = sorted(list(data_dir.glob("*.fits")) + list(data_dir.glob("*.fit")))
    print("=" * 75)
    print("DIGANTARA FULL-FRAME CANDIDATE DETECTOR (ALL 10 FITS IMAGES)")
    print(f"Total FITS files: {len(fits_files)}")
    print(f"Settings: k_sigma={args.k_sigma}, min_area={args.min_area}")
    print("Notice: Outputs are candidate proposals, strictly separated from ground truth.")
    print("=" * 75)

    all_records = []
    summary = []

    for f in fits_files:
        df_c = detect_candidates_in_fits(
            f, output_dir=output_dir, k_sigma=args.k_sigma, min_area=args.min_area
        )
        if not df_c.empty:
            all_records.append(df_c)
            cat_counts = df_c["candidate_category"].value_counts().to_dict()
            summary.append({
                "source_file": f.name,
                "total_candidates": len(df_c),
                "streaks": cat_counts.get("candidate_streak", 0),
                "blob_stars": cat_counts.get("candidate_blob_star", 0),
                "ambiguous_small": cat_counts.get("candidate_ambiguous_small", 0),
            })

    if all_records:
        combined_df = pd.concat(all_records, ignore_index=True)
        csv_path = output_dir / "all_10_images_candidate_catalog.csv"
        combined_df.to_csv(csv_path, index=False)
        print(f"\nSaved combined candidate catalog ({len(combined_df)} records) to: {csv_path.name}")

    summary_df = pd.DataFrame(summary)
    sum_csv = output_dir / "candidate_detection_summary.csv"
    summary_df.to_csv(sum_csv, index=False)

    print("\n" + "=" * 75)
    print("CANDIDATE DETECTION SUMMARY (ALL 10 IMAGES)")
    print("=" * 75)
    print(summary_df.to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
