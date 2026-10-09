import argparse
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label, find_objects


def estimate_background_stats(data_subsample):
    """
    Estimate background sky level and dispersion using median and median absolute deviation.
    Robust against bright stars and outlier pixels.
    """
    sub = data_subsample.astype(np.float32)
    median = float(np.median(sub))
    mad = float(np.median(np.abs(sub - median)))
    sigma = 1.4826 * mad
    if sigma <= 0.0:
        sigma = float(np.std(sub))
    if sigma <= 0.0:
        sigma = 1.0
    return median, sigma


def compute_shape_properties(component_mask, y_slice, x_slice):
    """
    Compute centroid, area, bounding box, and elongation using second moments of the binary patch.
    Elongation is calculated as the ratio of major to minor semi-axes (>= 1.0).
    """
    yy, xx = np.nonzero(component_mask)
    if len(yy) == 0:
        return None

    area = int(len(yy))
    global_y = yy + y_slice.start
    global_x = xx + x_slice.start

    cy = float(np.mean(global_y))
    cx = float(np.mean(global_x))

    bbox_ymin = int(y_slice.start)
    bbox_ymax = int(y_slice.stop)
    bbox_xmin = int(x_slice.start)
    bbox_xmax = int(x_slice.stop)

    # Second moments for orientation and elongation estimation
    if area < 5:
        elongation = 1.0
        theta_deg = 0.0
    else:
        y_cent = yy - np.mean(yy)
        x_cent = xx - np.mean(xx)

        mu20 = np.mean(x_cent ** 2)
        mu02 = np.mean(y_cent ** 2)
        mu11 = np.mean(x_cent * y_cent)

        # Eigenvalues of 2D covariance matrix
        delta = np.sqrt((mu20 - mu02) ** 2 + 4.0 * (mu11 ** 2))
        lambda1 = max(0.0, 0.5 * (mu20 + mu02 + delta))
        lambda2 = max(0.0, 0.5 * (mu20 + mu02 - delta))

        major = np.sqrt(lambda1)
        minor = np.sqrt(lambda2)

        if minor > 1e-4:
            elongation = float(major / minor)
        else:
            elongation = float(major / 1e-4)

        theta_rad = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
        theta_deg = float(np.degrees(theta_rad))

    return {
        "cx": round(cx, 2),
        "cy": round(cy, 2),
        "area": area,
        "bbox_xmin": bbox_xmin,
        "bbox_xmax": bbox_xmax,
        "bbox_ymin": bbox_ymin,
        "bbox_ymax": bbox_ymax,
        "elongation": round(elongation, 2),
        "orientation_deg": round(theta_deg, 2),
    }


def find_candidates_in_tile(tile_data, threshold_val, min_area, max_area, min_elongation):
    """
    Identify elongated connected components above threshold within a tile.
    """
    binary_map = tile_data > threshold_val
    if not np.any(binary_map):
        return []

    labeled_arr, num_features = label(binary_map)
    if num_features == 0:
        return []

    slices = find_objects(labeled_arr)
    candidates = []

    for idx, sl in enumerate(slices, 1):
        if sl is None:
            continue
        patch = (labeled_arr[sl] == idx)
        patch_area = int(np.sum(patch))

        if patch_area < min_area or patch_area > max_area:
            continue

        props = compute_shape_properties(patch, sl[0], sl[1])
        if props is None:
            continue

        if props["elongation"] >= min_elongation:
            candidates.append(props)

    return candidates


def inspect_image(fits_path, output_dir, n_sigma=4.0, min_area=15, max_area=2500, min_elongation=2.2, max_crops=6):
    """
    Scan a single FITS file for streak candidates, returning tabular records and saving verification crops.
    """
    fits_path = Path(fits_path)
    output_dir = Path(output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(fits_path, mode="readonly", memmap=False) as hdul:
        data = hdul[0].data

    if data is None:
        print(f"Warning: No data in {fits_path.name}")
        return []

    height, width = data.shape

    # Adaptive background thresholding per image
    subsample = data[::8, ::8]
    bg_median, bg_sigma = estimate_background_stats(subsample)
    threshold_val = bg_median + n_sigma * bg_sigma

    print(f"[{fits_path.name}] Dim: {width}x{height} | BG median: {bg_median:.1f}, sigma: {bg_sigma:.2f} | Threshold: {threshold_val:.1f}")

    # Process in 1024x1024 windows across image to manage memory
    step = 1024
    all_candidates = []

    for y0 in range(0, height, step):
        y1 = min(y0 + step, height)
        for x0 in range(0, width, step):
            x1 = min(x0 + step, width)
            tile = data[y0:y1, x0:x1]

            tile_cands = find_candidates_in_tile(tile, threshold_val, min_area, max_area, min_elongation)
            for c in tile_cands:
                # Convert back to full image coordinates
                global_record = {
                    "source_file": fits_path.name,
                    "cx": round(c["cx"] + x0, 2),
                    "cy": round(c["cy"] + y0, 2),
                    "bbox_xmin": c["bbox_xmin"] + x0,
                    "bbox_xmax": c["bbox_xmax"] + x0,
                    "bbox_ymin": c["bbox_ymin"] + y0,
                    "bbox_ymax": c["bbox_ymax"] + y0,
                    "area": c["area"],
                    "elongation": c["elongation"],
                    "orientation_deg": c["orientation_deg"],
                    "bg_median": round(bg_median, 2),
                    "bg_sigma": round(bg_sigma, 2),
                    "detection_threshold": round(threshold_val, 2),
                    "min_elongation_setting": min_elongation,
                }
                all_candidates.append(global_record)

    print(f"[{fits_path.name}] Found {len(all_candidates)} initial candidate(s).")

    # Generate verification preview crops for top elongated detections
    if all_candidates:
        sorted_cands = sorted(all_candidates, key=lambda x: x["elongation"], reverse=True)
        saved_crops = 0

        for i, cand in enumerate(sorted_cands[:max_crops], 1):
            cx, cy = int(cand["cx"]), int(cand["cy"])
            crop_radius = 80
            cy_min = max(0, cy - crop_radius)
            cy_max = min(height, cy + crop_radius)
            cx_min = max(0, cx - crop_radius)
            cx_max = min(width, cx + crop_radius)

            raw_crop = data[cy_min:cy_max, cx_min:cx_max].astype(np.float32)

            # Contrast stretch strictly for inspection display
            p_lo = np.percentile(raw_crop, 2.0)
            p_hi = np.percentile(raw_crop, 99.8)
            if p_hi <= p_lo:
                p_hi = p_lo + 1.0
            display_crop = np.clip((raw_crop - p_lo) / (p_hi - p_lo), 0.0, 1.0)

            fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
            ax.imshow(display_crop, cmap="gray", origin="lower")
            # Draw candidate centroid and bbox in local crop coordinates
            rel_cx = cand["cx"] - cx_min
            rel_cy = cand["cy"] - cy_min
            rel_x0 = cand["bbox_xmin"] - cx_min
            rel_y0 = cand["bbox_ymin"] - cy_min
            w_box = cand["bbox_xmax"] - cand["bbox_xmin"]
            h_box = cand["bbox_ymax"] - cand["bbox_ymin"]

            rect = plt.Rectangle((rel_x0, rel_y0), w_box, h_box, fill=False, color="#00ffcc", linewidth=1.5)
            ax.add_patch(rect)
            ax.plot(rel_cx, rel_cy, "r+", markersize=8)

            ax.set_title(
                f"Candidate #{i}: ({cx}, {cy})\nElongation: {cand['elongation']:.1f} | Area: {cand['area']} px",
                fontsize=9,
            )
            ax.axis("off")

            crop_filename = f"{fits_path.stem}_cand{i:02d}.png"
            fig.savefig(crops_dir / crop_filename, bbox_inches="tight")
            plt.close(fig)
            saved_crops += 1

        print(f"[{fits_path.name}] Saved {saved_crops} inspection crop(s) to {crops_dir.name}/")

    return all_candidates


def main():
    parser = argparse.ArgumentParser(description="Find candidate elongated streaks in FITS images for human inspection.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="output/streak_inspection")
    parser.add_argument("--file", type=str, default=None, help="Inspect a specific FITS file instead of all files.")
    parser.add_argument("--n-sigma", type=float, default=4.0, help="Threshold multiplier above background noise.")
    parser.add_argument("--min-area", type=int, default=15, help="Minimum connected component area in pixels.")
    parser.add_argument("--min-elongation", type=float, default=2.2, help="Minimum major/minor axis ratio.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        fits_files = [Path(args.file)]
    else:
        fits_files = sorted(list(data_dir.glob("*.fits")) + list(data_dir.glob("*.fit")))

    if not fits_files:
        print(f"No FITS images found in {data_dir}")
        return

    print("=" * 60)
    print("STREAK CANDIDATE INSPECTION TOOL (EXPLORATORY)")
    print("=" * 60)
    print(f"FITS count to inspect: {len(fits_files)}")
    print(f"Output directory:      {output_dir}")
    print(f"Settings: n_sigma={args.n_sigma}, min_area={args.min_area}, min_elongation={args.min_elongation}")
    print("Note: Candidates require manual verification; artifacts and stars may appear.")
    print("=" * 60)

    total_candidates = []
    for f in fits_files:
        cands = inspect_image(
            f,
            output_dir=output_dir,
            n_sigma=args.n_sigma,
            min_area=args.min_area,
            min_elongation=args.min_elongation,
        )
        total_candidates.extend(cands)

    csv_path = output_dir / "streak_candidates.csv"
    df = pd.DataFrame(total_candidates)
    df.to_csv(csv_path, index=False)
    print("=" * 60)
    print(f"Done. Found {len(total_candidates)} total candidate(s) across {len(fits_files)} file(s).")
    print(f"Candidate table saved to: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
