import argparse
import gc
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


TARGET_FILES = [
    "data/1a600998-b97c-4307-8632-6fcf573621ff.fits",
    "data/CAM_B_20260815T153133_manual_f000207.fits",
]


def estimate_local_background_and_noise(data, block_size=256):
    """
    Estimate coarse 2D background and robust noise maps.
    Assumptions:
      - Background varies smoothly over 256x256 spatial blocks.
      - Quantiles (q84 - q16) / 2 robustly estimate Gaussian-like noise dispersion.
      - Bilinear interpolation (scipy.ndimage.zoom) scales the grid to full resolution.
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

    # Bilinear interpolation to full frame
    zoom_factors = (rows / grid_med.shape[0], cols / grid_med.shape[1])
    bg_local = ndi.zoom(grid_med, zoom_factors, order=1)[:rows, :cols].astype(np.float32)
    noise_local = ndi.zoom(grid_noise, zoom_factors, order=1)[:rows, :cols].astype(np.float32)

    # Fallback for blocks with zero quantile noise due to integer discretization
    sub = data[::8, ::8].astype(np.float32)
    glob_q16, glob_q84 = np.percentile(sub, [16.0, 84.0])
    glob_noise = float((glob_q84 - glob_q16) / 2.0)
    del sub

    noise_local[noise_local <= 0.0] = glob_noise if glob_noise > 0.0 else 1.0

    return bg_local, noise_local


def compute_component_metrics(patch_mask, y_slice, x_slice, raw_data, bg_local, noise_local):
    """
    Calculate position, shape moments, and intensity metrics for a candidate component.
    Handles degenerate (collinear or tiny) components safely.
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

    # Extract pixel values and local background at component positions
    comp_pixels = raw_data[global_y, global_x].astype(np.float32)
    peak_intensity = float(np.max(comp_pixels))

    comp_bg = float(np.mean(bg_local[global_y, global_x]))
    comp_noise = float(np.mean(noise_local[global_y, global_x]))
    snr = (peak_intensity - comp_bg) / comp_noise if comp_noise > 0 else 0.0

    # Second moments for elongation estimation
    if area < 3:
        elongation = 1.0
    else:
        y_cent = yy - np.mean(yy)
        x_cent = xx - np.mean(xx)

        mu20 = np.mean(x_cent ** 2)
        mu02 = np.mean(y_cent ** 2)
        mu11 = np.mean(x_cent * y_cent)

        delta = np.sqrt((mu20 - mu02) ** 2 + 4.0 * (mu11 ** 2))
        lambda1 = max(0.0, 0.5 * (mu20 + mu02 + delta))
        lambda2 = max(0.0, 0.5 * (mu20 + mu02 - delta))

        major = np.sqrt(lambda1)
        minor = np.sqrt(lambda2)

        if minor > 1e-4:
            elongation = float(major / minor)
        else:
            elongation = float(major / 1e-4)

    # Distance to nearest 1024-pixel tile boundary
    dist_to_x_tile_boundary = min(cx % 1024, 1024 - (cx % 1024))
    dist_to_y_tile_boundary = min(cy % 1024, 1024 - (cy % 1024))
    min_dist_boundary = min(dist_to_x_tile_boundary, dist_to_y_tile_boundary)

    return {
        "cx": round(cx, 2),
        "cy": round(cy, 2),
        "bbox_xmin": bbox_xmin,
        "bbox_xmax": bbox_xmax,
        "bbox_ymin": bbox_ymin,
        "bbox_ymax": bbox_ymax,
        "area": area,
        "peak_intensity": round(peak_intensity, 2),
        "local_bg": round(comp_bg, 2),
        "local_noise": round(comp_noise, 2),
        "snr": round(float(snr), 2),
        "elongation": round(float(elongation), 2),
        "dist_to_tile_boundary": round(float(min_dist_boundary), 2),
    }


def sample_candidates_stratified(candidates_df, samples_per_category=4):
    """
    Sample candidate detections across elongation, area, SNR, and boundary proximity
    to provide a diverse set for manual review.
    Categories:
      1. Elongated candidates (possible streak candidates)
      2. Compact candidates (possible star / point-source candidates)
      3. Faint / low-SNR candidates (near detection limits)
      4. Tile boundary candidates (within 25px of 1024-tile boundaries)
      5. Ambiguous / intermediate candidates
    """
    sampled_indices = set()
    sampled_records = []

    def add_from_pool(pool, category_label, count):
        added = 0
        for idx, row in pool.iterrows():
            if idx not in sampled_indices:
                sampled_indices.add(idx)
                rec = row.to_dict()
                rec["sampling_category"] = category_label
                rec["review_status"] = "awaiting_human_review"
                sampled_records.append(rec)
                added += 1
                if added >= count:
                    break

    # 1. Elongated candidates
    pool_elong = candidates_df[candidates_df["elongation"] >= 2.2].sort_values("elongation", ascending=False)
    add_from_pool(pool_elong, "elongated_structure", samples_per_category)

    # 2. Compact candidates
    pool_compact = candidates_df[(candidates_df["elongation"] < 1.3) & (candidates_df["area"] >= 5)].sort_values("area", ascending=False)
    add_from_pool(pool_compact, "compact_source", samples_per_category)

    # 3. Faint / Low SNR candidates
    pool_faint = candidates_df[candidates_df["snr"] <= 6.0].sort_values("snr", ascending=True)
    add_from_pool(pool_faint, "faint_low_snr", samples_per_category)

    # 4. Candidates near 1024-pixel tile boundaries (checking boundary splitting behavior)
    pool_boundary = candidates_df[candidates_df["dist_to_tile_boundary"] <= 20.0].sort_values("dist_to_tile_boundary", ascending=True)
    add_from_pool(pool_boundary, "near_tile_boundary", samples_per_category)

    # 5. Ambiguous / intermediate candidates
    pool_ambig = candidates_df[(candidates_df["elongation"] >= 1.4) & (candidates_df["elongation"] < 2.2)].sort_values("elongation", ascending=False)
    add_from_pool(pool_ambig, "ambiguous_intermediate", samples_per_category)

    return pd.DataFrame(sampled_records)


def save_candidate_crop(raw_data, cand, out_path, crop_size=120):
    """
    Save an individual zoomed 120x120 crop of the raw data with display-only
    contrast stretching and quantitative overlay markers.
    """
    rows, cols = raw_data.shape
    cx, cy = int(cand["cx"]), int(cand["cy"])
    half = crop_size // 2

    y0 = max(0, cy - half)
    y1 = min(rows, y0 + crop_size)
    if y1 - y0 < crop_size:
        y0 = max(0, y1 - crop_size)

    x0 = max(0, cx - half)
    x1 = min(cols, x0 + crop_size)
    if x1 - x0 < crop_size:
        x0 = max(0, x1 - crop_size)

    crop = raw_data[y0:y1, x0:x1].astype(np.float32)

    # Display-only percentile stretch
    p1 = np.percentile(crop, 1.0)
    p99_5 = np.percentile(crop, 99.5)
    if p99_5 <= p1:
        p99_5 = p1 + 1.0
    norm_crop = np.clip((crop - p1) / (p99_5 - p1), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=120)
    ax.imshow(norm_crop, cmap="gray", origin="lower")

    # Local crop coordinates
    rel_x0 = cand["bbox_xmin"] - x0
    rel_y0 = cand["bbox_ymin"] - y0
    w_box = cand["bbox_xmax"] - cand["bbox_xmin"]
    h_box = cand["bbox_ymax"] - cand["bbox_ymin"]
    rel_cx = cand["cx"] - x0
    rel_cy = cand["cy"] - y0

    # Overlay bounding box & centroid
    rect = plt.Rectangle((rel_x0, rel_y0), w_box, h_box, fill=False, color="#00ffcc", linewidth=1.5)
    ax.add_patch(rect)
    ax.plot(rel_cx, rel_cy, "r+", markersize=9, markeredgewidth=1.5)

    title_text = (
        f"ID: {cand['candidate_id']} | Cat: {cand['sampling_category']}\n"
        f"Pos: ({cx}, {cy}) | Area: {cand['area']} px | Elong: {cand['elongation']:.2f}\n"
        f"Peak: {cand['peak_intensity']} ADU | SNR: {cand['snr']:.1f} | NearBnd: {cand['dist_to_tile_boundary']} px"
    )
    ax.set_title(title_text, fontsize=8, pad=6)
    ax.axis("off")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def process_image_candidates(filepath, output_dir, k_sigma=4.0, min_area=3, max_area=5000):
    """
    Run full-frame connected-component candidate detection on a single FITS file.
    Does not label objects as confirmed stars or streaks.
    """
    filepath = Path(filepath)
    print(f"\nProcessing full frame: {filepath.name}...")

    with fits.open(filepath, mode="readonly", memmap=False) as hdul:
        data = hdul[0].data

    rows, cols = data.shape
    print(f"  Frame dimensions: {cols}x{rows} | Data type: {data.dtype}")

    # Estimate 2D background and noise
    bg_local, noise_local = estimate_local_background_and_noise(data, block_size=256)
    threshold_map = bg_local + k_sigma * noise_local

    # Threshold mask (area >= 3 is a candidate filter, not proof of an astronomical object)
    mask = data > threshold_map
    del threshold_map

    labeled_arr, num_features = ndi.label(mask)
    del mask
    print(f"  Total raw thresholded components: {num_features}")

    slices = ndi.find_objects(labeled_arr)
    all_candidates = []

    for idx, sl in enumerate(slices, 1):
        if sl is None:
            continue
        patch = (labeled_arr[sl] == idx)
        patch_area = int(np.sum(patch))

        if patch_area < min_area or patch_area > max_area:
            continue

        props = compute_component_metrics(patch, sl[0], sl[1], data, bg_local, noise_local)
        if props is not None:
            props["source_file"] = filepath.name
            props["candidate_id"] = f"{filepath.stem}_c{len(all_candidates)+1:04d}"
            props["k_sigma_setting"] = k_sigma
            props["min_area_setting"] = min_area
            all_candidates.append(props)

    del labeled_arr
    del slices
    del bg_local
    del noise_local

    print(f"  Candidates meeting filter (area >= {min_area}): {len(all_candidates)}")
    if not all_candidates:
        return pd.DataFrame(), pd.DataFrame()

    cands_df = pd.DataFrame(all_candidates)

    # Sample representative candidates across categories
    sampled_df = sample_candidates_stratified(cands_df, samples_per_category=4)
    print(f"  Sampled {len(sampled_df)} candidates across categories for visual review.")

    # Save visual 120x120 crops
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    for _, cand_row in sampled_df.iterrows():
        crop_path = crops_dir / f"{cand_row['candidate_id']}.png"
        save_candidate_crop(data, cand_row, crop_path, crop_size=120)

    del data
    gc.collect()

    return cands_df, sampled_df


def main():
    parser = argparse.ArgumentParser(description="Full-frame candidate detection and review crop extraction.")
    parser.add_argument("--k-sigma", type=float, default=4.0, help="Local noise threshold multiplier (default: 4.0).")
    parser.add_argument("--min-area", type=int, default=3, help="Minimum candidate area in pixels (default: 3).")
    parser.add_argument("--output-dir", type=str, default="output/candidate_review", help="Output directory path.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PROSPECTIVE CANDIDATE REVIEW EXTRACTION (HUMAN REVIEW TRIAGE)")
    print("=" * 80)
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Settings: k_sigma = {args.k_sigma}, min_area = {args.min_area}")
    print("Assumption: Candidates are detections awaiting manual review, not confirmed objects.")
    print("=" * 80)

    summary_records = []
    all_sampled = []

    for f_path_str in TARGET_FILES:
        path = Path(f_path_str)
        if not path.exists():
            print(f"Warning: File {f_path_str} not found!")
            continue

        cands_df, sampled_df = process_image_candidates(
            path,
            output_dir=output_dir,
            k_sigma=args.k_sigma,
            min_area=args.min_area,
        )

        summary_records.append({
            "source_file": path.name,
            "total_candidates_found": len(cands_df),
            "sampled_candidates_reviewed": len(sampled_df),
            "k_sigma": args.k_sigma,
            "min_area": args.min_area,
        })

        if not sampled_df.empty:
            all_sampled.append(sampled_df)

    if all_sampled:
        combined_sampled = pd.concat(all_sampled, ignore_index=True)
        csv_path = output_dir / "reviewed_candidates.csv"
        combined_sampled.to_csv(csv_path, index=False)
        print(f"\nSaved {len(combined_sampled)} reviewed candidate metadata rows to: {csv_path.name}")

    summary_df = pd.DataFrame(summary_records)
    sum_csv_path = output_dir / "candidate_review_summary.csv"
    summary_df.to_csv(sum_csv_path, index=False)

    print("\n" + "=" * 80)
    print("CANDIDATE REVIEW SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print("Done. All candidate crops and metadata are saved under output/candidate_review/.\n")


if __name__ == "__main__":
    main()
