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
    Estimate 2D local background and robust quantile noise on 256x256 blocks.
    Bilinearly interpolated to full resolution.
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

    sub = data[::8, ::8].astype(np.float32)
    glob_q16, glob_q84 = np.percentile(sub, [16.0, 84.0])
    glob_noise = float((glob_q84 - glob_q16) / 2.0)
    del sub

    noise_local[noise_local <= 0.0] = glob_noise if glob_noise > 0.0 else 1.0
    return bg_local, noise_local


def measure_component_v2(patch_mask, y_slice, x_slice, raw_data, bg_local, noise_local):
    """
    Calculate bounding-box dimensions, regularized elongation, and physical metrics.
    Regularization:
      - Lambda2 (minor axis variance) is clamped to a minimum floor of 0.25 (equivalent
        to a physical PSF radius of ~0.5 pixels) to prevent near-zero division on 1-pixel-wide lines.
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

    # Bounding-box dimensions
    bbox_dx = max(1, bbox_xmax - bbox_xmin)
    bbox_dy = max(1, bbox_ymax - bbox_ymin)
    bbox_len = max(bbox_dx, bbox_dy)
    bbox_width = min(bbox_dx, bbox_dy)
    aspect_ratio = float(bbox_len / max(1, bbox_width))

    # Pixel intensity metrics
    comp_pixels = raw_data[global_y, global_x].astype(np.float32)
    peak_intensity = float(np.max(comp_pixels))
    comp_bg = float(np.mean(bg_local[global_y, global_x]))
    comp_noise = float(np.mean(noise_local[global_y, global_x]))
    snr = (peak_intensity - comp_bg) / comp_noise if comp_noise > 0 else 0.0

    # Second moments with safe floor on minor eigenvalue
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

        # Regularize: minimum variance floor of 0.25 (~0.5 pixel physical radius)
        reg_lambda2 = max(lambda2, 0.25)
        reg_elongation = float(np.sqrt(lambda1 / reg_lambda2))

    # Proximity to nearest 1024-pixel tile boundary
    dist_x = min(cx % 1024, 1024 - (cx % 1024))
    dist_y = min(cy % 1024, 1024 - (cy % 1024))
    min_dist_boundary = min(dist_x, dist_y)

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
        "local_bg": round(comp_bg, 2),
        "local_noise": round(comp_noise, 2),
        "snr": round(float(snr), 2),
        "reg_elongation": round(float(reg_elongation), 2),
        "dist_to_tile_boundary": round(float(min_dist_boundary), 2),
    }


def categorize_and_score_candidate(cand):
    """
    Assign review category and composite ranking score based on multi-parameter geometry and SNR.
    Categories (mutually exclusive, prioritizing extended features):
      1. elongated_candidate: area >= 8, bbox_len >= 8, aspect_ratio >= 2.0, reg_elongation >= 1.8
      2. short_linear_candidate: bbox_len >= 5, aspect_ratio >= 2.5, area < 8 (captures c0507-type features)
      3. faint_extended_candidate: bbox_len >= 7, snr <= 7.0, aspect_ratio >= 1.5
      4. compact_candidate: area >= 6, aspect_ratio <= 1.4, reg_elongation < 1.3
      5. unresolved_small_candidate: area < 6, not qualifying as short_linear
    """
    area = cand["area"]
    b_len = cand["bbox_len"]
    ar = cand["aspect_ratio"]
    elong = cand["reg_elongation"]
    snr = max(1.0, cand["snr"])

    if area >= 8 and b_len >= 8 and ar >= 1.8 and elong >= 1.8:
        category = "elongated_candidate"
        # Score prioritizes length, continuity, and moderate SNR
        score = b_len * np.log1p(snr) * min(elong, 8.0)

    elif (b_len >= 5 and ar >= 2.5 and area < 8) or (b_len >= 6 and ar >= 2.0):
        category = "short_linear_candidate"
        # Prioritizes aspect ratio and length for small linear tracks
        score = b_len * ar * np.log1p(snr)

    elif b_len >= 7 and snr <= 7.0 and ar >= 1.5:
        category = "faint_extended_candidate"
        # Prioritizes extended size at lower SNR
        score = b_len * ar / np.sqrt(snr)

    elif area >= 6 and ar <= 1.4 and elong < 1.3:
        category = "compact_candidate"
        # Symmetrical stars scored by area and peak SNR
        score = area * np.log1p(snr)

    else:
        category = "unresolved_small_candidate"
        score = float(area)

    return category, round(float(score), 2)


def save_candidate_crop_v2(raw_data, cand, out_path, crop_size=120):
    """
    Save 120x120 crop with display-only contrast stretch and detailed overlay metrics.
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

    p1 = np.percentile(crop, 1.0)
    p99_5 = np.percentile(crop, 99.5)
    if p99_5 <= p1:
        p99_5 = p1 + 1.0
    norm_crop = np.clip((crop - p1) / (p99_5 - p1), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=120)
    ax.imshow(norm_crop, cmap="gray", origin="lower")

    rel_x0 = cand["bbox_xmin"] - x0
    rel_y0 = cand["bbox_ymin"] - y0
    w_box = cand["bbox_xmax"] - cand["bbox_xmin"]
    h_box = cand["bbox_ymax"] - cand["bbox_ymin"]
    rel_cx = cand["cx"] - x0
    rel_cy = cand["cy"] - y0

    # Color overlay by category
    color_map = {
        "elongated_candidate": "#00ffcc",
        "short_linear_candidate": "#ff00ff",
        "faint_extended_candidate": "#ffff00",
        "compact_candidate": "#0099ff",
        "unresolved_small_candidate": "#ff9966",
    }
    box_color = color_map.get(cand["review_category"], "#00ffcc")

    rect = plt.Rectangle((rel_x0, rel_y0), w_box, h_box, fill=False, color=box_color, linewidth=1.5)
    ax.add_patch(rect)
    ax.plot(rel_cx, rel_cy, "r+", markersize=9, markeredgewidth=1.5)

    title_text = (
        f"[{cand['candidate_id']}] Cat: {cand['review_category']} (Score: {cand['rank_score']:.1f})\n"
        f"Len: {cand['bbox_len']} px | Width: {cand['bbox_width']} px | AR: {cand['aspect_ratio']:.1f} | Area: {cand['area']} px\n"
        f"Elong: {cand['reg_elongation']:.2f} | Peak: {cand['peak_intensity']} ADU | SNR: {cand['snr']:.1f} | NearBnd: {cand['dist_to_tile_boundary']} px"
    )
    ax.set_title(title_text, fontsize=7.5, pad=6)
    ax.axis("off")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def process_image_v2(filepath, output_dir, k_sigma=4.0, min_area=3, max_crops_per_cat=4):
    """
    Remeasure full-frame candidates with V2 regularized shape metrics and multi-feature ranking.
    """
    filepath = Path(filepath)
    print(f"\nProcessing full-frame V2 ranking: {filepath.name}...")

    with fits.open(filepath, mode="readonly", memmap=False) as hdul:
        data = hdul[0].data

    rows, cols = data.shape
    bg_local, noise_local = estimate_local_background_and_noise(data, block_size=256)
    threshold_map = bg_local + k_sigma * noise_local

    mask = data > threshold_map
    del threshold_map

    labeled_arr, num_features = ndi.label(mask)
    del mask
    print(f"  Raw connected components: {num_features}")

    slices = ndi.find_objects(labeled_arr)
    all_candidates = []

    for idx, sl in enumerate(slices, 1):
        if sl is None:
            continue
        patch = (labeled_arr[sl] == idx)
        patch_area = int(np.sum(patch))

        if patch_area < min_area or patch_area > 5000:
            continue

        props = measure_component_v2(patch, sl[0], sl[1], data, bg_local, noise_local)
        if props is not None:
            props["source_file"] = filepath.name
            props["candidate_id"] = f"{filepath.stem}_v2_{len(all_candidates)+1:04d}"
            props["k_sigma_setting"] = k_sigma
            props["min_area_setting"] = min_area

            cat, score = categorize_and_score_candidate(props)
            props["review_category"] = cat
            props["rank_score"] = score
            props["review_status"] = "awaiting_human_review"
            all_candidates.append(props)

    del labeled_arr
    del slices
    del bg_local
    del noise_local

    print(f"  Total measured candidates (area >= {min_area}): {len(all_candidates)}")
    if not all_candidates:
        return pd.DataFrame(), pd.DataFrame(), {}

    df_all = pd.DataFrame(all_candidates)

    # Category counts
    cat_counts = df_all["review_category"].value_counts().to_dict()

    # Sample top candidates per category
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    sampled_records = []
    categories = [
        "elongated_candidate",
        "short_linear_candidate",
        "faint_extended_candidate",
        "compact_candidate",
        "unresolved_small_candidate",
    ]

    for cat in categories:
        sub_df = df_all[df_all["review_category"] == cat].sort_values("rank_score", ascending=False)
        top_cat = sub_df.head(max_crops_per_cat)
        for _, cand_row in top_cat.iterrows():
            sampled_records.append(cand_row.to_dict())
            crop_path = crops_dir / f"{cand_row['candidate_id']}.png"
            save_candidate_crop_v2(data, cand_row, crop_path, crop_size=120)

    sampled_df = pd.DataFrame(sampled_records)
    print(f"  Generated {len(sampled_df)} representative crops under {crops_dir.name}/")

    del data
    gc.collect()

    return df_all, sampled_df, cat_counts


def main():
    parser = argparse.ArgumentParser(description="V2 Candidate ranking with regularized elongation and multi-metric criteria.")
    parser.add_argument("--k-sigma", type=float, default=4.0, help="Local noise threshold multiplier (default: 4.0).")
    parser.add_argument("--min-area", type=int, default=3, help="Minimum candidate area in pixels (default: 3).")
    parser.add_argument("--output-dir", type=str, default="output/candidate_review_v2", help="V2 Output directory.")
    parser.add_argument("--crops-per-cat", type=int, default=4, help="Max crops saved per category (default: 4).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("V2 CANDIDATE RANKING & STRATIFIED REVIEW (NON-DESTRUCTIVE)")
    print("=" * 80)
    print(f"Output directory: {output_dir.resolve()}")
    print("Notice: No class is confirmed; cosmic rays, artifacts, and real streaks share similar geometries.")
    print("=" * 80)

    all_sampled = []
    summary_records = []

    for f_str in TARGET_FILES:
        path = Path(f_str)
        if not path.exists():
            print(f"Warning: File {f_str} not found!")
            continue

        df_all, df_sampled, cat_counts = process_image_v2(
            path,
            output_dir=output_dir,
            k_sigma=args.k_sigma,
            min_area=args.min_area,
            max_crops_per_cat=args.crops_per_cat,
        )

        sum_row = {
            "source_file": path.name,
            "total_candidates": len(df_all),
            "sampled_reviewed": len(df_sampled),
        }
        sum_row.update(cat_counts)
        summary_records.append(sum_row)

        if not df_sampled.empty:
            all_sampled.append(df_sampled)

    if all_sampled:
        combined_sampled = pd.concat(all_sampled, ignore_index=True)
        csv_path = output_dir / "reviewed_candidates_v2.csv"
        combined_sampled.to_csv(csv_path, index=False)
        print(f"\nSaved {len(combined_sampled)} V2 reviewed candidates to: {csv_path.name}")

    sum_df = pd.DataFrame(summary_records).fillna(0)
    sum_csv_path = output_dir / "category_summary_v2.csv"
    sum_df.to_csv(sum_csv_path, index=False)

    print("\n" + "=" * 80)
    print("V2 CATEGORY BREAKDOWN SUMMARY")
    print("=" * 80)
    print(sum_df.to_string(index=False))
    print("=" * 80)
    print(f"Done. Preserved V1 results; all V2 outputs saved in: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
