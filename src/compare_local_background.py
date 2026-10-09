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


def estimate_global_stats(data):
    """
    Compute global median and robust noise using quantiles (q84 - q16) / 2
    on regularly subsampled pixels to remain fast and resilient to stars.
    """
    sub = data[::8, ::8].astype(np.float32)
    bg_median = float(np.median(sub))
    q16, q84 = np.percentile(sub, [16.0, 84.0])
    noise = float((q84 - q16) / 2.0)
    del sub
    return bg_median, noise


def compute_block_grid(data, block_size=256):
    """
    Divide data into complete and edge blocks of size block_size x block_size.
    Computes local median and local robust noise (q84 - q16) / 2 for each block.
    """
    rows, cols = data.shape
    r_starts = list(range(0, rows, block_size))
    c_starts = list(range(0, cols, block_size))

    n_r = len(r_starts)
    n_c = len(c_starts)

    grid_medians = np.zeros((n_r, n_c), dtype=np.float32)
    grid_noise = np.zeros((n_r, n_c), dtype=np.float32)

    for i, r in enumerate(r_starts):
        r_end = min(r + block_size, rows)
        for j, c in enumerate(c_starts):
            c_end = min(c + block_size, cols)
            block = data[r:r_end, c:c_end].astype(np.float32)

            med = float(np.median(block))
            q16, q84 = np.percentile(block, [16.0, 84.0])
            sigma = float((q84 - q16) / 2.0)

            grid_medians[i, j] = med
            grid_noise[i, j] = sigma

    return grid_medians, grid_noise


def interpolate_grid_to_full(grid, target_shape):
    """
    Bilinearly interpolate a coarse grid to the target full-frame image shape.
    """
    zoom_factors = (target_shape[0] / grid.shape[0], target_shape[1] / grid.shape[1])
    interpolated = ndi.zoom(grid, zoom_factors, order=1)

    # Crop or pad slightly if rounding caused off-by-one difference
    if interpolated.shape != target_shape:
        res = np.zeros(target_shape, dtype=np.float32)
        r_max = min(target_shape[0], interpolated.shape[0])
        c_max = min(target_shape[1], interpolated.shape[1])
        res[:r_max, :c_max] = interpolated[:r_max, :c_max]
        return res
    return interpolated.astype(np.float32)


def evaluate_mask(mask):
    """
    Count total masked pixels, total connected components,
    and components with area >= 3 pixels.
    """
    num_pixels = int(np.count_nonzero(mask))
    if num_pixels == 0:
        return 0, 0, 0

    labeled, num_components = ndi.label(mask)
    if num_components == 0:
        del labeled
        return num_pixels, 0, 0

    # Fast component area histogram
    areas = np.bincount(labeled.ravel())[1:]
    components_ge3 = int(np.count_nonzero(areas >= 3))

    del labeled
    del areas
    return num_pixels, int(num_components), components_ge3


def save_diagnostic_visuals(data, bg_local, mask_global_3, mask_local_3, out_path, title):
    """
    Save 4-panel visual comparison:
    1. Downsampled raw image (percentile stretch)
    2. Estimated 2D local background map
    3. Global k=3 threshold detection mask
    4. Local k=3 threshold detection mask
    """
    down = 8
    sub_data = data[::down, ::down].astype(np.float32)
    sub_bg = bg_local[::down, ::down]
    sub_m_glob = mask_global_3[::down, ::down]
    sub_m_loc = mask_local_3[::down, ::down]

    p2, p98 = np.percentile(sub_data, [2.0, 98.0])
    if p98 <= p2:
        p98 = p2 + 1.0
    norm_data = np.clip((sub_data - p2) / (p98 - p2), 0.0, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    im0 = axes[0, 0].imshow(norm_data, cmap="gray", origin="lower")
    axes[0, 0].set_title("Subsampled Raw Image (2%-98% Stretch)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(sub_bg, cmap="viridis", origin="lower")
    axes[0, 1].set_title(f"Estimated Local Background (Range: {sub_bg.min():.1f} - {sub_bg.max():.1f} ADU)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(sub_m_glob, cmap="hot", origin="lower")
    axes[1, 0].set_title("Global Threshold Mask (k=3)")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(sub_m_loc, cmap="hot", origin="lower")
    axes[1, 1].set_title("Local 2D Threshold Mask (k=3)")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def process_image(filepath, output_dir):
    filepath = Path(filepath)
    print(f"\nProcessing {filepath.name}...")

    with fits.open(filepath, mode="readonly", memmap=False) as hdul:
        data = hdul[0].data

    rows, cols = data.shape

    # 1. Global background & noise
    glob_bg, glob_noise = estimate_global_stats(data)
    print(f"  Global BG: {glob_bg:.2f}, Global noise: {glob_noise:.2f}")

    # 2. Local 2D background & noise map (256x256 blocks)
    grid_med, grid_noise = compute_block_grid(data, block_size=256)
    bg_local = interpolate_grid_to_full(grid_med, (rows, cols))
    noise_local = interpolate_grid_to_full(grid_noise, (rows, cols))

    # Guard against unreliable or zero local noise
    zero_noise_blocks = int(np.count_nonzero(grid_noise <= 0.0))
    if zero_noise_blocks > 0:
        print(f"  Notice: {zero_noise_blocks}/{grid_noise.size} blocks have zero quantile noise. Falling back to global noise for zero regions.")
        noise_local[noise_local <= 0.0] = glob_noise if glob_noise > 0.0 else 1.0

    print(f"  Local BG range: [{bg_local.min():.2f}, {bg_local.max():.2f}] | Local noise range: [{noise_local.min():.2f}, {noise_local.max():.2f}]")

    results = []
    masks_for_plot = {}

    for k in [3.0, 4.0]:
        # --- Global Method ---
        glob_thresh = glob_bg + k * glob_noise
        mask_glob = data > glob_thresh
        pix_g, comp_g, ge3_g = evaluate_mask(mask_glob)

        results.append({
            "file_name": filepath.name,
            "method": "Global",
            "k": k,
            "threshold_repr": f"{glob_thresh:.2f}",
            "thresh_min": round(glob_thresh, 2),
            "thresh_max": round(glob_thresh, 2),
            "thresholded_pixels": pix_g,
            "total_components": comp_g,
            "components_area_ge3": ge3_g,
            "notes": "Single scalar threshold",
        })

        if k == 3.0:
            masks_for_plot["glob_3"] = mask_glob
        del mask_glob

        # --- Local Method ---
        thresh_local_map = bg_local + k * noise_local
        mask_loc = data > thresh_local_map
        pix_l, comp_l, ge3_l = evaluate_mask(mask_loc)

        results.append({
            "file_name": filepath.name,
            "method": "Local_2D",
            "k": k,
            "threshold_repr": f"[{thresh_local_map.min():.2f} - {thresh_local_map.max():.2f}]",
            "thresh_min": round(float(thresh_local_map.min()), 2),
            "thresh_max": round(float(thresh_local_map.max()), 2),
            "thresholded_pixels": pix_l,
            "total_components": comp_l,
            "components_area_ge3": ge3_l,
            "notes": f"Interpolated 256x256 grid; {zero_noise_blocks} zero-noise blocks guarded",
        })

        if k == 3.0:
            masks_for_plot["loc_3"] = mask_loc
        del mask_loc
        del thresh_local_map

    # Save visual comparison
    plot_path = output_dir / f"{filepath.stem}_bg_comparison.png"
    save_diagnostic_visuals(
        data=data,
        bg_local=bg_local,
        mask_global_3=masks_for_plot["glob_3"],
        mask_local_3=masks_for_plot["loc_3"],
        out_path=plot_path,
        title=f"Background & Thresholding Comparison: {filepath.name}",
    )
    print(f"  Visual plot saved to: {plot_path.name}")

    del data
    del bg_local
    del noise_local
    del masks_for_plot
    gc.collect()

    return results


def main():
    output_dir = Path("output/local_background_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("GLOBAL VS. LOCAL BACKGROUND ESTIMATION COMPARISON")
    print("=" * 75)

    all_rows = []
    for f in TARGET_FILES:
        path = Path(f)
        if not path.exists():
            print(f"Warning: File {f} not found!")
            continue
        rows = process_image(path, output_dir)
        all_rows.extend(rows)

    summary_df = pd.DataFrame(all_rows)
    csv_path = output_dir / "comparison_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 75)
    print("COMPARISON RESULTS SUMMARY TABLE")
    print("=" * 75)
    display_cols = [
        "file_name",
        "method",
        "k",
        "threshold_repr",
        "thresholded_pixels",
        "total_components",
        "components_area_ge3",
    ]
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 75)
    print(f"Saved CSV: {csv_path.resolve()}\n")


if __name__ == "__main__":
    main()
