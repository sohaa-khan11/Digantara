import gc
from pathlib import Path
from astropy.io import fits
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def compute_block_medians(data, block_size=256):
    """
    Compute the min and max median intensity across complete, non-overlapping
    blocks of size block_size x block_size to quantify spatial background variation.
    """
    rows, cols = data.shape
    r_max = (rows // block_size) * block_size
    c_max = (cols // block_size) * block_size

    medians = []
    for r in range(0, r_max, block_size):
        for c in range(0, c_max, block_size):
            block = data[r : r + block_size, c : c + block_size]
            medians.append(float(np.median(block)))

    if not medians:
        return 0.0, 0.0

    return min(medians), max(medians)


def count_components_at_threshold(data, threshold):
    """
    Count connected components on the full image at a specified threshold.
    Releases the full-frame label array immediately to conserve memory.
    """
    # 1-byte boolean mask (approx 61 MB for 6380 x 9568)
    mask = data > threshold

    # ndi.label creates a 32-bit int array (approx 244 MB for 6380 x 9568)
    _, num_components = ndi.label(mask)

    del mask
    return int(num_components)


def diagnose_image(fits_path):
    """
    Perform full-frame diagnostics on a single FITS image.
    Avoids tile boundaries and measures background noise robustness.
    """
    fits_path = Path(fits_path)
    print(f"Diagnosing {fits_path.name}...")

    with fits.open(fits_path, mode="readonly", memmap=False) as hdul:
        # Keep native dtype when possible to reduce memory footprint
        data = hdul[0].data

    if data is None:
        return {
            "file_name": fits_path.name,
            "shape": "None",
            "dtype": "None",
            "bg_median": np.nan,
            "q16": np.nan,
            "q84": np.nan,
            "sigma_quantile": np.nan,
            "zero_pixel_count": 0,
            "block_min_median": np.nan,
            "block_max_median": np.nan,
            "components_3sigma": -1,
            "components_4sigma": -1,
            "status": "error_empty_data",
        }

    rows, cols = data.shape
    dtype_str = str(data.dtype)

    # 1. Count exact zero-valued pixels across full frame
    zero_pixel_count = int(np.count_nonzero(data == 0))

    # 2. Subsample regularly (every 8th pixel in both axes) for quantile estimation
    # 6380/8 * 9568/8 approx 950,000 pixels (fast and statistically representative)
    subsample = data[::8, ::8].astype(np.float32)

    bg_median = float(np.median(subsample))
    q16, q84 = np.percentile(subsample, [16.0, 84.0])
    q16 = float(q16)
    q84 = float(q84)

    # Quantile-based noise estimate for Gaussian-like background: (q84 - q16) / 2
    sigma_quantile = (q84 - q16) / 2.0

    # 3. Block-level background variation across complete 256x256 blocks
    block_min, block_max = compute_block_medians(data, block_size=256)

    # 4. Full-frame connected components at median + 3*sigma and median + 4*sigma
    if sigma_quantile <= 0.0 or np.isnan(sigma_quantile):
        comp_3sigma = -1
        comp_4sigma = -1
        status = "invalid_noise_skipped_components"
    else:
        thresh_3sigma = bg_median + 3.0 * sigma_quantile
        comp_3sigma = count_components_at_threshold(data, thresh_3sigma)

        thresh_4sigma = bg_median + 4.0 * sigma_quantile
        comp_4sigma = count_components_at_threshold(data, thresh_4sigma)
        status = "ok"

    # Explicitly release memory of full frame before returning
    del data
    del subsample
    gc.collect()

    return {
        "file_name": fits_path.name,
        "shape": f"{rows}x{cols}",
        "dtype": dtype_str,
        "bg_median": round(bg_median, 3),
        "q16": round(q16, 3),
        "q84": round(q84, 3),
        "sigma_quantile": round(sigma_quantile, 4),
        "zero_pixel_count": zero_pixel_count,
        "block_min_median": round(block_min, 3),
        "block_max_median": round(block_max, 3),
        "components_3sigma": comp_3sigma,
        "components_4sigma": comp_4sigma,
        "status": status,
    }


def main():
    data_dir = Path("data")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    fits_files = sorted(list(data_dir.glob("*.fits")) + list(data_dir.glob("*.fit")))
    if not fits_files:
        print(f"No FITS files found in {data_dir.resolve()}")
        return

    print("=" * 80)
    print("FITS DATASET FULL-FRAME DETECTION DIAGNOSTICS")
    print(f"Found {len(fits_files)} FITS image(s) in {data_dir}")
    print("=" * 80)

    records = []
    for f in fits_files:
        rec = diagnose_image(f)
        records.append(rec)

    df = pd.DataFrame(records)
    csv_path = output_dir / "dataset_detection_diagnostics.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("SUMMARY DIAGNOSTICS TABLE")
    print("=" * 80)
    display_cols = [
        "file_name",
        "dtype",
        "bg_median",
        "sigma_quantile",
        "zero_pixel_count",
        "block_min_median",
        "block_max_median",
        "components_3sigma",
        "components_4sigma",
        "status",
    ]
    print(df[display_cols].to_string(index=False))
    print("=" * 80)
    print(f"Diagnostics CSV saved to: {csv_path.resolve()}\n")


if __name__ == "__main__":
    main()
