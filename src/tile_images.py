import argparse
from pathlib import Path
from astropy.io import fits
import numpy as np
import pandas as pd


def tile_image(fits_path, output_dir, tile_size=1024):
    fits_path = Path(fits_path)
    output_dir = Path(output_dir)
    image_dir = output_dir / fits_path.stem
    image_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with fits.open(fits_path, mode="readonly", memmap=False) as hdul:
        data = hdul[0].data

    height, width = data.shape
    num_rows = int(np.ceil(height / tile_size))
    num_cols = int(np.ceil(width / tile_size))

    for row_idx in range(num_rows):
        y_min = row_idx * tile_size
        y_max = min(y_min + tile_size, height)
        valid_h = y_max - y_min

        for col_idx in range(num_cols):
            x_min = col_idx * tile_size
            x_max = min(x_min + tile_size, width)
            valid_w = x_max - x_min

            crop = data[y_min:y_max, x_min:x_max]

            if valid_h == tile_size and valid_w == tile_size:
                tile_data = crop
                padded = False
            else:
                tile_data = np.zeros((tile_size, tile_size), dtype=data.dtype)
                tile_data[:valid_h, :valid_w] = crop
                padded = True

            tile_name = f"{fits_path.stem}_r{row_idx:02d}_c{col_idx:02d}.npy"
            np.save(image_dir / tile_name, tile_data)

            records.append({
                "tile_name": tile_name,
                "source_file": fits_path.name,
                "row_idx": row_idx,
                "col_idx": col_idx,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "valid_width": valid_w,
                "valid_height": valid_h,
                "padded": padded,
                "dtype": str(tile_data.dtype),
            })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="tiles")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        fits_files = [Path(args.file)]
    else:
        fits_files = sorted(list(data_dir.glob("*.fits")) + list(data_dir.glob("*.fit")))

    if not fits_files:
        print(f"No FITS files found.")
        return

    all_records = []
    for i, fits_file in enumerate(fits_files, 1):
        print(f"[{i}/{len(fits_files)}] Tiling {fits_file.name}...")
        records = tile_image(fits_file, output_dir, tile_size=args.tile_size)
        all_records.extend(records)

    manifest_df = pd.DataFrame(all_records)
    manifest_path = output_dir / "tiles_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Saved {len(all_records)} tile records to {manifest_path}")


if __name__ == "__main__":
    main()
