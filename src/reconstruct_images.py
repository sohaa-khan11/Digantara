
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


def reconstruct_images(data_dir, tiles_dir, manifest_path, output_dir):
    data_dir = Path(data_dir)
    tiles_dir = Path(tiles_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    total = 0

    for fits_path in sorted(data_dir.glob("*.fits")):
        with fits.open(fits_path, mode="readonly", memmap=False) as hdul:
            original = hdul[0].data

        reconstructed = np.empty_like(original)
        rows = manifest[manifest["source_file"] == fits_path.name]

        if rows.empty:
            raise ValueError(f"No tiles found for {fits_path.name}")

        for _, row in rows.iterrows():
            tile_path = tiles_dir / fits_path.stem / row["tile_name"]
            tile = np.load(tile_path)

            y0, y1 = int(row["y_min"]), int(row["y_max"])
            x0, x1 = int(row["x_min"]), int(row["x_max"])
            valid_h, valid_w = y1 - y0, x1 - x0

            if tile.shape != (1024, 1024):
                raise ValueError(f"Unexpected tile shape: {tile_path}")

            reconstructed[y0:y1, x0:x1] = tile[:valid_h, :valid_w]

        if not np.array_equal(original, reconstructed):
            raise ValueError(f"Reconstruction mismatch: {fits_path.name}")

        output_path = output_dir / fits_path.name
        fits.writeto(output_path, reconstructed, overwrite=True)
        print(f"PASS: {fits_path.name} -> {output_path.name}")
        total += 1

    print(f"\nSuccessfully reconstructed {total} FITS images.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tiles-dir", default="tiles")
    parser.add_argument("--manifest", default="tiles/tiles_manifest.csv")
    parser.add_argument("--output-dir", default="output/reconstructed")
    args = parser.parse_args()

    reconstruct_images(
        args.data_dir, args.tiles_dir, args.manifest, args.output_dir
    )


if __name__ == "__main__":
    main()
