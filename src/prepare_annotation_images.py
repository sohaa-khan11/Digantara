import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def stretch_tile(tile_data, valid_w, valid_h):
    valid_region = tile_data[:valid_h, :valid_w].astype(np.float32)

    vmin = float(np.min(valid_region))
    vmax = float(np.max(valid_region))

    stretched_tile = np.zeros(tile_data.shape, dtype=np.uint8)

    # Constant intensity or fully zero valid region
    if vmax <= vmin:
        return stretched_tile

    # Percentile stretch: robust against hot pixels and cosmic rays
    p_low = float(np.percentile(valid_region, 1.0))
    p_high = float(np.percentile(valid_region, 99.8))

    if p_high <= p_low:
        p_high = vmax

    clipped = np.clip(valid_region, p_low, p_high)
    normalized = ((clipped - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)

    stretched_tile[:valid_h, :valid_w] = normalized
    return stretched_tile


def prepare_images(tiles_dir="tiles", manifest_path="tiles/tiles_manifest.csv", output_dir="annotation/images", limit=None):
    tiles_dir = Path(tiles_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path)
    if limit is not None and limit > 0:
        manifest_df = manifest_df.iloc[:limit]

    records = []
    total = len(manifest_df)
    print(f"Preparing {total} annotation image(s) into {output_dir}...")

    for idx, row in manifest_df.iterrows():
        source_stem = Path(row["source_file"]).stem
        tile_path = tiles_dir / source_stem / row["tile_name"]

        if not tile_path.exists():
            print(f"Skipping missing tile: {tile_path}")
            continue

        tile_data = np.load(tile_path)
        valid_w = int(row["valid_width"])
        valid_h = int(row["valid_height"])

        stretched = stretch_tile(tile_data, valid_w, valid_h)

        image_name = f"{Path(row['tile_name']).stem}.png"
        image_path = output_dir / image_name

        Image.fromarray(stretched, mode="L").save(image_path)

        record = row.to_dict()
        record["image_name"] = image_name
        record["image_width"] = stretched.shape[1]
        record["image_height"] = stretched.shape[0]
        records.append(record)

    out_manifest_df = pd.DataFrame(records)
    out_manifest_path = output_dir.parent / "images_manifest.csv"
    out_manifest_df.to_csv(out_manifest_path, index=False)
    print(f"Saved {len(records)} images and updated manifest at {out_manifest_path}")
    return out_manifest_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles-dir", type=str, default="tiles")
    parser.add_argument("--manifest", type=str, default="tiles/tiles_manifest.csv")
    parser.add_argument("--output-dir", type=str, default="annotation/images")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    prepare_images(
        tiles_dir=args.tiles_dir,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
