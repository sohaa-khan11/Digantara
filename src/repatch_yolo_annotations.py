"""
Repatch Tile-Level YOLO Annotations to Full-Frame Coordinate Space (9568 x 6380).

Requirements:
- Read human-verified XML annotations and tiles_manifest.csv.
- Project tile-relative polygon vertices to full-frame dimensions (9568 x 6380).
- Produce compliant full-size YOLO Ultralytics segmentation format (.txt).
- Generate a contrast-stretched full-frame overview PNG with overlaid polygons for visual inspection.
- Maintain strict provenance: only human-verified annotations are repatched; unreviewed tiles are NOT marked as background.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASS_MAPPING = {
    "blob_star": 0,
    "streak": 1,
}

FULL_WIDTH = 9568.0
FULL_HEIGHT = 6380.0


def repatch_annotations(xml_path, manifest_path, output_dir, data_dir=None):
    xml_path = Path(xml_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    vis_dir = output_dir / "visual_inspection"

    labels_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    # Map tile filename (stem) to row record
    manifest["stem"] = manifest["tile_name"].apply(lambda x: Path(x).stem)
    manifest_lookup = manifest.set_index("stem").to_dict(orient="index")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Group polygons by source FITS image
    grouped_by_image = {}

    for img_elem in root.findall(".//image"):
        img_name = img_elem.get("name")
        stem = Path(img_name).stem

        if stem not in manifest_lookup:
            print(f"Warning: Tile {stem} not found in manifest!")
            continue

        tile_info = manifest_lookup[stem]
        source_fits = tile_info["source_file"]
        x_offset = float(tile_info["x_min"])
        y_offset = float(tile_info["y_min"])

        if source_fits not in grouped_by_image:
            grouped_by_image[source_fits] = []

        for poly in img_elem.findall("polygon"):
            label = poly.get("label")
            if label not in CLASS_MAPPING:
                continue

            cls_id = CLASS_MAPPING[label]
            pts_str = poly.get("points")
            full_coords = []

            for pair in pts_str.strip().split(";"):
                if not pair.strip():
                    continue
                x_t, y_t = map(float, pair.split(","))

                # Map tile coordinate to full-frame space
                x_full = x_offset + x_t
                y_full = y_offset + y_t

                # Normalize to [0.0, 1.0] relative to 9568 x 6380
                x_norm = max(0.0, min(1.0, x_full / FULL_WIDTH))
                y_norm = max(0.0, min(1.0, y_full / FULL_HEIGHT))
                full_coords.extend([x_norm, y_norm])

            grouped_by_image[source_fits].append({
                "class_id": cls_id,
                "label": label,
                "tile_stem": stem,
                "coords": full_coords,
            })

    print(f"Repatching annotations across {len(grouped_by_image)} source FITS image(s)...")

    for src_fits, annotations in grouped_by_image.items():
        stem_fits = Path(src_fits).stem
        txt_out = labels_dir / f"{stem_fits}.txt"

        lines = []
        for ann in annotations:
            cls_id = ann["class_id"]
            coords_str = " ".join(f"{c:.6f}" for c in ann["coords"])
            lines.append(f"{cls_id} {coords_str}")

        with open(txt_out, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

        print(f"  -> {txt_out.name}: Saved {len(lines)} repatched polygon(s)")

        # Generate full-frame visual inspection figure if raw FITS is accessible
        if data_dir:
            fits_path = Path(data_dir) / src_fits
            if fits_path.exists():
                render_visual_inspection(fits_path, annotations, vis_dir / f"{stem_fits}_inspection.png")

    # Document annotated regions of interest to maintain honest label accounting
    roi_summary = []
    for src_fits, annotations in grouped_by_image.items():
        unique_tiles = sorted(list(set(a["tile_stem"] for a in annotations)))
        roi_summary.append({
            "source_fits": src_fits,
            "repatched_polygons": len(annotations),
            "annotated_tiles_count": len(unique_tiles),
            "annotated_tiles": ", ".join(unique_tiles),
            "unreviewed_tiles_count": 70 - len(unique_tiles),
            "provenance": "human_verified_cvat_export",
        })

    roi_df = pd.DataFrame(roi_summary)
    manifest_csv = output_dir / "repatched_annotation_manifest.csv"
    roi_df.to_csv(manifest_csv, index=False)
    print(f"Saved repatched annotation manifest to: {manifest_csv.name}")


def render_visual_inspection(fits_path, annotations, out_png):
    """
    Render a downsampled full-field image with repatched annotation polygons overlaid.
    """
    with fits.open(fits_path, mode="readonly", memmap=False) as hdul:
        data = hdul[0].data

    # Downsample by 8 for high-res fast inspection (1196 x 798)
    down = 8
    sub = data[::down, ::down].astype(np.float32)

    p1, p99 = np.percentile(sub, [1.0, 99.5])
    if p99 <= p1:
        p99 = p1 + 1.0
    norm = np.clip((sub - p1) / (p99 - p1), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    ax.imshow(norm, cmap="gray", origin="lower")

    for ann in annotations:
        coords = ann["coords"]
        xs = [coords[i] * (FULL_WIDTH / down) for i in range(0, len(coords) - 1, 2)]
        ys = [coords[i + 1] * (FULL_HEIGHT / down) for i in range(0, len(coords) - 1, 2)]

        color = "#00ffff" if ann["class_id"] == 1 else "#ff00ff"
        ax.plot(xs + [xs[0]], ys + [ys[0]], color=color, linewidth=1.2)
        ax.plot(xs[0], ys[0], "o", color=color, markersize=3)

    ax.set_title(
        f"Repatched Full-Size Annotation Inspection: {fits_path.name}\n"
        f"Magenta: blob_star (class 0) | Cyan: streak (class 1) | Total Polygons: {len(annotations)}",
        fontsize=10,
    )
    ax.axis("off")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Visual inspection overview: {out_png.name}")


def main():
    parser = argparse.ArgumentParser(description="Repatch tile annotations to full-size YOLO format.")
    parser.add_argument("--xml", type=str, default="annotation/cvat_export/annotations_combined_pilot4.xml")
    parser.add_argument("--manifest", type=str, default="tiles/tiles_manifest.csv")
    parser.add_argument("--output-dir", type=str, default="annotation/repatched_full_size")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    repatch_annotations(args.xml, args.manifest, args.output_dir, args.data_dir)


if __name__ == "__main__":
    main()
