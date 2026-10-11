"""
Automated Segmentation Annotation Pipeline for Digantara Assessment.

Processes all 700 tiles across 10 FITS images using the validated candidate masks
and calibrated morphology filters to export:
1. Tile-level YOLO segmentation labels (annotation/automated_yolo_tiles/labels/)
2. Repatched full-frame YOLO segmentation labels (annotation/repatched_full_size_automated/labels/)
3. Comprehensive per-tile and per-image manifest documenting human vs automated provenance.

Class mapping:
  blob_star = 0
  streak = 1
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import label, find_objects
from scipy.spatial import ConvexHull


def mask_to_segmentation_polygon(component_mask, offset_x=0.0, offset_y=0.0, min_dist=1.5, max_points=24):
    """
    Extracts a valid closed polygon from a binary component mask.
    Handles thin 1-pixel features via subpixel expansion to ensure non-degenerate 2D polygons.
    """
    yy, xx = np.nonzero(component_mask)
    if len(yy) < 3:
        return None

    pts = np.column_stack([xx.astype(np.float64) + offset_x, yy.astype(np.float64) + offset_y])

    # Guard against 1D degenerate collinear points
    if np.ptp(pts[:, 0]) < 1e-4:
        pts = np.vstack([pts + [-0.5, 0.0], pts + [0.5, 0.0]])
    if np.ptp(pts[:, 1]) < 1e-4:
        pts = np.vstack([pts + [0.0, -0.5], pts + [0.0, 0.5]])

    try:
        hull = ConvexHull(pts)
        h_pts = pts[hull.vertices]
        if len(h_pts) < 3:
            return None

        # Vertex reduction / simplification
        if len(h_pts) > 4:
            keep = [0]
            for i in range(1, len(h_pts)):
                if np.linalg.norm(h_pts[i] - h_pts[keep[-1]]) >= min_dist:
                    keep.append(i)
            h_pts = h_pts[keep]
            if len(h_pts) > 3 and np.linalg.norm(h_pts[-1] - h_pts[0]) < min_dist:
                h_pts = h_pts[:-1]

        if len(h_pts) > max_points:
            step = len(h_pts) / float(max_points)
            indices = [int(i * step) for i in range(max_points)]
            h_pts = h_pts[indices]

        if len(h_pts) < 3:
            return None
        return h_pts
    except Exception:
        return None


def run_automated_annotation(
    tiles_manifest_path="tiles/tiles_manifest.csv",
    candidate_catalog_path="output/automated_candidates/all_10_images_candidate_catalog.csv",
    masks_dir="output/automated_candidates/candidate_masks",
    output_tiles_dir="annotation/automated_yolo_tiles",
    output_full_dir="annotation/repatched_full_size_automated",
    streak_min_snr=10.0,
    star_min_snr=15.0,
    star_min_area=6,
    star_max_area=1500
):
    tiles_manifest = pd.read_csv(tiles_manifest_path)
    cands_df = pd.read_csv(candidate_catalog_path)
    masks_dir = Path(masks_dir)

    out_tiles_lbl = Path(output_tiles_dir) / "labels"
    out_full_lbl = Path(output_full_dir) / "labels"
    out_tiles_lbl.mkdir(parents=True, exist_ok=True)
    out_full_lbl.mkdir(parents=True, exist_ok=True)

    # Known human-reviewed tiles across pilot and validation
    human_reviewed_tiles = {
        "1a600998-b97c-4307-8632-6fcf573621ff_r00_c00",
        "1a600998-b97c-4307-8632-6fcf573621ff_r00_c01",
        "1a600998-b97c-4307-8632-6fcf573621ff_r00_c02",
        "1a600998-b97c-4307-8632-6fcf573621ff_r02_c09",
        "3e6beceb-b453-4e49-a222-b661eaaac4d0_r00_c09",
        "3e6beceb-b453-4e49-a222-b661eaaac4d0_r03_c04",
        "3e6beceb-b453-4e49-a222-b661eaaac4d0_r03_c05",
        "3e6beceb-b453-4e49-a222-b661eaaac4d0_r05_c01",
        "7d7fefcb-60ce-4f09-8572-4db66d030ff3_r02_c00",
        "7d7fefcb-60ce-4f09-8572-4db66d030ff3_r05_c06",
        "CAM_B_20260815T153133_manual_f000207_r01_c06",
        "CAM_B_20260815T153133_manual_f000207_r02_c02",
        "CAM_B_20260815T153133_manual_f000207_r02_c03",
        "CAM_B_20260815T153133_manual_f000207_r06_c09",
    }

    # Filter high-confidence candidates
    valid_streaks = cands_df[
        (cands_df["candidate_category"] == "candidate_streak") &
        (cands_df["snr"] >= streak_min_snr) &
        ((cands_df["bbox_len"] >= 6) | (cands_df["snr"] >= 15.0))
    ]
    valid_stars = cands_df[
        (cands_df["candidate_category"] == "candidate_blob_star") &
        (cands_df["snr"] >= star_min_snr) &
        (cands_df["area"] >= star_min_area) &
        (cands_df["area"] <= star_max_area)
    ]

    valid_cand_ids = set(valid_streaks["candidate_id"]).union(set(valid_stars["candidate_id"]))
    print(f"Qualified candidates: {len(valid_streaks)} streaks, {len(valid_stars)} stars ({len(valid_cand_ids)} total)")

    fits_groups = tiles_manifest.groupby("source_file")
    per_tile_records = []
    full_image_records = []

    for source_file, group in fits_groups:
        stem = Path(source_file).stem
        mask_npz = masks_dir / f"{stem}_candidate_mask.npz"
        if not mask_npz.exists():
            print(f"Warning: mask not found for {source_file}")
            continue

        with np.load(mask_npz) as ldf:
            full_mask = ldf["mask"]
        full_h, full_w = full_mask.shape

        # Filter candidate rows for this image
        img_cands = cands_df[(cands_df["source_file"] == source_file) & (cands_df["candidate_id"].isin(valid_cand_ids))]

        full_frame_polygons = []

        # Process each tile
        for _, tile_row in group.iterrows():
            tname = tile_row["tile_name"].replace(".npy", "")
            x_min = int(tile_row["x_min"])
            x_max = int(tile_row["x_max"])
            y_min = int(tile_row["y_min"])
            y_max = int(tile_row["y_max"])
            valid_w = int(tile_row["valid_width"])
            valid_h = int(tile_row["valid_height"])

            is_human = tname in human_reviewed_tiles

            # Extract tile region from full candidate mask
            valid_patch = full_mask[y_min:y_max, x_min:x_max]

            # Find candidates whose centroids fall into this tile's valid footprint
            tile_cands = img_cands[
                (img_cands["cx"] >= x_min) & (img_cands["cx"] < x_max) &
                (img_cands["cy"] >= y_min) & (img_cands["cy"] < y_max)
            ]

            tile_polygons = []
            star_count = 0
            streak_count = 0

            for _, cand in tile_cands.iterrows():
                cid = cand["candidate_id"]
                cat = cand["candidate_category"]
                cls_id = 1 if cat == "candidate_streak" else 0

                # Crop candidate sub-region within valid_patch
                sl_ymin = max(0, int(cand["bbox_ymin"]) - y_min)
                sl_ymax = min(valid_h, int(cand["bbox_ymax"]) - y_min)
                sl_xmin = max(0, int(cand["bbox_xmin"]) - x_min)
                sl_xmax = min(valid_w, int(cand["bbox_xmax"]) - x_min)

                if sl_ymax <= sl_ymin or sl_xmax <= sl_xmin:
                    continue

                comp = (valid_patch[sl_ymin:sl_ymax, sl_xmin:sl_xmax] > 0)
                if np.sum(comp) < 3:
                    continue

                # Local tile polygon
                poly_local = mask_to_segmentation_polygon(comp, offset_x=sl_xmin, offset_y=sl_ymin)
                if poly_local is None or len(poly_local) < 3:
                    continue

                # Tile-normalized coordinates (1024x1024 tile space)
                norm_local = []
                for pt in poly_local:
                    nx = np.clip(pt[0] / 1024.0, 0.0, 1.0)
                    ny = np.clip(pt[1] / 1024.0, 0.0, 1.0)
                    norm_local.extend([nx, ny])

                tile_polygons.append((cls_id, norm_local))

                # Full-frame polygon
                poly_full = mask_to_segmentation_polygon(comp, offset_x=x_min + sl_xmin, offset_y=y_min + sl_ymin)
                if poly_full is not None and len(poly_full) >= 3:
                    norm_full = []
                    for pt in poly_full:
                        nx = np.clip(pt[0] / float(full_w), 0.0, 1.0)
                        ny = np.clip(pt[1] / float(full_h), 0.0, 1.0)
                        norm_full.extend([nx, ny])
                    full_frame_polygons.append((cls_id, norm_full))

                if cls_id == 1:
                    streak_count += 1
                else:
                    star_count += 1

            # Write tile label file
            tile_txt_path = out_tiles_lbl / f"{tname}.txt"
            with open(tile_txt_path, "w") as fp:
                for cls_id, coords in tile_polygons:
                    coord_str = " ".join(f"{c:.6f}" for c in coords)
                    fp.write(f"{cls_id} {coord_str}\n")

            per_tile_records.append({
                "source_file": source_file,
                "tile_name": tname,
                "row_idx": tile_row["row_idx"],
                "col_idx": tile_row["col_idx"],
                "valid_width": valid_w,
                "valid_height": valid_h,
                "human_reviewed": is_human,
                "provenance": "human_verified_cvat" if is_human else "automated_pipeline_candidate",
                "pred_blob_star": star_count,
                "pred_streak": streak_count,
                "total_pred_instances": star_count + streak_count,
                "label_file": str(tile_txt_path.relative_to(output_tiles_dir))
            })

        # Write full-frame repatched label file for this source image
        full_txt_path = out_full_lbl / f"{stem}.txt"
        with open(full_txt_path, "w") as fp:
            for cls_id, coords in full_frame_polygons:
                coord_str = " ".join(f"{c:.6f}" for c in coords)
                fp.write(f"{cls_id} {coord_str}\n")

        full_image_records.append({
            "source_file": source_file,
            "total_full_polygons": len(full_frame_polygons),
            "pred_blob_stars": sum(1 for c, _ in full_frame_polygons if c == 0),
            "pred_streaks": sum(1 for c, _ in full_frame_polygons if c == 1),
            "label_file": str(full_txt_path.relative_to(output_full_dir)),
            "provenance": "automated_pipeline_repatched"
        })
        print(f"Processed {source_file}: {len(full_frame_polygons)} full-frame polygons across 70 tiles.")

    # Save manifests
    tile_manifest_df = pd.DataFrame(per_tile_records)
    tile_manifest_out = Path(output_tiles_dir) / "per_tile_automated_manifest.csv"
    tile_manifest_df.to_csv(tile_manifest_out, index=False)

    full_manifest_df = pd.DataFrame(full_image_records)
    full_manifest_out = Path(output_full_dir) / "repatched_automated_manifest.csv"
    full_manifest_df.to_csv(full_manifest_out, index=False)

    print(f"\nCompleted automated segmentation export!")
    print(f"Per-tile manifest: {tile_manifest_out} ({len(tile_manifest_df)} tiles)")
    print(f"Full-frame manifest: {full_manifest_out} ({len(full_manifest_df)} images)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated segmentation pipeline for Digantara assessment.")
    parser.add_argument("--tiles-manifest", default="tiles/tiles_manifest.csv")
    parser.add_argument("--candidate-catalog", default="output/automated_candidates/all_10_images_candidate_catalog.csv")
    parser.add_argument("--masks-dir", default="output/automated_candidates/candidate_masks")
    parser.add_argument("--output-tiles-dir", default="annotation/automated_yolo_tiles")
    parser.add_argument("--output-full-dir", default="annotation/repatched_full_size_automated")
    args = parser.parse_args()

    run_automated_annotation(
        tiles_manifest_path=args.tiles_manifest,
        candidate_catalog_path=args.candidate_catalog,
        masks_dir=args.masks_dir,
        output_tiles_dir=args.output_tiles_dir,
        output_full_dir=args.output_full_dir
    )
