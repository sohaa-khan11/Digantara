"""
Generate Inspection Contact Sheet for Automated Segmentation Annotations.

Extracts representative tile images across:
1. High-confidence streaks
2. Dense star fields (blob_stars)
3. Padded edge tiles (valid width 352 px, valid height 236 px)
4. Confirmed empty/background tiles (0 instances)
5. Human-verified reference tiles

Renders side-by-side: original raw FITS tile imagery and segmented YOLO polygon overlays.
Color conventions:
  blob_star (class 0): Lime
  streak (class 1): Cyan
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def resolve_tile_path(tile_manifest_df, tile_name):
    """
    Resolves the actual file path for a tile .npy file inside tiles/<source_stem>/<tile_name>.npy.
    """
    row = tile_manifest_df[tile_manifest_df["tile_name"] == tile_name]
    if row.empty:
        # Try finding with .npy extension
        row = tile_manifest_df[tile_manifest_df["tile_name"] == f"{tile_name}.npy"]
    
    if not row.empty:
        source_file = row.iloc[0]["source_file"]
        stem = Path(source_file).stem
        fname = f"{tile_name}.npy" if not tile_name.endswith(".npy") else tile_name
        cand_path = Path("tiles") / stem / fname
        if cand_path.exists():
            return cand_path

    # Fallback search across subdirectories in tiles/
    for sub in Path("tiles").iterdir():
        if sub.is_dir():
            cand = sub / f"{tile_name}.npy"
            if cand.exists():
                return cand
            cand = sub / tile_name
            if cand.exists():
                return cand

    return None


def generate_contact_sheet(
    output_path="annotation/automated_yolo_tiles/contact_sheet_inspection.png"
):
    tile_manifest_path = Path("annotation/automated_yolo_tiles/per_tile_automated_manifest.csv")
    tiles_info_path = Path("tiles/tiles_manifest.csv")

    if not tile_manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {tile_manifest_path}")

    tile_manifest = pd.read_csv(tile_manifest_path)
    tiles_info = pd.read_csv(tiles_info_path)

    # Representative cases to display
    # 1. High-confidence streak candidate in 4a8e...
    streak_candidates_1 = tile_manifest[(tile_manifest["pred_streak"] >= 1) & (tile_manifest["source_file"] == "4a8e6cd4-1f5a-4f3a-a858-0f905c2d0b9a.fits")]
    sample_streak_1 = streak_candidates_1.sort_values("pred_streak", ascending=False).iloc[0]

    # 2. High-confidence streak candidate in CAM_B...276
    streak_candidates_2 = tile_manifest[(tile_manifest["pred_streak"] >= 1) & (tile_manifest["source_file"] == "CAM_B_20260815T154244_manual_f000276.fits")]
    sample_streak_2 = streak_candidates_2.sort_values("pred_streak", ascending=False).iloc[0]

    # 3. Dense star field in 66ed...
    sample_dense_stars = tile_manifest[(tile_manifest["pred_blob_star"] >= 20) & (tile_manifest["source_file"] == "66ed4268-554e-4729-9e87-e556c2fd42b9.fits")].iloc[0]

    # 4. Human-reviewed star field in 3e6b... tile r03_c04
    sample_human_stars = tile_manifest[(tile_manifest["source_file"] == "3e6beceb-b453-4e49-a222-b661eaaac4d0.fits") & (tile_manifest["tile_name"] == "3e6beceb-b453-4e49-a222-b661eaaac4d0_r03_c04")].iloc[0]

    # 5. Right edge padded tile (col=9, valid_width=352)
    sample_edge_right = tile_manifest[(tile_manifest["col_idx"] == 9) & (tile_manifest["pred_blob_star"] > 0)].iloc[0]

    # 6. Bottom edge padded tile (row=6, valid_height=236)
    sample_edge_bottom = tile_manifest[(tile_manifest["row_idx"] == 6) & (tile_manifest["pred_blob_star"] > 0)].iloc[0]

    # 7. True background empty tile (0 instances)
    sample_empty = tile_manifest[tile_manifest["total_pred_instances"] == 0].iloc[0]

    # 8. Human-verified pilot tile in 1a60... tile r00_c01 (contains verified streak + stars)
    sample_human_pilot = tile_manifest[tile_manifest["tile_name"] == "1a600998-b97c-4307-8632-6fcf573621ff_r00_c01"].iloc[0]

    cases = [
        ("Streak Candidate (4a8e...)", sample_streak_1),
        ("Streak Candidate (CAM_B...276)", sample_streak_2),
        ("Dense Star Field (66ed...)", sample_dense_stars),
        ("Human-Reviewed Stars (3e6b...)", sample_human_stars),
        ("Padded Right Edge (Valid W=352 px)", sample_edge_right),
        ("Padded Bottom Edge (Valid H=236 px)", sample_edge_bottom),
        ("Background Tile (Empty Predictions)", sample_empty),
        ("Human Pilot Tile (1a60...r00_c01)", sample_human_pilot),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(18, 18), facecolor="#1a1a1a")
    fig.suptitle(
        "Digantara Automated YOLO Segmentation Inspection Contact Sheet\n(Raw FITS Tile Imagery vs Predicted / Human Instance Polygons)",
        fontsize=16,
        fontweight="bold",
        color="white",
        y=0.98
    )

    legend_elements = [
        Line2D([0], [0], color="lime", lw=2, label="blob_star (Class 0)"),
        Line2D([0], [0], color="cyan", lw=2, label="streak (Class 1)"),
        Line2D([0], [0], color="red", lw=1.5, ls="--", label="Valid Footprint Boundary")
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.95, 0.98), facecolor="#2b2b2b", edgecolor="gray", labelcolor="white")

    for i, (category_title, row) in enumerate(cases):
        tname = row["tile_name"]
        npy_path = resolve_tile_path(tiles_info, tname)
        txt_path = Path("annotation/automated_yolo_tiles/labels") / f"{tname}.txt"

        if npy_path is None or not npy_path.exists():
            print(f"ERROR: Tile image file not found for {tname}", file=sys.stderr)
            continue

        tile_img = np.load(npy_path).astype(np.float32)
        valid_w = int(row["valid_width"])
        valid_h = int(row["valid_height"])

        # Percentile contrast stretch for astronomical data
        # Mask valid pixels to compute robust background stretch
        valid_pixels = tile_img[:valid_h, :valid_w]
        p1, p99_5 = np.percentile(valid_pixels, [1.0, 99.5])
        stretch_denom = max(1.0, p99_5 - p1)
        tile_norm = np.clip((tile_img - p1) / stretch_denom, 0.0, 1.0)

        # Left panel: RAW FITS Tile
        ax_raw = axes[i // 2, (i % 2) * 2]
        ax_raw.imshow(tile_norm, cmap="gray", origin="upper")
        ax_raw.set_title(
            f"RAW: {category_title}\n[{tname[:28]}...]\nFootprint: {valid_w}x{valid_h} px",
            fontsize=8.5,
            color="white",
            pad=4
        )
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])
        for spine in ax_raw.spines.values():
            spine.set_color("#444444")

        # Draw footprint boundary if padded
        if valid_w < 1024 or valid_h < 1024:
            rect = plt.Rectangle((0, 0), valid_w, valid_h, fill=False, edgecolor="red", linestyle="--", linewidth=1.5)
            ax_raw.add_patch(rect)

        # Right panel: OVERLAY with Polygons
        ax_ov = axes[i // 2, (i % 2) * 2 + 1]
        ax_ov.imshow(tile_norm, cmap="gray", origin="upper")

        poly_count_star = 0
        poly_count_streak = 0

        if txt_path.exists():
            with open(txt_path) as fp:
                for line in fp:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    coords = [float(c) for c in parts[1:]]
                    xs = np.array(coords[0::2]) * 1024.0
                    ys = np.array(coords[1::2]) * 1024.0
                    color = "cyan" if cls_id == 1 else "lime"
                    ax_ov.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), color=color, linewidth=1.5)
                    if cls_id == 1:
                        poly_count_streak += 1
                    else:
                        poly_count_star += 1

        if valid_w < 1024 or valid_h < 1024:
            rect = plt.Rectangle((0, 0), valid_w, valid_h, fill=False, edgecolor="red", linestyle="--", linewidth=1.5)
            ax_ov.add_patch(rect)

        prov = "Human-Verified" if row["human_reviewed"] else "Automated Pipeline"
        ax_ov.set_title(
            f"OVERLAY ({prov})\nStars: {poly_count_star} | Streaks: {poly_count_streak}\nStatus: {'Negative Background' if row['total_pred_instances']==0 else 'Target Detections'}",
            fontsize=8.5,
            color="white",
            pad=4
        )
        ax_ov.set_xticks([])
        ax_ov.set_yticks([])
        for spine in ax_ov.spines.values():
            spine.set_color("#444444")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"SUCCESS: Contact sheet generated and saved to {output_path}")


if __name__ == "__main__":
    generate_contact_sheet()
