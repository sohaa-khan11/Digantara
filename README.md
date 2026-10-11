# FITS Optical Processing & YOLO Segmentation Pipeline

A modular Python framework for astronomical and Space Situational Awareness (SSA) imagery. The pipeline provides end-to-end tooling for 16-bit FITS metadata extraction, lossless non-overlapping image tiling with edge padding, reversible bit-exact full-frame reconstruction, contrast-adjusted display generation, CVAT annotation conversion to YOLO Ultralytics segmentation format, full-size coordinate repatching, and full-frame algorithmic candidate profiling.

---

## Key Capabilities

* **16-bit FITS Inspection:** Directly ingests scaled single-band FITS data (`BITPIX=16`, `BZERO=32768`), computing global quantiles, block-level background variation, and noise characteristics without altering raw integer values.
* **Lossless Tiling Engine:** Slices large astronomical frames ($9568 \times 6380$) into uniform $1024 \times 1024$ patches with selective right- and bottom-edge zero padding, recording spatial coordinates and valid footprints in a manifest.
* **Bit-Exact Image Inversion:** Reconstructs original full-size FITS frames from tiled numpy arrays with bit-for-bit exactness (`np.array_equal` verified across all pixels).
* **Multi-Scale Annotation Formatting:** Converts CVAT polygon XML exports into normalized YOLO Ultralytics segmentation labels (`.txt`), supporting both tile-level ($1024 \times 1024$) and repatched full-frame ($9568 \times 6380$) coordinate systems.
* **Adaptive Candidate Detection:** Detects point sources and extended tracks across full frames using local 2D median filtering and robust quantile noise modeling $((q_{84}-q_{16})/2)$ to overcome integer quantization and optical vignetting.

---

## Repository Structure

```
.
├── src/
│   ├── fits_inspector.py               # FITS header & distribution analysis
│   ├── run_exploration.py              # Exploratory batch runner
│   ├── tile_images.py                  # Non-overlapping 1024x1024 tiling with edge padding
│   ├── prepare_annotation_images.py    # Display-stretched 8-bit PNG generation
│   ├── reconstruct_images.py           # Reversible bit-exact FITS reconstruction
│   ├── convert_cvat_xml_to_yolo.py     # CVAT XML to YOLO segmentation converter
│   ├── repatch_yolo_annotations.py     # Tile to full-frame (9568x6380) YOLO repatcher
│   ├── detect_full_frame_candidates.py # Adaptive 2D candidate detector
│   ├── run_automated_segmentation_pipeline.py # Dataset-wide automated segmentation & repatching
│   ├── generate_contact_sheet.py       # Visual contact sheet inspection generator
│   ├── compare_local_background.py     # Global vs 2D local background diagnostics
│   ├── diagnose_dataset_detection.py   # Full-frame noise & component diagnostics
│   ├── inspect_streak_candidates.py    # Exploratory streak candidate tool
│   ├── rank_reviewed_candidates.py     # Multi-metric candidate ranking
│   └── diagnostics/
│       └── probe_image1.py             # Empirical background and component probe
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Excludes raw data, PDF documents, and binaries
└── README.md
```

---

## Environment Setup

Requires **Python 3.10+**. Install the dependencies defined in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Core libraries used: `astropy`, `numpy`, `scipy`, `matplotlib`, `pandas`, and `Pillow`.

---

## Pipeline Usage

All scripts support clear command-line options with sensible defaults:

### 1. Inspect FITS Files
Extract observation metadata, header cards, and pixel distribution statistics:
```bash
python src/run_exploration.py --data-dir data --output-dir output --num-samples 2
```

### 2. Tile Full Images
Divide $9568 \times 6380$ images into non-overlapping $1024 \times 1024$ tiles, padding right/bottom borders as needed:
```bash
python src/tile_images.py --data-dir data --output-dir tiles --tile-size 1024
```

### 3. Reconstruct Full-Size Imagery
Verify loss-free reversibility by reconstructing the full FITS frames from tiles using `tiles_manifest.csv`:
```bash
python src/reconstruct_images.py --data-dir data --tiles-dir tiles --manifest tiles/tiles_manifest.csv --output-dir output/reconstructed
```

### 4. Prepare Annotation Display Images
Generate 8-bit contrast-stretched PNG images for annotation interfaces without modifying the raw `.npy` tiles:
```bash
python src/prepare_annotation_images.py --tiles-dir tiles --manifest tiles/tiles_manifest.csv --output-dir annotation/images
```

### 5. Convert CVAT XML to YOLO Segmentation (Human Annotations)
Convert exported CVAT polygon annotations to normalized YOLO segmentation format:
```bash
# Example for validation set
python src/convert_cvat_xml_to_yolo.py --xml "annotation/cvat_export/validation_10.xml" --images-dir "annotation/cvat_validation_set" --output-dir "annotation/yolo_validation/labels"
```

### 6. Repatch Human Annotations to Full-Frame Coordinate Space
Project tile-level normalized coordinates back into the original $9568 \times 6380$ sensor frame:
```bash
# Example for validation set
python src/repatch_yolo_annotations.py --xml "annotation/cvat_export/validation_10.xml" --manifest "tiles/tiles_manifest.csv" --output-dir "annotation/repatched_full_size_validation" --data-dir "data"
```

### 7. Run Full-Frame Candidate Detection
Scan full images for prospective point sources and linear tracks using adaptive 2D background modeling ($4\sigma$, area $\ge 3$ px):
```bash
python src/detect_full_frame_candidates.py --data-dir data --output-dir output/automated_candidates --k-sigma 4.0 --min-area 3
```

### 8. Run Dataset-Wide Automated Segmentation & Repatching
Generate tile-level and full-frame repatched YOLO segmentation labels across all 700 tiles using calibrated SNR and geometry criteria:
```bash
python src/run_automated_segmentation_pipeline.py --tiles-manifest tiles/tiles_manifest.csv --candidate-catalog output/automated_candidates/all_10_images_candidate_catalog.csv --masks-dir output/automated_candidates/candidate_masks --output-tiles-dir annotation/automated_yolo_tiles --output-full-dir annotation/repatched_full_size_automated
```

### 9. Generate Visual Inspection Contact Sheet
Render representative 16-panel grayscale comparison of raw FITS tiles against predicted/human polygon overlays:
```bash
python src/generate_contact_sheet.py
```

---

## Technical Specifications & Conventions

### Image & Sensor Geometry
* **Sensor Resolution:** $9568 \times 6380$ pixels (columns $\times$ rows).
* **Data Format:** Single-band 16-bit unsigned integer (`uint16`) FITS.
* **Operational Modes:** Supports low-background staring frames (12-bit ADC, $[0, 4095]$) and high-background tracking frames (full 16-bit, $[0, 65535]$ with optical vignetting).

### Tiling Matrix & Boundary Padding
Dividing $9568 \times 6380$ into $1024 \times 1024$ tiles yields a $7 \text{ rows} \times 10 \text{ columns}$ grid (70 tiles per image):
* **Interior Tiles:** Exact $1024 \times 1024$ unpadded regions.
* **Right Edge Tiles (Column 9):** Valid data occupies $x \in [0, 352]$; columns $[352, 1024]$ are zero-padded.
* **Bottom Edge Tiles (Row 6):** Valid data occupies $y \in [0, 236]$; rows $[236, 1024]$ are zero-padded.
* **Corner Tile (Row 6, Column 9):** Contains a $236 \times 352$ valid patch with dual-axis zero padding.

### Class Definitions & Classification Rules
* **Class 0 (`blob_star`):** Symmetric stellar point sources.
  - Candidate rule: $\text{area} \ge 5\text{ px}$, aspect ratio $AR \le 1.4$, moment elongation $\epsilon < 1.3$.
  - Automated export filter: $\text{SNR} \ge 15.0$, $6 \le \text{area} \le 1500\text{ px}$.
* **Class 1 (`streak`):** Linear satellite or orbital debris trails.
  - Candidate rule: $(\text{length} \ge 8\text{ px} \land AR \ge 1.8 \land \epsilon \ge 1.8 \land \text{area} \ge 8\text{ px}) \lor (\text{length} \ge 5\text{ px} \land AR \ge 2.5)$.
  - Automated export filter: $\text{SNR} \ge 10.0 \land (\text{length} \ge 6\text{ px} \lor \text{SNR} \ge 15.0)$.
* **YOLO Ultralytics Segmentation Syntax:**
  ```
  <class_id> x1 y1 x2 y2 ... xn yn
  ```
  All vertex coordinates $(x_i, y_i)$ are floating-point values strictly normalized to $[0.0, 1.0]$. Verified images containing zero annotations produce valid 0-byte `.txt` files (representing verified negative background).

---

## Validation & Known Limitations

1. **Reconstruction Equality:** All 10 full-frame images reconstructed via `reconstruct_images.py` pass bit-for-bit exact equality (`np.array_equal`) against raw source data.
2. **Label Provenance Separation:**
   - **Human Verified:** 14 tiles across pilot (`annotation/yolo_pilot4/`, `annotation/repatched_full_size/`) and validation sets (`annotation/yolo_validation/`, `annotation/repatched_full_size_validation/`).
   - **Automated Candidates:** 700 tiles in `annotation/automated_yolo_tiles/` and 10 full-frame files in `annotation/repatched_full_size_automated/`.
   - Manifests explicitly flag provenance (`human_verified_cvat` vs. `automated_pipeline_candidate`). Unreviewed tiles with zero automated predictions are algorithmically empty, not confirmed negatives.
3. **Discrete Quantization:** For frames where background levels are $\le 1.0\text{ ADU}$, single-pixel Poisson noise fluctuations reach discrete integer thresholds. An empirical minimum area floor ($\text{area} \ge 3\text{ px}$) is enforced during initial detection to suppress spurious 1-pixel thermal noise while preserving unresolved optical point spread functions.
4. **Candidate Review Requirement:** Algorithmic candidate detections serve as a triage tool for human annotators; they are not confirmed celestial or orbital objects without human review.

---

## Reproducibility

To verify the pipeline on local data:
1. Ensure raw FITS files are located in `data/`.
2. Run the tiling command (`src/tile_images.py`).
3. Run reconstruction verification (`src/reconstruct_images.py`) to confirm bit-exact preservation.
4. Convert human annotations using `src/convert_cvat_xml_to_yolo.py` and `src/repatch_yolo_annotations.py`.
5. Execute full-frame candidate detection (`src/detect_full_frame_candidates.py`).
6. Run the automated segmentation and repatching pipeline (`src/run_automated_segmentation_pipeline.py`).
7. Generate visual inspection contact sheets (`src/generate_contact_sheet.py`).
