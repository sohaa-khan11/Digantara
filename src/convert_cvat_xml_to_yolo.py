import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_MAPPING = {
    "blob_star": 0,
    "streak": 1,
}


def parse_points(points_str, img_w, img_h):
    coords = []
    pairs = points_str.strip().split(";")
    for pair in pairs:
        if not pair.strip():
            continue
        parts = pair.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid coordinate pair: '{pair}'")
        x = float(parts[0].strip())
        y = float(parts[1].strip())

        # Normalize to [0, 1]
        x_norm = max(0.0, min(1.0, x / img_w))
        y_norm = max(0.0, min(1.0, y / img_h))
        coords.extend([x_norm, y_norm])

    if len(coords) < 6:
        raise ValueError(f"Polygon requires at least 3 points, got {len(coords) // 2}")

    return coords


def convert_annotations(xml_path, images_dir, output_dir):
    xml_path = Path(xml_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    stats = {
        "total_images": 0,
        "missing_images": [],
        "labels_by_class": {cls_name: 0 for cls_name in CLASS_MAPPING},
        "generated_files": 0,
    }

    image_nodes = root.findall("image")
    stats["total_images"] = len(image_nodes)

    for img_elem in image_nodes:
        img_name = img_elem.get("name")
        img_w = float(img_elem.get("width"))
        img_h = float(img_elem.get("height"))

        img_path = images_dir / img_name
        if not img_path.exists():
            stats["missing_images"].append(img_name)
            continue

        label_lines = []
        for poly in img_elem.findall("polygon"):
            label_name = poly.get("label")
            if label_name not in CLASS_MAPPING:
                raise ValueError(f"Unknown label '{label_name}' in image {img_name}")

            cls_id = CLASS_MAPPING[label_name]
            stats["labels_by_class"][label_name] += 1

            points_str = poly.get("points")
            norm_coords = parse_points(points_str, img_w, img_h)

            coords_formatted = " ".join(f"{c:.6f}" for c in norm_coords)
            label_lines.append(f"{cls_id} {coords_formatted}")

        out_txt = output_dir / f"{Path(img_name).stem}.txt"
        with open(out_txt, "w", encoding="utf-8") as f:
            if label_lines:
                f.write("\n".join(label_lines) + "\n")

        stats["generated_files"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="annotation/cvat_export/annotations.xml")
    parser.add_argument("--images-dir", type=str, default="annotation/images")
    parser.add_argument("--output-dir", type=str, default="annotation/yolo_pilot/labels")
    args = parser.parse_args()

    stats = convert_annotations(args.xml, args.images_dir, args.output_dir)

    print("=" * 50)
    print("CVAT XML -> YOLO Segmentation Conversion Report")
    print("=" * 50)
    print(f"Total XML images:       {stats['total_images']}")
    print(f"Missing image matches:  {len(stats['missing_images'])}")
    if stats["missing_images"]:
        for m in stats["missing_images"]:
            print(f"  - Missing: {m}")
    print("Polygons by class:")
    for cls_name, count in stats["labels_by_class"].items():
        print(f"  - {cls_name} (class {CLASS_MAPPING[cls_name]}): {count}")
    print(f"Generated label files:  {stats['generated_files']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
