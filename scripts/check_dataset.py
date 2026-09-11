"""Sanity-check a YOLO-format dataset export before training on it.

For each split listed in data.yaml (train, val, test) it reports:
  - number of images
  - images with zero annotations (background negatives: expected, not errors)
  - total boxes, boxes-per-image distribution, boxes per class

and flags label lines Ultralytics would reject. Ultralytics does not skip just
the bad line: it drops the WHOLE image from training with a one-line
"ignoring corrupt image/label" warning, which is easy to miss. Flagged:
  - lines that don't have exactly 5 values (class x_center y_center w h)
  - class ids that aren't integers in [0, number of classes)
  - coordinates outside [0, 1]

The number of classes is read from data.yaml, so the same script works on the
original 8-class export and on the collapsed single-class copy.

Usage:
    python scripts/check_dataset.py datasets/raw_export/data.yaml
    python scripts/check_dataset.py datasets/collapsed/data.yaml

Exits with code 1 if any problems were found, 0 otherwise.
"""

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# The image extensions Ultralytics loads (ultralytics.data.utils.IMG_FORMATS).
IMAGE_EXTENSIONS = {
    "avif", "bmp", "dng", "heic", "heif", "jp2", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp",
}

SPLITS = ("train", "val", "test")

COORDINATE_NAMES = ("x_center", "y_center", "width", "height")


@dataclass
class SplitReport:
    name: str
    images_dir: Path | None
    num_images: int = 0
    num_boxes: int = 0
    empty_label_files: int = 0    # background image whose .txt exists but is empty
    missing_label_files: int = 0  # background image with no .txt at all
    boxes_per_image: Counter = field(default_factory=Counter)  # {boxes in image: number of images}
    boxes_per_class: Counter = field(default_factory=Counter)  # {class id: number of boxes}
    problems: list[str] = field(default_factory=list)

    @property
    def zero_annotation_images(self) -> int:
        # Ultralytics treats an empty .txt and a missing .txt the same way:
        # the image is a background negative with no objects in it.
        return self.empty_label_files + self.missing_label_files


def read_class_names(data: dict) -> list[str]:
    """Return class names by index, following the same rules as Ultralytics."""
    names = data.get("names")
    if names is None:
        if "nc" not in data:
            sys.exit("error: data.yaml has neither 'names' nor 'nc'")
        return [f"class_{i}" for i in range(data["nc"])]
    # Ultralytics accepts names as a list or as a {index: name} mapping.
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def dataset_root(yaml_path: Path, data: dict) -> Path:
    """The directory that split paths in data.yaml are relative to.

    Same rule as Ultralytics: the `path:` key if set, otherwise the folder
    containing data.yaml. Roboflow exports don't set `path:`. Note that
    Ultralytics resolves a relative `path:` against the current working
    directory, so a relative one behaves differently depending on where you
    run from.
    """
    if data.get("path"):
        return Path(data["path"]).resolve()
    return yaml_path.parent


def resolve_split_dir(root: Path, entry: str) -> Path:
    """Turn a split entry such as '../train/images' into a directory path.

    Roboflow writes split paths as '../train/images' even though train/ sits
    next to data.yaml, so taken literally they point one level too high.
    Ultralytics works around this: if the literal path doesn't exist and
    starts with '../', it retries with the '../' removed. We do the same, so
    this script checks the folder that training will actually read.
    """
    candidate = (root / entry).resolve()
    if not candidate.exists() and entry.startswith("../"):
        candidate = (root / entry[3:]).resolve()
    return candidate


def label_path_for(image_path: Path) -> Path:
    """Find the label file for an image, using the YOLO folder convention.

    Replace the last 'images' folder in the path with 'labels', and the file
    extension with '.txt':
        train/images/photo.rf.abc.jpeg  ->  train/labels/photo.rf.abc.txt
    """
    parts = list(image_path.parts)
    last_images_index = len(parts) - 1 - parts[::-1].index("images")
    parts[last_images_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def check_label_line(line: str, num_classes: int) -> tuple[int | None, list[str]]:
    """Check one 'class x_center y_center width height' line.

    Returns the class id (None if it couldn't be read) and a list of
    problems (empty if the line is fine).
    """
    values = line.split()
    if len(values) != 5:
        # A polygon (segmentation) label, for example, has 1 + 2N values.
        return None, [f"expected 5 values (class x_center y_center width height), found {len(values)}"]

    problems = []

    class_id = None
    try:
        # Strict int(): a class written as "0.0" is flagged rather than
        # accepted, because it means whatever wrote the file wrote floats.
        class_id = int(values[0])
    except ValueError:
        problems.append(f"class id {values[0]!r} is not an integer")
    else:
        if not 0 <= class_id < num_classes:
            problems.append(f"class id {class_id} is outside 0..{num_classes - 1}")

    for coordinate_name, text in zip(COORDINATE_NAMES, values[1:]):
        try:
            value = float(text)
        except ValueError:
            problems.append(f"{coordinate_name} {text!r} is not a number")
            continue
        if not 0.0 <= value <= 1.0:
            problems.append(f"{coordinate_name} {value} is outside [0, 1]")

    return class_id, problems


def check_split(name: str, images_dir: Path, num_classes: int) -> SplitReport:
    report = SplitReport(name=name, images_dir=images_dir)

    if "images" not in images_dir.parts:
        report.problems.append(
            f"{images_dir} has no 'images' folder in its path, so label files can't be located"
        )
        return report

    # rglob, not iterdir: Ultralytics also picks up images in subfolders.
    image_paths = sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower().lstrip(".") in IMAGE_EXTENSIONS
    )

    for image_path in image_paths:
        report.num_images += 1
        label_path = label_path_for(image_path)

        if not label_path.exists():
            report.missing_label_files += 1
            report.boxes_per_image[0] += 1
            continue

        boxes_in_image = 0
        lines = label_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            boxes_in_image += 1
            class_id, line_problems = check_label_line(line, num_classes)
            if class_id is not None:
                report.boxes_per_class[class_id] += 1
            for problem in line_problems:
                report.problems.append(f"{label_path.name}, line {line_number}: {problem}")

        if boxes_in_image == 0:
            report.empty_label_files += 1
        report.num_boxes += boxes_in_image
        report.boxes_per_image[boxes_in_image] += 1

    return report


def combine(reports: list[SplitReport]) -> SplitReport:
    """Add up per-split reports into one covering the whole dataset."""
    total = SplitReport(name="all splits", images_dir=None)
    for report in reports:
        total.num_images += report.num_images
        total.num_boxes += report.num_boxes
        total.empty_label_files += report.empty_label_files
        total.missing_label_files += report.missing_label_files
        # Counter.update() adds counts together rather than replacing them.
        total.boxes_per_image.update(report.boxes_per_image)
        total.boxes_per_class.update(report.boxes_per_class)
    return total


def print_report(report: SplitReport, class_names: list[str]) -> None:
    location = f"  {report.images_dir}" if report.images_dir else ""
    print(f"\n== {report.name} =={location}")
    print(f"  images            {report.num_images:6d}")
    print(
        f"  zero annotations  {report.zero_annotation_images:6d}"
        f"  ({report.empty_label_files} empty .txt, {report.missing_label_files} missing .txt)"
    )
    print(f"  boxes             {report.num_boxes:6d}")

    print("  boxes per image:")
    most_images = max(report.boxes_per_image.values(), default=0)
    for boxes_in_image in sorted(report.boxes_per_image):
        num_images = report.boxes_per_image[boxes_in_image]
        bar = "#" * round(40 * num_images / most_images)
        print(f"    {boxes_in_image:3d} boxes: {num_images:5d} images  {bar}")

    # Every class is listed, including ones with zero boxes: a class missing
    # from a split (e.g. absent from val) is exactly what this should reveal.
    print("  boxes per class:")
    for class_id, class_name in enumerate(class_names):
        print(f"    {class_id:3d}  {class_name:<20} {report.boxes_per_class[class_id]:6d}")
    unknown_ids = sorted(set(report.boxes_per_class) - set(range(len(class_names))))
    for class_id in unknown_ids:
        print(f"    {class_id:3d}  {'(not in names)':<20} {report.boxes_per_class[class_id]:6d}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity-check a YOLO-format dataset export.")
    parser.add_argument("data_yaml", type=Path, help="path to the export's data.yaml")
    args = parser.parse_args()

    yaml_path = args.data_yaml.resolve()
    if not yaml_path.is_file():
        parser.error(f"{yaml_path} not found")

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    problems = []  # problems with data.yaml itself; per-split ones live on each report
    notes = []     # worth knowing, but not errors

    class_names = read_class_names(data)
    if "nc" in data and data["nc"] != len(class_names):
        problems.append(
            f"data.yaml: nc is {data['nc']} but names lists {len(class_names)} classes "
            "(Ultralytics refuses to train on this)"
        )

    root = dataset_root(yaml_path, data)
    print(f"Dataset: {yaml_path}")
    print(f"Classes: {len(class_names)}")

    reports = []
    for split in SPLITS:
        entry = data.get(split)
        # train and val are required for training; test is optional.
        missing = notes if split == "test" else problems
        if entry is None:
            missing.append(f"data.yaml has no '{split}' entry")
            continue
        if not isinstance(entry, str):
            problems.append(f"'{split}' is not a single directory path; this script only supports that form")
            continue
        images_dir = resolve_split_dir(root, entry)
        if not images_dir.is_dir():
            missing.append(f"'{split}' is listed in data.yaml but {images_dir} does not exist; skipped")
            continue
        reports.append(check_split(split, images_dir, len(class_names)))

    for report in reports:
        print_report(report, class_names)
    if len(reports) > 1:
        print_report(combine(reports), class_names)

    if notes:
        print("\nNotes:")
        for note in notes:
            print(f"  - {note}")

    all_problems = problems + [f"[{r.name}] {p}" for r in reports for p in r.problems]
    if all_problems:
        print(f"\nProblems ({len(all_problems)}):")
        for problem in all_problems:
            print(f"  - {problem}")
        return 1

    print("\nNo problems found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
