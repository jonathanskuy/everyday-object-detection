"""Collapse a multi-class YOLO export into the single-class detector dataset.

The detector is class-agnostic: it learns WHERE objects are, not which object
each one is. So it trains on a copy of the export whose labels all say class
0. The original export is never modified — its per-class labels are what
seeds the Stage 2 reference index.

Which splits exist, and how many classes there are, are read from the
export's own data.yaml. Nothing here assumes "train and valid": an export
with a test split gets one too.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

# The single class the detector trains on. "object" is a YOLO class name in
# data.yaml only; never use it as a Python identifier.
CLASS_NAME = "object"

# Splits Ultralytics understands, in the order they are reported.
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitStats:
    """What was copied for one split."""

    images: int
    labels: int
    boxes: int
    backgrounds: int


def collapse_labels(text: str) -> str:
    """Rewrite one label file's contents with every class id set to 0.

    Empty text stays empty: an empty label file marks a background image,
    which is a deliberate negative example and must be kept.
    """
    lines = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        parts[0] = "0"
        lines.append(" ".join(parts))
    return "\n".join(lines)


def collapse_split(images_dir: Path, output_dir: Path) -> SplitStats:
    """Copy one split's images and rewrite its labels into `output_dir`.

    `images_dir` is the split's images folder; its labels are found by YOLO's
    convention (../labels). Images with no label file are copied as they are:
    they are background images, and the detector needs them.
    """
    labels_dir = images_dir.parent / "labels"
    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    images = labels = boxes = backgrounds = 0
    for img_path in sorted(p for p in images_dir.glob("*") if p.is_file()):
        shutil.copy2(img_path, out_images / img_path.name)
        images += 1

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            backgrounds += 1
            continue

        collapsed = collapse_labels(label_path.read_text())
        (out_labels / label_path.name).write_text(collapsed)
        labels += 1
        if collapsed:
            boxes += len(collapsed.splitlines())
        else:
            backgrounds += 1

    return SplitStats(images, labels, boxes, backgrounds)


def collapse_dataset(source_yaml: Path, output_dir: Path) -> dict[str, SplitStats]:
    """Collapse every split listed in `source_yaml` into `output_dir`.

    Also writes output_dir/data.yaml describing the copy. That file lists its
    splits as paths relative to itself and deliberately has NO `path` key:
    Ultralytics then resolves them against the file's own folder, so the
    dataset works on any machine. An absolute path would break as soon as the
    project moved, e.g. to Colab.

    Returns the per-split statistics, so the caller can report what happened
    instead of this deciding how to print it.
    """
    source = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    base = Path(source["path"]) if source.get("path") else source_yaml.parent

    stats: dict[str, SplitStats] = {}
    folders: dict[str, str] = {}
    for split in SPLITS:
        if split not in source or not source[split]:
            continue
        # Roboflow exports write "../train/images", relative to the data.yaml.
        # Ultralytics strips the "../" when that path does not exist, so try
        # the same two candidates here rather than assuming either layout.
        listed = Path(source[split])
        candidates = [base / listed]
        if not listed.is_absolute() and listed.parts and listed.parts[0] == "..":
            candidates.append(base / Path(*listed.parts[1:]))
        images_dir = next((c for c in candidates if c.is_dir()), None)
        if images_dir is None:
            continue    # listed but not present, e.g. "test" in an export without one

        # The copy keeps the export's own folder names: Roboflow calls the
        # validation split "valid", while the data.yaml key is "val". Renaming
        # it would leave the copy laid out differently from the original.
        folders[split] = images_dir.parent.name
        stats[split] = collapse_split(images_dir, output_dir / folders[split])

    data_yaml = {split: f"{folders[split]}/images" for split in stats}
    data_yaml["nc"] = 1
    data_yaml["names"] = [CLASS_NAME]
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return stats
