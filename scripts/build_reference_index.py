"""Build the Stage 2 reference index from the original, per-class dataset export.

The raw Roboflow export keeps its object classes (bag, phone, watch, ...),
which is exactly what the index needs: every labelled box is a reference crop
with its name already attached. This is why the export is kept untouched
next to the single-class copy the detector trains on.

Usage:
    python scripts/build_reference_index.py                 # add the train split
    python scripts/build_reference_index.py --reset         # replace what is stored
    python scripts/build_reference_index.py --split val     # e.g. to add more references

By default only the TRAIN split is indexed, so the validation split's crops
stay unseen and can measure identification accuracy honestly.

Crops are cut with the identification settings from the config, the same ones
the API uses, so reference and query crops are prepared identically.

Assumption: the class name is used as both object_id and object_name, which
is only right if each class is ONE physical item. If a class ever covers
several distinct items (two different phones), the labels need per-item names.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from ultralytics.data.utils import check_det_dataset

from object_detection.config.loader import load_config
from object_detection.identification.cropping import iter_labelled_crops
from object_detection.identification.index import create_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Stage 2 reference index.")
    parser.add_argument(
        "--split",
        default="train",
        help="which split of the raw export to index (default: train)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete each class's existing references first, instead of adding to them",
    )
    args = parser.parse_args()

    cfg = load_config()
    identification = cfg.identification

    # Ultralytics reads the export's own data.yaml: split paths and class
    # names come from the dataset, never from this script, and its Roboflow
    # "../" path quirk is handled for us.
    dataset = check_det_dataset(str(cfg.data.source_yaml))
    if args.split not in dataset or not dataset[args.split]:
        available = [split for split in ("train", "val", "test") if dataset.get(split)]
        sys.exit(f"error: split {args.split!r} is not in {cfg.data.source_yaml}; available: {available}")

    class_names = dataset["names"]  # {0: "bag", 1: "glasses", ...}
    images_dir = Path(dataset[args.split])
    img_paths = sorted(p for p in images_dir.glob("*") if p.is_file())

    print(f"Export:     {cfg.data.source_yaml}")
    print(f"Split:      {args.split} ({len(img_paths)} images from {images_dir})")
    print(f"Embedder:   {identification.embedder} / {identification.model}")
    print(f"Index:      {identification.collection} at {identification.qdrant_location}")
    print(f"Crops:      padding {identification.crop_padding}, minimum size {identification.min_crop_size}px")

    # Loading the model happens here, and downloads it on first use.
    index = create_index(identification)
    try:
        if args.reset:
            for name in class_names.values():
                index.remove_object(name)
            print(f"Reset:      removed existing references for {len(class_names)} classes")

        stored = Counter()
        # Crops are embedded and stored one at a time, as they are produced,
        # so memory stays flat however large the dataset is.
        for name, crop in iter_labelled_crops(
            img_paths, class_names, identification.crop_padding, identification.min_crop_size
        ):
            index.add_references([crop], object_id=name, object_name=name)
            stored[name] += 1
    finally:
        # Local mode locks its folder; without this, nothing else can open
        # the index until this process exits.
        index.close()

    print("\nReferences stored:")
    for class_id, name in class_names.items():
        print(f"  {class_id:3d}  {name:<20} {stored[name]:5d}")
    print(f"  total{'':<22} {sum(stored.values()):5d}")

    missing = [name for name in class_names.values() if not stored[name]]
    if missing:
        print(f"\nWarning: no references stored for {len(missing)} class(es): {', '.join(missing)}")
        print("Those objects can never be identified. Check the split and the labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
