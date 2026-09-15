"""Build the Stage 2 reference index from the original, per-class dataset export.

The raw Roboflow export keeps its object classes (bag, phone, watch, ...),
which is exactly what the index needs: every labelled box is a reference crop
with its name already attached. This is why the export is kept untouched
next to the single-class copy the detector trains on.

Planned usage:
    python scripts/build_reference_index.py

Planned steps:
    1. Load the config: raw export location, index settings, crop padding
       and minimum size.
    2. Read class names from the export's data.yaml (never hardcoded), and
       labelled boxes from its TRAIN split. The validation split is kept out
       of the index so its crops can test identification accuracy.
    3. Crop every labelled box with identification.cropping.crop_detection,
       using the same padding and minimum size the API will use.
    4. Add the crops to the index, grouped by class, using the class name as
       both object_id and object_name.

Assumption to confirm: using the class name as the identity is only right if
each class is ONE physical item. If a class ever covers several distinct
items (two different phones), the labels need per-item names instead.

Status: scaffold only. Nothing is implemented yet.
"""


def main() -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
