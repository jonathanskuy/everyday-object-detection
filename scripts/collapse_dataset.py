"""Create the single-class dataset the detector trains on.

Copies the export named by `data.source_yaml` in the config to
`data.detector_yaml`'s folder, with every label's class set to 0. The
original export is left untouched: its per-class labels seed the Stage 2
reference index.

Usage:
    python scripts/collapse_dataset.py
    python scripts/collapse_dataset.py --overwrite     # rebuild from scratch

Splits and classes are read from the export's data.yaml, so an export with a
test split gets a collapsed test split too.

This replaces the collapse cells in notebooks/00_data_preparation.ipynb, which
handled only train and valid, and wrote an absolute path into the copy's
data.yaml.
"""

import argparse
import shutil
import sys

from object_detection.config.loader import load_config
from object_detection.utils.dataset import collapse_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Collapse the export into a single-class dataset.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete the existing collapsed dataset first, instead of refusing to touch it",
    )
    args = parser.parse_args()

    cfg = load_config()
    source_yaml = cfg.data.source_yaml
    output_dir = cfg.data.detector_yaml.parent

    if not source_yaml.exists():
        sys.exit(f"error: {source_yaml} does not exist (is the export downloaded?)")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            sys.exit(
                f"error: {output_dir} already exists. Re-run with --overwrite to rebuild it.\n"
                f"       Copying into it would mix old and new files, and a shrinking dataset "
                f"would silently keep images that are no longer in the export."
            )
        shutil.rmtree(output_dir)

    print(f"Source: {source_yaml}")
    print(f"Output: {output_dir}\n")

    stats = collapse_dataset(source_yaml, output_dir)
    if not stats:
        sys.exit(f"error: no splits from {source_yaml} were found on disk")

    print(f"{'split':<8}{'images':>8}{'labels':>8}{'boxes':>8}{'backgrounds':>13}")
    for split, split_stats in stats.items():
        print(
            f"{split:<8}{split_stats.images:>8}{split_stats.labels:>8}"
            f"{split_stats.boxes:>8}{split_stats.backgrounds:>13}"
        )

    print(f"\nWrote {output_dir / 'data.yaml'}")
    if "test" not in stats:
        print(
            "\nNote: the export has no test split. Evaluation and thresholds are then\n"
            "chosen on the same data used to pick the model, which flatters the results."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
