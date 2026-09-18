"""Compare embedding models on identification accuracy and unknown-object rejection.

Every candidate is measured the same way, on the same crops, so the only
difference is the model:

  1. an index is built from the TRAIN split's labelled crops, and the VAL
     split's crops are identified against it -> top-1 accuracy;
  2. a second index is built with one object left out, and every val crop is
     scored against it -> how cleanly an unknown object can be rejected.

Usage:
    python scripts/compare_embedders.py
    python scripts/compare_embedders.py --held-out phone
    python scripts/compare_embedders.py --model dinov2:facebook/dinov2-base:518

Each --model is "kind:name" with an optional ":image size" (dinov2 only).
Given none, the list in DEFAULT_CANDIDATES is used. Models are downloaded on
first use, and the indexes are temporary: the project's own index is never
touched.

The result is evidence for a choice, not the choice itself. On a small
dataset the differences are a couple of crops wide, i.e. within noise.
"""

import argparse
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from ultralytics.data.utils import check_det_dataset

from object_detection.config.loader import IdentificationConfig, load_config
from object_detection.identification.cropping import iter_labelled_crops
from object_detection.identification.embedders import Embedder, create_embedder
from object_detection.identification.evaluation import (
    best_scores,
    fill_index,
    separation,
    separation_pairs,
    top1_accuracy,
)
from object_detection.identification.index import LocalEmbeddingIndex

# (embedder kind, model name, image size; 0 = the config's value)
DEFAULT_CANDIDATES = [
    ("fastembed", "Qdrant/Unicom-ViT-B-16", 0),
    ("fastembed", "Qdrant/clip-ViT-B-32-vision", 0),
    ("fastembed", "Qdrant/resnet50-onnx", 0),
    ("dinov2", "facebook/dinov2-small", 224),
    ("dinov2", "facebook/dinov2-base", 224),
]


def parse_candidate(value: str) -> tuple[str, str, int]:
    """Turn "kind:name[:image_size]" into a candidate tuple."""
    parts = value.split(":")
    if len(parts) == 2:
        return parts[0], parts[1], 0
    if len(parts) == 3 and parts[2].isdigit():
        return parts[0], parts[1], int(parts[2])
    raise argparse.ArgumentTypeError(f"expected kind:name or kind:name:image_size, got {value!r}")


def temporary_index(embedder: Embedder, prefix: str) -> LocalEmbeddingIndex:
    """An index of its own, in a temporary folder, so the project's index is untouched.

    LocalEmbeddingIndex directly rather than create_index, because every
    candidate here computes its vectors locally, and the same embedder object
    is reused for both of a candidate's indexes instead of loading the model
    twice.
    """
    return LocalEmbeddingIndex(embedder, tempfile.mkdtemp(prefix=prefix), "comparison")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare embedding models for identification.")
    parser.add_argument("--held-out", default=None,
                        help="object to leave out of the second index, or 'all' to repeat for every class "
                             "and add up the results (default: the first class)")
    parser.add_argument("--model", action="append", type=parse_candidate, dest="models",
                        help="candidate as kind:name[:image_size]; repeatable")
    args = parser.parse_args()
    candidates = args.models or DEFAULT_CANDIDATES

    cfg = load_config()
    identification = cfg.identification
    dataset = check_det_dataset(str(cfg.data.source_yaml))
    class_names = dataset["names"]
    # Every class in turn, or just one. Repeating over all of them stops the
    # result depending on which object happened to be chosen: with `watch`
    # held out ResNet50 looked perfect, with `phone` it did not.
    if args.held_out == "all":
        held_out_classes = list(class_names.values())
    else:
        held_out_classes = [args.held_out or next(iter(class_names.values()))]
        if held_out_classes[0] not in class_names.values():
            sys.exit(
                f"error: {held_out_classes[0]!r} is not a class in {cfg.data.source_yaml}; "
                f"classes: {list(class_names.values())}"
            )

    splits = {}
    for split in ("train", "val"):
        splits[split] = sorted(p for p in Path(dataset[split]).glob("*") if p.is_file())

    def crops(split):
        # A fresh generator each time: crops are made again per candidate
        # rather than kept in memory, which matters on large datasets.
        return iter_labelled_crops(splits[split], class_names, identification.crop_padding, identification.min_crop_size)

    print(f"Dataset:  {cfg.data.source_yaml}")
    print(f"Splits:   {len(splits['train'])} train images, {len(splits['val'])} val images")
    print(f"Held out: {', '.join(held_out_classes)}")
    print(f"Crops:    padding {identification.crop_padding}, minimum size {identification.min_crop_size}px\n")

    header = (
        f"{'embedder':<32}{'dim':>5}{'top-1':>9}{'AUC':>7}"
        f"{'unknown rej.':>14}{'known kept':>12}{'sec/crop':>10}"
    )
    print(header)
    print("-" * len(header))

    for kind, model_name, image_size in candidates:
        config = replace(
            identification,
            embedder=kind,
            model=model_name,
            image_size=image_size or identification.image_size,
        )

        embedder = create_embedder(config)

        full = temporary_index(embedder, "qdrant_cmp_")
        started = time.perf_counter()
        stored = fill_index(full, crops("train"))
        seconds_per_crop = (time.perf_counter() - started) / stored
        correct, total = top1_accuracy(full, crops("val"))
        full.close()

        # Each held-out object gets its own index, and the results are added
        # up, so no single object decides the answer.
        rejected = unknown_seen = kept = known_seen = 0
        wins = pairs = 0.0
        for held_out in held_out_classes:
            without = temporary_index(embedder, "qdrant_cmp_holdout_")
            fill_index(without, crops("train"), skip=held_out)
            unknown_scores, known_scores = best_scores(without, crops("val"), held_out)
            without.close()
            if not unknown_scores or not known_scores:
                continue      # that object has no crops in this split

            counts = separation(unknown_scores, known_scores)
            rejected += counts.unknown_rejected
            unknown_seen += counts.unknown_total
            kept += counts.known_kept
            known_seen += counts.known_total
            class_wins, class_pairs = separation_pairs(unknown_scores, known_scores)
            wins += class_wins
            pairs += class_pairs

        label = f"{model_name}@{config.image_size}" if kind == "dinov2" else model_name
        print(
            f"{label:<32}{embedder.dimension:>5}{correct / total:>9.2f}{wins / pairs:>7.3f}"
            f"{f'{rejected}/{unknown_seen}':>14}{f'{kept}/{known_seen}':>12}{seconds_per_crop:>10.3f}"
        )

    print(
        "\nAUC          = how often a known crop outscores an unknown one, over every pair."
        "\n               1.0 is perfect separation, 0.5 is a coin flip. Robust to odd crops."
        "\nunknown rej. = held-out crops rejected by a threshold that keeps every known crop."
        "\nknown kept   = known crops still identified by a threshold that rejects every held-out crop."
        "\n               Both counts hinge on the single most extreme crop, so read them next to the AUC."
        "\nTimings are rough: they include whatever else the machine was doing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
