"""Load the project YAML config into typed, read-only objects.

Usage:
    from object_detection.config.loader import load_config

    cfg = load_config()                  # default.yaml
    cfg = load_config("my_run.yaml")     # any other config file
    cfg.train.epochs, cfg.data.detector_yaml, cfg.inference.weights
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")

# object_detection/config/loader.py -> parents[2] is the repository root.
# Relative paths in the config are resolved against this rather than the
# current working directory: a notebook runs from notebooks/, a script from
# the repo root, and CWD-relative paths would point somewhere different in
# each. This relies on an editable install (`pip install -e .`), which keeps
# the package inside the repo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# frozen=True makes the config read-only once loaded, so no code path can
# quietly change a value that another part of the run already used.
@dataclass(frozen=True)
class ModelConfig:
    weights: Path


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch: int
    imgsz: int
    seed: int
    name: str


@dataclass(frozen=True)
class DataConfig:
    source_yaml: Path
    detector_yaml: Path


@dataclass(frozen=True)
class InferenceConfig:
    weights: Path
    conf: float


@dataclass(frozen=True)
class IdentificationConfig:
    crop_padding: float
    min_crop_size: int
    embedder: str
    model: str
    # A folder for Qdrant's local mode, or a server URL. Kept as a string
    # because it can be either; see _resolve_location.
    qdrant_location: str
    collection: str
    unknown_threshold: float


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    train: TrainConfig
    data: DataConfig
    inference: InferenceConfig
    identification: IdentificationConfig


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_location(value: str) -> str:
    """Resolve a Qdrant location: a URL stays as it is, a folder becomes absolute.

    The same setting names both a server ("http://localhost:6333") and a
    local-mode folder. Folders are resolved against the repo root like every
    other path, so a notebook and the API use the same database.
    """
    if value.startswith(("http://", "https://")):
        return value
    return str(_resolve_path(value))


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read a YAML config file and return it as a Config.

    A missing or misspelled key raises an error here, at load time, rather
    than surfacing later as a confusing failure halfway through training.
    Paths are not checked for existence: the single-class dataset does not
    exist until the collapse step has run, the pretrained checkpoint not
    until Ultralytics first downloads it, and loading the config should not
    depend on either.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # In `model` and `inference`, `weights` is the only path, so only that key
    # is resolved; anything else (like `conf`) passes through unchanged.
    model_section = dict(raw["model"])
    model_section["weights"] = _resolve_path(model_section["weights"])

    inference_section = dict(raw["inference"])
    inference_section["weights"] = _resolve_path(inference_section["weights"])

    identification_section = dict(raw["identification"])
    identification_section["qdrant_location"] = _resolve_location(identification_section["qdrant_location"])

    return Config(
        # `**section` passes each YAML key as a keyword argument, which is
        # what makes an unknown or missing key an immediate TypeError.
        model=ModelConfig(**model_section),
        train=TrainConfig(**raw["train"]),
        # Every key in `data` is a path, so resolve them all the same way.
        data=DataConfig(**{key: _resolve_path(value) for key, value in raw["data"].items()}),
        inference=InferenceConfig(**inference_section),
        identification=IdentificationConfig(**identification_section),
    )
