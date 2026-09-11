"""Load the project YAML config into typed, read-only objects.

Usage:
    from object_detection.config.loader import load_config

    cfg = load_config()                  # default.yaml
    cfg = load_config("my_run.yaml")     # any other config file
    cfg.train.epochs, cfg.data.detector_yaml, cfg.inference.conf
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
    weights: str


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch: int
    imgsz: int
    seed: int


@dataclass(frozen=True)
class DataConfig:
    source_yaml: Path
    detector_yaml: Path


@dataclass(frozen=True)
class InferenceConfig:
    conf: float


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    train: TrainConfig
    data: DataConfig
    inference: InferenceConfig


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read a YAML config file and return it as a Config.

    A missing or misspelled key raises an error here, at load time, rather
    than surfacing later as a confusing failure halfway through training.
    Paths are not checked for existence: the single-class dataset does not
    exist until the collapse step has run, and loading the config should not
    depend on that.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        # `**section` passes each YAML key as a keyword argument, which is
        # what makes an unknown or missing key an immediate TypeError.
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        # Every key in `data` is a path, so resolve them all the same way.
        data=DataConfig(**{key: _resolve_path(value) for key, value in raw["data"].items()}),
        inference=InferenceConfig(**raw["inference"]),
    )
