"""Tests for loading the YAML config."""

from pathlib import Path

import pytest
import yaml

from object_detection.config.loader import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.train.epochs > 0
    assert 0 < cfg.inference.conf < 1


def test_relative_paths_are_resolved_against_the_repo_root():
    cfg = load_config()
    for path in (cfg.model.weights, cfg.data.source_yaml, cfg.data.detector_yaml, cfg.inference.weights):
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert path.is_relative_to(PROJECT_ROOT)


def write_modified_config(tmp_path, change):
    """Copy default.yaml into tmp_path with `change` applied to its contents."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    change(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_missing_key_fails_at_load_time(tmp_path):
    path = write_modified_config(tmp_path, lambda raw: raw["train"].pop("epochs"))
    with pytest.raises(TypeError, match="epochs"):
        load_config(path)


def test_misspelled_key_fails_at_load_time(tmp_path):
    path = write_modified_config(tmp_path, lambda raw: raw["train"].update(epoch=raw["train"].pop("epochs")))
    with pytest.raises(TypeError):
        load_config(path)


def test_config_values_cannot_be_changed_after_loading():
    cfg = load_config()
    with pytest.raises(AttributeError):
        cfg.inference.conf = 0.9
