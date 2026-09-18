"""Tests for create_embedder / create_index: the config decides which implementation is used."""

from dataclasses import replace

import pytest

import object_detection.identification.embedders as embedders_module
from object_detection.config.loader import load_config
from object_detection.identification.embedders import create_embedder
from object_detection.identification.index import LocalEmbeddingIndex, create_index


class FakeFastEmbedEmbedder:
    """Stands in for FastEmbedEmbedder so no model is downloaded in tests."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    name = "fake/model"
    dimension = 4


def test_fastembed_is_built_with_the_configured_model(monkeypatch):
    monkeypatch.setattr(embedders_module, "FastEmbedEmbedder", FakeFastEmbedEmbedder)
    config = load_config().identification

    embedder = create_embedder(config)

    assert isinstance(embedder, FakeFastEmbedEmbedder)
    assert embedder.model_name == config.model


class FakeDinoV2Embedder:
    """Stands in for DinoV2Embedder so no model is downloaded in tests."""

    def __init__(self, model_name: str, image_size: int):
        self.model_name = model_name
        self.image_size = image_size

    name = "fake/dinov2"
    dimension = 4


def test_dinov2_is_built_with_the_configured_model_and_image_size(monkeypatch):
    monkeypatch.setattr(embedders_module, "DinoV2Embedder", FakeDinoV2Embedder)
    config = replace(load_config().identification, embedder="dinov2", model="facebook/dinov2-base")

    embedder = create_embedder(config)

    assert isinstance(embedder, FakeDinoV2Embedder)
    assert embedder.model_name == "facebook/dinov2-base"
    assert embedder.image_size == config.image_size


def test_an_unknown_embedder_name_is_rejected():
    config = replace(load_config().identification, embedder="something-else")
    with pytest.raises(ValueError, match="Unknown embedder"):
        create_embedder(config)


def test_create_index_builds_a_local_index_at_the_configured_location(monkeypatch, tmp_path):
    monkeypatch.setattr(embedders_module, "FastEmbedEmbedder", FakeFastEmbedEmbedder)
    config = replace(
        load_config().identification,
        qdrant_location=str(tmp_path / "qdrant"),
        collection="factory_test",
    )

    index = create_index(config)
    try:
        assert isinstance(index, LocalEmbeddingIndex)
        # The collection was created with the fake embedder's vector size.
        assert (tmp_path / "qdrant").exists()
    finally:
        index.close()
