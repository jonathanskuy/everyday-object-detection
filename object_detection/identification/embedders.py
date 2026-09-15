"""Turn square crops into vectors, one implementation per embedding model.

Every embedder has the same interface, "crops in, vectors out", so the
local-embedding index (index.py) works with any of them and switching models
is a config change. Each implementation owns everything model-specific:
input size, resampling and colour normalisation.

Qdrant Cloud Inference has no embedder here: it computes vectors on Qdrant's
servers during insert and search, so our code never turns a crop into a
vector itself. It is handled by CloudInferenceIndex in index.py instead.

Planned comparison (FastEmbed vs Qdrant Cloud Inference vs DINOv2): these
classes, together with index.py, are what makes it a matter of configuration.

Status: scaffold only. Nothing is implemented yet, and the libraries the
implementations will need (fastembed, transformers) are not dependencies yet.
"""

from abc import ABC, abstractmethod

from PIL import Image


class Embedder(ABC):
    """Turns crops into fixed-length vectors describing what they look like.

    Similar-looking crops must get vectors that are close together under
    cosine similarity, which is the distance the Qdrant collection will use.

    ABC (abstract base class) means this class only defines the interface:
    it can't be used directly, and every subclass must implement the methods
    marked @abstractmethod.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier of the model, e.g. "Qdrant/Unicom-ViT-B-16".

        Stored alongside the index, because vectors from different models
        are not comparable: searching an index built with another model would
        return meaningless matches without any error.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of every vector this embedder produces, e.g. 768.

        Needed to create the Qdrant collection, which has a fixed vector size.
        """

    @abstractmethod
    def embed(self, crops: list[Image.Image]) -> list[list[float]]:
        """Return one vector per crop, in the same order.

        Takes a list rather than one crop because models run much faster on
        a batch than on the same crops one at a time (a bakery tray can
        produce dozens of crops from a single photo).
        """


class FastEmbedEmbedder(Embedder):
    """Embeds crops with one of FastEmbed's image models, run locally.

    FastEmbed is Qdrant's embedding library. It runs models in ONNX format
    (fast on CPU, no PyTorch needed) and does the model's resizing and
    normalisation itself. Its image models include
    "Qdrant/clip-ViT-B-32-vision", "Qdrant/resnet50-onnx",
    "Qdrant/Unicom-ViT-B-32" and "Qdrant/Unicom-ViT-B-16".
    """

    def __init__(self, model_name: str):
        """Load the FastEmbed model `model_name` (downloaded on first use)."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    def embed(self, crops: list[Image.Image]) -> list[list[float]]:
        raise NotImplementedError


class DinoV2Embedder(Embedder):
    """Embeds crops with Meta's DINOv2, a self-supervised vision model (Apache-2.0).

    Trained on images alone, so its vectors capture fine visual detail
    rather than text-described categories, which suits telling look-alike
    items apart. Used frozen (no training).

    Decisions to settle during implementation, each worth measuring:
      - which vector to use: the CLS token (one summary vector for the
        image) or the average of the patch vectors;
      - input size: any multiple of 14 pixels; larger keeps more detail but
        is slower;
      - the Hugging Face processor centre-crops by default. Crops arrive
        square from cropping.py so nothing should be cut, but its settings
        must be checked rather than assumed.
    """

    def __init__(self, model_name: str, image_size: int):
        """Load `model_name` (e.g. "facebook/dinov2-base") for inputs of `image_size` pixels."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    def embed(self, crops: list[Image.Image]) -> list[list[float]]:
        raise NotImplementedError
