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

Status: FastEmbedEmbedder is implemented; DinoV2Embedder is still a stub.
"""

from abc import ABC, abstractmethod

from fastembed import ImageEmbedding
from PIL import Image

from object_detection.config.loader import IdentificationConfig


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
        """Load the FastEmbed model `model_name` (downloaded and cached on first use)."""
        self._model_name = model_name
        self._model = ImageEmbedding(model_name=model_name)
        # FastEmbed publishes each model's vector length; reading it here
        # means the collection's vector size always matches the model, with
        # no number to keep in sync by hand.
        self._dimension = next(
            entry["dim"] for entry in ImageEmbedding.list_supported_models() if entry["model"] == model_name
        )

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, crops: list[Image.Image]) -> list[list[float]]:
        # FastEmbed accepts PIL images directly and does the model's own
        # resizing and normalisation internally, which is why cropping.py
        # leaves the crops at their natural size.
        # It returns a generator of NumPy arrays; .tolist() turns each into
        # the plain list of floats that Qdrant expects.
        return [vector.tolist() for vector in self._model.embed(crops)]


class DinoV2Embedder(Embedder):
    """Embeds crops with Meta's DINOv2, a self-supervised vision model (Apache-2.0).

    Trained on images alone, so its vectors capture fine visual detail
    rather than text-described categories, which suits telling look-alike
    items apart. Used frozen (no training).

    Decisions made here, each worth revisiting with a measurement:
      - the CLS token is used as the image's vector. A vision transformer
        splits an image into patches and adds one extra "class" token that
        collects information from all of them, which is the usual choice for
        a whole-image descriptor. Averaging the patch tokens is the common
        alternative.
      - the processor's own resizing is set to image_size and its
        CENTRE-CROP IS TURNED OFF. By default it would resize to 256 and cut
        the middle 224, which would trim the edges of crops that arrive
        already square from cropping.py.
    """

    def __init__(self, model_name: str, image_size: int):
        """Load `model_name` (e.g. "facebook/dinov2-base") for inputs of `image_size` pixels."""
        # Imported here, not at the top of the file: transformers takes
        # several seconds to import, and everything that touches the index
        # (the API, notebooks, the drawing helpers) would pay that cost even
        # when another embedder is configured.
        from transformers import AutoImageProcessor, AutoModel

        self._model_name = model_name
        self._image_size = image_size
        self._processor = AutoImageProcessor.from_pretrained(
            model_name,
            size={"height": image_size, "width": image_size},
            do_center_crop=False,
        )
        import torch

        self._model = AutoModel.from_pretrained(model_name)
        # eval() turns off training-only behaviour (dropout and such); the
        # model is used frozen, never trained.
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        # hidden_size is the model's vector length: 384 for small, 768 for
        # base, 1024 for large. Read from the model so it can't drift.
        self._dimension = self._model.config.hidden_size

    @property
    def name(self) -> str:
        # The input size changes the vectors, so it is part of the identity
        # recorded with the index: the same model at 224 and at 518 px does
        # not produce comparable vectors.
        return f"{self._model_name}@{self._image_size}px"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, crops: list[Image.Image]) -> list[list[float]]:
        import torch

        # inference_mode() skips the bookkeeping PyTorch does for training,
        # which is faster and uses less memory.
        with torch.inference_mode():
            inputs = self._processor(images=crops, return_tensors="pt").to(self._device)
            outputs = self._model(**inputs)
            # last_hidden_state is (images, tokens, dimension); token 0 is the
            # CLS token, so [:, 0] takes one vector per image.
            vectors = outputs.last_hidden_state[:, 0]
        return vectors.cpu().tolist()


def create_embedder(config: IdentificationConfig) -> Embedder:
    """Build the embedder named in the config.

    Keeping this in one place means every caller (the index-building script,
    the API, notebooks) picks its model the same way: by changing the config,
    never by importing a specific class.
    """
    if config.embedder == "fastembed":
        return FastEmbedEmbedder(config.model)
    if config.embedder == "dinov2":
        return DinoV2Embedder(config.model, config.image_size)
    raise ValueError(f"Unknown embedder {config.embedder!r}: expected 'fastembed' or 'dinov2'.")
