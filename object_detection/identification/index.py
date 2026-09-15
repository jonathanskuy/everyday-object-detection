"""Store reference objects in Qdrant and search them by crop.

The index takes CROPS, not vectors. With FastEmbed and DINOv2 our own code
computes the vector, but with Qdrant Cloud Inference the image is sent to
Qdrant, which computes the vector on its servers during the request, so our
code never has a vector to pass. Taking crops lets all three options share
one interface: code that uses an index (identify.py, the index-building
script, the comparison notebook) is identical whichever option is behind it.

Each stored vector carries a payload with its object's identity, so a search
returns object_id and object_name directly. One object usually has several
reference vectors (different angles, backgrounds), all with the same
object_id; each vector also has its own internal Qdrant point ID, which is
not the object_id.

Status: scaffold only. Nothing is implemented yet, and qdrant-client is not a
dependency yet.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from object_detection.identification.embedders import Embedder


@dataclass(frozen=True)
class Match:
    """One search result: a stored reference object and how similar it is to the query crop."""

    object_id: str
    object_name: str
    # Cosine similarity between the query crop's vector and the reference's
    # vector: higher means more similar. Becomes `match_score` in the API.
    score: float


class ReferenceIndex(ABC):
    """A searchable collection of reference objects."""

    @abstractmethod
    def add_references(self, crops: list[Image.Image], object_id: str, object_name: str) -> None:
        """Store reference crops for one object, all under the same identity.

        Calling it again for an existing object_id adds more references to
        that object; it does not replace the existing ones.
        """

    @abstractmethod
    def search(self, crops: list[Image.Image], limit: int) -> list[list[Match]]:
        """Return the `limit` most similar stored references for each crop, best first.

        One list of matches per crop, in the same order as `crops`. Batched
        for the same reason as Embedder.embed.
        """

    @abstractmethod
    def remove_object(self, object_id: str) -> None:
        """Delete every reference stored for `object_id` (e.g. a retired product)."""


class LocalEmbeddingIndex(ReferenceIndex):
    """An index whose vectors are computed by our own code, then stored in Qdrant.

    Used for FastEmbed and DINOv2: `embedder` turns crops into vectors, and
    Qdrant stores and searches them.
    """

    def __init__(self, embedder: Embedder, location: str, collection_name: str):
        """Connect to Qdrant and create the collection if it doesn't exist.

        Args:
            embedder: turns crops into vectors. Its name is recorded with the
                collection, and opening a collection built with a different
                model must fail loudly, since mixed vectors can't be compared.
            location: a folder path for Qdrant's local mode (inside this
                process, one process at a time), or a URL such as
                "http://localhost:6333" for a Qdrant server, e.g. in Docker.
                Only this value changes when moving between the two.
            collection_name: which collection to use in that Qdrant instance.
        """
        raise NotImplementedError

    def add_references(self, crops: list[Image.Image], object_id: str, object_name: str) -> None:
        raise NotImplementedError

    def search(self, crops: list[Image.Image], limit: int) -> list[list[Match]]:
        raise NotImplementedError

    def remove_object(self, object_id: str) -> None:
        raise NotImplementedError


class CloudInferenceIndex(ReferenceIndex):
    """An index on Qdrant Cloud, which computes the vectors on its own servers.

    Crops are sent to Qdrant Cloud as images; Qdrant embeds them with
    `model_name` while inserting and searching.

    Before using it with real data: images leave company infrastructure, so
    this needs approval under the company's data policy.
    """

    def __init__(self, url: str, collection_name: str, model_name: str):
        """Connect to the Qdrant Cloud cluster at `url`.

        The API key is read from the QDRANT_API_KEY environment variable,
        never from the config file: the config is committed to git, and a
        key in git history counts as leaked.
        """
        raise NotImplementedError

    def add_references(self, crops: list[Image.Image], object_id: str, object_name: str) -> None:
        raise NotImplementedError

    def search(self, crops: list[Image.Image], limit: int) -> list[list[Match]]:
        raise NotImplementedError

    def remove_object(self, object_id: str) -> None:
        raise NotImplementedError
