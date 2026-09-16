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

Status: LocalEmbeddingIndex is implemented; CloudInferenceIndex is still a stub.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image
from qdrant_client import QdrantClient, models

from object_detection.identification.embedders import Embedder

# Payload keys. EMBEDDER_KEY records which model produced a vector: vectors
# from different models are not comparable, and a mismatch would otherwise
# return confident nonsense instead of an error.
OBJECT_ID_KEY = "object_id"
OBJECT_NAME_KEY = "object_name"
EMBEDDER_KEY = "embedder"


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

    def close(self) -> None:
        """Release the connection, if the implementation holds one.

        Matters for Qdrant's local mode, which locks its folder: nothing else
        can open the index until it is closed.
        """


class LocalEmbeddingIndex(ReferenceIndex):
    """An index whose vectors are computed by our own code, then stored in Qdrant.

    Used for FastEmbed and DINOv2: `embedder` turns crops into vectors, and
    Qdrant stores and searches them.
    """

    def __init__(self, embedder: Embedder, location: str, collection_name: str):
        """Connect to Qdrant and create the collection if it doesn't exist.

        Args:
            embedder: turns crops into vectors. Its name is stored with every
                point, and opening a collection built with a different model
                fails loudly, since mixed vectors can't be compared.
            location: a folder path for Qdrant's local mode (inside this
                process, one process at a time), or a URL such as
                "http://localhost:6333" for a Qdrant server, e.g. in Docker.
                Only this value changes when moving between the two.
            collection_name: which collection to use in that Qdrant instance.
        """
        self._embedder = embedder
        self._collection = collection_name
        if location.startswith(("http://", "https://")):
            self._client = QdrantClient(url=location)
        else:
            self._client = QdrantClient(path=location)

        if self._client.collection_exists(collection_name):
            self._check_same_embedder()
        else:
            self._client.create_collection(
                collection_name,
                # Cosine distance compares the *direction* of two vectors and
                # ignores their length, which is the usual choice for
                # similarity between embeddings.
                vectors_config=models.VectorParams(
                    size=embedder.dimension, distance=models.Distance.COSINE
                ),
            )

    def _check_same_embedder(self) -> None:
        """Fail if the existing collection was built with a different embedding model."""
        points, _ = self._client.scroll(self._collection, limit=1, with_payload=True)
        if not points:
            return  # empty collection: nothing to conflict with
        stored = points[0].payload.get(EMBEDDER_KEY)
        if stored != self._embedder.name:
            raise ValueError(
                f"Collection {self._collection!r} was built with embedder {stored!r}, "
                f"but {self._embedder.name!r} is configured. Vectors from different "
                f"models are not comparable: rebuild the index, or use a different collection."
            )

    def add_references(self, crops: list[Image.Image], object_id: str, object_name: str) -> None:
        if not crops:
            return
        vectors = self._embedder.embed(crops)
        points = [
            models.PointStruct(
                # Each reference crop is its own point with its own random id;
                # object_id lives in the payload, because one object has many
                # reference points.
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    OBJECT_ID_KEY: object_id,
                    OBJECT_NAME_KEY: object_name,
                    EMBEDDER_KEY: self._embedder.name,
                },
            )
            for vector in vectors
        ]
        self._client.upsert(self._collection, points=points)

    def search(self, crops: list[Image.Image], limit: int) -> list[list[Match]]:
        if not crops:
            return []
        vectors = self._embedder.embed(crops)
        # One request per crop, sent together: a tray of pastries is searched
        # in a single round trip instead of dozens.
        responses = self._client.query_batch_points(
            self._collection,
            requests=[
                models.QueryRequest(query=vector, limit=limit, with_payload=True) for vector in vectors
            ],
        )
        return [
            [
                Match(
                    object_id=point.payload[OBJECT_ID_KEY],
                    object_name=point.payload[OBJECT_NAME_KEY],
                    score=point.score,
                )
                for point in response.points
            ]
            for response in responses
        ]

    def remove_object(self, object_id: str) -> None:
        # Deletes by payload rather than by point id: one object has many
        # points, and the caller only knows the object's identity.
        self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=OBJECT_ID_KEY, match=models.MatchValue(value=object_id)
                        )
                    ]
                )
            ),
        )

    def close(self) -> None:
        self._client.close()


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
