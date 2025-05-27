from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self):
        logger.info(
            f"Connecting to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
        )
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        logger.info("Successfully connected to Qdrant")

        self.vector_size = settings.EMBEDDING_DIM
        self.distance = Distance.COSINE

    def recreate_collection(self, collection_name: str):
        logger.info(f"Recreating collection: {collection_name}")
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
        )

    def upsert_chunks(self, collection_name: str, chunks: List[Dict[str, Any]]):
        logger.info(f"Upserting {len(chunks)} chunks into {collection_name}")
        points = []
        for chunk in chunks:
            points.append(
                PointStruct(
                    id=chunk["chunk_index"],
                    vector=chunk["embedding"],
                    payload={
                        "text": chunk["text"],
                        "chunk_index": chunk["chunk_index"],
                    },
                )
            )
        self.client.upsert(collection_name=collection_name, points=points, wait=True)

    def search(self, collection_name: str, query_vector: List[float], limit: int = 5):
        logger.info(f"Searching {collection_name} for top {limit} results")
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )
