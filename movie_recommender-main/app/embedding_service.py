from typing import List, Union

from loguru import logger
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(
            f"Embedding model loaded successfully (dimension: {self.get_dimension()})"
        )

    def encode(
        self, text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text.

        Args:
            text: Single text string or list of texts

        Returns:
            Embedding vector(s) as list(s) of floats
        """
        embeddings = self.model.encode(text)

        # Convert to list format
        if isinstance(text, str):
            return embeddings.tolist()
        else:
            return [emb.tolist() for emb in embeddings]

    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
