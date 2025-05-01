from typing import Any, Protocol

from ..models.embeddings import Embedding


class IEmbeddingModel(Protocol):
    async def aembedding(
        self,
        queries: list[str],
        *_: Any,
        **__: Any,
    ) -> list[Embedding]: ...
