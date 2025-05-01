from typing import Any, Protocol

from agent.models.document import ScoredChunks


class ISearch(Protocol):
    async def asearch(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> ScoredChunks: ...
