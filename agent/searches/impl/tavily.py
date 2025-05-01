import asyncio
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, Literal, TypedDict
from tavily import TavilyClient

from agent.models.document import ScoredChunk, ScoredChunks, Chunk, WebsearchMetdata


class SearchResult(TypedDict):
    url: str
    raw_content: str | None
    content: str | None
    title: str
    score: float


class SearchResponse(TypedDict):
    results: list[SearchResult]


class TavilyWebSearch:
    def __init__(self, api_key: str, search_executor: Executor | None = None) -> None:
        self.api_key = api_key
        self.client = TavilyClient(api_key)
        self.search_executor = search_executor or ThreadPoolExecutor()

    def search(
        self,
        query: str,
        topk: int = 3,
        timerange: Literal["day", "week", "month", "year"] = "week",
    ) -> ScoredChunks:
        if query == "":
            raise ValueError("Query cannot be empty")

        response: SearchResponse = self.client.search(
            query,
            max_results=topk,
            include_raw_content=True,  # it's necessary to get full content
            time_range=timerange,
        )

        scored_chunks = [
            ScoredChunk(
                chunk=Chunk(
                    text=result["raw_content"] or "",
                    metadata=WebsearchMetdata(
                        url=result["url"],
                    ),
                ),
                score=result["score"],
            )
            for result in filter(
                lambda element: element["raw_content"] is not None, response["results"]
            )
        ]

        return ScoredChunks(scored_chunks).sort()

    async def asearch(
        self,
        query: str,
        topk: int = 3,
        timerange: Literal["day", "week", "month", "year"] = "week",
        *args: Any,
        **kwargs: Any,
    ) -> ScoredChunks:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.search_executor, self.search, query, topk, timerange
        )
