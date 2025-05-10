import asyncio
from concurrent.futures import Executor, ThreadPoolExecutor
import logging
from typing import Any, Literal, TypedDict
from pydantic import BaseModel
from tavily import TavilyClient
import tiktoken

from agent.models.document import ScoredChunk, ScoredChunks, Chunk, WebsearchMetdata

logger = logging.getLogger(__name__)


class SearchResult(TypedDict):
    url: str
    raw_content: str | None
    content: str | None
    title: str
    score: float


class SearchResponse(TypedDict):
    results: list[SearchResult]


class TavilySettings(BaseModel):
    chunks_per_source: int = 3
    search_depth: Literal["advanced"] = "advanced"
    timerange: Literal["day", "week", "month", "year"] = "month"
    encoding_model_name: str = "gpt-4o"
    token_limit: int = 8000


class TavilyWebSearch:
    def __init__(
        self,
        api_key: str,
        executor_search: Executor | None = None,
        settings: TavilySettings | None = None,
    ) -> None:
        self.api_key = api_key
        self.client = TavilyClient(api_key)
        # TODO: ProcessPoolExecutor
        self.executor_search = executor_search or ThreadPoolExecutor()
        self.settings = settings or TavilySettings()
        self.encoding = tiktoken.encoding_for_model(self.settings.encoding_model_name)

    def count_num_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def search(
        self,
        query: str,
        topk: int = 3,
    ) -> ScoredChunks:
        if query == "":
            raise ValueError("Query cannot be empty")

        response: SearchResponse = self.client.search(
            query,
            max_results=topk,
            include_raw_content=True,  # it's necessary to get full content
            time_range=self.settings.timerange,
            chunks_per_source=self.settings.chunks_per_source,
            search_depth=self.settings.search_depth,
        )

        scored_chunks = []
        for result in response["results"]:
            if result["raw_content"] is None:
                logger.warning("Result %s has no raw content", result["url"])
                continue

            num_tokens = self.count_num_tokens(result["raw_content"])
            if num_tokens > self.settings.token_limit:
                logger.warning(
                    "Result %s exceeds token limit %d: %d tokens",
                    result["url"],
                    self.settings.token_limit,
                    num_tokens,
                )
                continue
            scored_chunks.append(
                ScoredChunk(
                    chunk=Chunk(
                        text=result["raw_content"] or "",
                        metadata=WebsearchMetdata(
                            url=result["url"],
                        ),
                    ),
                    score=result["score"],
                )
            )

        return ScoredChunks(scored_chunks).sort()

    async def asearch(
        self,
        query: str,
        topk: int = 3,
        *args: Any,
        **kwargs: Any,
    ) -> ScoredChunks:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor_search, self.search, query, topk
        )
