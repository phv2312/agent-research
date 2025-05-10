from pydantic import BaseModel, Field

from agent.programs.base import BaseProgram


class SearchQueries(BaseModel):
    queries: list[str] = Field(
        description="Comprehensive list of search queries.",
    )


class SearchQueriesProgram(BaseProgram[SearchQueries]):
    ModelOutCls = SearchQueries
