from pydantic import BaseModel, Field

from .base import BaseProgram


class Queries(BaseModel):
    queries: list[str] = Field(
        description="Comprehensive list of search queries.",
    )


class QueriesProgram(BaseProgram[Queries]):
    ModelOutCls = Queries
