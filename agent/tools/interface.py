from typing import Protocol

from pydantic import BaseModel

from agent.tools.models import ToolSchema


class ITool[ParamsT: BaseModel, ResponseT](Protocol):
    name: str
    description: str
    ParamsCls: type[ParamsT]

    @property
    def schema(self) -> ToolSchema: ...

    async def __call__(self, params: ParamsT) -> ResponseT: ...
