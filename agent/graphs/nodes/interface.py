from typing import Any, Protocol

from langgraph.types import Command
from pydantic import BaseModel


class INode[
    NodeInputT: BaseModel,
    NodeOutputT: BaseModel | Command[Any],
](Protocol):
    async def process(
        self,
        state: NodeInputT,
        *_: Any,
        **__: Any,
    ) -> NodeOutputT: ...
