from typing import Protocol
from pydantic import BaseModel


class IGraphNode[InputT: BaseModel, OutputT: BaseModel](Protocol):
    async def process(self, node_input: InputT) -> OutputT: ...
