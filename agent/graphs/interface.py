from typing import Protocol

from .models import InputT, OutputT


class IGraphNode(Protocol):
    async def process(self, node_input: InputT) -> OutputT: ...
