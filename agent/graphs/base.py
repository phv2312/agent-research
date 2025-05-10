from abc import ABC, abstractmethod
from typing import cast
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel


class BaseGraphNode[InputT: BaseModel, OutputT: BaseModel](ABC):
    InputCls: type[InputT]
    OutputCls: type[OutputT]

    def __init__(self) -> None:
        self.compiled_graph = self.build_graph()

    async def ainvoke(
        self,
        input_data: InputT,
    ) -> OutputT:
        return cast(OutputT, await self.compiled_graph.ainvoke(input_data))

    @abstractmethod
    def build_graph(self) -> CompiledGraph: ...
