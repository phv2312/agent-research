from abc import ABC, abstractmethod
from typing import Any

from langgraph.types import Command, StreamWriter
from pydantic import BaseModel

from .models import State


class BaseNode[NodeInputT: State, NodeOutputT: BaseModel | Command[Any]](ABC):
    @abstractmethod
    async def process(
        self,
        state: NodeInputT,
        writer: StreamWriter,
        *_: Any,
        **__: Any,
    ) -> NodeOutputT:
        raise NotImplementedError
