from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel

from agent.models.messages import AssistantMessage, SystemMessage, UserMessage


class IProgram[ModelOutT: BaseModel](Protocol):
    async def aprocess(
        self,
        message: UserMessage | None = None,
        system_message: SystemMessage | None = None,
        history: Sequence[UserMessage | AssistantMessage] | None = None,
    ) -> ModelOutT: ...
