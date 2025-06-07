from enum import StrEnum, auto
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field

from agent.models.document import ScoredChunks
from .messages import AssistantMessage


class StreamEvent(StrEnum):
    CHAT = auto()
    CHUNKS = auto()
    INTERRUPT = auto()


class BaseStreamData[DataT: BaseModel](BaseModel):
    event: StreamEvent
    data: DataT


class Interrupt(BaseModel):
    is_interrupted: bool


class StreamChatData(BaseStreamData[AssistantMessage]):
    event: Literal[StreamEvent.CHAT] = Field(StreamEvent.CHAT)
    data: AssistantMessage

    @classmethod
    def from_message(cls, content: str) -> Self:
        return cls(data=AssistantMessage(content=content))


class StreamChunksData(BaseStreamData[ScoredChunks]):
    event: Literal[StreamEvent.CHUNKS] = Field(StreamEvent.CHUNKS)
    data: ScoredChunks


class StreamInterruptData(BaseStreamData[Interrupt]):
    event: Literal[StreamEvent.INTERRUPT] = Field(StreamEvent.INTERRUPT)
    data: Interrupt


StreamData = Annotated[
    StreamChatData | StreamChunksData | StreamInterruptData,
    Field(discriminator="event"),
]
