from enum import StrEnum, auto
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class StreamEvent(StrEnum):
    expert = auto()


class BaseStreamEvent[dataT: BaseModel](BaseModel):
    event: StreamEvent
    data: dataT


class ExpertData(BaseModel):
    section_id: str
    token: str


class StreamExpertEvent(BaseStreamEvent[ExpertData]):
    event: Literal[StreamEvent.expert] = StreamEvent.expert


StreamData = Annotated[
    StreamExpertEvent,
    Field(
        discriminator="event",
    ),
]
