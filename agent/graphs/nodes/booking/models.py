import operator
from enum import StrEnum, auto
from typing import Annotated

from pydantic import BaseModel, Field

from agent.models.messages import AssistantMessage, UserMessage
from agent.programs.impl.booking import BookingAIResponse


class State(BaseModel):
    query: UserMessage
    history: Annotated[
        list[UserMessage | AssistantMessage],
        operator.add,
    ] = Field(default_factory=list)
    feedbacks: list[str] | None = None
    booking_response: BookingAIResponse | None = None


class Nodes(StrEnum):
    COORDINATOR = auto()
    END = "__end__"
    FAQ = auto()
    OPERATION = auto()
    OPERATION_FEEDBACK = auto()
