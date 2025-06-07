from typing import Literal, Union
from pydantic import BaseModel, Field

from agent.models.booking import OperationType, TicketInformation
from agent.programs.base import BaseProgram


class BaseOperation(BaseModel):
    operator: OperationType


class UpdateOperation(BaseModel):
    operator: Literal[OperationType.UPDATE] = Field(
        default=OperationType.UPDATE, description="The operation type"
    )

    # TODO:
    # Instead of having previous_ticket and new_ticket,
    # Later should have [<id>, UpdatedTicket]
    previous_ticket: TicketInformation = Field(
        description="The previous/original ticket information"
    )
    new_ticket: TicketInformation = Field(description="The new ticket information")


class CreateOperation(BaseModel):
    operator: Literal[OperationType.CREATE] = Field(
        default=OperationType.CREATE, description="The operation type"
    )
    ticket: TicketInformation = Field(description="The ticket to be created")


class DeleteOperation(BaseModel):
    operator: Literal[OperationType.DELETE] = Field(
        default=OperationType.DELETE, description="The operation type"
    )
    # TODO: only the id of the ticket is enough
    ticket: list[TicketInformation] = Field(description="List of ticket to be deleted")


class ReadOperation(BaseModel):
    operator: Literal[OperationType.READ] = Field(
        default=OperationType.READ, description="The operation type"
    )
    # TODO: only the id of the ticket is enough
    tickets: list[TicketInformation] = Field(
        description="List of ticket information to be read"
    )


BookingOperation = Union[
    ReadOperation | CreateOperation | UpdateOperation | DeleteOperation,
]


class BookingAIResponse(BaseModel):
    request: BookingOperation | None = Field(
        default=None, description="The operation to be performed"
    )
    followup_query: str = Field(default="", description="The followup query")


class BookingOperationProgram(BaseProgram[BookingAIResponse]):
    ModelOutCls = BookingAIResponse
