from enum import StrEnum
from pathlib import Path
from typing import Self
from pydantic import RootModel, Field, BaseModel


class OperationType(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    READ = "read"


class TicketInformation(BaseModel):
    transport: str = Field(
        description="The type of transport (e.g., flight, train, bus)"
    )
    departure_time: str = Field(description="The departure time of the ticket")
    destination: str = Field(description="The destination of the ticket")
    booking_reference: str = Field(description="The booking reference of the ticket")

    @property
    def content(self) -> str:
        return (
            f"Transport: {self.transport}, "
            f"Departure Time: {self.departure_time}, "
            f"Destination: {self.destination}, "
            f"Booking Reference: {self.booking_reference}"
        )


class Tickets(RootModel[list[TicketInformation]]):
    @property
    def content(self) -> str:
        return "\n".join(ticket.content for ticket in self.root)

    @classmethod
    def from_json_file(
        cls,
        filepath: Path,
    ) -> Self:
        with open(filepath, "r") as f:
            return cls.model_validate_json(f.read())

    def add(self, ticket: TicketInformation) -> Self:
        return self.model_copy(
            update={
                "root": [*self.root, ticket],
            }
        )

    def update(
        self,
        original_ticket: TicketInformation,
        updated_ticket: TicketInformation,
    ) -> Self:
        filtered_tickets = [ticket for ticket in self.root if ticket != original_ticket]

        return self.model_copy(
            update={
                "root": [*filtered_tickets, updated_ticket],
            }
        )

    def delete(self, original_ticket: TicketInformation) -> Self:
        filtered_tickets = [ticket for ticket in self.root if ticket != original_ticket]

        return self.model_copy(
            update={
                "root": filtered_tickets,
            }
        )
