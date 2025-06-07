import logging
from typing import Any, Literal
from jinja2 import Template
from langgraph.types import Command, StreamWriter, interrupt

from agent.models.booking import Tickets
from agent.models.messages import UserMessage
from agent.models.stream import StreamChatData
from agent.programs.interface import IProgram

from .base import BaseNode
from .models import State, Nodes

logger = logging.getLogger(__name__)


class OperationNode(BaseNode[State, Command[Literal[Nodes.OPERATION_FEEDBACK]]]):
    def __init__(
        self,
        program: IProgram[Any],
        prompt_template: Template,
        tickets: Tickets,
    ):
        self.program = program
        self.prompt_template = prompt_template
        self.tickets = tickets

    async def process(
        self, state: State, writer: StreamWriter, *_: Any, **__: Any
    ) -> Command[Literal[Nodes.OPERATION_FEEDBACK]]:
        prompt_content = self.prompt_template.render(
            user_ticket_info=self.tickets.content,
            user_query=state.query.content,
            feedbacks=state.feedbacks or [],
        )
        logger.debug(f"Prompt content: {prompt_content}")

        response = await self.program.aprocess(
            message=UserMessage(content=prompt_content),
        )

        return Command(
            goto=Nodes.OPERATION_FEEDBACK,
            update={
                "booking_response": response,
            },
        )


class OperationFeedbackNode(
    BaseNode[State, Command[Literal[Nodes.END, Nodes.OPERATION]]]
):
    async def process(
        self, state: State, writer: StreamWriter, *_: Any, **__: Any
    ) -> Command[Literal[Nodes.END, Nodes.OPERATION]]:
        booking_response = state.booking_response
        if booking_response is None:
            logger.warning("No booking response")
            return Command(
                goto=Nodes.END,
            )

        if booking_response.followup_query != "":
            feedback_response = interrupt(booking_response.followup_query)

            return Command(
                goto=Nodes.OPERATION,
                update={
                    "feedbacks": [
                        *(state.feedbacks or []),
                        feedback_response,
                    ]
                },
            )

        if booking_response.request:
            operation_content = (
                f"I will perform: {booking_response.request.operator} on "
                f"data: {booking_response.request.model_dump()}. Thanks"
            )
            writer(StreamChatData.from_message(operation_content))

        return Command(
            goto=Nodes.END,
        )
