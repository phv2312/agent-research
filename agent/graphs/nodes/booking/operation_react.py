import logging
from typing import Any, Literal
from jinja2 import Template
from langgraph.types import Command, StreamWriter

from agent.models.booking import Tickets
from agent.models.messages import UserMessage
from agent.models.stream import StreamChatData
from agent.graphs.prebuilt.react import (
    ReactAgentWorkflow,
    State as ReactState,
)
from .base import BaseNode
from .models import State, Nodes


logger = logging.getLogger(__name__)


class ReactOperationNode(BaseNode[State, Command[Literal[Nodes.END]]]):
    def __init__(
        self, prompt_template: Template, tickets: Tickets, react: ReactAgentWorkflow
    ):
        self.prompt_template = prompt_template
        self.tickets = tickets
        self.react = react

    async def process(
        self, state: State, writer: StreamWriter, *_: Any, **__: Any
    ) -> Command[Literal[Nodes.END]]:
        prompt_content = self.prompt_template.render(
            user_ticket_info=self.tickets.content,
            user_query=state.query.content,
            feedbacks=state.feedbacks or [],
        )
        logger.debug(f"Prompt content: {prompt_content}")

        response = await self.react.process(
            ReactState(
                messages=[UserMessage(content=prompt_content)],
            )
        )
        ai_message = response["messages"][-1]
        writer(
            StreamChatData.from_message(
                f"[Confirmation - TODO: route to SQL agent] {str(ai_message.content)}"
            )
        )
        return Command(
            goto=Nodes.END,
        )
