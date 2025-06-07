from enum import StrEnum
import logging
from typing import Any, Literal, TypedDict

from jinja2 import Template
from langgraph.types import Command, StreamWriter

from agent.chats.interface import IChatModel
from agent.models.messages import UserMessage
from agent.models.stream import StreamChatData

from .base import BaseNode
from .models import State, Nodes


logger = logging.getLogger(__name__)


class ToolSchemaFunction(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolSchema(TypedDict):
    type: Literal["function"]
    function: ToolSchemaFunction


class ToolNames(StrEnum):
    FAQ = "faq"
    BOOKING = "booking"


class ToolFactory:
    @staticmethod
    def faq() -> ToolSchema:
        return ToolSchema(
            type="function",
            function=ToolSchemaFunction(
                name="faq",
                description="Answers questions about ticket policies and general information",
                parameters={},
            ),
        )

    @staticmethod
    def booking() -> ToolSchema:
        return ToolSchema(
            type="function",
            function=ToolSchemaFunction(
                name="booking",
                description="Handles ticket booking and modification requests",
                parameters={},
            ),
        )


class CoordinatorNode(
    BaseNode[
        State,
        Command[Literal[Nodes.END, Nodes.OPERATION, Nodes.FAQ],],
    ],
):
    def __init__(self, chat_model: IChatModel, prompt_template: Template) -> None:
        self.chat_model = chat_model
        self.prompt_template = prompt_template

    async def process(
        self,
        state: State,
        writer: StreamWriter,
        *_: Any,
        **__: Any,
    ) -> Command[Literal[Nodes.END, Nodes.OPERATION, Nodes.FAQ]]:
        prompt_content: str = self.prompt_template.render(
            query=state.query.content,
            history=[
                {
                    "role": message.role,
                    "content": str(message.content),
                }
                for message in state.history
            ],
        )

        message_generator = self.chat_model.astream(
            UserMessage(content=prompt_content),
            temperature=0.0,
            tools=[
                ToolFactory.faq(),
                ToolFactory.booking(),
            ],
        )

        async for message in message_generator:
            for tool_call in message.tool_calls or []:
                if tool_call is None or tool_call.function is None:
                    continue

                match tool_call.function.name:
                    case ToolNames.FAQ:
                        logger.info("FAQ tool call")
                        return Command(
                            goto=Nodes.FAQ,
                        )
                    case ToolNames.BOOKING:
                        logger.info("Booking tool call")
                        return Command(
                            goto=Nodes.OPERATION,
                        )
                    case _:
                        raise ValueError(
                            f"Unknown tool call: {tool_call.function.name}",
                        )

            if message.tool_calls is None:
                writer(StreamChatData.from_message(str(message.content)))

        return Command(goto=Nodes.END)
