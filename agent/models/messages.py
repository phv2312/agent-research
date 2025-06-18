import base64
from collections.abc import Sequence
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Literal, cast
from pydantic import BaseModel, BeforeValidator, Discriminator, Field, RootModel
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)


def encode_image_base64(imagepath: Path) -> str:
    with open(imagepath, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_image}"


class MessageRole(StrEnum):
    system = auto()
    user = auto()
    assistant = auto()


class ContentType(StrEnum):
    image = "image_url"
    text = "text"


class BaseContent(BaseModel):
    type: ContentType


class ImageURL(BaseModel):
    url: Annotated[str, BeforeValidator(lambda path: encode_image_base64(path))]
    detail: Literal["low", "medium", "high"] = "low"


class ImageContent(BaseContent):
    type: Literal[ContentType.image] = ContentType.image
    image_url: ImageURL

    @classmethod
    def from_path(
        cls, path: Path, detail: Literal["low", "medium", "high"] = "low"
    ) -> "ImageContent":
        return cls(image_url=ImageURL(url=str(path), detail=detail))


class TextContent(BaseContent):
    type: Literal[ContentType.text] = ContentType.text
    text: str


MessageContent = Annotated[ImageContent | TextContent, Discriminator("type")]


class BaseMessage(BaseModel):
    role: MessageRole
    content: str | Annotated[list[MessageContent], Field(min_length=1)]


class UserMessage(BaseMessage):
    role: Literal[MessageRole.user] = MessageRole.user


class AssistantMessage(BaseMessage):
    role: Literal[MessageRole.assistant] = MessageRole.assistant
    tool_calls: list[ChatCompletionMessageToolCall | ChoiceDeltaToolCall] | None = None


class SystemMessage(BaseMessage):
    role: Literal[MessageRole.system] = MessageRole.system


class ToolResponseMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


Message = Annotated[
    UserMessage | AssistantMessage | SystemMessage | ToolResponseMessage,
    Field(
        discriminator="role",
    ),
]


class Messages(RootModel[list[Message]]):
    @classmethod
    def from_conversation(
        cls,
        message: UserMessage | None = None,
        system_message: SystemMessage | None = None,
        history: Sequence[AssistantMessage | UserMessage] | None = None,
    ) -> "Messages":
        if message is None and system_message is None:
            raise ValueError("Either message or system_message must be provided.")

        messages: list[Message] = [
            *(history or []),
        ]

        if message:
            messages = [message, *messages]

        if system_message:
            messages = [
                system_message,
                *messages,
            ]

        return cls(root=messages)

    def as_list(self) -> list[Message]:
        return self.root

    def as_openai_list(self) -> list[ChatCompletionMessageParam]:
        return [
            cast(ChatCompletionMessageParam, msg.model_dump()) for msg in self.as_list()
        ]
