import base64
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Any, Literal
from pydantic import BaseModel, BeforeValidator, Discriminator, Field


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

    def to_openai_message(self) -> dict[str, Any]:
        match self.content:
            case str():
                return {
                    "role": self.role.name.lower(),
                    "content": self.content,
                }
            case list():
                return {
                    "role": self.role.name.lower(),
                    "content": [content.model_dump() for content in self.content],
                }
            case _:
                raise ValueError("Invalid content in message")


class UserMessage(BaseMessage):
    role: Literal[MessageRole.user] = MessageRole.user


class AssistantMessage(BaseMessage):
    role: Literal[MessageRole.assistant] = MessageRole.assistant


class SystemMessage(BaseMessage):
    role: Literal[MessageRole.system] = MessageRole.system
