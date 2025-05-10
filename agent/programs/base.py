from collections.abc import Sequence
from typing import cast

from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from agent.models.messages import AssistantMessage, SystemMessage, UserMessage
from .exc import ParsedResultError


class BaseProgram[ModelOutT: BaseModel]:
    ModelOutCls: type[BaseModel]

    def __init__(
        self,
        api_key: str,
        api_version: str,
        azure_endpoint: str,
        deployment_name: str,
    ) -> None:
        self.openai = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )
        self.deployment_name = deployment_name

    async def aprocess(
        self,
        message: UserMessage | None = None,
        system_message: SystemMessage | None = None,
        history: Sequence[UserMessage | AssistantMessage] | None = None,
    ) -> ModelOutT:
        if message is None and system_message is None:
            raise ValueError("Either message or system_message must be provided.")

        all_messages = [
            *[msg.to_openai_message() for msg in (history or [])],
        ]

        if message:
            all_messages = [
                *all_messages,
                message.to_openai_message(),
            ]

        if system_message:
            all_messages = [
                system_message.to_openai_message(),
                *all_messages,
            ]

        completion = await self.openai.beta.chat.completions.parse(
            model=self.deployment_name,
            messages=cast(list[ChatCompletionMessageParam], all_messages),
            response_format=self.ModelOutCls,
        )
        first_choice = completion.choices[0].message
        if first_choice.refusal:
            raise ParsedResultError(first_choice.refusal)
        return cast(ModelOutT, first_choice.parsed)
