from collections.abc import AsyncGenerator, Sequence
from typing import Any
from openai import AsyncAzureOpenAI
from agent.models.messages import UserMessage, SystemMessage, AssistantMessage, Messages


class OpenAIChatModel:
    def __init__(
        self,
        api_key: str,
        api_version: str,
        azure_endpoint: str,
        deployment_name: str,
    ):
        self.openai = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )
        self.deployment_name = deployment_name

    async def astream(
        self,
        message: UserMessage | None = None,
        system_message: SystemMessage | None = None,
        history: Sequence[AssistantMessage | UserMessage] | None = None,
        temperature: float = 0.1,
        max_completion_tokens: int | None = None,
        *_: Any,
        **__: Any,
    ) -> AsyncGenerator[AssistantMessage, None]:
        messages = Messages.from_conversation(
            message=message,
            system_message=system_message,
            history=history,
        ).as_openai_list()

        async for response in await self.openai.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        ):
            if len(response.choices) == 0:
                continue
            delta = response.choices[0].delta
            if delta is None or not delta.content:
                continue
            yield AssistantMessage(content=delta.content)

    async def achat(
        self,
        message: UserMessage | None = None,
        system_message: SystemMessage | None = None,
        history: Sequence[AssistantMessage | UserMessage] | None = None,
        temperature: float = 0.1,
        max_completion_tokens: int | None = None,
        *_: Any,
        **__: Any,
    ) -> AssistantMessage:
        messages = Messages.from_conversation(
            message=message,
            system_message=system_message,
            history=history,
        ).as_openai_list()

        response = await self.openai.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )

        if len(response.choices) == 0:
            raise ValueError("No response from OpenAI")

        return AssistantMessage(content=response.choices[0].message.content or "")
