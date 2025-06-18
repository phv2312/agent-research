from collections.abc import AsyncGenerator, Sequence
from typing import Any
from openai import AsyncAzureOpenAI
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam,
)
from agent.models.messages import AssistantMessage, Messages


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
        messages: Messages,
        temperature: float = 0.1,
        max_completion_tokens: int | None = None,
        *_: Any,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        **__: Any,
    ) -> AsyncGenerator[AssistantMessage, None]:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools

        if max_completion_tokens:
            kwargs["max_completion_tokens"] = max_completion_tokens

        async for response in await self.openai.chat.completions.create(
            model=self.deployment_name,
            messages=messages.as_openai_list(),
            stream=True,
            temperature=temperature,
            **kwargs,
        ):
            if len(response.choices) == 0:
                continue
            delta = response.choices[0].delta
            if delta is None:
                continue
            yield AssistantMessage(
                content=delta.content or "",
                tool_calls=delta.tool_calls,
            )

    async def achat(
        self,
        messages: Messages,
        temperature: float = 0.1,
        max_completion_tokens: int | None = None,
        *_: Any,
        tools: Sequence[ChatCompletionToolParam] | None = None,
        **__: Any,
    ) -> AssistantMessage:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools

        if max_completion_tokens:
            kwargs["max_completion_tokens"] = max_completion_tokens

        response = await self.openai.chat.completions.create(
            model=self.deployment_name,
            messages=messages.as_openai_list(),
            temperature=temperature,
            **kwargs,
        )

        if len(response.choices) == 0:
            raise ValueError("No response from OpenAI")

        message = response.choices[0].message

        return AssistantMessage(
            content=message.content or "",
            tool_calls=message.tool_calls or [],
        )
