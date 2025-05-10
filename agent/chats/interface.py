from collections.abc import AsyncGenerator, Sequence
from typing import Any, Protocol

from ..models.messages import SystemMessage, UserMessage, AssistantMessage


class IChatModel(Protocol):
    async def achat(
        self,
        message: UserMessage | None = None,
        system_message: SystemMessage | None = None,
        history: Sequence[AssistantMessage | UserMessage] | None = None,
        *_: Any,
        **__: Any,
    ) -> AssistantMessage: ...

    # Due to the bugs of mypy, we should define as def instead of async def
    # https://github.com/python/mypy/issues/12662
    def astream(
        self,
        message: UserMessage | None = None,
        system_message: SystemMessage | None = None,
        history: Sequence[AssistantMessage | UserMessage] | None = None,
        *_: Any,
        **__: Any,
    ) -> AsyncGenerator[AssistantMessage, None]: ...
