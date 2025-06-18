from collections.abc import AsyncGenerator
from typing import Any, Protocol

from ..models.messages import AssistantMessage, Messages


class IChatModel(Protocol):
    async def achat(
        self,
        messages: Messages,
        *_: Any,
        **__: Any,
    ) -> AssistantMessage: ...

    # Due to the bugs of mypy, we should define as def instead of async def
    # https://github.com/python/mypy/issues/12662
    def astream(
        self,
        messages: Messages,
        *_: Any,
        **__: Any,
    ) -> AsyncGenerator[AssistantMessage, None]: ...
