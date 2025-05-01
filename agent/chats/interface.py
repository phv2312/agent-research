from collections.abc import Sequence
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
