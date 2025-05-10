from typing import Protocol


class ITextSplitter(Protocol):
    async def asplit_text(
        self,
        text: str,
    ) -> list[str]: ...
