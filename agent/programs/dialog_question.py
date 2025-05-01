from typing import Iterator
from pydantic import BaseModel, Field

from .base import BaseProgram


class DialogQuestion(BaseModel):
    question: list[str] = Field(..., description="List of questions/queries")

    @property
    def content(self) -> str:
        return "\n".join(self.question).strip()

    def __len__(self) -> int:
        return len(self.question)

    def iter(self) -> Iterator[str]:
        return iter(self.question)


class DialogQuestionProgram(BaseProgram[DialogQuestion]):
    ModelOutCls = DialogQuestion
