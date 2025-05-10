from pydantic import BaseModel, Field

from agent.programs.base import BaseProgram


class DialogQuestion(BaseModel):
    question: str = Field(..., description="question to be asked")

    @property
    def content(self) -> str:
        return self.question


class DialogQuestionProgram(BaseProgram[DialogQuestion]):
    ModelOutCls = DialogQuestion
