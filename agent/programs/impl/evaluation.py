from pydantic import BaseModel, Field
from agent.programs.base import BaseProgram


class EvaluationResponse(BaseModel):
    score: float = Field(
        default=0.0,
        description="Score adherence from 0.0 (completely non-compliant) to 1.0 (perfectly compliant)",
    )
    reasoning: str = Field(
        default="", description="Short & clear reasoning for the score"
    )
    rules: list[str] = Field(
        default_factory=list,
        description="Identify which rules are relevant to the evaluation",
    )


class EvaluationProgram(BaseProgram[EvaluationResponse]):
    ModelOutCls = EvaluationResponse
