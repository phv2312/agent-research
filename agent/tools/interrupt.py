from langgraph.types import interrupt
from pydantic import BaseModel, Field
from .base import BaseTool


class InterruptedParams(BaseModel):
    content: str = Field(
        description="The query you want to clarify with the user, or ask for information"
    )


class InterruptedResponse(BaseModel):
    content: str = Field(description="User feedbacks")


class InterruptedTool(BaseTool[InterruptedParams, InterruptedResponse]):
    name = "interrupt"
    description = "Gather information from user"
    ParamsCls = InterruptedParams

    async def __call__(self, params: InterruptedParams) -> InterruptedResponse:
        feedback = interrupt(params.content)
        return InterruptedResponse(content=feedback)
