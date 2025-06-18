from abc import abstractmethod
from pydantic import BaseModel
from .models import ToolSchema, ToolSchemaFunction


class BaseTool[ParamsT: BaseModel, ResponseT]:
    name: str
    description: str
    ParamsCls: type[ParamsT]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            type="function",
            function=ToolSchemaFunction(
                name=self.name,
                description=self.description,
                parameters=self.ParamsCls.model_json_schema(),
            ),
        )

    @abstractmethod
    async def __call__(self, params: ParamsT) -> ResponseT:
        raise NotImplementedError
