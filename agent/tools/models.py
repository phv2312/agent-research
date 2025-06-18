from typing import Any, Literal, TypedDict


class ToolSchemaFunction(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolSchema(TypedDict):
    type: Literal["function"]
    function: ToolSchemaFunction
