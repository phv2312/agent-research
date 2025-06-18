import json
import logging
import operator
from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Send
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from agent.tools.interface import ITool
from agent.chats.interface import IChatModel
from agent.models.messages import (
    AssistantMessage,
    Message,
    Messages,
    ToolResponseMessage,
)


logger = logging.getLogger(__name__)


class State(BaseModel):
    messages: Annotated[list[Message], operator.add] = Field(default_factory=list)


class ReactAgentWorkflow:
    def __init__(
        self,
        chat_model: IChatModel,
        tools: list[ITool[Any, Any]],
        human_tools: list[ITool[Any, Any]],
    ) -> None:
        self.chat_model = chat_model
        self.tools = tools
        self.human_tools = human_tools

        self.mp_tools = {tool.name: tool for tool in self.tools}
        self.mp_human_tools = {tool.name: tool for tool in self.human_tools}
        self.graph = self.build()

    def build(
        self,
    ) -> CompiledGraph:
        async def _build_agent_node(
            state: State,
        ) -> dict[Literal["messages"], Any]:
            # TODO: not support stream now.
            response = await self.chat_model.achat(
                messages=Messages(state.messages),
                tools=[tool.schema for tool in self.tools + self.human_tools],
            )

            return {"messages": state.messages + [response]}

        async def _decide_tool_calls(state: State) -> list[Send]:
            if len(state.messages) < 1:
                return [Send(node="__end__", arg=state)]

            last_message = state.messages[-1]
            if isinstance(last_message, AssistantMessage):
                send_signals: list[Send] = []
                for tool_call in last_message.tool_calls or []:
                    if tool_call.function is None:
                        continue
                    name = tool_call.function.name
                    message = AssistantMessage(content="", tool_calls=[tool_call])

                    node = "tools"
                    if name in self.mp_human_tools:
                        node = "human_tools"

                    send_signals.append(Send(node=node, arg=State(messages=[message])))

                return send_signals

            return [Send(node="__end__", arg=state)]

        def _validate_tool_call(state: State) -> ChatCompletionMessageToolCall:
            if len(state.messages) < 1:
                raise ValueError("Expected > 1 message")

            last_message = state.messages[-1]
            if not isinstance(last_message, AssistantMessage):
                raise ValueError("Expected the last message should be AssistantMessage")

            if last_message.tool_calls is None or len(last_message.tool_calls) != 1:
                raise ValueError(
                    "Expected the last message tool_calls should not be None"
                )

            tool_call = last_message.tool_calls[0]

            if not isinstance(tool_call, ChatCompletionMessageToolCall):
                raise ValueError(
                    "Expected the last message tool_calls should be ChatCompletionMessageToolCall"
                )

            return tool_call

        async def _tools(state: State) -> dict[str, Any]:
            tool_call = _validate_tool_call(state)
            params_dict: dict[str, Any] = json.loads(tool_call.function.arguments)

            caller = self.mp_tools.get(tool_call.function.name)
            if caller is None:
                raise ValueError(f"Can not find tool-name: {tool_call.function.name}.")

            logger.debug("Calling tool: %s with params: %s", caller.name, params_dict)
            tool_response = await caller(caller.ParamsCls.model_validate(params_dict))
            return {
                "messages": [
                    ToolResponseMessage(
                        role="tool",
                        tool_call_id=tool_call.id,
                        content=str(tool_response),
                    )
                ]
            }

        async def _human_tools(state: State) -> dict[str, Any]:
            tool_call = _validate_tool_call(state)
            params_dict: dict[str, Any] = json.loads(tool_call.function.arguments)

            caller = self.mp_human_tools.get(tool_call.function.name)
            if caller is None:
                raise ValueError(
                    f"Can not find human tool-name: {tool_call.function.name}."
                )

            logger.debug("Calling tool: %s with params: %s", caller.name, params_dict)
            params = caller.ParamsCls.model_validate(params_dict)
            response = await caller(params)

            return {
                "messages": [
                    ToolResponseMessage(
                        role="tool", tool_call_id=tool_call.id, content=str(response)
                    )
                ]
            }

        return (
            StateGraph(State)
            .add_node("agent", _build_agent_node)
            .add_node("tools", _tools)
            .add_node("human_tools", _human_tools)
            .add_edge("tools", "agent")
            .add_edge("human_tools", "agent")
            .add_conditional_edges(
                "agent",
                _decide_tool_calls,
            )
            .set_entry_point("agent")
            .compile()
        )

    async def process(
        self,
        state: State,
    ) -> dict[str, Any]:
        response = await self.graph.ainvoke(
            input=state,
        )
        return response
