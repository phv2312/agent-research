import logging
from functools import cached_property
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, cast

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.types import Command, Interrupt as LGInterrupt
from langchain_core.runnables import RunnableConfig

from agent.container import Container
from agent.graphs.nodes.booking.coordinator import CoordinatorNode
from agent.graphs.nodes.booking.faq import FAQNode

# from agent.graphs.nodes.booking.operation import OperationNode, OperationFeedbackNode
from agent.graphs.nodes.booking.operation_react import ReactOperationNode
from agent.graphs.nodes.booking.models import Nodes, State
from agent.graphs.prebuilt.react import ReactAgentWorkflow
from agent.tools.interrupt import InterruptedTool
from agent.models.booking import Tickets
from agent.models.stream import (
    Interrupt,
    StreamChatData,
    StreamChunksData,
    StreamData,
    StreamInterruptData,
)
from agent.models.messages import AssistantMessage, UserMessage
from agent.prompts import BookingJinja2Prompts, PromptsFactory


logger = logging.getLogger(__name__)

InterruptData = dict[Literal["__interrupt__"], tuple[LGInterrupt, ...]]


class GraphDependencies:
    def __init__(self, samplepath: Path) -> None:
        self.samplepath = samplepath

    @cached_property
    def container(self) -> Container:
        return Container()

    @cached_property
    def booking_prompts(self) -> BookingJinja2Prompts:
        return PromptsFactory.booking()

    @cached_property
    def coordinator(self) -> CoordinatorNode:
        return CoordinatorNode(
            chat_model=self.container.chats.get("azure_openai"),
            prompt_template=self.booking_prompts.get("coordinator"),
        )

    @cached_property
    def faq(self) -> FAQNode:
        return FAQNode(
            chat_model=self.container.chats.get("azure_openai"),
            vectordb=self.container.vectordbs.get("milvus"),
            embedding_model=self.container.embeddings.get("azure_openai"),
            prompt_template=self.booking_prompts.get("faq"),
        )

    @cached_property
    def operation(self) -> ReactOperationNode:
        return ReactOperationNode(
            react=ReactAgentWorkflow(
                chat_model=self.container.chats.get("azure_openai"),
                tools=[],
                human_tools=[InterruptedTool()],
            ),
            prompt_template=self.booking_prompts.get("operation_react"),
            tickets=Tickets.from_json_file(self.samplepath),
        )


class BookingAssistantGraph:
    def __init__(self, dependencies: GraphDependencies) -> None:
        self.deps = dependencies
        self.graph = self.build()

    def build(self) -> CompiledGraph:
        builder = (
            StateGraph(State)
            .add_node(Nodes.COORDINATOR, self.deps.coordinator.process)
            .add_node(Nodes.FAQ, self.deps.faq.process)
            .add_node(
                Nodes.OPERATION,
                self.deps.operation.process,
            )
            # .add_node(Nodes.OPERATION_FEEDBACK, self.deps.operation_feedback.process)
            .set_entry_point(Nodes.COORDINATOR)
        )

        return builder.compile(checkpointer=MemorySaver())

    async def stream_async_answer(
        self,
        query: str,
        conversation_id: str,
        history: list[UserMessage | AssistantMessage] | None = None,
        *_: Any,
        is_interrupted: bool = False,
    ) -> AsyncGenerator[StreamData, None]:
        run_config = {
            "configurable": {
                "thread_id": conversation_id,
            },
        }

        logger.info(
            "Running graph with config: %s, is_interrupted: %s, history: %s, query: %s",
            run_config,
            is_interrupted,
            history,
            query,
        )

        graph_input: State | Command[Any]
        if is_interrupted:
            graph_input = Command(resume=query)
        else:
            graph_input = State(
                query=UserMessage(content=query),
                history=(history or []),
                feedbacks=None,
                booking_response=None,
            )

        async for data in self.graph.astream(
            graph_input,
            config=cast(RunnableConfig, run_config),
            stream_mode=["custom", "updates"],
            # subgraphs=True,
        ):
            # print(data)
            mode, event = data
            match mode:
                case "custom":
                    if isinstance(event, StreamChatData | StreamChunksData):
                        yield event
                case "updates":
                    if "__interrupt__" in event:
                        interrupt_data = cast(InterruptData, event)
                        yield StreamChatData(
                            data=AssistantMessage(
                                content=interrupt_data["__interrupt__"][0].value
                            ),
                        )

            _is_interrupted = mode == "updates" and "__interrupt__" in event
            if _is_interrupted != is_interrupted:
                yield StreamInterruptData(
                    data=Interrupt(
                        is_interrupted=_is_interrupted,
                    ),
                )
                is_interrupted = _is_interrupted
