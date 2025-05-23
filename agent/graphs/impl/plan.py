from dataclasses import dataclass
import logging

from langgraph.graph.graph import CompiledGraph
from langgraph.graph import StateGraph

from pydantic import BaseModel, Field
from agent import prompts
from agent.chats.interface import IChatModel
from agent.graphs.base import BaseGraphNode
from agent.models.document import ScoredChunks
from agent.models.messages import UserMessage
from agent.programs.outline import Outline, OutlineReportProgram
from agent.searches import IWebSearch


logger = logging.getLogger(__name__)


class TopicPlanningInput(BaseModel):
    message: str = Field(..., description="User query to generate outline")
    feedback: str = Field("", description="Feedback from user to optimize the planning")


class TopicPlanningOutput(BaseModel):
    outline: Outline | None = None


class TopicPlanningState(TopicPlanningInput, TopicPlanningOutput):
    saerched_results: ScoredChunks = Field(
        default_factory=lambda: ScoredChunks([]),
        description="Search results from web search",
    )


@dataclass
class TopicPlanningSettings:
    max_completion_tokens: int = 4096 * 10
    websearch_topk: int = 2


class TopicPlanningGraph(BaseGraphNode[TopicPlanningInput, TopicPlanningOutput]):
    InputCls = TopicPlanningInput
    OutputCls = TopicPlanningOutput

    def __init__(
        self,
        websearch: IWebSearch,
        chat_model: IChatModel,
        outline_program: OutlineReportProgram,
        settings: TopicPlanningSettings | None = None,
    ) -> None:
        self.websearch = websearch
        self.chat_model = chat_model
        self.outline_program = outline_program
        self.settings = settings or TopicPlanningSettings()

        super().__init__()

    async def search(
        self,
        planning_input: TopicPlanningInput,
    ) -> TopicPlanningState:
        logger.info("Search web for query; %s", planning_input.message)
        searched_results: ScoredChunks = await self.websearch.asearch(
            query=planning_input.message,
            topk=self.settings.websearch_topk,
        )
        return TopicPlanningState(
            message=planning_input.message,
            feedback=planning_input.feedback,
            saerched_results=searched_results,
        )

    async def plan(self, state: TopicPlanningState) -> TopicPlanningOutput:
        message = UserMessage(
            content=prompts.REPORT_PLANNER.render(
                topic=state.message,
                context=state.saerched_results.context,
                feedback=state.feedback,
            )
        )
        outline = await self.outline_program.aprocess(
            message=message,
        )
        return TopicPlanningOutput(
            outline=outline,
        )

    def build_graph(self) -> CompiledGraph:
        return (
            StateGraph(
                TopicPlanningState,
                input=TopicPlanningInput,
                output=TopicPlanningOutput,
            )
            .add_node("search", self.search)
            .add_node("plan", self.plan)
            .add_edge("search", "plan")
            .set_entry_point("search")
            .compile()
        )
