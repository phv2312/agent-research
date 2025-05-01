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
from agent.programs.macro_report import Outline, OutlineReportProgram
from agent.searches import ISearch


logger = logging.getLogger(__name__)


class TopicPlanningInput(BaseModel):
    message: str = Field(..., description="User query to generate outline")
    feedback: str = Field("", description="Feedback from user to optimize the planning")


class TopicPlanningOutput(BaseModel):
    outline: Outline | None = None


class TopicPlanningState(TopicPlanningInput, TopicPlanningOutput): ...


@dataclass
class TopicPlanningSettings:
    max_completion_tokens: int = 4096 * 10
    websearch_topk: int = 5


class TopicPlanningGraph(BaseGraphNode[TopicPlanningInput, TopicPlanningOutput]):
    InputCls = TopicPlanningInput
    OutputCls = TopicPlanningOutput

    def __init__(
        self,
        websearch: ISearch,
        chat_model: IChatModel,
        outline_program: OutlineReportProgram,
        settings: TopicPlanningSettings | None = None,
    ) -> None:
        self.websearch = websearch
        self.chat_model = chat_model
        self.outline_program = outline_program
        self.settings = settings or TopicPlanningSettings()

        super().__init__()

    async def plan(self, planing_input: TopicPlanningInput) -> TopicPlanningOutput:
        logger.info("Search web for query; %s", planing_input.message)
        searched_results: ScoredChunks = await self.websearch.asearch(
            query=planing_input.message,
            topk=self.settings.websearch_topk,
        )
        message = UserMessage(
            content=prompts.REPORT_PLANNER.render(
                topic=planing_input.message,
                context=searched_results.context,
                feedback=planing_input.feedback,
            )
        )

        outline = await self.outline_program.aprocess(
            message=message,
        )

        return TopicPlanningOutput(
            outline=outline,
        )

    def build_graph(self) -> CompiledGraph:
        builder = StateGraph(
            TopicPlanningState,
            input=TopicPlanningInput,
            output=TopicPlanningOutput,
        )

        builder.add_node("plan", self.plan)
        builder.set_entry_point("plan")

        return builder.compile()
