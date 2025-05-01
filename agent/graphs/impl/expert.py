from concurrent.futures import Executor, ProcessPoolExecutor
import logging
import asyncio
import operator
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph.graph import CompiledGraph
from langgraph.graph import StateGraph, END
from langchain_text_splitters import TokenTextSplitter

from agent import prompts
from agent.chats.interface import IChatModel
from agent.models.document import ScoredChunks
from agent.models.messages import AssistantMessage, MessageRole, UserMessage
from agent.programs import (
    Section,
    DialogQuestion,
    DialogQuestionProgram,
    QueriesProgram,
)
from agent.graphs.base import BaseGraphNode
from agent.searches import ISearch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DialogInput(BaseModel):
    topic: str
    section: Section


class DialogConversation(BaseModel):
    conversations: Annotated[list[UserMessage | AssistantMessage], operator.add] = (
        Field(default_factory=list)
    )
    question: DialogQuestion | None = None
    search_results: Annotated[list[ScoredChunks], operator.add] = Field(
        default_factory=list
    )

    @property
    def conversation_context(self) -> str:
        context = ""
        for conversation in self.conversations:
            match conversation.role:
                case MessageRole.user:
                    context += f"Question: {conversation.content}\n\n"
                case MessageRole.assistant:
                    context += f"Answer: {conversation.content}\n\n"
                case _:
                    raise ValueError("Invalid role in conversation")
        return context

    @property
    def summarization_context(self) -> str:
        return ScoredChunks([]).extend(self.search_results).sort().context


class Summarization(BaseModel):
    section_id: str
    content: str


class DialogOutput(BaseModel):
    summarizations: Annotated[list[Summarization], operator.add] = Field(
        default_factory=list
    )


class DialogState(DialogInput, DialogOutput, DialogConversation): ...


class DialogExpertSettings(BaseModel):
    topk: int = Field(default=2, description="Top k results for web-search")
    max_questions: int = Field(
        default=5, description="Max questions before summarization"
    )
    max_queries: int = Field(
        default=1, description="Max queries to decompose the question"
    )

    chunk_size: int = Field(
        default=4096 * 20, description="Max completion tokens for chat model"
    )
    chunk_overlap: int = Field(default=0, alias="chunk overlap")
    encoding_model_name: str = Field(default="gpt-4o", alias="encoding model name")
    split_tokens_num_workers: int = Field(
        default=2, alias="num workers for split tokens"
    )


class DialogExpertGraph(BaseGraphNode[DialogInput, DialogOutput]):
    InputCls = DialogInput
    OutputCls = DialogOutput

    def __init__(
        self,
        chat_model: IChatModel,
        websearch: ISearch,
        dialog_question_program: DialogQuestionProgram,
        queries_program: QueriesProgram,
        settings: DialogExpertSettings | None = None,
        executor_split_tokens: Executor | None = None,
    ) -> None:
        self.chat_model = chat_model
        self.websearch = websearch
        self.dialog_question_program = dialog_question_program
        self.queries_program = queries_program
        self.settings = settings or DialogExpertSettings()
        self.executor_split_tokens = executor_split_tokens or ProcessPoolExecutor(
            max_workers=self.settings.split_tokens_num_workers
        )

        super().__init__()

    async def update_section(self, dialog_input: DialogInput) -> DialogState:
        logger.info("Section: %s", dialog_input.section)
        return DialogState(
            section=dialog_input.section,
            topic=dialog_input.topic,
        )

    @staticmethod
    def split_tokens(
        encoding_model_name: str, chunk_size: int, chunk_overlap: int, text: str
    ) -> list[str]:
        return TokenTextSplitter(
            model_name=encoding_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ).split_text(text)

    async def ask_question(self, state: DialogState) -> DialogConversation:
        message = UserMessage(
            content=prompts.QUERY_BY_EXPERT.render(
                topic=state.topic,
                section_topic=state.section.title,
                number_of_queries=self.settings.max_queries,
                conversation=state.conversation_context,
            )
        )
        question = await self.dialog_question_program.aprocess(
            message=message,
        )

        return DialogConversation(
            question=question,
        )

    async def should_continue(self, state: DialogState) -> Literal["no", "yes"]:
        if len(state.question or []) == 0:
            return "no"

        num_questions = len(state.conversations) // 2
        if num_questions >= self.settings.max_questions:
            logger.warning("Reached max number of conversations")
            return "no"

        return "yes"

    async def answer_question(self, state: DialogState) -> DialogConversation:
        if state.question is None:
            raise ValueError("No question to answer")

        retrieval_results_list: list[ScoredChunks] = await asyncio.gather(
            *[
                self.websearch.asearch(
                    query=query,
                    topk=self.settings.topk,
                )
                for query in state.question.iter()
            ]
        )

        conversations: list[UserMessage | AssistantMessage] = []
        flatten_search_results: list[ScoredChunks] = []
        for retrieval_results, question in zip(
            retrieval_results_list, state.question.iter()
        ):
            conversations.extend(
                [
                    UserMessage(content=question),
                    AssistantMessage(
                        content=retrieval_results.context,
                    ),
                ]
            )
            flatten_search_results.append(retrieval_results)

        return DialogConversation(
            conversations=conversations,
            search_results=flatten_search_results,
        )

    async def summarize(self, state: DialogState) -> DialogOutput:
        # Prevent exceeding the max token limit
        loop = asyncio.get_event_loop()
        splitted_tokens = await loop.run_in_executor(
            self.executor_split_tokens,
            self.split_tokens,
            self.settings.encoding_model_name,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
            state.summarization_context,
        )

        if len(splitted_tokens) == 0:
            raise ValueError("No tokens to summarize")

        chat_response = await self.chat_model.achat(
            message=UserMessage(
                content=prompts.TOPIC_SUMMARIZATION.render(
                    section_title=state.section.title,
                    context=splitted_tokens[0],
                )
            ),
        )

        return DialogOutput(
            summarizations=[
                Summarization(
                    section_id=state.section.id,
                    content=str(chat_response.content),
                )
            ],
        )

    def build_graph(self) -> CompiledGraph:
        builder = StateGraph(
            DialogState,
            input=DialogInput,
            output=DialogOutput,
        )
        builder.add_node(
            "update-section",
            self.update_section,
        )
        builder.add_node(
            "ask-question",
            self.ask_question,
        )
        builder.add_node(
            "answer-question",
            self.answer_question,
        )
        builder.add_node(
            "summarize",
            self.summarize,
        )
        builder.set_entry_point("update-section")
        builder.add_edge("update-section", "ask-question")
        builder.add_conditional_edges(
            "ask-question",
            self.should_continue,
            {"no": "summarize", "yes": "answer-question"},
        )
        builder.add_edge("answer-question", "ask-question")
        builder.add_edge("summarize", END)

        return builder.compile()
