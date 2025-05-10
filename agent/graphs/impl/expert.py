from concurrent.futures import Executor, ProcessPoolExecutor
import logging
import asyncio
import operator
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.types import StreamWriter
from langgraph.graph.graph import CompiledGraph
from langgraph.graph import StateGraph, END
from langchain_text_splitters import TokenTextSplitter

from agent import prompts
from agent.chats.interface import IChatModel
from agent.models.document import ScoredChunks
from agent.models.messages import (
    AssistantMessage,
    BaseMessage,
    MessageRole,
    UserMessage,
)
from agent.programs import (
    Section,
    DialogQuestion,
    DialogQuestionProgram,
    QueriesProgram,
)
from agent.graphs.base import BaseGraphNode
from agent.searches import ISearch
from agent.graphs.models import StreamExpertEvent, ExpertData

logger = logging.getLogger(__name__)


class DialogInput(BaseModel):
    topic: str
    section: Section


class DialogConversation(BaseModel):
    conversations: Annotated[list[BaseMessage], operator.add] = Field(
        default_factory=list
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
    topk: int = Field(default=3, description="Top k results for web-search")
    max_messages: int = Field(default=5, description="Max messages in the conversation")
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

    async def ask_question(
        self, state: DialogState, writer: StreamWriter
    ) -> DialogConversation:
        message = UserMessage(
            content=prompts.QUERY_BY_EXPERT.render(
                section_topic=state.section.title,
                section_description=state.section.description,
                conversation=state.conversation_context,
            )
        )
        question = await self.dialog_question_program.aprocess(
            message=message,
        )

        writer(
            StreamExpertEvent(
                data=ExpertData(
                    section_id=state.section.id,
                    token=f"\n\n---\n\n### Question: {question.content}\n\n",
                )
            )
        )

        return DialogConversation(
            question=question,
        )

    async def should_continue(
        self, state: DialogState, writer: StreamWriter
    ) -> Literal["no", "yes"]:
        num_messages = len(state.conversations)
        if num_messages >= self.settings.max_messages:
            logger.warning(
                "Reached max messages %d/%d of conversations",
                num_messages,
                self.settings.max_messages,
            )
            return "no"

        return "yes"

    async def answer_question(
        self, state: DialogState, writer: StreamWriter
    ) -> DialogConversation:
        if state.question is None:
            raise ValueError("Question is not set")

        retrieval_results: ScoredChunks = await self.websearch.asearch(
            query=state.question.content,
            topk=self.settings.topk,
        )

        full_text = ""
        async for event in self.chat_model.astream(
            message=UserMessage(
                content=prompts.ANSWER_BY_EXPERT.render(
                    user_query=state.question.content,
                    context=retrieval_results.context,
                )
            ),
        ):
            full_text += str(event.content)
            writer(
                StreamExpertEvent(
                    data=ExpertData(
                        section_id=state.section.id,
                        token=str(event.content),
                    )
                )
            )

        conversations = [
            UserMessage(content=state.question.content),
            AssistantMessage(
                content=full_text,
            ),
        ]

        return DialogConversation(
            conversations=conversations,
            search_results=[retrieval_results],
        )

    async def summarize(self, state: DialogState, writer: StreamWriter) -> DialogOutput:
        # Prevent exceeding the max token limit
        loop = asyncio.get_event_loop()
        splitted_tokens = await loop.run_in_executor(
            self.executor_split_tokens,
            self.split_tokens,
            self.settings.encoding_model_name,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
            state.conversation_context,
        )

        if len(splitted_tokens) == 0:
            raise ValueError("No tokens to summarize")

        writer(
            StreamExpertEvent(
                data=ExpertData(
                    section_id=state.section.id,
                    token="\n\n---\n\n### Summarization:\n",
                )
            )
        )

        full_text = ""
        async for event in self.chat_model.astream(
            message=UserMessage(
                content=prompts.TOPIC_SUMMARIZATION.render(
                    conversation=splitted_tokens[0],
                    section_title=state.section.title,
                )
            ),
        ):
            full_text += str(event.content)
            writer(
                StreamExpertEvent(
                    data=ExpertData(
                        section_id=state.section.id,
                        token=str(event.content),
                    )
                )
            )

        return DialogOutput(
            summarizations=[
                Summarization(
                    section_id=state.section.id,
                    content=full_text,
                )
            ],
        )

    def build_graph(self) -> CompiledGraph:
        return (
            StateGraph(
                DialogState,
                input=DialogInput,
                output=DialogOutput,
            )
            .add_node("update-section", self.update_section)
            .add_node("ask-question", self.ask_question)
            .add_node("answer-question", self.answer_question)
            .add_node("summarize", self.summarize)
            .add_edge("update-section", "ask-question")
            .add_edge("ask-question", "answer-question")
            .add_conditional_edges(
                "answer-question",
                self.should_continue,
                {"no": "summarize", "yes": "ask-question"},
            )
            .add_edge("summarize", END)
            .set_entry_point("update-section")
            .compile()
        )
