from dataclasses import dataclass
import logging
from typing import Any, Literal

from jinja2 import Template
from langgraph.types import Command, StreamWriter

from agent.chats.interface import IChatModel
from agent.embeddings.interface import IEmbeddingModel
from agent.models.document import ScoredChunks
from agent.models.messages import UserMessage
from agent.models.stream import StreamChatData, StreamChunksData
from agent.storages.vectordb.milvus import Milvus
from .base import BaseNode
from .models import State, Nodes


logger = logging.getLogger(__name__)


@dataclass
class FAQSettings:
    top_k: int = 10
    temperature: float = 0.2


class FAQNode(BaseNode[State, Command[Literal[Nodes.END]]]):
    def __init__(
        self,
        chat_model: IChatModel,
        vectordb: Milvus,
        embedding_model: IEmbeddingModel,
        prompt_template: Template,
        settings: FAQSettings | None = None,
    ) -> None:
        self.chat_model = chat_model
        self.vectodb = vectordb
        self.embedding_model = embedding_model
        self.prompt_template = prompt_template
        self.settings = settings or FAQSettings()

    async def process(
        self, state: State, writer: StreamWriter, *_: Any, **__: Any
    ) -> Command[Literal[Nodes.END]]:
        query_embedding = await self.embedding_model.aembedding(
            [str(state.query.content)]
        )
        chunks: ScoredChunks = await self.vectodb.search(
            query=query_embedding[0],
            top_k=self.settings.top_k,
        )

        writer(StreamChunksData(data=chunks))
        prompt_content: str = self.prompt_template.render(
            retrieved_context=chunks.context,
            user_query=state.query.content,
        )

        async for message in self.chat_model.astream(
            UserMessage(content=prompt_content),
            temperature=self.settings.temperature,
            history=state.history,
        ):
            writer(StreamChatData.from_message(str(message.content)))

        return Command(goto=Nodes.END)
