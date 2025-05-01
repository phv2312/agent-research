from functools import cached_property
from pathlib import Path

from agent.programs.queries import QueriesProgram
from agent.searches import (
    TavilyWebSearch,
    DuckduckgoWebSearch,
)
from agent.tools.hybrid.core import HybridSearch

from .chats import IChatModel, OpenAIChatModel
from .embeddings import IEmbeddingModel, SmallOpenAIEmbeddingModel
from .extractors import (
    IExtractor,
    PDFExtractor,
)
from .storages.vectordb import Milvus
from .programs import (
    DialogQuestionProgram,
    OutlineReportProgram,
)

from .env import Env
from .storages.local import Storage


class Container:
    def __init__(self, env: Env | None = None, storage: Storage | None = None) -> None:
        self.env = env or Env()
        self.storage = storage or Storage(imagedir=Path("images"))

    @cached_property
    def chat_model(self) -> IChatModel:
        return OpenAIChatModel(
            api_key=self.env.openai_api_key,
            api_version=self.env.openai_api_version,
            azure_endpoint=self.env.openai_azure_endpoint,
            deployment_name=self.env.openai_chat_deployment_name,
        )

    @cached_property
    def embedding_model(self) -> IEmbeddingModel:
        return SmallOpenAIEmbeddingModel(
            api_key=self.env.openai_api_key,
            api_version=self.env.openai_api_version,
            azure_endpoint=self.env.openai_azure_endpoint,
            deployment_name=self.env.openai_embedding_deployment_name,
        )

    @cached_property
    def pdf_extractor(self) -> IExtractor:
        return PDFExtractor(self.storage)

    @cached_property
    def milvus(self) -> Milvus:
        return Milvus(
            uri=self.env.milvus_uri,
            collection_name=self.env.milvus_collection_name,
        )

    @cached_property
    def tavily_websearch(self) -> TavilyWebSearch:
        return TavilyWebSearch(
            api_key=self.env.tavily_api_key,
        )

    @cached_property
    def duckduckgo_websearch(self) -> DuckduckgoWebSearch:
        return DuckduckgoWebSearch()

    @cached_property
    def hybrid_search(self) -> HybridSearch:
        return HybridSearch(
            websearch=self.tavily_websearch,
            milvus=self.milvus,
            embedding_model=self.embedding_model,
        )

    @cached_property
    def outline_report_program(self) -> OutlineReportProgram:
        return OutlineReportProgram(
            api_key=self.env.openai_api_key,
            api_version=self.env.openai_api_version,
            azure_endpoint=self.env.openai_azure_endpoint,
            deployment_name=self.env.openai_chat_deployment_name,
        )

    @cached_property
    def queries_decomposition_program(self) -> QueriesProgram:
        return QueriesProgram(
            api_key=self.env.openai_api_key,
            api_version=self.env.openai_api_version,
            azure_endpoint=self.env.openai_azure_endpoint,
            deployment_name=self.env.openai_chat_deployment_name,
        )

    @cached_property
    def dialog_question_program(self) -> DialogQuestionProgram:
        return DialogQuestionProgram(
            api_key=self.env.openai_api_key,
            api_version=self.env.openai_api_version,
            azure_endpoint=self.env.openai_azure_endpoint,
            deployment_name=self.env.openai_chat_deployment_name,
        )
