from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Callable

from agent.programs import (
    IProgram,
    SearchQueriesProgram,
    DialogQuestionProgram,
    OutlineReportProgram,
)
from agent.searches import (
    TavilyWebSearch,
    IWebSearch,
)
from agent.tools.hybrid.core import HybridSearch
from agent.chats import IChatModel, OpenAIChatModel
from agent.embeddings import IEmbeddingModel, SmallOpenAIEmbeddingModel
from agent.extractors import (
    IExtractor,
    PDFExtractor,
)
from agent.storages.vectordb import Milvus
from agent.env import Env
from agent.storages.local import Storage


class BaseProvider[T](ABC):
    @property
    def supported_models(self) -> list[str]:
        return list(self.mp_name_init.keys())

    @property
    @abstractmethod
    def mp_name_init(self) -> dict[str, Callable[[], T]]:
        raise NotImplementedError()

    def get(self, model_name: str) -> T:
        if model_name not in self.mp_name_init:
            raise ValueError(f"Unknown model name: {model_name}")

        return self.mp_name_init[model_name]()


class ChatProvider(BaseProvider[IChatModel]):
    def __init__(self, env: Env) -> None:
        self.env = env

    @property
    def mp_name_init(self) -> dict[str, Callable[[], IChatModel]]:
        return {
            "azure_openai": self.init_azure_openai,
        }

    @lru_cache(maxsize=1)
    def init_azure_openai(self) -> OpenAIChatModel:
        return OpenAIChatModel(
            api_key=self.env.openai_api_key,
            api_version=self.env.openai_api_version,
            azure_endpoint=self.env.openai_azure_endpoint,
            deployment_name=self.env.openai_chat_deployment_name,
        )


class EmbeddingProvider(BaseProvider[IEmbeddingModel]):
    def __init__(self, env: Env) -> None:
        self.env = env

    @property
    def mp_name_init(self) -> dict[str, Callable[[], IEmbeddingModel]]:
        return {
            "azure_openai": self.init_azure_openai,
        }

    @lru_cache(maxsize=1)
    def init_azure_openai(self) -> SmallOpenAIEmbeddingModel:
        return SmallOpenAIEmbeddingModel(
            api_key=self.env.openai_api_key,
            api_version=self.env.openai_api_version,
            azure_endpoint=self.env.openai_azure_endpoint,
            deployment_name=self.env.openai_embedding_deployment_name,
        )


class ExtractorProvider(BaseProvider[IExtractor]):
    def __init__(self, env: Env, storage: Storage) -> None:
        self.env = env
        self.storage = storage

    @property
    def mp_name_init(self) -> dict[str, Callable[[], IExtractor]]:
        return {
            "pdf": self.init_pdf_extractor,
        }

    @lru_cache(maxsize=1)
    def init_pdf_extractor(self) -> PDFExtractor:
        return PDFExtractor(self.storage)


class VectorDBProvider(BaseProvider[Milvus]):
    def __init__(self, env: Env) -> None:
        self.env = env

    @property
    def mp_name_init(self) -> dict[str, Callable[[], Milvus]]:
        return {
            "milvus": self.init_milvus,
        }

    @lru_cache(maxsize=1)
    def init_milvus(self) -> Milvus:
        return Milvus(
            uri=self.env.milvus_uri,
            collection_name=self.env.milvus_collection_name,
        )


class WebSearchProvider(BaseProvider[IWebSearch]):
    def __init__(self, env: Env) -> None:
        self.env = env

    @property
    def mp_name_init(self) -> dict[str, Callable[[], IWebSearch]]:
        return {
            "tavily": self.init_tavily_websearch,
        }

    @lru_cache(maxsize=1)
    def init_tavily_websearch(self) -> TavilyWebSearch:
        return TavilyWebSearch(api_key=self.env.tavily_api_key)


class ProgramProvider(BaseProvider[IProgram[Any]]):
    def __init__(self, env: Env) -> None:
        self.env = env

    @property
    def mp_name_init(self) -> dict[str, Callable[[], IProgram[Any]]]:
        return {
            "search_queries": self.init_search_queries_program,
            "outline_report": self.init_outline_report_program,
            "dialog_question": self.init_dialog_question_program,
        }

    @lru_cache(maxsize=1)
    def init_search_queries_program(self) -> SearchQueriesProgram:
        return SearchQueriesProgram(
            api_key=self.env.openai_api_key,
            api_version=self.env.openai_api_version,
            azure_endpoint=self.env.openai_azure_endpoint,
            deployment_name=self.env.openai_chat_deployment_name,
        )

    @lru_cache(maxsize=1)
    def init_outline_report_program(self) -> OutlineReportProgram:
        return OutlineReportProgram(
            api_key=self.env.openai_api_key,
            api_version=self.env.openai_api_version,
            azure_endpoint=self.env.openai_azure_endpoint,
            deployment_name=self.env.openai_chat_deployment_name,
        )

    @lru_cache(maxsize=1)
    def init_dialog_question_program(self) -> DialogQuestionProgram:
        return DialogQuestionProgram(
            api_key=self.env.openai_api_key,
            api_version=self.env.openai_api_version,
            azure_endpoint=self.env.openai_azure_endpoint,
            deployment_name=self.env.openai_chat_deployment_name,
        )


class Container:
    def __init__(self, env: Env | None = None, storage: Storage | None = None) -> None:
        self.env = env or Env()
        self.storage = storage or Storage(imagedir=Path("images"))

    @cached_property
    def chat_provider(self) -> ChatProvider:
        return ChatProvider(self.env)

    @cached_property
    def embedding_provider(self) -> EmbeddingProvider:
        return EmbeddingProvider(self.env)

    @cached_property
    def extractor_provider(self) -> ExtractorProvider:
        return ExtractorProvider(
            self.env,
            self.storage,
        )

    @cached_property
    def vectordb_provider(self) -> VectorDBProvider:
        return VectorDBProvider(self.env)

    @cached_property
    def websearch_provider(self) -> WebSearchProvider:
        return WebSearchProvider(self.env)

    @cached_property
    def hybrid_search(self) -> HybridSearch:
        return HybridSearch(
            websearch=self.websearch_provider.get("tavily"),
            milvus=self.vectordb_provider.get("milvus"),
            embedding_model=self.embedding_provider.get("azure_openai"),
        )

    @cached_property
    def program_provider(self) -> ProgramProvider:
        return ProgramProvider(self.env)
