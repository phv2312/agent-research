from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Callable, Literal

from agent.programs import (
    IProgram,
    SearchQueriesProgram,
    DialogQuestionProgram,
    OutlineReportProgram,
)
from agent.text_splitters import (
    ITextSplitter,
    LangchainTextSplitter,
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


class BaseProvider[NameT, ReturnT](ABC):
    @property
    def supported_models(self) -> list[NameT]:
        return list(self.mp_name_init.keys())

    @property
    @abstractmethod
    def mp_name_init(self) -> dict[NameT, Callable[[], ReturnT]]:
        raise NotImplementedError()

    def get(self, model_name: NameT) -> ReturnT:
        if model_name not in self.mp_name_init:
            raise ValueError(f"Unknown model name: {model_name}")

        return self.mp_name_init[model_name]()


class ChatProvider(BaseProvider[Literal["azure_openai"], IChatModel]):
    def __init__(self, env: Env) -> None:
        self.env = env

    @property
    def mp_name_init(self) -> dict[Literal["azure_openai"], Callable[[], IChatModel]]:
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


class EmbeddingProvider(BaseProvider[Literal["azure_openai"], IEmbeddingModel]):
    def __init__(self, env: Env) -> None:
        self.env = env

    @property
    def mp_name_init(
        self,
    ) -> dict[Literal["azure_openai"], Callable[[], IEmbeddingModel]]:
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


class TextSplitterProvider(
    BaseProvider[
        Literal["langchain"],
        ITextSplitter,
    ]
):
    def __init__(self, env: Env) -> None:
        self.env = env

    @property
    def mp_name_init(self) -> dict[Literal["langchain"], Callable[[], ITextSplitter]]:
        return {
            "langchain": self.init_langchain_text_splitter,
        }

    @lru_cache(maxsize=1)
    def init_langchain_text_splitter(self) -> LangchainTextSplitter:
        return LangchainTextSplitter()


class ExtractorProvider(
    BaseProvider[
        Literal["pdf"],
        IExtractor,
    ]
):
    def __init__(
        self, env: Env, storage: Storage, text_splitter_provider: TextSplitterProvider
    ) -> None:
        self.env = env
        self.storage = storage
        self.text_splitter_provider = text_splitter_provider

    @property
    def mp_name_init(self) -> dict[Literal["pdf"], Callable[[], IExtractor]]:
        return {
            "pdf": self.init_pdf_extractor,
        }

    @lru_cache(maxsize=1)
    def init_pdf_extractor(self) -> PDFExtractor:
        return PDFExtractor(self.storage, self.text_splitter_provider.get("langchain"))


class VectorDBProvider(
    BaseProvider[
        Literal["milvus"],
        Milvus,
    ]
):
    def __init__(self, env: Env) -> None:
        self.env = env

    @property
    def mp_name_init(self) -> dict[Literal["milvus"], Callable[[], Milvus]]:
        return {
            "milvus": self.init_milvus,
        }

    @lru_cache(maxsize=1)
    def init_milvus(self) -> Milvus:
        return Milvus(
            uri=self.env.milvus_uri,
            collection_name=self.env.milvus_collection_name,
        )


class WebSearchProvider(
    BaseProvider[
        Literal["tavily"],
        IWebSearch,
    ]
):
    def __init__(self, env: Env, text_splitter_provider: TextSplitterProvider) -> None:
        self.env = env
        self.text_splitter_provider = text_splitter_provider

    @property
    def mp_name_init(self) -> dict[Literal["tavily"], Callable[[], IWebSearch]]:
        return {
            "tavily": self.init_tavily_websearch,
        }

    @lru_cache(maxsize=1)
    def init_tavily_websearch(self) -> TavilyWebSearch:
        return TavilyWebSearch(
            self.env.tavily_api_key,
            splitter=self.text_splitter_provider.get("langchain"),
        )


class ProgramProvider(
    BaseProvider[
        Literal["search_queries", "outline_report", "dialog_question"],
        IProgram[Any],
    ]
):
    def __init__(self, env: Env) -> None:
        self.env = env

    @property
    def mp_name_init(
        self,
    ) -> dict[
        Literal["search_queries", "outline_report", "dialog_question"],
        Callable[[], IProgram[Any]],
    ]:
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
            self.text_splitter_provider,
        )

    @cached_property
    def vectordb_provider(self) -> VectorDBProvider:
        return VectorDBProvider(self.env)

    @cached_property
    def websearch_provider(self) -> WebSearchProvider:
        return WebSearchProvider(self.env, self.text_splitter_provider)

    @cached_property
    def text_splitter_provider(self) -> TextSplitterProvider:
        return TextSplitterProvider(self.env)

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
