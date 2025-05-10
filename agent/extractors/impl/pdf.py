import asyncio
from concurrent.futures import Executor, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pymupdf
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_text_splitters import TokenTextSplitter

from agent.batched import Batched

from ...storages.local import Storage
from ...models.document import Chunk, Document, DocumentMetadata


class PDFExtractorSettings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True)
    chunk_size: int = Field(default=1024, alias="pdf_extractor_chunk_size")
    chunk_overlap: int = Field(default=256, alias="pdf_extractor_chunk_overlap")
    encoding_model_name: str = Field(
        default="gpt-4o", alias="pdf_extractor_encoding_model_name"
    )

    number_executor_split_tokens: int = Field(
        default=2, alias="pdf_extractor_number_executor_split_tokens"
    )
    batch_size: int = Field(default=2, alias="pdf_extractor_batch_size")


class PDFExtractor:
    def __init__(
        self,
        storage: Storage,
        settings: PDFExtractorSettings | None = None,
        executor_split_tokens: Executor | None = None,
    ):
        self.storage = storage
        self.settings = settings or PDFExtractorSettings()
        self.executor_split_tokens = executor_split_tokens or ProcessPoolExecutor(
            max_workers=self.settings.number_executor_split_tokens
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

    async def asplit_tokens(self, text: str) -> list[str]:
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(
            self.executor_split_tokens,
            self.split_tokens,
            self.settings.encoding_model_name,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
            text,
        )

    async def aextract(self, filepath: Path, *_: Any, **__: Any) -> Document:
        pages_content: list[str] = []
        pages_imagepath: list[str] = []
        with pymupdf.Document(filepath) as document:
            for pageidx, page in enumerate(document, start=1):
                pages_content.append(page.get_text())

                relpath = self.storage.gen_path(
                    reldir=filepath.name, name=f"page{pageidx}"
                )
                self.storage.save_image(page.get_pixmap(), relpath)
                pages_imagepath.append(relpath)

        splitted_texts_list: list[list[str]] = []
        for batched_pages_content in Batched.iter(
            pages_content, batch_size=self.settings.batch_size
        ):
            splitted_texts_list.extend(
                await asyncio.gather(
                    *[
                        self.asplit_tokens(page_content)
                        for page_content in batched_pages_content
                    ]
                )
            )

        chunks = []
        for pageidx, (splitted_texts, imagepath) in enumerate(
            zip(splitted_texts_list, pages_imagepath), start=1
        ):
            chunks.extend(
                [
                    Chunk(
                        text=splitted_text,
                        metadata=DocumentMetadata(
                            filename=str(filepath),
                            pageidx=pageidx,
                            rendered_page_path=imagepath,
                        ),
                    )
                    for splitted_text in splitted_texts
                ]
            )

        return Document(
            filename=str(filepath),
            chunks=chunks,
        )
