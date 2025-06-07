from pathlib import Path
import time

import logging
import glob
import asyncio

from agent.container import Container


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    container = Container()

    filedir: str = "datas/references/booking"
    for filepath in glob.glob(f"{filedir}/*.pdf"):
        filepath = Path(filepath)
        logger.info("Processing %s", filepath)

        extract_start_time = time.perf_counter()
        document = await container.extractors.get("pdf").aextract(filepath)
        logger.info("Extracting time: %.3f", time.perf_counter() - extract_start_time)

        embed_start_time = time.perf_counter()
        embeddings = await container.embeddings.get("azure_openai").aembedding(
            [chunk.text for chunk in document.chunks]
        )
        logger.info("Embedding time: %.3f", time.perf_counter() - embed_start_time)

        milvus_start_time = time.perf_counter()
        await container.vectordbs.get("milvus").add(document.chunks, embeddings)
        logger.info(
            "Milvus indexing time: %.3f", time.perf_counter() - milvus_start_time
        )


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
