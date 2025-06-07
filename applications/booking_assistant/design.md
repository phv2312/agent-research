# Architecture

There're two workflows, chat & indexing:

- **Chat**: route query from user to the corresponding services (FAQ or Ticket Operation).

- **Indexing**: input internal policy, convert document(s) to chunks and insert to database for later FAQ usage.


## Chat Workflow

### Architecture

The overall graph is illustrated:

![chat-graph](./assets/graph.png)

The reponsibility of each component:

- **Coordinator**: route the user request to the corresponding services, be able of handling small talks/ greeting.

- **FAQ**: inheritted by RAG system, which can search for the internal similar documents inside databse, then answer query from user.

- **Operation**: to be honest, it should be named as *after-service* operation, be responsible for parsing the user query to the corresponding sql statements.

- **Operation Feedback**: in case client missing any information makes **Operation** node fail to parse. This node will gather more information from user by asking follow-up query. Inherrited by the Human In The Loop (HITL) system.

### How to support
The current workflow is mostly for text channel. In case of multimodal input, let create another node, called as `multimodal_processor` to help convert image/voice to text beforehand.

![chat-graph-multimodal](./assets/multimodal-graph.png)

## Indexing Workflow

### Architecture

The indexing workflow can be illustrated:

![indexing-workflow](./assets/indexing.png)

The reponsibility of each component:

- **Extractor** & **Text Spiltter**: given unstructured document, convert to chunks with metadata.
- **Embedding**: convert text to vector
- **Insert-DB**: means insert to Database, currenly only milvus is supported.

## Components

| Component | Description | Implementations | Tech Stack & Reasoning |
|:----------|:------------|:----------------|:-----------------------|
| chats | Contains implementations of chat model interfaces and utilities for interacting with LLM chat models like ChatGPT | - OpenAI chat | - OpenAI is the leading provider of advanced LLMs like ChatGPT. |
| embeddings | Houses embedding model implementations for transforming text into vector representations | - OpenAI embeddings | - Same reason with OpenAI Chat. |
| extractors | Contains utilities for extracting information and data from various sources | - PDF Extractor | - Pymupdf: one of the leading frameworks in PDF parser.
| graphs | Implements workflow graphs and node-based processing systems for AI agent operations | - Nodes & Graph for Booking assistant | - Langgraph because it supports many core features: orchestrating, streaming, async native support...  |
| models | Contains Pydantic data models and schemas that define the structure of data used throughout the system | - Message, Stream event | - Pydantic: simply is the best. |
| programs | Houses LLM programs that generate structured data using Pydantic models | - Booking operations | - OpenAI structured output parsing
| storages | Contains storage implementations, possibly including vector database integrations | - Local storage<br>- Milvus | - Milvus is the leading vectorDB, support cluster for large scale, many indexes support and tutotirals, rapid feature development.
| text_splitters | Implements text chunking and splitting utilities for processing large text documents | - Langchain text splitter | - langchain-core because langgraph will install them already.
