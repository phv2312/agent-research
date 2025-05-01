# AI Research Agent

An AI-powered research agent designed to process, analyze, and interact with various data sources using state-of-the-art AI models and vector databases.

## Core Components

| Component | Features |
|-----------|----------|
| Chat Models | • Implementation of various LLM Models |
| Embedding Models | • Implementation of various Embedding Models |
| Vector Storage | • Vector database (Milvus) integration<br>|
| Search Capabilities | • Tavily web search integration<br>• DuckDuckGo web search support |
| Programs | • Structured Component Parser with LLM |
| Extractors | • PDF document extraction<br>• Support for other document formats |
| Environment Configuration | • OpenAI settings (API keys, endpoints)<br>• Milvus configuration<br>• Web search API settings<br>• Local storage configuration |

## Project Structure

```
agent/
├── chats/
├── embeddings/
├── extractors/
├── graphs/
├── models/
├── programs/
├── searches/
├── storages/
└── tools/
```

## Installation

1. Ensure you have **Python 3.12+** installed
2. Install UV package manager:
   ```bash
   pip install uv
   ```

3. Clone the repository:
   ```bash
   git clone https://github.com/phv2312/agent-research.git
   ```

4. Create and activate a virtual environment:
   ```bash
   uv sync --all-groups
   ```

5. Copy `.env.example` to `.env` and fill your own credentials:
   ```bash
   cp .env.example .env
   ```
