# AI Research Agent

An AI-powered research agent designed to process, analyze, and interact with various data sources using state-of-the-art AI models and vector databases.

## Core Components

| Component | Features |
|-----------|----------|
| Chat Models | • Implementation of various LLM Models |
| Embedding Models | • Implementation of various Embedding Models |
| Vector Storage | • Milvus vector database integration<br>• Support for vector similarity search<br>• Dynamic field configuration<br>• Configurable consistency levels<br>• Batch processing capabilities |
| Search Capabilities | • Hybrid search combining multiple sources<br>• Vector similarity search<br>• Web search integration<br>• Tavily web search integration<br>• DuckDuckGo web search support |
| Programs | • Base program framework for AI operations<br>• Dialog Question Program<br>• Outline Report Program<br>• Queries Decomposition Program |
| Extractors | • PDF document extraction<br>• Support for other document formats |
| Environment Configuration | • OpenAI settings (API keys, endpoints)<br>• Milvus configuration<br>• Web search API settings<br>• Local storage configuration |

## Project Structure

```
agent/
├── chats/           # Chat model implementations
├── embeddings/      # Embedding model implementations
├── extractors/      # Document extraction utilities
├── graphs/          # Graph-based operations
├── models/          # Data models and schemas
├── programs/        # AI program implementations
├── searches/        # Search implementations
├── storages/        # Storage implementations
└── tools/           # Utility tools and helpers
```

## Installation

1. Ensure you have Python 3.12+ installed
2. Install UV package manager:
   ```bash
   pip install uv
   ```

3. Clone the repository:
   ```bash
   git clone <repository-url>
   cd stock
   ```

4. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

5. Install dependencies using UV:
   ```bash
   uv pip install -r requirements.txt
   ```

6. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

7. Configure the required environment variables:
   ```bash
   # OpenAI
   openai_api_key=<your-key>
   openai_azure_endpoint=<your-endpoint>
   openai_api_version=2024-08-01-preview
   openai_chat_deployment_name=gpt-4.1-nano
   openai_embedding_deployment_name=text-embedding-3-small

   # Milvus
   milvus_collection_name=research
   milvus_uri="milvus-lite.db"
   milvus_token=""

   # Tavily
   tavily_api_key=<your-key>
   ```

## Contributing

We welcome contributions to improve the AI Research Agent! Here's how you can help:

1. **Fork the Repository**
   - Create a fork of this repository on GitHub
   - Clone your fork locally

2. **Set Up Development Environment**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   pre-commit install
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**
   - Follow the existing code style and conventions
   - Add or update tests as needed
   - Update documentation to reflect your changes

5. **Code Quality Requirements**
   - All code must pass MyPy type checking:
     ```bash
     mypy agent/ tests/
     ```
   - Additional MyPy dependencies required:
     - pydantic
     - pymilvus
     - tavily-python
     - openai
     - langgraph
   - Code must be formatted using Ruff:
     ```bash
     ruff check --fix .
     ruff format .
     ```
   - Pre-commit hooks will automatically check:
     - Trailing whitespace
     - File endings
     - YAML syntax
     - Large file additions (max 1MB)
     - Type hints (MyPy)
     - Code formatting (Ruff)

6. **Submit a Pull Request**
   - Push your changes to your fork
   - Create a pull request from your branch to our main branch
   - Provide a clear description of the changes
   - Link any relevant issues
   - Ensure all pre-commit checks pass

7. **Code Review**
   - Wait for maintainers to review your PR
   - Make any requested changes
   - Communicate in the PR comments for any questions

### Development Guidelines

- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Follow PEP 8 style guidelines
- Keep functions focused and single-purpose
- Write unit tests for new functionality
- Update documentation when adding new features
