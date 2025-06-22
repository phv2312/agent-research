import asyncio
import pandas as pd
import yaml
import json
import logging
import os
from pathlib import Path
import aiofiles
from jinja2 import Template
from pydantic import BaseModel, ValidationError

import gradio as gr
from gradio.components.file import File

from agent.batched import Batched
from agent.programs.impl.evaluation import EvaluationResponse
from agent.container import Container
from agent.env import Env
from agent.models.messages import UserMessage
from applications.conversation_eval.indexing import ConstantIngestor
from applications.conversation_eval.chat import ChatParser, CallParser, ConversationType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


container = None
embedding_model = None
milvus = None
evaluation_program = None
chat_parser = ChatParser()
call_parser = CallParser()
ingestor = ConstantIngestor()

topk = 3
batch_size = 5
prompt_path = "agent/prompts/conversation_eval/groundedness.md"
with open(prompt_path, "r") as file:
    prompt_template = Template(file.read())


class EvaluationError(BaseModel):
    error: str


class ConstantsFile(BaseModel):
    rules: dict[str, list[str]]

    @classmethod
    async def from_file(cls, file_path: Path) -> "ConstantsFile":
        async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
            content = await file.read()

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {str(e)}")

        return cls(rules=data)


def load_environment(
    openai_key: str,
    openai_endpoint: str,
    openai_version: str,
    chat_deployment: str,
    embedding_deployment: str,
    milvus_collection: str,
    milvus_uri: str,
    milvus_token: str,
):
    global container, embedding_model, milvus, evaluation_program
    try:
        container = Container(
            Env(
                tavily_api_key="",
                milvus_collection_name=milvus_collection,
                milvus_uri=milvus_uri,
                milvus_token=milvus_token,
                openai_api_key=openai_key,
                openai_azure_endpoint=openai_endpoint,
                openai_api_version=openai_version,
                openai_chat_deployment_name=chat_deployment,
                openai_embedding_deployment_name=embedding_deployment,
            )
        )
        embedding_model = container.embeddings.get("azure_openai")
        milvus = container.vectordbs.get("milvus")
        evaluation_program = container.programs.get("evaluation")

        return "✅ Services initialized successfully"
    except ValidationError as e:
        return f"❌ Invalid configuration: {e}"
    except Exception as e:
        return f"❌ Error loading environment: {str(e)}"


async def index_constants_file(filepath: Path) -> str:
    try:
        if not all([embedding_model, milvus]):
            return (
                "❌ Services not initialized. Please configure environment first.",
                "",
            )

        # mostly validate
        await ConstantsFile.from_file(filepath)

        chunks = await ingestor.ingest(filepath=filepath)
        logger.info("Created %d chunks from constants", len(chunks))

        embeddings = await embedding_model.aembedding([chunk.text for chunk in chunks])
        await milvus.add(chunks, embeddings)

        success_msg = f"✅ Successfully indexed {len(chunks)} constants from the file"
        return success_msg
    except ValidationError as e:
        error_msg = "\n".join([f"- {error['msg']}" for error in e.errors()])
        return False, f"❌ Validation failed:\n{error_msg}"
    except Exception as e:
        logger.error("Error indexing file: %s", str(e))
        return f"❌ Error indexing file: {str(e)}"


async def index_file_wrapper(file: File):
    if file is None:
        return "❌ Please upload a file", ""

    return await index_constants_file(Path(file.name))


async def analyze_conversation(
    conversation_text: str, conversation_type: ConversationType, placeholder_json: str
) -> list[EvaluationResponse | EvaluationError]:
    temp_file = Path("/tmp/temp_conversation.txt")
    async with aiofiles.open(temp_file, "w", encoding="utf-8") as file:
        await file.write(conversation_text)

    try:
        if not all([embedding_model, milvus, evaluation_program]):
            return [
                {
                    "Error": "Services not initialized. Please configure environment first."
                }
            ]

        try:
            placeholder = (
                json.loads(placeholder_json) if placeholder_json.strip() else {}
            )
        except json.JSONDecodeError:
            logger.warning(
                "Invalid JSON format for placeholders, using empty dictionary."
            )
            placeholder = {}

        match conversation_type:
            case ConversationType.chat:
                logger.info("Parsing chat conversation")
                mp_messages = await chat_parser.parse(temp_file)
            case ConversationType.call:
                logger.info("Parsing call conversation")
                mp_messages = await call_parser.parse(temp_file)

        responses: list[EvaluationResponse] = []
        counter = 1
        num_ai_messages = len(mp_messages["assistant"])
        for ai_contents in Batched.iter(
            mp_messages["assistant"], batch_size=batch_size
        ):
            embeddings = await embedding_model.aembedding(ai_contents)

            retrieved_chunks_list = await asyncio.gather(
                *[milvus.search(embedding, top_k=topk) for embedding in embeddings]
            )

            tasks: list[asyncio.Task[EvaluationResponse]] = []
            for ai_content, retrieved_chunks in zip(ai_contents, retrieved_chunks_list):
                constants = [
                    (
                        scored_chunk.chunk.metadata.rendered_page_path,
                        scored_chunk.chunk.text,
                    )
                    for scored_chunk in retrieved_chunks.root
                ]

                prompt_content = prompt_template.render(
                    bot_message=ai_content,
                    placeholders=json.dumps(placeholder, indent=2),
                    style_constants=constants,
                )

                tasks.append(
                    evaluation_program.aprocess(
                        message=UserMessage(content=prompt_content)
                    )
                )
            responses.extend(await asyncio.gather(*tasks))
            gr.Info(
                "[%.3d/%.3d]Evaluated messages"
                % (counter * batch_size, num_ai_messages),
                duration=2,
            )
            counter += 1

        return responses
    except Exception as e:
        logger.error("Error analyzing conversation: %s", str(e))
        return [EvaluationError(error=str(e))]
    finally:
        temp_file.unlink(missing_ok=True)


async def analyze_conversation_wrapper(
    conversation_text: str, conversation_type: ConversationType, placeholder_json: str
) -> tuple[str, pd.DataFrame | None]:
    if not conversation_text.strip():
        return "❌ Please provide conversation text", None

    try:
        results = await analyze_conversation(
            conversation_text, conversation_type, placeholder_json
        )

        if len(results) == 0:
            return (
                "❌ No results found. Please check your conversation text, types and constants.",
                None,
            )

        match results[0]:
            case EvaluationError():
                error_msg = results[0].error
                return f"❌ Error: {error_msg}", None
            case EvaluationResponse():
                df = pd.DataFrame([result.model_dump() for result in results])
                return f"✅ Analyzed {len(results)} assistant messages", df

    except Exception as e:
        return f"❌ Error: {str(e)}", None


async def load_from_env_file() -> tuple[str, ...]:
    """Load configuration from .env file and return field values."""
    try:
        env = Env()
        return (
            env.openai_api_key or "",
            env.openai_azure_endpoint or "",
            env.openai_api_version or "",
            env.openai_chat_deployment_name or "",
            env.openai_embedding_deployment_name or "",
            env.milvus_collection_name or "",
            env.milvus_uri or "",
            env.milvus_token or "",
        )
    except Exception as e:
        logger.error("Failed to load from .env file: %s", str(e))
        return ("", "", "", "", "", "", "", "")


with gr.Blocks(title="Conversation Evaluation Tool", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Conversation Evaluation Tool")
    gr.Markdown(
        "Configure environment, upload constants files and analyze conversations for compliance and quality."
    )

    with gr.Tabs():
        with gr.Tab("⚙️ Environment"):
            gr.Markdown("## Environment Configuration")
            gr.Markdown(
                "Configure your Azure OpenAI and Milvus connections based on your environment settings."
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### OpenAI Settings")
                    openai_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="Your OpenAI API key",
                        value=os.getenv("OPENAI_API_KEY", ""),
                    )
                    openai_endpoint_input = gr.Textbox(
                        label="OpenAI Azure Endpoint",
                        placeholder="https://your-resource.openai.azure.com/",
                        value=os.getenv("OPENAI_AZURE_ENDPOINT", ""),
                    )
                    openai_version_input = gr.Textbox(
                        label="OpenAI API Version",
                        placeholder="2024-08-01-preview",
                        value=os.getenv("OPENAI_API_VERSION", "2024-08-01-preview"),
                    )
                    chat_deployment_input = gr.Textbox(
                        label="Chat Deployment Name",
                        placeholder="gpt-4.1-mini",
                        value=os.getenv("OPENAI_CHAT_DEPLOYMENT_NAME", ""),
                    )
                    embedding_deployment_input = gr.Textbox(
                        label="Embedding Deployment Name",
                        placeholder="text-embedding-3-small",
                        value=os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME", ""),
                    )

                with gr.Column():
                    gr.Markdown("### Milvus Settings")
                    milvus_collection_input = gr.Textbox(
                        label="Milvus Collection Name",
                        placeholder="conversation_constants",
                        value=os.getenv("MILVUS_COLLECTION_NAME", ""),
                    )
                    milvus_uri_input = gr.Textbox(
                        label="Milvus URI",
                        placeholder="http://localhost:19530",
                        value=os.getenv("MILVUS_URI", ""),
                    )
                    milvus_token_input = gr.Textbox(
                        label="Milvus Token",
                        placeholder="Optional authentication token",
                        type="password",
                        value=os.getenv("MILVUS_TOKEN", ""),
                    )

            with gr.Row():
                load_env_file_btn = gr.Button(
                    "Load from .env File", variant="secondary"
                )
                load_env_btn = gr.Button(
                    "Load Environment & Initialize Services", variant="primary"
                )

            env_status = gr.Textbox(
                label="Status",
                interactive=False,
                value="Please configure and load environment to use the application.",
            )

            load_env_file_btn.click(
                fn=load_from_env_file,
                outputs=[
                    openai_key_input,
                    openai_endpoint_input,
                    openai_version_input,
                    chat_deployment_input,
                    embedding_deployment_input,
                    milvus_collection_input,
                    milvus_uri_input,
                    milvus_token_input,
                ],
            )

            load_env_btn.click(
                fn=load_environment,
                inputs=[
                    openai_key_input,
                    openai_endpoint_input,
                    openai_version_input,
                    chat_deployment_input,
                    embedding_deployment_input,
                    milvus_collection_input,
                    milvus_uri_input,
                    milvus_token_input,
                ],
                outputs=[env_status],
            )

        with gr.Tab("📁 Indexing"):
            gr.Markdown("## Upload Constants File")
            gr.Markdown(
                "Upload a YAML file containing conversation constants/templates."
            )

            with gr.Row():
                with gr.Column():
                    file_upload = gr.File(
                        label="Upload YAML File",
                        file_types=[".yaml", ".yml"],
                        type="filepath",
                    )
                    index_btn = gr.Button("Index File", variant="primary")

                with gr.Column():
                    index_status = gr.Textbox(label="Status", interactive=False)

            index_btn.click(
                fn=index_file_wrapper, inputs=[file_upload], outputs=[index_status]
            )

        with gr.Tab("💬 Conversation Analysis"):
            gr.Markdown("## Analyze Conversation")
            gr.Markdown(
                "Paste conversation text and analyze it against indexed constants."
            )

            with gr.Row():
                with gr.Column():
                    conversation_input = gr.Textbox(
                        label="Conversation Text",
                        placeholder=(
                            "Paste your conversation here...\n\nFor Chat format:\nUser: Hello\nAssistant: "
                            "Hi there!\n\nFor Call format:\n[2024-01-01 10:00] Agent: Hello\n[2024-01-01 10:01] Human: Hi"
                        ),
                    )

                    conversation_type = gr.Radio(
                        choices=[ConversationType.chat, ConversationType.call],
                        label="Conversation Type",
                        value=ConversationType.chat,
                    )

                    placeholder_input = gr.Textbox(
                        label="Placeholders (JSON)",
                        placeholder=(
                            '{"customer_pronoun": "Mr", "full_name": "John Cena", "customer_name": "John Cena", '
                            '"customer_first_name": "John"}'
                        ),
                        lines=3,
                    )

                    analyze_btn = gr.Button("Analyze Conversation", variant="primary")

                with gr.Column():
                    analysis_status = gr.Textbox(
                        label="Analysis Status", interactive=False
                    )

                    download_btn = gr.DownloadButton(
                        label="Download Results as Excel", visible=False
                    )

            results_df = gr.Dataframe(
                label="Analysis Results", visible=False, wrap=True
            )

            async def update_ui_after_analysis(
                conversation_text: str,
                conversation_type: ConversationType,
                placeholder_json: str,
            ):
                status, df = await analyze_conversation_wrapper(
                    conversation_text, conversation_type, placeholder_json
                )

                if df is not None:
                    pathout = "conv_analysis.xlsx"
                    df.to_excel(
                        pathout, sheet_name="Conversation_Analysis", index=False
                    )
                    return (
                        status,
                        gr.Dataframe(
                            value=df,
                            visible=True,
                            show_search="search",
                        ),
                        gr.DownloadButton(value=pathout, visible=True),
                    )
                else:
                    return (
                        status,
                        gr.Dataframe(visible=False),
                        gr.DownloadButton(visible=False),
                    )

            analyze_btn.click(
                fn=update_ui_after_analysis,
                inputs=[conversation_input, conversation_type, placeholder_input],
                outputs=[analysis_status, results_df, download_btn],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
