# LangGraph: Orchestration Framework for LLM Applications

## Key Concepts at a Glance

### 1. Two Core Patterns
- **🔄 Workflow Pattern**: Fixed execution path (e.g., RAG workflows)
- **🤖 Agent Pattern**: LLM-determined path (e.g., ReAct agents)

![Workflow & Agent Pattern](./assets/workflows.png)

A trade-off exists between reliability and an agent's controllability.

![Reliability Curve](./assets/reliability-curve.png)

### 2. Building Blocks

![nodes-edges-state](./assets/nodes-edges-state.png)

- **Nodes**: Functions or subgraphs that define actions
- **Edges**: Define the execution flow
- **State**: Shared context passed between nodes

### 3. Essential Features

- **👥 Human-in-the-Loop (HITL)**
    LangGraph supports a checkpointer (`one of the important features of langgraph`) to save and reload the state of each node.

    ```python
    input -> node-a -> [pause for feedback] -> [user feedback] -> [restore at the beginning of node-a] -> node-b -> end
    ```

    [Visualization of checkpointers](https://langchain-ai.github.io/langgraph/concepts/persistence/#get-state-history) implemented behind the scene:

    ![checkpointers](./assets/persistence-checkpointers.png)

- **⚡ Real-time Streaming**
    Supports multiple streaming methods: `values`, `updates`, `custom`, or any combination of these.

    - `updates`: Emits results after the execution of each node.
    - `values`: Emits the shared state whenever it changes.
    - `custom`: Primarily used to stream responses from an LLM provider (especially necessary when using providers other than LangChain).

- **🔌 Provider Agnostic**
    LangGraph focuses solely on orchestration. You can use any LLM provider such as LangChain, LlamaIndex, or OpenAI.

- **⚙️ Parallelization, Subgraphs, and Async Support**
    - A node can be a function or a subgraph.
    - Nodes can execute in parallel automatically.
        - If `node-a -> node-b` and `node-a -> node-c`, then `node-b` and `node-c` can run concurrently.
    - Fully asynchronous-compatible.

### 4. Demo

The script [demo.py](./demo.py) demonstrates:

- Defining a simple graph to understand the basics of `nodes - edges - state`:
    - Graph diagram
    - Difference between streaming events in `graph.astream()`

- Basic Human-in-the-Loop (HITL) support:
    - Checkpointer
    - Interrupt / Resume
    - Usage of [Command](https://langchain-ai.github.io/langgraph/concepts/low_level/#send) / [Send](https://langchain-ai.github.io/langgraph/concepts/low_level/#send) to control graph flow.

- Streaming responses from a specific graph node:
    - Using `stream_mode="custom"`


### 5. Further Read

- [Langgraph Persistence Review](https://langchain-ai.github.io/langgraph/concepts/persistence/)

    * Overview about langgraph persistence layer. When langgraph save the snapshot of nodes, how the persistence layer can help fault-tolerance, ...
    * What is `thread` ? ~ `conversation-id`
    * Checkpoint
        - get-state
        - get-state-history
        - replay
        - update-state ? Can it be used as the alternative method for `Command(resume=<...>)`?
        - time-travel

- [Langgraph examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
