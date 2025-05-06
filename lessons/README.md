# Langgraph: Orchestration Framework for LLM Applications

## Key Concepts at a Glance

### 1. Two Core Patterns
- **🔄 Workflow Pattern**: Fixed execution path (e.g., RAG workflows)
- **🤖 Agent Pattern**: LLM-determined path (e.g., ReAct agents)

![Workflow & Agent Pattern](./assets/workflows.png)

The trade-off between reliability and agent's controlability

![Reliability Curve](./assets/reliability-curve.png)

### 2. Essential Features
- **👥 Human-in-the-Loop (HITL)**

    * Langgraph supports checkpointer to save and reload state of each node.

    ```python
    input -> node-a -> [pause for feedback] -> [user feedback] -> [restore at the beginning of node-a] -> node-b -> end
    ```

- **⚡ Real-time Streaming**
    * Support multiples streaming method: `values`, `updates`, or `custom` or combination of those.

        - `updates`: results after execution of each node.
        - `values`: the shared state whenever it changes.
        - `custom`: mostly use to stream response from LLM provider (it's necessary when use difference provider than Langchain).

- **🔌 Provider Agnostic**
    * Focus on orchestrator level only. We can use arbitrary LLM provider such as Langchain, LLamaindex, or OpenAI.

- **⚙️ Parallelization & Subgraphs & Async**
    * The node can be a function or a subgraph.
    * Automatic parallelly node execution.
        - IF `node-a -> node-b & node-a -> node-c`, then `node-b` and `node-c` can be executed at the same time.
    * Asynchronous compatibility.

### 3. Building Blocks
![nodes-edges-state](./assets//nodes-edges-state.png)

- **Nodes**: Functions/subgraphs defining actions
- **Edges**: Define execution flow
- **State**: Shared context between nodes

### 4. Demo

The below script [demo.py](.demo.py) demonstrates:
- Define the simple graph to grasp basic understanding of `nodes - edges - states`.
    * graph diagram
    * difference between streaming events in `graph.astream()`
- Baisc Human In The Loop (HITL) supports.
    * checkpointer
    * interrupt / resume

- Streaming responses from the specified graph node.
    * `stream_mode=custom`
