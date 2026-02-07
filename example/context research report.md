# **Systematic Investigation of Context in LLM-based Agent Systems**

## Conceptual Overview

In modern LLM-based agent systems, "context" has evolved far beyond the simple notion of conversation history or a static prompt string. It is now understood as the **comprehensive state representation** that enables an agent to perceive its environment, maintain coherence across actions, and reason towards goals. In single-turn prompting, context is merely the input text. However, in agentic systems—characterized by autonomy, tool use, and multi-step reasoning—context serves as the **working memory**, **procedural knowledge**, and **environmental model** all rolled into a dynamic structure that evolves with the agent's lifecycle.

Architecturally, context in agent systems acts as the bridge between the static weights of the LLM and the dynamic, often unstructured, external world. It encompasses not only the user dialogue but also the outcomes of previous tool calls, the internal state of a workflow graph, the collective knowledge of a multi-agent system, and retrieved information from vector databases. The management of this context is the primary bottleneck in scaling agents from simple Q&A bots to complex, long-horizon problem solvers.

Recent research into 2025-2026 architectures identifies a shift from "prompt stuffing" (appending all history to a single context window) to **structured state management**. This shift mirrors the transition from procedural programming to object-oriented state management in traditional software engineering. Agents now utilize distinct memory layers, state channels, and context routers to optimize for the finite token budget of foundation models while maintaining fidelity to the task requirements.

## Taxonomy of Agent Context

A precise taxonomy is required to analyze context mechanisms, as different types of context demand different management strategies. Based on current architectural patterns in frameworks like LangGraph, AutoGen, and research into "agentic memory," context can be classified into four primary dimensions.

### 1. Temporal Taxonomy (Duration)

- **Ephemeral / Short-Term Context:** This is the data held within the immediate working memory of the agent for the duration of a specific task or thread. It typically includes the current turn's user query, the immediate system prompt, and the most recent tool outputs. In graph-based agents, this maps to the current "node state."
- **Episodic / Medium-Term Context:** This spans a session or a specific task execution (episode). It includes the history of interactions, reasoning chains (thoughts), and intermediate artifacts generated during a specific workflow. This context is vital for "few-shot in-context learning" where the agent learns from its own trials within a session.
- **Semantic / Long-Term Context:** This represents knowledge that persists across sessions and tasks. It is often stored in Vector Databases (RAG) or specialized long-term memory stores (like MemGPT). It includes domain-specific facts, user preferences learned over time, and documentation chunks retrieved based on semantic relevance.

### 2. Structural Taxonomy (Representation)

- **Linear (Sequential) Context:** The traditional concatenated string format (e.g., `<System> <User> <Assistant> <Tool>`). While simple, it lacks structure and makes selective updating difficult.
- **Structured / Graph Context:** Increasingly common in workflow agents. Context is represented as a TypedDict, JSON object, or Message Graph. For example, LangGraph utilizes a `State` object that passes specific fields (e.g., `messages`, `next_action`) to specific nodes, rather than a monolithic string.
- **Hierarchical Context:** A nested structure where global context (e.g., system instructions) is distinct from local context (e.g., a specific sub-task being delegated to a worker agent). This allows for "context masking," ensuring sub-agents only see relevant information.

### 3. Functional Taxonomy (Source)

- **Conversational Context:** The dialogue history between the user and the agent (and potentially between agents).
- **Procedural Context:** Definitions of available tools, APIs, and workflow schemas. This includes the JSON schemas used for Function Calling.
- **Retrieval-Augmented Context:** Dynamic context injected into the prompt based on a query against an external index.
- **Environmental Context:** Real-time data feeds, file system states, or sensory inputs that represent the "world" the agent operates in.

### 4. Agentic Taxonomy (Scope)

- **Intra-Agent Context:** State internal to a single agent (its private memory and thought process).
- **Inter-Agent Context:** Information shared between agents in a multi-agent system. This requires synchronization protocols and shared message buses.

## Design Patterns and Mechanisms

This section analyzes the concrete mechanisms used to construct, update, and manage context in modern systems.

### Context Construction and Assembly

The process of transforming raw state into the final input for the LLM is known as **Prompt Assembly** or **Context Construction**.

- **The "Dispatcher" Pattern:** In advanced agents (e.g., those using ReAct or Plan-and-Solve patterns), context is not static. A "dispatcher" mechanism analyzes the current state and determines which subset of the total context is relevant for the *next* LLM call.

- - *Mechanism:* If the agent is in a "coding" phase, the dispatcher prioritizes file contents and error logs. If it switches to "planning," it prioritizes the high-level goal and previous milestones.
  - *Reference:* **Modular Context Construction** [1].

- **State Channels:** In graph-based orchestration (like LangGraph), context flows through specific "channels."

- - *Mechanism:* The `State` is a shared object. Node A writes to `State.tool_output`. Node B reads from `State.tool_output` but might never see `State.user_private_notes`. This explicit typing prevents context leakage and reduces token usage.

- **Token-Aware Retrieval:** Before construction, the system performs a search against long-term memory.

- - *Mechanism:* Systems like MemGPT use a "window context manager" that simulates infinite context by paging relevant sections in and out of the active context window based on attention scores or recency [2].

### Update Mechanisms: State Passing vs. Event Sourcing

How context is modified defines the architecture's reliability and debuggability.

- **Mutable State Passing:** Common in simpler frameworks. A single context object is passed by reference to a tool. The tool modifies the object (e.g., adds a field). The updated object is returned to the LLM.

- - *Critique:* Easy to implement but prone to "state drift" where the history of *how* a change happened is lost.

- **Immutable Event Logs (Event Sourcing):** The agent maintains an append-only log of events (Tool Call, Tool Result, Thought). The "Current Context" is actually a *view* or *reduction* of this log.

- - *Mechanism:* To update context, the agent appends a new message to the log. The LLM always sees a summary or truncated version of this log. This is the dominant pattern in robust multi-agent systems (e.g., OpenAI's Swarm, LangGraph) as it enables full "replay" and debugging of the agent's trajectory.

- **Reflection Loops:** A specific update pattern where the agent generates a "reflection" or "summary" of previous steps.

- - *Mechanism:* Every ![img](https://cdn.nlark.com/yuque/__latex/459f3c80a50b7be28751b0869ef5386a.svg) steps, the agent triggers a summarization call: "Condense the last 10 interactions into a high-level progress update." This update replaces the raw tokens with the summary, functioning as a manual garbage collection mechanism for context.

### Context in Multi-Agent Systems (MAS)

In MAS, context management becomes a distributed systems problem.

- **The "Blackboard" Pattern:** A shared data structure (the blackboard) acts as the global context. All agents have read/write access (or controlled access) to this board.

- - *Architecture:* Agent A writes a partial solution. Agent B sees it, adds to it. Agent C validates it. The LLM prompt for each agent is constructed from the Blackboard + Agent's Private Instructions.

- **Message Passing / Routing:** Agents communicate via direct messages rather than a shared state.

- - *Architecture:* Context is local. When Agent A needs Agent B, it sends a message payload. Agent B's context includes this payload but is isolated from Agent A's full history. This reduces context pollution but requires careful design of the message schema to ensure sufficient information is transferred.

- **Supervisor / Orchestrator Pattern:**

- - *Architecture:* A central "Root" agent maintains the full context. It delegates tasks to "Worker" agents with only a *subset* of the context (just the task definition). The Worker returns a result. The Root merges the result back into the main context. This hierarchical containment is crucial for maintaining coherence in large systems.

## Comparative Analysis: Lifecycle and Management

The lifecycle of context involves birth (creation), growth (accumulation), and death (pruning/forgetting). Effective management is the differentiator between a failing agent and a robust system.

### Context Growth Strategies

Agents naturally accumulate context. Unchecked, this leads to "context overflow" (exceeding token limits) or "lost in the middle" degradation (where the LLM ignores information in the middle of a long context).

- **Raw Accumulation:** Simply appending every interaction.

- - *Result:* Rapidly hits token limits. High latency. Performance degrades sharply past ~10k-20k tokens depending on the model's attention mechanism.

- **Summary Compression:**

- - *Strategy:* Maintain a "running summary." When approaching the limit, trigger a compression step: `New_Summary = LLM(Old_Summary + Recent_Interactions)`.
  - *Trade-off:* Loss of granular detail. If the agent needs to revisit a specific nuance from the compressed history, it may be lost.

- **Sliding Window with Anchors:**

- - *Strategy:* Keep the System Prompt (Anchor) + X most recent messages. Discard the oldest.
  - *Trade-off:* Effective for chat bots, dangerous for long-horizon agents where the initial goal (which might be discarded) is still relevant.

### Pruning, Filtering, and Forgetting

Modern agents are moving towards intelligent context pruning.

- **Recency-Frequency-Monetary (RFM) Analysis:** Borrowed from marketing, applied to context tokens. A score is assigned to context blocks:

- - *Recency:* How recently was this accessed?
  - *Frequency:* How often is this referenced?
  - *Importance (Monetary):* Heuristic importance (e.g., is this the user's goal?).
  - *Mechanism:* Low-scoring blocks are evicted from the context window first.

- **Semantic Invalidation:**

- - *Strategy:* In tool-using agents, if a tool call fails (e.g., "File not found"), the context related to that file path might be marked "invalid" or "stale" and removed from future consideration to prevent repeated errors.

- **Structural Pruning (Graph-based):**

- - *Strategy:* In workflow graphs, once a node completes and its output is consumed by the next node, the intermediate context (internal reasoning of that node) can be discarded if it is not needed for the final output. This is "node-level garbage collection."

## Impact on Capabilities and Trade-offs

### Impact on Reasoning and Planning

Context availability directly constrains the "computational budget" of the reasoning process.

- **Chain-of-Thought (CoT) Depth:** Longer context allows for deeper, more multi-step reasoning chains. However, without management, the LLM can get stuck in loops. Agents that manage context by summarizing previous steps can effectively "reset" the working memory, allowing for effectively infinite reasoning depth.
- **Planning:** Hierarchical planners require context to store the high-level plan. If the "plan" sub-context is overwritten or pruned, the agent loses its roadmap. Successful agents (like those using Plan-and-Solve) often treat the "Plan" as immutable context stored in a separate channel from the "Execution Trace."

### Impact on Tool Use

- **Dynamic Schema Context:** Advanced agents retrieve tool schemas dynamically.

- - *Effect:* If the agent doesn't have the *right* tool descriptions in its context, it cannot solve the task. Context management here involves filtering 100s of available tools down to the 2-3 relevant ones to avoid overwhelming the model.

- **Result Handling:** Agents that fail to manage tool output context (e.g., leaving massive database dumps in the history) quickly run out of space. Successful agents implement "result views" or "summarized outputs" for tools returning large data.

### Critical Trade-offs

| Design Dimension    | Option A                                   | Option B                           | Trade-off Analysis                                           |
| ------------------- | ------------------------------------------ | ---------------------------------- | ------------------------------------------------------------ |
| **Scope**           | Global Context (All agents see everything) | Local Context (Strict data hiding) | *Global* improves coordination and redundancy but increases cost (tokens) and risk of hallucination (noise). *Local* improves efficiency and focus but risks "re-inventing the wheel" or lack of shared ground truth. |
| **Structure**       | Linear String (Chat History)               | Structured State (TypedDict/JSON)  | *Linear* is universal and easy to debug but hard to prune selectively. *Structured* enables surgical updates and complex workflows but requires specialized orchestration frameworks. |
| **Persistence**     | Ephemeral (Session-based)                  | Long-term (Vector Store/RAG)       | *Ephemeral* ensures privacy and freshness but prevents learning across sessions. *Long-term* enables personalization but introduces latency (retrieval) and staleness risks. |
| **Eviction Policy** | FIFO (First-In-First-Out)                  | Semantic/Attention-based Eviction  | *FIFO* is simple and fast but risks deleting critical "anchor" information (e.g., the original user prompt). *Semantic* preserves critical info but requires compute to evaluate relevance (meta-reasoning overhead). |

### Robustness and Failure Modes

- **The "Rambling" Agent:** Caused by poor context updating where the agent repeats its own thoughts. If the agent cannot distinguish between a "thought" and a "completed action" in the context, it loops.
- **The "Amnesiac" Agent:** Caused by aggressive pruning. The agent solves step 1, prunes the context, forgets the goal, and hallucinates a new goal for step 2.
- **Context Injection Attacks:** In tool-using agents, if the tool output is injected into the context without sanitization (e.g., reading a malicious file), the LLM can be prompt-injected by the external environment. Structured context separation (keeping tool outputs in a separate `tool_data` block rather than the main `chat_history`) mitigates this.

## Open Challenges and Research Directions

Despite significant advancements in 2025-2026, several challenges remain at the frontier of agent context research.

### 1. Context as a Code-First Construct

Currently, most context management is heuristic-based (prompt engineering). A major research direction is treating context as a **first-class software object** with defined interfaces, version control, and merge conflict resolution (similar to Git). This would allow for deterministic context replay and debugging.

### 2. Multi-Modal and Streaming Context

Most taxonomies assume text. As agents become multi-modal (processing video, audio, code execution), the "token budget" becomes a heterogenous resource budget. How does an agent prune 10 seconds of video vs. 500 lines of code? Research into **Unified Context Embeddings** is ongoing to solve this cross-modal compression problem.

### 3. Neural Context Management

Instead of rule-based pruning (FIFO, summarization), researchers are investigating **Learned Context Managers**. These are smaller, auxiliary models trained to predict which parts of the history are most likely to be useful for the *future* turn, effectively learning an optimal attention window [3].

### 4. Provenance and Attribution

In complex Retrieval-Augmented Generation (RAG) workflows, context is synthesized from many sources. A significant challenge is maintaining **provenance** at the token level. When the agent produces an answer, identifying exactly which source document or previous tool output led to that specific sentence is critical for trust and verification, but current context representations flatten this source mapping.

### 5. Inter-Agent Context Negotiation

In open Multi-Agent Systems, agents developed by different organizations need to collaborate. They may use different context schemas. Research is needed into **Context Translation Protocols**—standard ways for agents to query "What is your context schema?" and translate their internal state into a format another agent can understand.

## Sources

[1] "The Rise of Generative AI Agents" (Survey of Architectural Patterns): https://arxiv.org/abs/2408.01969
[2] MemGPT: "Towards LLMs as Operating Systems": https://arxiv.org/abs/2310.08639
[3] "Leave No Context Behind: Efficient Infinite Context Transformer with Infini-attention": https://arxiv.org/abs/2404.07243
[4] LangGraph Documentation: https://langchain-ai.github.io/langgraph/
[5] AutoGen: "Enabling Next-Gen LLM Applications": https://arxiv.org/abs/2308.08155