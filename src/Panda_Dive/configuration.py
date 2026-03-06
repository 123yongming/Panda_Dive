"""Configuration module for Panda_Dive research system."""

import os
from enum import Enum
from typing import Any, List

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    # TODO: 增加支持搜索的模型以及更多api
    """搜索API枚举类。."""

    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    NONE = "none"


class MCPConfig(BaseModel):
    """MCP配置类."""

    url: str | None = Field(default=None, optional=True)
    tools: List[str] | None = Field(default=None, optional=True)
    auth_required: bool | None = Field(default=False, optional=True)


class Configuration(BaseModel):
    """DeepResearch 全局配置类。."""

    # Researcher config
    search_api: SearchAPI = Field(
        default=SearchAPI.DUCKDUCKGO,
        description="搜索API",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "duckduckgo",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "DuckDuckGo", "value": SearchAPI.DUCKDUCKGO.value},
                    {"label": "None", "value": SearchAPI.NONE.value},
                ],
            }
        },
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of iterations to run the researcher for Research Supervisor.",
            }
        },
    )
    max_react_tool_calls: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 6,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calls to allow the Research Agent to make in a single iteration.",
            }
        },
    )

    mcp_config: MCPConfig | None = Field(
        default=None,
        description="MCP服务配置",
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration",
            }
        },
    )
    mcp_prompt: str | None = Field(
        default=None,
        description="MCP服务提示",
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it.",
            }
        },
    )

    # Retrieval quality config (Phase 1 tunables)
    query_variants: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Number of query variants to generate for retrieval quality enhancement",
            }
        },
    )
    relevance_threshold: float = Field(
        default=0.7,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.7,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Minimum relevance score threshold for retrieved documents",
            }
        },
    )
    rerank_top_k: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10,
                "min": 1,
                "max": 100,
                "step": 1,
                "description": "Number of top documents to return after reranking",
            }
        },
    )
    rerank_weight_source: str = Field(
        default="auto",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "auto",
                "description": "Weighting strategy for source credibility in reranking",
                "options": [
                    {"label": "Auto", "value": "auto"},
                    {"label": "High", "value": "high"},
                    {"label": "Medium", "value": "medium"},
                    {"label": "Low", "value": "low"},
                ],
            }
        },
    )

    # General config
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models",
            }
        },
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research",
            }
        },
    )
    enable_steering: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable human-in-the-loop steering checkpoints after each supervisor round",
            }
        },
    )
    steering_command_prefix: str = Field(
        default="/steer",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "/steer",
                "description": "Command prefix used to provide steering instructions at checkpoint (for example: /steer focus on academic sources)",
            }
        },
    )
    steering_continue_command: str = Field(
        default="/continue",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "/continue",
                "description": "Command used at steering checkpoint to continue without changing direction",
            }
        },
    )
    max_concurrent_research_units: int = Field(
        default=4,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of concurrent research units to run at once",
            }
        },
    )

    # Memory and context config
    memory_enabled: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable long-term memory extraction, retrieval, and injection.",
            }
        },
    )
    memory_namespace_template: str = Field(
        default="memory.owner.{owner}",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "memory.owner.{owner}",
                "description": (
                    "Dot-delimited namespace template for memory persistence and retrieval. "
                    "Supported variables: {owner}, {thread_id}, {topic_hash}."
                ),
            }
        },
    )
    memory_confidence_threshold: float = Field(
        default=0.75,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.75,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Minimum confidence required for writing memory facts.",
            }
        },
    )
    memory_novelty_threshold: float = Field(
        default=0.30,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.30,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Minimum novelty required for memory fact acceptance.",
            }
        },
    )
    memory_retrieval_top_k: int = Field(
        default=8,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8,
                "min": 1,
                "max": 50,
                "step": 1,
                "description": "Top-K memory facts to retrieve before prompt injection.",
            }
        },
    )
    memory_max_injection_tokens: int = Field(
        default=1200,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 1200,
                "min": 100,
                "max": 4000,
                "step": 50,
                "description": "Maximum token budget for memory injection block.",
            }
        },
    )
    memory_recency_half_life_days: int = Field(
        default=14,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 14,
                "min": 1,
                "max": 365,
                "step": 1,
                "description": "Half-life (days) used in memory recency decay ranking.",
            }
        },
    )
    memory_require_citations: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Require source URLs for accepting new memory facts.",
            }
        },
    )
    memory_max_facts_per_namespace: int = Field(
        default=500,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 500,
                "min": 50,
                "max": 5000,
                "step": 50,
                "description": "Maximum number of facts allowed per memory namespace.",
            }
        },
    )
    memory_backend: str = Field(
        default="sqlite",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "sqlite",
                "description": "Storage backend for long-term memory.",
                "options": [
                    {"label": "SQLite", "value": "sqlite"},
                    {"label": "LangGraph Runtime Store", "value": "langgraph_store"},
                ],
            }
        },
    )
    memory_sqlite_path: str = Field(
        default=".memory/memory.sqlite3",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": ".memory/memory.sqlite3",
                "description": "SQLite database path for long-term memory.",
            }
        },
    )
    memory_sqlite_journal_mode: str = Field(
        default="WAL",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "WAL",
                "description": "SQLite journal mode.",
                "options": [
                    {"label": "WAL", "value": "WAL"},
                    {"label": "DELETE", "value": "DELETE"},
                ],
            }
        },
    )
    memory_sqlite_busy_timeout_ms: int = Field(
        default=5000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 5000,
                "min": 100,
                "max": 60000,
                "step": 100,
                "description": "SQLite busy timeout in milliseconds.",
            }
        },
    )
    memory_search_candidates: int = Field(
        default=200,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 200,
                "min": 20,
                "max": 5000,
                "step": 10,
                "description": "Candidate count for lexical BM25 recall.",
            }
        },
    )
    memory_ann_enabled: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable ANN vector retrieval when vectorlite extension is available.",
            }
        },
    )
    memory_ann_max_elements: int = Field(
        default=200000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 200000,
                "min": 1000,
                "max": 2000000,
                "step": 1000,
                "description": "Maximum elements for vectorlite HNSW index.",
            }
        },
    )
    memory_ann_candidates: int = Field(
        default=200,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 200,
                "min": 10,
                "max": 5000,
                "step": 10,
                "description": "Top candidate count retrieved from ANN/linear vector search.",
            }
        },
    )
    memory_rrf_k: int = Field(
        default=60,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 60,
                "min": 1,
                "max": 200,
                "step": 1,
                "description": "RRF rank constant (1 / (k + rank)).",
            }
        },
    )
    memory_rrf_candidate_max: int = Field(
        default=400,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 400,
                "min": 20,
                "max": 10000,
                "step": 10,
                "description": "Maximum candidates kept per retrieval channel before RRF fusion.",
            }
        },
    )
    memory_embedding_enabled: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable embedding-based vector reranking for memory retrieval.",
            }
        },
    )
    memory_embedding_provider: str = Field(
        default="siliconflow_openai_compatible",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "siliconflow_openai_compatible",
                "description": "Embedding provider type for memory vector reranking.",
            }
        },
    )
    memory_embedding_model: str = Field(
        default="BAAI/bge-m3",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "BAAI/bge-m3",
                "description": "Embedding model name for memory vector reranking.",
            }
        },
    )
    memory_embedding_base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "https://api.siliconflow.cn/v1",
                "description": "Embedding API base URL.",
            }
        },
    )
    memory_embedding_api_key: str | None = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Embedding API key. Prefer setting via environment variable.",
            }
        },
    )
    memory_vector_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.6,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Deprecated. Kept for backward compatibility; ignored by RRF retrieval.",
            }
        },
    )

    # Model Configuration
    summarization_model: str = Field(
        default="deepseek-chat",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "deepseek-chat",
                "description": "Model for summarizing research results from Tavily search results",
            }
        },
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model",
            }
        },
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization",
            }
        },
    )
    research_model: str = Field(
        default="deepseek-chat",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "deepseek-chat",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API.",
            }
        },
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model",
            }
        },
    )
    compression_model: str = Field(
        default="deepseek-chat",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "deepseek-chat",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API.",
            }
        },
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model",
            }
        },
    )
    final_report_model: str = Field(
        default="deepseek-chat",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "deepseek-chat",
                "description": "Model for writing the final report from all research findings",
            }
        },
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model",
            }
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "Configuration":
        """Extract Configuration from RunnableConfig."""
        configuration = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {}
        for field_name in field_names:
            configured_value = configuration.get(field_name)
            if configured_value is not None:
                values[field_name] = configured_value
                continue
            env_value = os.environ.get(field_name.upper())
            if env_value is not None:
                values[field_name] = env_value
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic config class."""

        arbitrary_types_allowed = True
