import os
from enum import Enum
from typing import Any, List

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    # TODO: 增加支持搜索的模型以及更多api
    """搜索API枚举类。"""

    TAVILY = "tavily"
    NONE = "none"


class MCPConfig(BaseModel):
    """MCP配置类"""

    url: str | None = Field(default=None, optional=True)
    tools: List[str] | None = Field(default=None, optional=True)
    auth_required: bool | None = Field(default=False, optional=True)


class Configuration(BaseModel):
    """DeepResearch 全局配置类。"""

    # Researcher config
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        description="搜索API",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
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
        configuration = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(
                field_name.upper(), configuration.get(field_name)
            )
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        arbitrary_types_allowed = True
