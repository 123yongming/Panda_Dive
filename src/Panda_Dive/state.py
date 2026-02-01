"""定义状态节点以及structured数据"""

import operator
from typing import Annotated

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

"""
    状态节点定义
"""


def override_reducer(current_value, new_value):
    """状态节点reducer"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)


class AgentInputState(MessagesState):
    """通用agent输入状态节点
    已经继承了messagesstate中的消息
    """


class AgentState(MessagesState):
    """通用agent状态节点
    1、message
    2、reserach data
    """

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str | None
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str


class SupervisorState(MessagesState):
    """supervisor状态节点"""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0


class ResearcherState(MessagesState):
    """researcher状态节点"""

    researcher_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_topic: str
    tool_call_iterations: int = 0
    compress_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    rewritten_queries: Annotated[list[str], override_reducer] = []
    relevance_scores: Annotated[list[dict[str, float]], override_reducer] = []
    reranked_results: Annotated[list[str], override_reducer] = []
    quality_notes: Annotated[list[str], override_reducer] = []


class ResearcherOutputState(MessagesState):
    """researcher输出状态节点, to: Supervisor"""

    compress_research: str
    raw_notes: Annotated[list[str], override_reducer] = []


"""
    LLM对应的structured output and tool
    调用方法：
        model.with_structured_output(ClassName)
        model.bind_tools([ClassName1, ClassName2])
"""


class ConductResearch(BaseModel):
    """tool,研究执行计划"""

    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )


class ResearchComplete(BaseModel):
    """tool,研究完成"""


class Summary(BaseModel):
    """研究总结，绑定summarize_webpage_prompt"""

    summary: str
    key_excerpts: str


class ClarifyWithUser(BaseModel):
    """澄清用户需求（澄清模型输出）"""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )


class ResearchQuestion(BaseModel):
    """研究简报，相当于用户问题的改写（简报模型输出）"""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )
