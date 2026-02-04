"""Main Graph."""

import asyncio
import logging
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .configuration import Configuration
from .prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from .retrieval_quality import (
    _format_search_results,
    _parse_search_results,
    rerank_results,
    rewrite_query_for_retrieval,
    score_retrieval_quality,
)
from .state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from .utils import (
    anthropic_websearch_called,
    create_chat_model,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    supports_structured_output,
    think_tool,
)

# 初始化base 模型
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


async def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["write_research_brief", "__end__"]]:
    """分析用户消息，如果研究范围不明确，则提出澄清问题。.

    该函数判断用户的请求是否需要在继续研究之前进行澄清。
    如果禁用澄清或不需要澄清，则直接进入研究阶段。

    参数:
        state: 当前代理状态，包含用户消息
        config: 运行时配置，包含模型设置和偏好

    返回:
        Command：要么以澄清问题结束，要么继续撰写研究简报
    """
    # 1、判断是否需要澄清
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")
    # 2、配置模型
    messages = state["messages"]
    model = create_chat_model(
        model_name=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        api_key=get_api_key_for_model(configurable.research_model, config),
        tags=["langsmith:nostream"],
    )

    # 检查是否支持 structured output
    if supports_structured_output(configurable.research_model):
        clarification_model = model.with_structured_output(ClarifyWithUser).with_retry(
            stop_after_attempt=configurable.max_structured_output_retries
        )
    else:
        # 对于不支持 structured output 的模型，使用 JSON 解析
        clarification_model = model.with_retry(
            stop_after_attempt=configurable.max_structured_output_retries
        )
    # 3、调用模型
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

    # 4、处理模型输出
    if supports_structured_output(configurable.research_model):
        # 支持 structured output 的模型，直接使用响应
        if response.need_clarification:
            return Command(
                goto=END, update={"messages": [AIMessage(content=response.question)]}
            )
        else:
            return Command(
                goto="write_research_brief",
                update={"messages": [AIMessage(content=response.verification)]},
            )
    else:
        # 不支持 structured output 的模型，解析 JSON 响应
        import json

        try:
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            parsed = json.loads(response_text)
            need_clarification = parsed.get("need_clarification", False)

            if need_clarification:
                return Command(
                    goto=END,
                    update={
                        "messages": [AIMessage(content=parsed.get("question", ""))]
                    },
                )
            else:
                return Command(
                    goto="write_research_brief",
                    update={
                        "messages": [AIMessage(content=parsed.get("verification", ""))]
                    },
                )
        except json.JSONDecodeError:
            # JSON 解析失败，默认不需要澄清
            return Command(
                goto="write_research_brief",
                update={
                    "messages": [
                        AIMessage(
                            content=str(response.content)
                            if hasattr(response, "content")
                            else str(response)
                        )
                    ]
                },
            )


async def write_research_brief(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["research_supervisor"]]:
    """将用户消息转化为结构化研究简报并初始化监督者。.

    该函数分析用户消息并生成聚焦的研究简报，
    用于指导研究监督者。同时设置初始监督者
    上下文，包含适当的提示与指令。

    参数:
        state: 当前代理状态，包含用户消息
        config: 运行时配置，包含模型设置

    返回:
        Command：继续前往研究监督者并携带已初始化的上下文
    """
    # 1、配置简报模型
    configurable = Configuration.from_runnable_config(config)
    model = create_chat_model(
        model_name=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        api_key=get_api_key_for_model(configurable.research_model, config),
        tags=["langsmith:nostream"],
    )

    # 检查是否支持 structured output
    if supports_structured_output(configurable.research_model):
        research_model = model.with_structured_output(ResearchQuestion).with_retry(
            stop_after_attempt=configurable.max_structured_output_retries
        )
    else:
        research_model = model.with_retry(
            stop_after_attempt=configurable.max_structured_output_retries
        )
    # 2、调用模型
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])), date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])

    # 3、处理响应
    if supports_structured_output(configurable.research_model):
        # 支持 structured output 的模型，直接使用响应
        research_brief = response.research_brief
    else:
        # 不支持 structured output 的模型，解析 JSON 响应
        import json

        try:
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            parsed = json.loads(response_text)
            research_brief = parsed.get("research_brief", response_text)
        except json.JSONDecodeError:
            # JSON 解析失败，直接使用响应文本
            research_brief = (
                str(response.content) if hasattr(response, "content") else str(response)
            )

    # 4、使用研究简报初始化supervisor模型
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations,
    )
    return Command(
        goto="research_supervisor",
        update={
            "research_brief": research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=research_brief),
                ],
            },
        },
    )


async def supervisor(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor_tools"]]:
    """主导研究监督者，负责规划研究策略并委派任务给研究人员。.

    监督者分析研究简报，决定如何将研究分解为可管理的任务。它可以使用 think_tool 进行战略规划，
    使用 ConductResearch 将任务委派给子研究人员，或者在满意研究结果时使用 ResearchComplete。

    参数:
        state: 当前监督者状态，包含消息和研究上下文
        config: 运行时配置，包含模型设置

    返回:
        Command: 继续前往 supervisor_tools 进行工具执行
    """
    # 1、配置监管者模型
    configurable = Configuration.from_runnable_config(config)
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    research_model = (
        create_chat_model(
            model_name=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            tags=["langsmith:nostream"],
        )
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    # 2、调用模型
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    # 3、更新状态以及执行工具
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


async def supervisor_tools(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor", "__end__"]]:
    """执行监督者调用的工具，包括研究委派和战略思考。.

    此函数处理三种类型的监督者工具调用：
    1. think_tool - 继续对话的战略反思
    2. ConductResearch - 将研究任务委派给子研究人员
    3. ResearchComplete - 标志研究阶段完成

    参数:
        state: 当前监督者状态，包含消息和迭代计数
        config: 运行时配置，包含研究限制和模型设置

    返回:
        命令，继续监督循环或结束研究阶段
    """
    # 1、检查当前状态以及退出条件
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    # 退出标准
    # 1. 超过最大迭代次数
    # 2. 没有工具调用
    # 3. 研究完成工具调用
    exceeded_allowed_iterations = (
        research_iterations > configurable.max_researcher_iterations
    )
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
            },
        )
    # 2、处理所有的工具调用
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    # 处理think_tool
    think_tool_calls = [
        tool_call
        for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(
            ToolMessage(
                content=f"Reflection recorded: {reflection_content}",
                name="think_tool",
                tool_call_id=tool_call["id"],
            )
        )
    # 处理 ConductResearch 工具调用
    conduct_research_calls = [
        tool_call
        for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]
    if conduct_research_calls:
        try:
            allowed_conduct_research_calls = conduct_research_calls[
                : configurable.max_concurrent_research_units
            ]
            overflow_conduct_research_calls = conduct_research_calls[
                configurable.max_concurrent_research_units :
            ]
            research_tasks = [
                researcher_subgraph.ainvoke(
                    {
                        "researcher_messages": [
                            HumanMessage(content=tool_call["args"]["research_topic"])
                        ],
                        "research_topic": tool_call["args"]["research_topic"],
                    },
                    config,
                )
                for tool_call in allowed_conduct_research_calls
            ]
            tool_results = await asyncio.gather(*research_tasks)
            for observation, tool_call in zip(
                tool_results, allowed_conduct_research_calls
            ):
                all_tool_messages.append(
                    ToolMessage(
                        content=observation.get(
                            "compressed_research",
                            "Error synthesizing research report: Maximum retries exceeded",
                        ),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(
                    ToolMessage(
                        content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                        name="ConductResearch",
                        tool_call_id=overflow_call["id"],
                    )
                )
            raw_notes_concat = "\n".join(
                [
                    "\n".join(observation.get("raw_notes", []))
                    for observation in tool_results
                ]
            )

            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
        except Exception as e:
            # 任何错误都会跳出supervisor-researcher子图执行，直接生成final report
            # debug
            logging.exception("supervisor_tools: researcher_subgraph failed")
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token limit exceeded or other error - end research phase
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", ""),
                    },
                )
            all_tool_messages.append(
                ToolMessage(
                    content=f"Error running ConductResearch: {e}",
                    name="ConductResearch",
                    tool_call_id=tool_call["id"],
                )
            )
    # 3、更新状态，重回supervisor
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(goto="supervisor", update=update_payload)


# Supervisor Graph构建
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()

# 可视化查看supervisor子图
# print(supervisor_subgraph.get_graph().draw_ascii())
# print(supervisor_subgraph.get_graph().draw_mermaid())


async def researcher(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher_tools"]]:
    """独立的个体研究员，专注于特定主题进行深入研究。.

    该研究员由监督者分配具体研究主题，并使用
    可用工具（搜索、think_tool、MCP 工具）收集全面信息。
    可在搜索之间使用 think_tool 进行战略规划。

    Args:
        state: 当前研究员状态，包含消息与主题上下文
        config: 运行时配置，包含模型设置与工具可用性

    Returns:
        Command 以继续前往 researcher_tools 执行工具
    """
    # 1、加载配置与工具
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get(
        "researcher_messages", []
    )  # supervisor tool传入的topic
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    # 2、配置research 模型
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", date=get_today_str()
    )
    research_model = (
        create_chat_model(
            model_name=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            tags=["langsmith:nostream"],
        )
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    # 3、生成research回复
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
        },
    )


async def execute_tool_safely(tool, args, config):
    """安全执行工具，处理异常情况。.

    Args:
        tool: 要执行的工具对象
        args: 工具调用参数
        config: 运行时配置

    Returns:
        工具执行结果或错误信息字符串
    """
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher", "compress_research"]]:
    """执行研究员调用的工具，包括搜索工具和战略思考。.

    该函数处理各类研究员工具调用：
    1. think_tool - 继续研究对话的战略反思
    2. 搜索工具（tavily_search、web_search）- 信息收集
    3. MCP 工具 - 外部工具集成
    4. ResearchComplete - 标记单个研究任务完成

    参数:
        state: 当前研究员状态，包含消息与迭代计数
        config: 运行时配置，包含研究限制与工具设置

    返回:
        Command：继续研究循环或进入压缩阶段
    """
    # 1、检查当前状态以及退出条件
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    # 如果没有工具调用，停止
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = openai_websearch_called(
        most_recent_message
    ) or anthropic_websearch_called(most_recent_message)
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    # 2、处理工具调用
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
        for tool in tools
    }
    quality_model = create_chat_model(
        model_name=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        api_key=get_api_key_for_model(configurable.research_model, config),
        tags=["langsmith:nostream"],
    )
    tool_calls = most_recent_message.tool_calls
    rewritten_queries: list[str] = []
    relevance_scores: list[dict[str, float]] = []
    reranked_urls: list[str] = []
    quality_notes: list[str] = []

    search_tool_names = {"tavily_search", "web_search", "duckduckgo_search"}
    for tool_call in tool_calls:
        if tool_call["name"] not in search_tool_names:
            continue
        if not tool_call.get("args"):
            tool_call["args"] = {}
        queries = tool_call.get("args", {}).get("queries")
        if not queries:
            continue
        original_query = queries[0]
        rewritten = await rewrite_query_for_retrieval(
            original_query,
            {"queries": queries},
            quality_model,
            config,
        )
        combined_queries = []
        for query in rewritten + list(queries):
            if query not in combined_queries:
                combined_queries.append(query)
        tool_call["args"]["queries"] = combined_queries
        rewritten_queries.extend(rewritten)
        quality_notes.append(
            f"Rewrote query '{original_query}' into {len(rewritten)} variants"
        )
    tool_outputs = []
    tool_execution_pairs = []
    for tool_call in tool_calls:
        tool = tools_by_name.get(tool_call["name"])
        if tool is None:
            tool_outputs.append(
                ToolMessage(
                    content=f"Tool '{tool_call['name']}' not available.",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
            continue
        tool_execution_pairs.append((tool_call, tool))

    tool_execution_tasks = [
        execute_tool_safely(tool, tool_call["args"], config)
        for tool_call, tool in tool_execution_pairs
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    for observation, (tool_call, _tool) in zip(observations, tool_execution_pairs):
        content = observation
        if tool_call["name"] in search_tool_names and isinstance(observation, str):
            parsed_results = _parse_search_results(observation)
            if parsed_results:
                scored_results = await score_retrieval_quality(
                    parsed_results,
                    tool_call.get("args", {}).get("queries", [""])[0],
                    quality_model,
                    config,
                )
                reranked_results = await rerank_results(
                    scored_results,
                    tool_call.get("args", {}).get("queries", [""])[0],
                    quality_model,
                    config,
                )
                content = _format_search_results(reranked_results)
                relevance_scores.extend(
                    [
                        {
                            "url": result.get("url", ""),
                            "score": float(result.get("score", 0.0)),
                        }
                        for result in reranked_results
                    ]
                )
                reranked_urls.extend(
                    [result.get("url", "") for result in reranked_results]
                )
                quality_notes.append(f"Reranked {len(reranked_results)} search results")
        tool_outputs.append(
            ToolMessage(
                content=content,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    # 3、检查退出条件
    exceeded_iterations = (
        state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    )
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )
    if exceeded_iterations or research_complete_called:
        # 满足条件，退出循环到compress
        return Command(
            goto="compress_research",
            update={
                "researcher_messages": tool_outputs,
                "rewritten_queries": rewritten_queries,
                "relevance_scores": relevance_scores,
                "reranked_results": reranked_urls,
                "quality_notes": quality_notes,
            },
        )
    return Command(
        goto="researcher",
        update={
            "researcher_messages": tool_outputs,
            "rewritten_queries": rewritten_queries,
            "relevance_scores": relevance_scores,
            "reranked_results": reranked_urls,
            "quality_notes": quality_notes,
        },
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """压缩并综合研究结果，生成简洁、结构化的摘要。.

    该函数获取研究员工作中积累的所有研究结果、工具输出和 AI 消息，
    并将其提炼为清晰、全面的摘要，同时保留所有重要信息和发现。

    参数:
        state: 当前研究员状态，包含已积累的研究消息
        config: 运行时配置，包含压缩模型设置

    返回:
        字典，包含压缩后的研究摘要和原始笔记
    """
    # 1、配置压缩模型
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = create_chat_model(
        model_name=configurable.compression_model,
        max_tokens=configurable.compression_model_max_tokens,
        api_key=get_api_key_for_model(configurable.compression_model, config),
        tags=["langsmith:nostream"],
    )
    # 2、准备压缩提示
    researcher_messages = state.get("researcher_messages", [])
    researcher_messages.append(
        HumanMessage(content=compress_research_simple_human_message)
    )
    # 3、调用压缩模型
    synthesis_attempts = 0
    max_attempts = 3
    while synthesis_attempts < max_attempts:
        try:
            compression_prompt = compress_research_system_prompt.format(
                date=get_today_str()
            )
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            response = await synthesizer_model.ainvoke(messages)
            raw_notes_content = "\n".join(
                [
                    str(message.content)
                    for message in filter_messages(
                        researcher_messages, include_types=["tool", "ai"]
                    )
                ]
            )
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content],
            }
        except Exception as e:
            synthesis_attempts += 1
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            continue
    # 4、处理失败情况
    raw_notes_content = "\n".join(
        [
            str(message.content)
            for message in filter_messages(
                researcher_messages, include_types=["tool", "ai"]
            )
        ]
    )

    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content],
    }


# Researcher Subgraph 构建
researcher_builder = StateGraph(
    ResearcherState,
    output=ResearcherOutputState,
    config_schema=Configuration,
)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()

# print(researcher_subgraph.get_graph().draw_ascii())
# print(researcher_subgraph.get_graph().draw_mermaid())


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """生成最终综合研究报告。.

    该函数汇总所有已收集的研究发现，使用配置的“报告生成模型”将其综合成结构清晰、内容全面的最终报告。

    参数:
        state: 包含研究发现与上下文的代理状态
        config: 包含模型设置与 API 密钥的运行时配置

    返回:
        字典，包含最终报告与已清空的状态
    """
    # 1、抽取所有发现
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    # 2、配置报告生成模型
    configurable = Configuration.from_runnable_config(config)
    writer_model = create_chat_model(
        model_name=configurable.final_report_model,
        max_tokens=configurable.final_report_model_max_tokens,
        api_key=get_api_key_for_model(configurable.final_report_model, config),
        tags=["langsmith:nostream"],
    )
    # 3、调用报告生成模型
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    while current_retry <= max_retries:
        try:
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str(),
            )
            final_report = await writer_model.ainvoke(
                [HumanMessage(content=final_report_prompt)]
            )
            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state,
            }
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                if current_retry == 1:
                    model_token_limit = get_model_token_limit(
                        configurable.final_report_model
                    )
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [
                                AIMessage(
                                    content="Report generation failed due to token limits"
                                )
                            ],
                            **cleared_state,
                        }
                    findings_token_limit = model_token_limit * 4
                else:
                    findings_token_limit = int(findings_token_limit * 0.9)
                findings = findings[:findings_token_limit]
                continue
            else:
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [
                        AIMessage(content="Report generation failed due to an error")
                    ],
                    **cleared_state,
                }
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [
            AIMessage(content="Report generation failed after maximum retries")
        ],
        **cleared_state,
    }


# 构建Main graph
deep_researcher_builder = StateGraph(
    AgentState, input=AgentInputState, config_schema=Configuration
)

"""
主图
├── clarify_with_user
├── write_research_brief
├── research_supervisor (supervisor_subgraph)
│   ├── supervisor
│   └── supervisor_tools
└── final_report_generation

supervisor_tools 在运行时动态调用:
└── researcher_subgraph
    ├── researcher
    ├── researcher_tools
    └── compress_research
"""

deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

deep_researcher = deep_researcher_builder.compile()
# print(deep_researcher.get_graph().draw_ascii())
# print(deep_researcher.get_graph().draw_mermaid())
